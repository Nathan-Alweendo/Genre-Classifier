"""Core library for Project 3. Nothing to run here; the numbered scripts import it.
Own STFT -> mel filterbank -> DCT/MFCC -> clip features, plus data/cache/SVM helpers."""
import time, json
from functools import lru_cache
import numpy as np

FS = 22050
CLIP_SECONDS = 30
CLIP_LEN = FS*CLIP_SECONDS
BASE = dict(window="hann", N=1024, hop=512, n_mels=40)
KEYS = ("window", "N", "hop", "n_mels")
core = lambda cfg: {k: cfg[k] for k in KEYS}

# ---------------- STFT ----------------
def make_window(name, N):
    n = np.arange(N)
    if name == "rect":     return np.ones(N)
    if name == "hann":     return 0.5 - 0.5*np.cos(2*np.pi*n/N)
    if name == "hamming":  return 0.54 - 0.46*np.cos(2*np.pi*n/N)
    if name == "blackman": return 0.42 - 0.5*np.cos(2*np.pi*n/N) + 0.08*np.cos(4*np.pi*n/N)
    raise ValueError(name)

def my_fft(frames, fft_fn=None):
    """frames (n_frames, N) -> (n_frames, N//2+1). fft_fn = your lab FFT (slow, per frame);
    default numpy rfft (same numbers, fast) for the big runs."""
    if fft_fn is None:
        return np.fft.rfft(frames, axis=1)
    N = frames.shape[1]
    return np.array([np.asarray(fft_fn(f))[:N//2+1] for f in frames])

def my_stft(x, N=1024, hop=512, window="hann", center=True, fft_fn=None):
    if center:
        x = np.pad(x, N//2, mode="reflect")
    n_frames = 1 + (len(x) - N)//hop
    idx = np.arange(N)[None, :] + hop*np.arange(n_frames)[:, None]
    frames = x[idx] * make_window(window, N)[None, :]
    return my_fft(frames, fft_fn).T                      # (bins, frames) = [k, m]

# ---------------- mel / MFCC ----------------
hz2mel = lambda f: 2595.0*np.log10(1.0 + f/700.0)
mel2hz = lambda m: 700.0*(10**(m/2595.0) - 1.0)

@lru_cache(maxsize=None)
def mel_filterbank(n_mels, N, fs=FS):
    freqs = np.arange(N//2 + 1)*fs/N
    edges = mel2hz(np.linspace(hz2mel(0.0), hz2mel(fs/2), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for b in range(n_mels):
        lo, mid, hi = edges[b], edges[b+1], edges[b+2]
        fb[b] = np.maximum(0, np.minimum((freqs - lo)/(mid - lo), (hi - freqs)/(hi - mid)))
    return fb

@lru_cache(maxsize=None)
def dct_matrix(n_mfcc, B):
    n = np.arange(n_mfcc)[:, None]; b = np.arange(B)[None, :]
    M = np.cos(np.pi*n*(b + 0.5)/B)*np.sqrt(2.0/B)
    M[0] /= np.sqrt(2.0)
    return M

def log_mel(x, N=1024, hop=512, window="hann", n_mels=40):
    S = np.abs(my_stft(x, N, hop, window))**2
    return 10*np.log10(np.maximum(mel_filterbank(n_mels, N) @ S, 1e-10))   # (n_mels, frames)

def my_mfcc(x, n_mfcc=20, **cfg):
    return dct_matrix(n_mfcc, cfg["n_mels"]) @ log_mel(x, **cfg)

def clip_features(x, **cfg):
    c = my_mfcc(x, **cfg)
    return np.concatenate([c.mean(axis=1), c.std(axis=1)])    # 40-d vector per clip

def empty_filters(n_mels, N):
    return int((mel_filterbank(n_mels, N).sum(axis=1) == 0).sum())

def verify_vs_librosa():
    import librosa
    t = np.arange(FS*3)/FS; rng = np.random.default_rng(0)
    x = np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*3000*t) + 0.05*rng.standard_normal(t.size)
    for w in ["rect", "hann", "hamming", "blackman"]:
        ref = librosa.stft(x, n_fft=1024, hop_length=256, win_length=1024,
                           window="boxcar" if w == "rect" else w, center=True)
        print(f"STFT {w:9s} max abs err vs librosa = {np.abs(my_stft(x,1024,256,w)-ref).max():.2e}")
    fb = librosa.filters.mel(sr=FS, n_fft=1024, n_mels=40, htk=True, norm=None)
    print(f"mel filterbank   max abs err vs librosa = {np.abs(mel_filterbank(40,1024)-fb).max():.2e}")
    S = librosa.feature.melspectrogram(y=x, sr=FS, n_fft=1024, hop_length=512, n_mels=40,
                                       htk=True, norm=None, window="hann")
    ref = librosa.feature.mfcc(S=librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=None),
                               n_mfcc=20, dct_type=2, norm="ortho")
    print(f"MFCC             max abs err vs librosa = {np.abs(my_mfcc(x, **BASE)-ref).max():.2e}")

# ---------------- ablation grid ----------------
def configs():
    """15 unique configs: baseline, window sweep, N sweep (hop=N/2), mel-band sweep, N x hop grid."""
    out = {tuple(BASE.items()): dict(BASE)}
    def add(c): out[tuple(c.items())] = c
    for w in ["rect", "hann", "hamming", "blackman"]: add({**BASE, "window": w})
    for N in (512, 1024, 2048):                         add({**BASE, "N": N, "hop": N//2})
    for b in (20, 40, 64, 128):                         add({**BASE, "n_mels": b})
    for div in (4, 2, 1):
        for N in (512, 1024, 2048):                     add({**BASE, "N": N, "hop": N//div})
    return list(out.values())

# ---------------- data tables ----------------
def load_fma_table(root):
    import pandas as pd
    tr = pd.read_csv(f"{root}/fma_metadata/tracks.csv", index_col=0, header=[0, 1])
    # NB: read_csv gives plain strings, so "<= 'small'" would match ALL subsets alphabetically. Use ==.
    sm = tr[tr["set", "subset"].astype(str) == "small"]
    if len(sm) != 8000 or sm["track", "genre_top"].nunique() != 8:
        print(f"WARNING: fma_small should have 8000 tracks / 8 genres, got {len(sm)} / {sm['track','genre_top'].nunique()}")
    df = pd.DataFrame({"path": [f"{root}/fma_small/{i//1000:03d}/{i:06d}.mp3" for i in sm.index],
                       "genre": sm["track", "genre_top"].values,
                       "split": sm["set", "split"].values})
    return df.dropna().reset_index(drop=True)            # official split

def load_gtzan_table(root, drop_files=(), seed=0):
    """No official split exists: random stratified 80/10/10 by FILE. GTZAN repeats tracks, so
    this LEAKS and results are optimistic. State that in the report."""
    import glob, os, pandas as pd
    rows = [(p, os.path.basename(os.path.dirname(p))) for p in sorted(glob.glob(f"{root}/*/*.wav"))
            if os.path.basename(p) not in set(drop_files)]
    df = pd.DataFrame(rows, columns=["path", "genre"]); df["split"] = "training"
    rng = np.random.default_rng(seed)
    for _, idx in df.groupby("genre").groups.items():
        idx = rng.permutation(np.array(idx)); n = len(idx)
        df.loc[idx[:n//10], "split"] = "validation"
        df.loc[idx[n//10:n//5], "split"] = "test"
    return df

def subsample(df, per_genre, seed=0):
    n = min(per_genre, df.genre.value_counts().min())
    return df.groupby("genre").sample(n=n, random_state=seed).sort_index().reset_index(drop=True)

# ---------------- audio cache ----------------
def load_audio(path):
    """30 s, mono, 22,050 Hz, padded/trimmed to a fixed length."""
    try:
        import librosa
        y, _ = librosa.load(path, sr=FS, mono=True, duration=CLIP_SECONDS)
    except ImportError:                                  # fallback for plain wav at FS (testing only)
        from scipy.io import wavfile
        _, y = wavfile.read(path)
        if np.issubdtype(y.dtype, np.integer): y = y/32768.0
        y = y.mean(axis=1) if y.ndim > 1 else y
    y = np.asarray(y, dtype=np.float32)[:CLIP_LEN]
    return np.pad(y, (0, CLIP_LEN - len(y)))

def cache_paths(ds, cache_dir="cache"):
    return f"{cache_dir}/{ds}_audio.npy", f"{cache_dir}/{ds}_table.csv"

def load_cached(ds, cache_dir="cache", per_genre_limit=None):
    import pandas as pd
    apath, tpath = cache_paths(ds, cache_dir)
    df = pd.read_csv(tpath); df = df[df.ok].reset_index(drop=True)
    if per_genre_limit:
        df = df.groupby("genre").head(per_genre_limit).reset_index(drop=True)
    return df, apath

def read_clip(apath, row):
    return np.load(apath, mmap_mode="r")[row].astype(np.float32)/32768.0

# ---------------- feature extraction (parallel, from cache) ----------------
def _feat_one(apath, row, cfg):  return clip_features(read_clip(apath, row), **core(cfg))
def _lmel_one(apath, row, cfg):  return log_mel(read_clip(apath, row), **core(cfg)).astype(np.float32)

def extract_features(apath, rows, cfg, n_jobs=-1):
    from joblib import Parallel, delayed
    return np.stack(Parallel(n_jobs=n_jobs, batch_size=16)(delayed(_feat_one)(apath, r, cfg) for r in rows))

def extract_logmels(apath, rows, cfg, n_jobs=-1):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, batch_size=16)(delayed(_lmel_one)(apath, r, cfg) for r in rows)

def measure_ms(fn, rows, apath, cfg, n=10):
    """Single-thread latency per clip in ms (what a real-time system would pay)."""
    t0 = time.perf_counter()
    for r in rows[:n]: fn(apath, r, cfg)
    return 1000*(time.perf_counter() - t0)/min(n, len(rows))

# ---------------- evaluation ----------------
def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    accs = [np.mean(y_true[i] == y_pred[i]) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot))]
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))

def evaluate_svm(df, X, cfg, classes):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    y = df["genre"].values; sp = df["split"].values
    tr, va, te = sp == "training", sp == "validation", sp == "test"
    fit = tr | va
    t0 = time.time()
    gs = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                      {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 1e-3, 1e-2]},
                      cv=PredefinedSplit(np.where(va[fit], 0, -1)), refit=False)
    gs.fit(X[fit], y[fit])                                # tuned on the official validation split
    bp = gs.best_params_
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=bp["svc__C"], gamma=bp["svc__gamma"]))
    model.fit(X[tr], y[tr])                               # final model: TRAIN only (same data as the CNN)
    train_s = time.time() - t0
    t0 = time.time(); pred = model.predict(X[te]); pred_ms = 1000*(time.time() - t0)/te.sum()
    lo, hi = bootstrap_ci(y[te], pred)
    return dict(cfg, val_acc=float(gs.best_score_), acc=float(accuracy_score(y[te], pred)),
                acc_lo=lo, acc_hi=hi, macro_f1=float(f1_score(y[te], pred, average="macro")),
                train_s=train_s, predict_ms=pred_ms, dim=X.shape[1], n_test=int(te.sum()),
                empty_filters=empty_filters(cfg["n_mels"], cfg["N"]),
                cm=confusion_matrix(y[te], pred, labels=classes).tolist(), best_params=str(bp))
