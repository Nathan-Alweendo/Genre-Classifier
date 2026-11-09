"""STEP 2. Decode every clip ONCE and save it, so later scripts never touch mp3/wav again.
Usage: python 02_cache_audio.py --dataset gtzan   (then again with --dataset fma)
Reads : config.py paths
Writes: cache/<ds>_audio.npy (int16, clips x 661500) and cache/<ds>_table.csv (path, genre, split, row, ok)"""
import argparse, os
import numpy as np
import genre_dsp as g, config

def _decode(path):
    try:
        return g.load_audio(path), True
    except Exception as e:
        print("skip", path, type(e).__name__); return np.zeros(g.CLIP_LEN, np.float32), False

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=["gtzan", "fma"])
    ds = ap.parse_args().dataset
    from joblib import Parallel, delayed
    df = (g.load_gtzan_table(config.GTZAN_ROOT, config.GTZAN_DROP) if ds == "gtzan"
          else g.load_fma_table(config.FMA_ROOT))
    df = g.subsample(df, config.PER_GENRE[ds])
    print(df.groupby(["genre", "split"]).size().unstack(fill_value=0))
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    apath, tpath = g.cache_paths(ds, config.CACHE_DIR)
    mm = np.lib.format.open_memmap(apath, mode="w+", dtype=np.int16, shape=(len(df), g.CLIP_LEN))
    ok = np.zeros(len(df), bool)
    for s in range(0, len(df), 64):
        res = Parallel(n_jobs=-1)(delayed(_decode)(p) for p in df.path.values[s:s+64])
        for i, (y, good) in enumerate(res):
            mm[s+i] = (np.clip(y, -1, 1)*32767).astype(np.int16); ok[s+i] = good
        print(f"decoded {min(s+64, len(df))}/{len(df)}", end="\r")
    mm.flush()
    df["row"] = np.arange(len(df)); df["ok"] = ok
    df.to_csv(tpath, index=False)
    print(f"\nsaved {apath} and {tpath}; {ok.sum()} ok, {(~ok).sum()} skipped")
