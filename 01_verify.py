"""STEP 1. Checks your code is correct BEFORE any experiment. Run first. Produces no files.
 (a) own windows/STFT/mel/MFCC vs librosa   (b) YOUR lab FFT inside the STFT vs numpy and librosa"""
import time
import numpy as np
import genre_dsp as g
import lab_fft

if __name__ == "__main__":
    print("=== (a) own front end vs librosa (errors near 1e-6 or smaller = good) ===")
    try:
        g.verify_vs_librosa()
    except ImportError:
        print("librosa not installed: pip install librosa")
    print("\n=== (b) your lab FFT inside the STFT ===")
    t = np.arange(g.FS*2)/g.FS
    x = np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*2500*t)
    t0 = time.time(); ref = g.my_stft(x, 1024, 512, "hann");                    t_np = time.time() - t0
    t0 = time.time(); own = g.my_stft(x, 1024, 512, "hann", fft_fn=lab_fft.fft); t_own = time.time() - t0
    print(f"lab FFT vs numpy rfft  max abs err = {np.abs(own - ref).max():.2e}")
    print(f"time: numpy {t_np*1000:.1f} ms | lab FFT {t_own*1000:.0f} ms  ({t_own/t_np:.0f}x slower)")
    try:
        import librosa
        lib = librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window="hann", center=True)
        print(f"lab-FFT STFT vs librosa  max abs err = {np.abs(own - lib).max():.2e}")
    except ImportError:
        pass
    print("\nBulk runs use numpy's rfft (identical numbers, much faster); your FFT is verified above.")
