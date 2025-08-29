import os
import warnings
import numpy as np

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

SR = 44100
warnings.filterwarnings("ignore", message=r".*__audioread_load.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"PySoundFile failed.*")

def load_audio(path: str, target_sr: int = SR) -> tuple:
    """Return mono float32 [-1,1], sr."""
    if not os.path.exists(path):
        return np.array([], dtype=np.float32), target_sr
        
    if LIBROSA_AVAILABLE:
        try:
            y, _ = librosa.load(path, sr=target_sr, mono=True)
            y = y.astype(np.float32)
            m = float(np.max(np.abs(y))) if y.size else 0.0
            if m > 0:
                y = y / m
            return y, target_sr
        except Exception as e:
            print(f"Librosa load failed: {e}")
            # Fall through to wave method
    
    try:
        import wave
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            ch = wf.getnchannels()
            data = wf.readframes(n)
            dtype = np.int16 if wf.getsampwidth() == 2 else np.int8
            arr = np.frombuffer(data, dtype=dtype).astype(np.float32)
            
            # Normalize based on bit depth
            if dtype == np.int16:
                arr /= 32768.0
            else:
                arr /= 128.0
                
            if ch > 1:
                arr = arr.reshape(-1, ch).mean(axis=1)
            if sr != target_sr and arr.size:
                t_old = np.linspace(0, len(arr)/sr, len(arr), endpoint=False)
                t_new = np.linspace(0, len(arr)/sr, int(len(arr)*target_sr/sr), endpoint=False)
                arr = np.interp(t_new, t_old, arr).astype(np.float32)
                sr = target_sr
            m = float(np.max(np.abs(arr))) if arr.size else 0.0
            if m > 0:
                arr = arr / m
            return arr, sr
    except Exception as e:
        print(f"Wave load failed: {e}")
        return np.array([], dtype=np.float32), target_sr

def time_stretch(y: np.ndarray, speed: float) -> np.ndarray:
    """speed>1 → faster (shorter). If no librosa, naive resample (pitch changes)."""
    speed = max(0.25, min(2.0, float(speed)))
    if not y.size or abs(speed - 1.0) < 1e-6:
        return y.astype(np.float32)
        
    if LIBROSA_AVAILABLE:
        try:
            return librosa.effects.time_stretch(y, rate=speed).astype(np.float32)
        except Exception:
            # Fallback to interpolation if librosa fails
            pass
            
    n_new = max(1, int(len(y) / speed))
    t_old = np.linspace(0, 1, len(y), endpoint=False)
    t_new = np.linspace(0, 1, n_new, endpoint=False)
    return np.interp(t_new, t_old, y).astype(np.float32)

def to_stereo(mono: np.ndarray) -> np.ndarray:
    """Return (2, n) float32."""
    if mono.size == 0:
        return np.zeros((2, 0), dtype=np.float32)
    return np.vstack([mono, mono]).astype(np.float32, copy=False)

def to_int16_stereo(st: np.ndarray) -> np.ndarray:
    """(2, n) float32 -> (n, 2) int16 contiguous."""
    if st.size == 0:
        return np.zeros((0, 2), dtype=np.int16)
        
    st = np.clip(st, -1.0, 1.0)
    arr = (st * 32767.0).astype(np.int16, copy=False).T
    return np.ascontiguousarray(arr)  # (n,2) C-contiguous

def apply_fades(x: np.ndarray, fin_sec: float, fout_sec: float, sr: int = SR) -> np.ndarray:
    """In-place linear fades."""
    n = len(x)
    if n == 0:
        return x
    if fin_sec > 0:
        m = min(n, int(fin_sec * sr))
        if m > 0:
            x[:m] *= np.linspace(0, 1, m, dtype=np.float32)
    if fout_sec > 0:
        m = min(n, int(fout_sec * sr))
        if m > 0:
            x[-m:] *= np.linspace(1, 0, m, dtype=np.float32)
    return x