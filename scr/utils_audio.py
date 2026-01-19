from pathlib import Path
from typing import Union, Optional
import numpy as np
import soundfile as sf
import scipy.signal

TARGET_SR = 16000

def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav.astype(np.float32)
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    wav = scipy.signal.resample_poly(wav, up, down).astype(np.float32)
    return wav

def load_resample_mono(path: Union[str, Path], target_sr: int = TARGET_SR) -> Optional[np.ndarray]:
    p = str(path)

    # leitura com soundfile
    try:
        wav, sr = sf.read(p, always_2d=False)
    except Exception:
        return None

    if wav is None:
        return None

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)
    # normalização 
    maxv = float(np.max(np.abs(wav))) if wav.size else 0.0
    if maxv > 0:
        wav = wav / maxv

    wav = _resample(wav, sr, target_sr)
    return wav
