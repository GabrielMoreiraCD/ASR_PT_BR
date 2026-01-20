# utils_audio.py
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Optional, Union
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
    return scipy.signal.resample_poly(wav, up, down).astype(np.float32)


def _normalize(wav: np.ndarray) -> np.ndarray:
    wav = wav.astype(np.float32, copy=False)
    maxv = float(np.max(np.abs(wav))) if wav.size else 0.0
    if maxv > 0:
        wav = wav / maxv
    return wav


def _ffmpeg_to_wav_16k_mono(src_path: Union[str, Path], out_wav: Union[str, Path]) -> None:
    """
    Extrai áudio do mp4/m4a etc para wav PCM 16k mono.
    Requer ffmpeg disponível no PATH.
    """
    src_path = str(src_path)
    out_wav = str(out_wav)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", src_path,
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-f", "wav",
        out_wav,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src_path}\n{p.stderr[:2000]}")


def safe_audio_duration_sec(path: Union[str, Path]) -> float:
    """
    Para wav/flac: lê header com soundfile.
    Para mp4: tenta extrair duration via ffprobe se existir; se não existir, retorna NaN.
    """
    p = str(path)
    ext = Path(p).suffix.lower()

    try:
        if ext in [".wav", ".flac", ".ogg"]:
            info = sf.info(p)
            if info.frames and info.samplerate:
                return float(info.frames) / float(info.samplerate)
            return float("nan")
    except Exception:
        pass

    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", p]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except Exception:
        return float("nan")


def load_resample_mono(path: Union[str, Path], target_sr: int = TARGET_SR, cache_wav_dir: Optional[Path] = None) -> Optional[np.ndarray]:

    p = Path(path)
    ext = p.suffix.lower()

    # Convertendo formatos de audio p\ wav 16k mono
    if ext in [".mp4", ".m4a", ".mov", ".mkv", ".webm"]:
        if cache_wav_dir is None:
            cache_wav_dir = p.parent / "_wav_cache"
        cache_wav_dir.mkdir(parents=True, exist_ok=True)

        out_wav = cache_wav_dir / (p.stem + ".wav")
        if (not out_wav.exists()) or out_wav.stat().st_size == 0:
            _ffmpeg_to_wav_16k_mono(p, out_wav)
        p = out_wav

    try:
        wav, sr = sf.read(str(p), always_2d=False)
    except Exception:
        return None

    if wav is None:
        return None

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = _normalize(wav)
    wav = _resample(wav, sr, target_sr)
    return wav
