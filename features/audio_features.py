"""
audio_features.py — Tach audio tu video, tinh MFCC + dac trung pho.

Port tu Kaggle notebook (cell 18). Yeu cau ffmpeg da cai san tren he thong
(kiem tra: `ffmpeg -version`).
"""

import os
import subprocess
import tempfile

import librosa
import numpy as np


def extract_audio_features(video_path: str, n_mfcc: int = 40) -> np.ndarray | None:
    """
    Tach audio tu video bang ffmpeg, tinh MFCC(n_mfcc) + spectral centroid +
    RMS energy + zero-crossing rate (moi loai lay mean va std) -> 1 vector
    dac trung co do dai co dinh cho moi video.

    Tra ve None neu video khong co audio / ffmpeg loi / audio qua ngan.
    """
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", tmp_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        if result.returncode != 0 or not os.path.exists(tmp_wav):
            return None

        y, sr = librosa.load(tmp_wav, sr=16000)
        if len(y) < sr * 0.1:  # audio qua ngan/loi
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rms_energy = librosa.feature.rms(y=y)
        zero_crossing = librosa.feature.zero_crossing_rate(y)

        feature_vector = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            spectral_centroid.mean(axis=1), spectral_centroid.std(axis=1),
            rms_energy.mean(axis=1), rms_energy.std(axis=1),
            zero_crossing.mean(axis=1), zero_crossing.std(axis=1),
        ])
        return feature_vector
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)