"""
optical_flow_v1.py — Dac trung optical flow (Farneback), thong ke toan cuc.

LUU Y: day la 1 trong 4 huong duoc THU NGHIEM trong qua trinh phat trien,
KHONG nam trong 2 model cuoi cung duoc chon de deploy (audio + R3D). Giu
lai file nay de tai lieu hoa day du qua trinh thu nghiem cua team.

Port tu Kaggle notebook (cell 29).
"""

import cv2
import numpy as np


def extract_optical_flow_features(
    video_path: str,
    resize_dim: tuple[int, int] = (160, 120),
    max_frames: int = 90,
    sample_every: int = 2,
) -> np.ndarray | None:
    """
    Doc video, tinh optical flow (Farneback) giua cac frame lien tiep, tra
    ve vector dac trung co dinh tom tat chuyen dong toan video.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        prev_gray = None
        magnitudes = []
        angles = []
        frame_count = 0

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % sample_every != 0:
                continue

            frame = cv2.resize(frame, resize_dim)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitudes.append(mag.mean())
                angles.append(ang.mean())

            prev_gray = gray

        cap.release()

        if len(magnitudes) < 2:
            return None

        magnitudes = np.array(magnitudes)
        angles = np.array(angles)

        return np.array([
            magnitudes.mean(), magnitudes.std(), magnitudes.max(), magnitudes.min(),
            np.percentile(magnitudes, 25), np.percentile(magnitudes, 75),
            angles.mean(), angles.std(),
            np.diff(magnitudes).std() if len(magnitudes) > 2 else 0,
            len(magnitudes),
        ])

    except Exception:
        return None