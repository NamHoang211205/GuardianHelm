"""
optical_flow_v2.py — Ban cai tien cua optical flow v1: them segment theo
thoi gian, dac trung "post-peak stillness", va gia toc (dao ham bac 2).

LUU Y: day la 1 trong 4 huong duoc THU NGHIEM, KHONG nam trong 2 model cuoi
cung duoc chon de deploy (audio + R3D). Giu lai de tai lieu hoa qua trinh
thu nghiem cua team.

Port tu Kaggle notebook (cell 40).
"""

import cv2
import numpy as np


def extract_optical_flow_features_v2(
    video_path: str,
    resize_dim: tuple[int, int] = (160, 120),
    max_frames: int = 90,
    sample_every: int = 1,
    n_segments: int = 4,
) -> np.ndarray | None:
    """
    Cai tien so voi v1:
      - Chia video thanh n_segments doan theo thoi gian, tinh stat rieng
        tung doan (giu duoc thong tin THOI DIEM xay ra chuyen dong manh).
      - Them dac trung "do tinh sau dinh" (post-peak stillness) - dac
        trung rieng cua nga (chuyen dong manh roi bat dong dot ngot).
      - sample_every=1 (lay du frame hon, khong bo bot) de khong mat chi tiet.
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

        if len(magnitudes) < n_segments * 2:
            return None

        magnitudes = np.array(magnitudes)
        angles = np.array(angles)
        n = len(magnitudes)

        # 1. Global stats
        global_feats = [
            magnitudes.mean(), magnitudes.std(), magnitudes.max(), magnitudes.min(),
            np.percentile(magnitudes, 25), np.percentile(magnitudes, 75),
            angles.mean(), angles.std(),
            np.diff(magnitudes).std() if n > 2 else 0,
        ]

        # 2. Segment-based stats
        segment_feats = []
        boundaries = np.linspace(0, n, n_segments + 1, dtype=int)
        for i in range(n_segments):
            seg = magnitudes[boundaries[i]:boundaries[i + 1]]
            segment_feats += ([seg.mean(), seg.std()] if len(seg) > 0 else [0, 0])

        # 3. Post-peak stillness
        peak_idx = np.argmax(magnitudes)
        peak_position_ratio = peak_idx / n

        if peak_idx < n - 1:
            after_peak = magnitudes[peak_idx + 1:]
            post_peak_mean = after_peak.mean() if len(after_peak) > 0 else 0
            post_peak_ratio = post_peak_mean / (magnitudes[peak_idx] + 1e-6)
        else:
            post_peak_mean = 0
            post_peak_ratio = 0

        stillness_feats = [peak_position_ratio, post_peak_mean, post_peak_ratio]

        # 4. Acceleration (dao ham bac 2)
        if n > 3:
            accel = np.diff(np.diff(magnitudes))
            accel_feats = [accel.mean(), accel.std(), np.abs(accel).max()]
        else:
            accel_feats = [0, 0, 0]

        return np.array(global_feats + segment_feats + stillness_feats + accel_feats + [n])

    except Exception:
        return None