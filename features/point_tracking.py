"""
point_tracking.py — Dac trung theo doi diem dac trung (KLT / Lucas-Kanade).

LUU Y: day la 1 trong 4 huong duoc THU NGHIEM, KHONG nam trong 2 model cuoi
cung duoc chon de deploy (audio + R3D). Giu lai de tai lieu hoa qua trinh
thu nghiem cua team.

Port tu Kaggle notebook (cell 46).
"""

import cv2
import numpy as np


def extract_point_tracking_features(
    video_path: str,
    resize_dim: tuple[int, int] = (320, 240),
    max_frames: int = 90,
    max_corners: int = 50,
    n_segments: int = 4,
) -> np.ndarray | None:
    """
    Theo doi 1 nhom diem dac trung (KLT tracker) qua video. Trich xuat:
      - Ty le diem bi mat track theo thoi gian (point loss rate) - tang
        vot khi rung/nhoe manh.
      - Do hon loan huong di chuyen giua cac diem (direction incoherence).
      - Do lon dich chuyen trung binh cua cac diem con track duoc.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        lk_params = dict(winSize=(15, 15), maxLevel=2,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict(maxCorners=max_corners, qualityLevel=0.01, minDistance=7, blockSize=7)

        ret, old_frame = cap.read()
        if not ret:
            return None
        old_frame = cv2.resize(old_frame, resize_dim)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        if p0 is None:
            return None

        initial_point_count = len(p0)
        loss_ratios = []
        direction_incoherence = []
        displacement_means = []

        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret or p0 is None or len(p0) == 0:
                break
            frame_count += 1

            frame = cv2.resize(frame, resize_dim)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            p1, status, _err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

            if p1 is None:
                loss_ratios.append(1.0)
                break

            good_new = p1[status == 1]
            good_old = p0[status == 1]

            loss_ratio = 1.0 - (len(good_new) / initial_point_count)
            loss_ratios.append(loss_ratio)

            if len(good_new) > 1:
                displacement = good_new - good_old
                angles = np.arctan2(displacement[:, 1], displacement[:, 0])
                magnitudes = np.linalg.norm(displacement, axis=1)
                direction_incoherence.append(np.std(angles))
                displacement_means.append(magnitudes.mean())

            old_gray = gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            if len(p0) < max_corners * 0.3:
                new_pts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
                if new_pts is not None:
                    p0 = new_pts

        if len(loss_ratios) < 3:
            return None

        loss_ratios = np.array(loss_ratios)
        direction_incoherence = np.array(direction_incoherence) if direction_incoherence else np.array([0])
        displacement_means = np.array(displacement_means) if displacement_means else np.array([0])

        global_feats = [
            loss_ratios.mean(), loss_ratios.max(), loss_ratios.std(),
            direction_incoherence.mean(), direction_incoherence.max(), direction_incoherence.std(),
            displacement_means.mean(), displacement_means.max(), displacement_means.std(),
        ]

        n = len(loss_ratios)
        boundaries = np.linspace(0, n, n_segments + 1, dtype=int)
        segment_feats = []
        for i in range(n_segments):
            seg = loss_ratios[boundaries[i]:boundaries[i + 1]]
            segment_feats.append(seg.mean() if len(seg) > 0 else 0)

        loss_spike = np.diff(loss_ratios).max() if n > 1 else 0

        return np.array(global_feats + segment_feats + [loss_spike, n])

    except Exception:
        return None