"""
cv_module.py — Track 1 (CV), phien ban toi uu LATENCY
Chi dung 2 tin hieu (bo lane detection vi qua nang va rui ro false positive
o duong khong vach ke - xem ly do trong thao luan truoc):
  1. Sparse Optical Flow (Lucas-Kanade) - do do "hon loan" chuyen dong
  2. Horizon/sky ratio - do ty le vung sang phia tren khung hinh

3 don bay giam latency da ap dung:
  - Downsample manh truoc khi xu ly (mac dinh 160x120)
  - Sparse optical flow thay vi dense (nhanh hon Farneback nhieu lan)
  - Frame skipping (chi xu ly 1/N frame, mac dinh N=3)
"""

import time
from collections import deque

import cv2
import numpy as np

from interface import CVResult

# ========================= CONFIG =========================
PROCESS_WIDTH = 160          # resize chieu rong truoc khi xu ly - GIAM SO NAY neu can nhanh hon nua
PROCESS_HEIGHT = 120
FRAME_SKIP = 3                # chi xu ly 1/N frame - TANG SO NAY neu can nhanh hon nua
MAX_CORNERS = 30              # so diem dac trung track cho optical flow - cang it cang nhanh
WINDOW_SEC = 2.0               # cua so thoi gian de tinh anomaly (giay)
FLOW_MAGNITUDE_THRESHOLD = 8.0   # TUNE theo du lieu that - do lon flow trung binh coi la bat thuong
FLOW_VARIANCE_THRESHOLD = 2.5    # TUNE theo du lieu that - do phan tan huong flow coi la hon loan
SKY_RATIO_LOW_THRESHOLD = 0.05   # TUNE - neu ty le "vung sang phia tren" thap hon nguong nay lien tuc -> nghi ngo

# Lucas-Kanade params (giu mac dinh la du dung, khong can chinh nhieu)
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
FEATURE_PARAMS = dict(maxCorners=MAX_CORNERS, qualityLevel=0.3, minDistance=7, blockSize=7)


class CVAnomalyDetector:
    """
    Giu state giua cac lan goi (frame truoc, diem dang track) de tinh
    optical flow tang dan - khong tinh lai tu dau moi frame.
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.frame_count = 0
        # luu lai cac gia tri trong 1 cua so thoi gian de tinh thong ke
        self.flow_history = deque()
        self.sky_history = deque()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Downsample + chuyen xam - buoc quan trong nhat de giam latency."""
        small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray

    def _compute_sparse_flow(self, gray: np.ndarray) -> tuple[float, float]:
        """
        Tra ve (magnitude_trung_binh, huong_phuong_sai) cua sparse optical flow.
        Nhanh hon Farneback (dense) nhieu lan vi chi track ~30 diem thay vi
        toan bo pixel.
        """
        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            self.prev_gray = gray
            return 0.0, 0.0

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **LK_PARAMS
        )

        if new_points is None:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            self.prev_gray = gray
            return 0.0, 0.0

        good_new = new_points[status == 1]
        good_old = self.prev_points[status == 1]

        if len(good_new) < 5:
            # mat qua nhieu diem track (thuong xay ra khi chuyen dong qua manh/mo)
            # - ban than day cung la 1 dau hieu bat thuong, tra ve gia tri cao
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            self.prev_gray = gray
            return FLOW_MAGNITUDE_THRESHOLD * 2, FLOW_VARIANCE_THRESHOLD * 2

        deltas = good_new - good_old
        magnitudes = np.linalg.norm(deltas, axis=1)
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])

        mag_mean = float(np.mean(magnitudes))
        angle_variance = float(np.var(angles))

        # cap nhat state cho lan goi tiep theo
        self.prev_gray = gray
        # dinh ky tim lai diem dac trung moi (tranh mat het diem theo thoi gian)
        if self.frame_count % 10 == 0 or len(good_new) < MAX_CORNERS // 2:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
        else:
            self.prev_points = good_new.reshape(-1, 1, 2)

        return mag_mean, angle_variance

    def _compute_sky_ratio(self, gray: np.ndarray) -> float:
        """
        Uoc luong nhanh ty le 'vung troi/canh vat' o nua tren khung hinh.
        Heuristic don gian: dem ty le pixel sang (gia dinh troi/canh vat sang
        hon mat duong) trong 1/3 tren cung cua khung hinh - cuc nhanh, khong
        can segmentation model.
        """
        top_third = gray[: gray.shape[0] // 3, :]
        bright_pixel_ratio = float(np.mean(top_third > 120))  # nguong 120/255 - TUNE theo dieu kien anh sang thuc te
        return bright_pixel_ratio

    def process_frame(self, frame: np.ndarray) -> CVResult | None:
        """
        Goi ham nay cho MOI frame doc duoc tu camera - ham se tu frame-skip
        ben trong, tra ve None cho cac frame bi bo qua (khong tinh gi ca,
        cuc re) va chi tra ve CVResult that su o cuoi moi cua so thoi gian.
        """
        self.frame_count += 1
        now = time.time()

        if self.frame_count % FRAME_SKIP != 0:
            return None  # bo qua frame nay hoan toan - khong tinh toan gi, day la diem tiet kiem latency chinh

        gray = self._preprocess(frame)
        mag, ang_var = self._compute_sparse_flow(gray)
        sky_ratio = self._compute_sky_ratio(gray)

        self.flow_history.append((now, mag, ang_var))
        self.sky_history.append((now, sky_ratio))

        # xoa cac gia tri qua cu ra khoi cua so thoi gian
        cutoff = now - WINDOW_SEC
        while self.flow_history and self.flow_history[0][0] < cutoff:
            self.flow_history.popleft()
        while self.sky_history and self.sky_history[0][0] < cutoff:
            self.sky_history.popleft()

        if len(self.flow_history) < 2:
            return CVResult(anomaly_score=0.0, is_suspect=False,
                             signal_source="warming_up", timestamp=now)

        avg_mag = float(np.mean([m for _, m, _ in self.flow_history]))
        avg_ang_var = float(np.mean([a for _, _, a in self.flow_history]))
        avg_sky = float(np.mean([s for _, s in self.sky_history]))

        flow_score = min(1.0, (avg_mag / FLOW_MAGNITUDE_THRESHOLD +
                                avg_ang_var / FLOW_VARIANCE_THRESHOLD) / 2)
        horizon_score = 1.0 if avg_sky < SKY_RATIO_LOW_THRESHOLD else 0.0

        # fusion don gian: lay max giua 2 tin hieu (uu tien khong bo sot ca that)
        anomaly_score = max(flow_score, horizon_score)
        is_suspect = anomaly_score > 0.6  # TUNE nguong nay bang du lieu that

        source = "optical_flow" if flow_score > horizon_score else "horizon"

        return CVResult(
            anomaly_score=anomaly_score,
            is_suspect=is_suspect,
            signal_source=source,
            timestamp=now,
            debug_info={"flow_mag": avg_mag, "flow_var": avg_ang_var, "sky_ratio": avg_sky},
        )


# ========================= DEMO / BENCHMARK LATENCY =========================

def detect_anomaly(frames: list[np.ndarray], fps: float = 30.0) -> CVResult:
    """Ham khop dung interface.py - xu ly 1 cua so frame co san (dung khi test offline)."""
    detector = CVAnomalyDetector()
    result = None
    for f in frames:
        r = detector.process_frame(f)
        if r is not None:
            result = r
    return result if result else CVResult(anomaly_score=0.0, is_suspect=False,
                                            signal_source="none", timestamp=time.time())


if __name__ == "__main__":
    # Benchmark latency thuc te bang webcam - chay thu de do FPS xu ly duoc
    cap = cv2.VideoCapture(0)
    detector = CVAnomalyDetector()

    print("Dang benchmark latency... nhan 'q' de thoat")
    frame_times = deque(maxlen=100)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        result = detector.process_frame(frame)
        t1 = time.time()

        if result is not None:
            frame_times.append(t1 - t0)
            avg_latency_ms = np.mean(frame_times) * 1000
            print(f"anomaly={result.anomaly_score:.2f} suspect={result.is_suspect} "
                  f"latency={avg_latency_ms:.1f}ms nguon={result.signal_source}")

        cv2.imshow("frame (nhan q de thoat)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
