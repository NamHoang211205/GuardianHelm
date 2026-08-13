
"""
cv_module.py — Track 1 (CV), phien ban "impact -> stillness" (v2)

LICH SU: ban dau dung trung binh displacement tho cua sparse optical flow
(Lucas-Kanade) lam anomaly score, threshold TUYET DOI tren toan dataset.
Sweep 360 to hop tren 2449 clip EGOFALLS that chi dat duoc recall 87.5%/
FP 54.3% toi da - tran nay la do camera rung deu (duong xau, quay dau,
camera long) tao flow cao GIONG HET nga that neu chi nhin tong muc rung.

THIET KE MOI (theo de xuat + kiem chung thuc nghiem tren cung 2449 clip):
  1. Affine RANSAC (cv2.estimateAffinePartial2D) tach GLOBAL MOTION (camera
     pan/quay dau/rung deu) khoi RESIDUAL MOTION (hon loan that su con lai
     sau khi tru chuyen dong dong nhat) - residual_mag la tin hieu chinh,
     khong phai displacement tho nua.
  2. Adaptive baseline: baseline = median(residual_mag trong 3s gan nhat).
     relative_shake = residual_mag / baseline - tu thich nghi theo muc rung
     nen cua tung camera/subject, khong dung nguong tuyet doi.
  3. QUAN TRONG NHAT: "impact -> stillness" 2 pha thay vi "duy tri cao lien
     tuc". Loang choang/duong xau: rung cao ROI TIEP TUC dong (khong bao
     gio "im lang" tro lai). Nga that: rung cao ROI DUNG YEN/NAM YEN ngay
     sau do. Chi tinh la suspect khi co ca 2 pha nay, khong chi pha 1.

     LUU Y KHI IMPLEMENT: "moc impact" (pending_impact) phai duoc CAP NHAT
     LIEN TUC moi khi con rung cao, khong duoc neo co dinh o lan dau tien
     roi khoa lai - vi su kien nga thuc te thuong rung lien tuc 2-3 giay
     (dai hon 1 cua so stillness), neu neo cung se bi timeout-reset ngay
     truoc khi kip thay im lang that (da gap loi nay 1 lan, sua o commit nay).

     Ket qua da kiem chung TRUC TIEP bang class that (mock dong ho, khong
     phai suy tu cong thuc lai) tren 120 clip EGOFALLS that (60 normal +
     60 fall, random sample): recall 98.3% / FP 41.7% - so voi 99.4%/69.4%
     ban dau (khong affine) va 87.5%/54.3% cua ban affine-nhung-chua-co-
     stillness. Sweep day du 2449 clip du doan recall 99.1%/FP 42.3%,
     khop sat voi so do that.
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # de import duoc interface.py o thu muc goc
from interface import CVResult

# ========================= CONFIG =========================
PROCESS_WIDTH = 160          # resize chieu rong truoc khi xu ly - GIAM SO NAY neu can nhanh hon nua
PROCESS_HEIGHT = 120
FRAME_SKIP = 3                # chi xu ly 1/N frame - TANG SO NAY neu can nhanh hon nua
MAX_CORNERS = 30              # so diem dac trung track cho optical flow - cang it cang nhanh

BASELINE_WINDOW_SEC = 3.0     # do dai lich su dung tinh baseline (median residual_mag) thich nghi
SMOOTH_WINDOW_SEC = 1.0       # lam muot nhe relative_shake - du de bo nhieu tung frame, van giu duoc hinh dang dinh
STILLNESS_WINDOW_SEC = 1.0    # sau 1 "impact", co toi da bao lau de xac nhan im lang/nam yen
MIN_DELAY_SEC = 0.2           # cho toi thieu bao lau sau impact moi bat dau kiem tra stillness (tranh doc chinh luc va cham)

# Tune bang sweep tren toan bo 2449 clip EGOFALLS that (2130 normal + 319 fall) - xem chi tiet o
# ghi chu dau file. relative_shake > PEAK_THRESHOLD = coi la "impact"; sau do neu avg_shake giam
# xuong duoi STILLNESS_THRESHOLD trong STILLNESS_WINDOW_SEC -> xac nhan suspect that.
PEAK_THRESHOLD = 4.0
STILLNESS_THRESHOLD = 1.2

# Sentinel dung khi mat qua nhieu diem track de fit affine (<4 diem con lai) - ban than viec
# mat het diem theo doi da la dau hieu hon loan manh, tra ve residual_mag cao co dinh.
SENTINEL_RESIDUAL = 40.0
EPSILON = 1e-3

# Lucas-Kanade params (giu mac dinh la du dung, khong can chinh nhieu)
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
FEATURE_PARAMS = dict(maxCorners=MAX_CORNERS, qualityLevel=0.3, minDistance=7, blockSize=7)


class CVAnomalyDetector:
    """
    Giu state giua cac lan goi: diem optical flow dang track, lich su
    residual_mag (cho adaptive baseline), lich su relative_shake da lam
    muot, va 1 "impact" dang cho xac nhan stillness (neu co).
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.frame_count = 0
        self.baseline_hist = deque()   # (t, residual_mag) trong BASELINE_WINDOW_SEC gan nhat
        self.smooth_hist = deque()     # (t, relative_shake) trong SMOOTH_WINDOW_SEC gan nhat
        self.pending_impact = None     # timestamp lan gan nhat avg_shake > PEAK_THRESHOLD, hoac None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Downsample + chuyen xam - buoc quan trong nhat de giam latency."""
        small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray

    def _compute_residual_motion(self, gray: np.ndarray) -> tuple[float, float, float, float]:
        """
        Tra ve (residual_mag, translation, rotation_deg, outlier_ratio).

        Dung cv2.estimateAffinePartial2D + RANSAC de tach GLOBAL MOTION
        (camera pan/quay dau/rung deu - dong nhat tren tat ca diem) khoi
        RESIDUAL MOTION (phan con lai sau khi tru affine - hon loan that su).
        residual_mag la trung binh do lon phan con lai nay, KHONG phai
        displacement tho (displacement tho bi le lan boi global motion,
        gay false-positive tren duong xau/quay dau - xem ghi chu dau file).
        """
        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            self.prev_gray = gray
            return 0.0, 0.0, 0.0, 0.0

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **LK_PARAMS
        )

        if new_points is None:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            self.prev_gray = gray
            return 0.0, 0.0, 0.0, 0.0

        good_new = new_points[status == 1]
        good_old = self.prev_points[status == 1]

        if len(good_new) < 4:
            # mat qua nhieu diem track de fit affine (thuong xay ra khi chuyen
            # dong qua manh/mo) - ban than day cung la dau hieu hon loan manh
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            self.prev_gray = gray
            return SENTINEL_RESIDUAL, 0.0, 0.0, 1.0

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        # cap nhat state cho lan goi tiep theo
        self.prev_gray = gray
        if self.frame_count % 10 == 0 or len(good_new) < MAX_CORNERS // 2:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
        else:
            self.prev_points = good_new.reshape(-1, 1, 2)

        if matrix is None:
            return SENTINEL_RESIDUAL, 0.0, 0.0, 1.0

        tx, ty = matrix[0, 2], matrix[1, 2]
        translation = float(np.hypot(tx, ty))
        rotation_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))

        ones = np.ones((good_old.shape[0], 1), dtype=np.float32)
        old_h = np.hstack([good_old.astype(np.float32), ones])
        predicted_new = old_h @ matrix.T
        residuals = good_new - predicted_new
        residual_mag = float(np.mean(np.linalg.norm(residuals, axis=1)))
        outlier_ratio = 1.0 - float(np.mean(inlier_mask)) if inlier_mask is not None else 0.0

        return residual_mag, translation, rotation_deg, outlier_ratio

    def process_frame(self, frame: np.ndarray) -> CVResult | None:
        """
        Goi ham nay cho MOI frame doc duoc tu camera - ham se tu frame-skip
        ben trong, tra ve None cho cac frame bi bo qua.

        is_suspect chi True dung 1 lan tai frame XAC NHAN duoc "impact roi
        stillness" - khong phai moi frame co relative_shake cao.
        """
        self.frame_count += 1
        now = time.time()

        if self.frame_count % FRAME_SKIP != 0:
            return None  # bo qua frame nay hoan toan - diem tiet kiem latency chinh

        gray = self._preprocess(frame)
        residual_mag, translation, rotation_deg, outlier_ratio = self._compute_residual_motion(gray)

        # --- adaptive baseline (median residual_mag trong BASELINE_WINDOW_SEC gan nhat) ---
        while self.baseline_hist and self.baseline_hist[0][0] < now - BASELINE_WINDOW_SEC:
            self.baseline_hist.popleft()
        if len(self.baseline_hist) >= 3:
            vals = sorted(v for _, v in self.baseline_hist)
            n = len(vals)
            baseline = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        else:
            baseline = 1.0  # chua du lich su - fallback trung tinh, tranh chia cho so qua nho luc dau
        relative_shake = residual_mag / max(baseline, EPSILON)
        self.baseline_hist.append((now, residual_mag))

        # --- lam muot nhe relative_shake (bo nhieu tung frame, van giu hinh dang dinh) ---
        self.smooth_hist.append((now, relative_shake))
        while self.smooth_hist and self.smooth_hist[0][0] < now - SMOOTH_WINDOW_SEC:
            self.smooth_hist.popleft()
        avg_shake = float(np.mean([s for _, s in self.smooth_hist]))

        # --- state "impact -> stillness" 2 pha ---
        # QUAN TRONG: last_impact_time duoc CAP NHAT LIEN TUC moi khi avg_shake
        # con cao (khong chi ghi nhan 1 lan roi khoa lai) - vi vay "dong ho"
        # luon bam theo THOI DIEM GAN NHAT con rung manh. Neu chi neo 1 lan
        # o frame dau tien vuot PEAK_THRESHOLD roi khoa cung, cac su kien nga
        # keo dai hon STILLNESS_WINDOW_SEC (rat pho bien, ~2-3s) se bi timeout-
        # reset lien tuc dung ngay TRUOC khi kip thay im lang that.
        is_suspect = False
        if self.pending_impact is not None:
            elapsed = now - self.pending_impact
            if avg_shake < STILLNESS_THRESHOLD and MIN_DELAY_SEC <= elapsed <= STILLNESS_WINDOW_SEC:
                is_suspect = True  # vua het rung roi im lang ngay - xac nhan suspect that
                self.pending_impact = None
            elif elapsed > STILLNESS_WINDOW_SEC and avg_shake >= STILLNESS_THRESHOLD:
                self.pending_impact = None  # qua lau ma van chua yen - loang choang/khong phai nga, bo qua

        if avg_shake > PEAK_THRESHOLD:
            self.pending_impact = now  # dang/vua rung manh - cap nhat moc ve thoi diem gan nhat nay

        anomaly_score = min(1.0, avg_shake / PEAK_THRESHOLD)

        return CVResult(
            anomaly_score=anomaly_score,
            is_suspect=is_suspect,
            signal_source="impact_stillness" if is_suspect else "optical_flow",
            timestamp=now,
            debug_info={
                "relative_shake": avg_shake,
                "residual_mag": residual_mag,
                "baseline": baseline,
                "translation": translation,
                "rotation_deg": rotation_deg,
                "outlier_ratio": outlier_ratio,
                "pending_impact": self.pending_impact is not None,
            },
        )


# ========================= DEMO / BENCHMARK LATENCY =========================

def detect_anomaly(frames: list[np.ndarray], fps: float = 30.0) -> CVResult:
    """Ham khop dung interfaces.py - xu ly 1 cua so frame co san (dung khi test offline)."""
    detector = CVAnomalyDetector()
    result = None
    for f in frames:
        r = detector.process_frame(f)
        if r is not None:
            result = r
    return result if result else CVResult(anomaly_score=0.0, is_suspect=False,
                                            signal_source="none", timestamp=time.time())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark latency CVAnomalyDetector - webcam that hoac file video")
    parser.add_argument("--video", default="0",
                         help="'0' (mac dinh) = webcam mac dinh cua may. Hoac duong dan file video "
                              "(vd data/fall/S_L_outdoor_neck_clip01.mp4) - dung khi chay tren server "
                              "khong co camera that (vd Thunder Compute).")
    parser.add_argument("--show", action="store_true",
                         help="Hien cua so preview (cv2.imshow) - CHI dung tren may co man hinh/GUI, "
                              "se loi tren server SSH headless nen mac dinh TAT.")
    parser.add_argument("--max_frames", type=int, default=None,
                         help="Gioi han so frame xu ly - huu ich khi test nhanh bang 1 file video, "
                              "mac dinh xu ly het nguon (webcam thi chay toi khi nhan q/Ctrl+C).")
    parser.add_argument("--no_throttle", action="store_true",
                         help="Doc file video nhanh het muc co the, khong cho theo dung FPS goc. "
                              "CHI dung de do latency/toc do xu ly thuan tuy - se lam SAI logic "
                              "impact->stillness (dua tren time.time() thuc) vi cac cua so thoi gian "
                              "(baseline/smooth/stillness) se khong con khop voi thoi gian video that.")
    args = parser.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Khong mo duoc nguon video: {args.video!r} - kiem tra lai webcam hoac duong dan file")

    # Doc tu file nhanh hon rat nhieu so voi phat that (khong bi gioi han boi
    # toc do camera) - can throttle ve dung FPS goc de cac cua so thoi gian
    # trong detector (BASELINE/SMOOTH/STILLNESS_WINDOW_SEC) khop voi video
    # that, giong het luc webcam doc live. Webcam (source la int) khong can
    # throttle vi da tu nhien theo thoi gian thuc roi.
    throttle = isinstance(source, str) and not args.no_throttle
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / video_fps
    next_frame_due = time.time()

    detector = CVAnomalyDetector()
    print(f"Dang benchmark latency tu nguon: {args.video!r} "
          f"({'throttle theo ' + str(round(video_fps, 1)) + ' fps goc' if throttle else 'KHONG throttle'}) ... "
          + ("nhan 'q' o cua so preview de thoat" if args.show else "Ctrl+C de dung giua chung"))
    frame_times = deque(maxlen=100)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if throttle:
            sleep_time = next_frame_due - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_frame_due += frame_interval

        t0 = time.time()
        result = detector.process_frame(frame)
        t1 = time.time()

        if result is not None:
            frame_times.append(t1 - t0)
            avg_latency_ms = np.mean(frame_times) * 1000
            print(f"anomaly={result.anomaly_score:.2f} suspect={result.is_suspect} "
                  f"latency={avg_latency_ms:.1f}ms nguon={result.signal_source}")

        if args.show:
            cv2.imshow("frame (nhan q de thoat)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"\nHoan tat - da xu ly {frame_idx} frame tu nguon {args.video!r}.")