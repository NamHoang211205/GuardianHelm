"""
GuardianHelm end-to-end pipeline:

Camera
→ heuristic
→ deep-learning confirmation
→ audio
→ decision
→ SMS/SQLite retry queue
"""

import argparse
import time

import cv2

from audio_module import ask_and_classify
from cv_heuristic.cv_module import CVAnomalyDetector
from cv_hybrid.hybrid_detector import HybridFallDetector
from decision_module import (
    decide_action,
    retry_pending_alerts,
    send_alert,
)
from interface import ActionType


RETRY_INTERVAL_SEC = 30.0
COOLDOWN_AFTER_DECISION_SEC = 5.0


def build_detector(args):
    if args.cv_mode == "heuristic":
        print("[MAIN] CV mode: heuristic")
        return CVAnomalyDetector()

    print("[MAIN] CV mode: hybrid")

    return HybridFallDetector(
        checkpoint=args.checkpoint,
        buffer_size=args.buffer_size,
        heuristic_gate=args.heuristic_gate,
        deep_threshold=args.deep_threshold,
        confirm_count=args.confirm_count,
        device=args.device,
    )


def reset_detector(detector, args):
    if hasattr(detector, "reset"):
        detector.reset()
        return detector

    return build_detector(args)


def main():
    parser = argparse.ArgumentParser(
        description="GuardianHelm hybrid pipeline"
    )

    parser.add_argument(
        "--video",
        default="0",
        help="'0' cho webcam hoặc đường dẫn video",
    )

    parser.add_argument(
        "--gps",
        nargs=2,
        type=float,
        default=[21.0285, 105.8542],
        metavar=("LAT", "LON"),
    )

    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--cv_mode",
        choices=["heuristic", "hybrid"],
        default="heuristic",
        help=(
            "'heuristic' (mặc định) chạy ngay không cần setup gì thêm - đã "
            "kiểm chứng trên 2449 clip EGOFALLS thật (recall 98.3%%/FP 41.7%%). "
            "'hybrid' cần --checkpoint trỏ tới model đã train bằng "
            "cv_deep_learning/train_model.py (chưa có checkpoint nào được "
            "train trong repo này - dùng hybrid sẽ báo lỗi nếu thiếu file)."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoint.pt",
    )

    parser.add_argument(
        "--heuristic_gate",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--deep_threshold",
        type=float,
        default=None,
        help=(
            "Ghi đè threshold trong checkpoint; "
            "mặc định dùng threshold đã lưu"
        ),
    )

    parser.add_argument(
        "--confirm_count",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--buffer_size",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Ví dụ: cuda, cuda:0 hoặc cpu",
    )

    args = parser.parse_args()

    source = (
        int(args.video)
        if args.video.isdigit()
        else args.video
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(
            f"Không mở được video: {args.video!r}"
        )

    # Đọc từ file nhanh hơn phát thật rất nhiều (không bị giới hạn tốc độ
    # camera) - CVAnomalyDetector/HybridFallDetector dùng time.time() thực
    # cho các cửa sổ baseline/smooth/stillness, nên cần throttle về đúng FPS
    # gốc khi test bằng file, nếu không các cửa sổ đó sẽ không khớp với thời
    # gian video thật (dẫn tới bỏ sót sự kiện ngã). Webcam (source là int)
    # không cần throttle vì đã tự nhiên theo thời gian thực.
    throttle = isinstance(source, str)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / video_fps
    next_frame_due = time.time()

    detector = build_detector(args)
    gps = tuple(args.gps)
    last_retry = time.time()

    print(
        f"GuardianHelm đang chạy; "
        f"source={args.video!r}; gps={gps}"
    )

    try:
        while True:
            ok, frame = cap.read()

            if throttle:
                sleep_time = next_frame_due - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_frame_due += frame_interval

            if not ok:
                print("Hết video hoặc mất camera.")
                break

            cv_result = detector.process_frame(frame)

            now = time.time()

            if (
                now - last_retry
                >= RETRY_INTERVAL_SEC
            ):
                retry_pending_alerts()
                last_retry = now

            if cv_result is None:
                continue

            if cv_result.deep_score is not None:
                print(
                    "[CV] "
                    f"heuristic={cv_result.heuristic_score:.3f} "
                    f"deep={cv_result.deep_score:.3f} "
                    f"hybrid={cv_result.anomaly_score:.3f} "
                    f"decision={cv_result.decision.value} "
                    f"latency={cv_result.inference_ms:.1f}ms"
                )

            if not cv_result.is_suspect:
                continue

            print(
                "[MAIN] Fall confirmed; "
                "starting audio confirmation"
            )

            audio_result = ask_and_classify(
                cv_result,
                timeout_sec=args.timeout_sec,
            )

            decision = decide_action(
                cv_result,
                audio_result,
            )

            if (
                decision.action
                == ActionType.ESCALATE
            ):
                sent = send_alert(
                    decision,
                    gps,
                )

                print(
                    "[MAIN] Alert result:",
                    "sent" if sent else "queued",
                )

            time.sleep(
                COOLDOWN_AFTER_DECISION_SEC
            )

            detector = reset_detector(
                detector,
                args,
            )

    except KeyboardInterrupt:
        print("\nĐã dừng GuardianHelm.")

    finally:
        cap.release()


if __name__ == "__main__":
    main()