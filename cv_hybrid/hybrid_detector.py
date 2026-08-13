import time
from collections import deque

import numpy as np

from cv_deep_learning.inference import DeepFallClassifier
from cv_heuristic.cv_module import CVAnomalyDetector
from interface import CVDecision, CVResult


class HybridFallDetector:
    """
    Cascade detector:

    Layer 1:
        Heuristic nhẹ chạy liên tục.

    Layer 2:
        Deep model chỉ chạy khi heuristic vượt gate.

    Audio chỉ được kích hoạt khi deep model xác nhận fall đủ số
    lần liên tiếp.
    """

    def __init__(
        self,
        checkpoint: str,
        buffer_size: int = 48,
        heuristic_gate: float = 0.45,
        deep_threshold: float | None = None,
        confirm_count: int = 2,
        device: str | None = None,
    ):
        if buffer_size < 16:
            raise ValueError(
                "buffer_size phải lớn hơn hoặc bằng 16"
            )

        if confirm_count < 1:
            raise ValueError(
                "confirm_count phải lớn hơn hoặc bằng 1"
            )

        self.checkpoint = checkpoint
        self.buffer_size = buffer_size
        self.heuristic_gate = heuristic_gate
        self.deep_threshold_override = deep_threshold
        self.confirm_count = confirm_count
        self.device = device

        self.deep = DeepFallClassifier(
            checkpoint=checkpoint,
            device=device,
            threshold=deep_threshold,
        )

        self.reset()

    def reset(self):
        self.heuristic = CVAnomalyDetector()
        self.frames = deque(
            maxlen=self.buffer_size
        )
        self.consecutive_positive = 0

    def _normal_result(
        self,
        heuristic_result: CVResult,
    ) -> CVResult:
        return CVResult(
            anomaly_score=heuristic_result.anomaly_score,
            heuristic_score=heuristic_result.anomaly_score,
            deep_score=None,
            is_suspect=False,
            decision=CVDecision.NORMAL,
            signal_source="heuristic",
            timestamp=heuristic_result.timestamp,
            inference_ms=0.0,
            debug_info={
                **heuristic_result.debug_info,
                "deep_model_called": False,
                "positive_streak": 0,
            },
        )

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> CVResult | None:
        # copy để tránh camera/backend tái sử dụng vùng nhớ frame.
        self.frames.append(frame.copy())

        heuristic_result = (
            self.heuristic.process_frame(frame)
        )

        # Frame bị heuristic skip.
        if heuristic_result is None:
            return None

        heuristic_score = (
            heuristic_result.anomaly_score
        )

        if (
            heuristic_score < self.heuristic_gate
            or len(self.frames) < 16
        ):
            self.consecutive_positive = 0
            return self._normal_result(
                heuristic_result
            )

        deep_score, inference_ms = (
            self.deep.predict(list(self.frames))
        )

        deep_positive = (
            deep_score >= self.deep.threshold
        )

        if deep_positive:
            self.consecutive_positive += 1
        else:
            self.consecutive_positive = 0

        confirmed = (
            self.consecutive_positive
            >= self.confirm_count
        )

        # Deep model là nguồn xác nhận chính.
        hybrid_score = (
            0.25 * heuristic_score
            + 0.75 * deep_score
        )

        if confirmed:
            decision = CVDecision.FALL
        else:
            decision = CVDecision.SUSPECT

        return CVResult(
            anomaly_score=hybrid_score,
            heuristic_score=heuristic_score,
            deep_score=deep_score,
            is_suspect=confirmed,
            decision=decision,
            signal_source="hybrid",
            timestamp=time.time(),
            inference_ms=inference_ms,
            debug_info={
                **heuristic_result.debug_info,
                "deep_model_called": True,
                "deep_positive": deep_positive,
                "deep_threshold": self.deep.threshold,
                "positive_streak": (
                    self.consecutive_positive
                ),
                "required_confirmations": (
                    self.confirm_count
                ),
            },
        )