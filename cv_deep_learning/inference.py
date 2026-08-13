from pathlib import Path
from time import perf_counter

import torch

from cv_deep_learning.preprocessing import preprocess_frames


def build_model():
    from torchvision.models.video import mc3_18

    model = mc3_18(weights=None)
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        2,
    )
    return model


class DeepFallClassifier:
    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        threshold: float | None = None,
    ):
        checkpoint_path = Path(checkpoint)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {checkpoint_path}"
            )

        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        saved = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model = build_model()

        # Hỗ trợ cả checkpoint mới có metadata và state_dict cũ.
        if (
            isinstance(saved, dict)
            and "model_state" in saved
        ):
            state_dict = saved["model_state"]
            saved_threshold = float(
                saved.get("threshold", 0.5)
            )
        else:
            state_dict = saved
            saved_threshold = 0.5

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.threshold = (
            float(threshold)
            if threshold is not None
            else saved_threshold
        )

        print(
            "[DEEP] Loaded model:",
            checkpoint_path,
            f"device={self.device}",
            f"threshold={self.threshold:.3f}",
        )

    @torch.inference_mode()
    def predict(
        self,
        frames,
    ) -> tuple[float, float]:
        """
        Return:
            fall_probability, inference_ms
        """
        started = perf_counter()

        clip = preprocess_frames(frames)
        clip = clip.to(
            self.device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(clip)
            fall_probability = (
                torch.softmax(logits, dim=1)[0, 1].item()
            )

        inference_ms = (
            perf_counter() - started
        ) * 1000.0

        return fall_probability, inference_ms