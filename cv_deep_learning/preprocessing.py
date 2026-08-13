import cv2
import numpy as np
import torch


NUM_FRAMES = 16
FRAME_SIZE = 112

KINETICS_MEAN = torch.tensor(
    [0.43216, 0.394666, 0.37645],
    dtype=torch.float32,
).view(3, 1, 1, 1)

KINETICS_STD = torch.tensor(
    [0.22803, 0.22145, 0.216989],
    dtype=torch.float32,
).view(3, 1, 1, 1)


def sample_frame_indices(
    total_frames: int,
    num_frames: int = NUM_FRAMES,
) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Video không có frame")

    return np.linspace(
        0,
        total_frames - 1,
        num_frames,
    ).astype(np.int64)


def preprocess_frames(
    frames: list[np.ndarray],
    num_frames: int = NUM_FRAMES,
) -> torch.Tensor:
    """
    Input:
        Danh sách frame BGR từ OpenCV.

    Output:
        Tensor (1, C, T, H, W), RGB và Kinetics-normalized.
    """
    if not frames:
        raise ValueError("Cần ít nhất một frame")

    indices = sample_frame_indices(
        len(frames),
        num_frames,
    )

    processed = []

    for index in indices:
        frame = cv2.resize(
            frames[index],
            (FRAME_SIZE, FRAME_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed.append(frame)

    clip = np.stack(processed).astype(np.float32) / 255.0

    tensor = torch.from_numpy(clip).permute(3, 0, 1, 2)
    tensor = (tensor - KINETICS_MEAN) / KINETICS_STD

    return tensor.unsqueeze(0)