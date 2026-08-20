"""
r3d_embedding.py — Trich embedding 512 chieu tu R3D-18 pretrained (Kinetics-400).

Port tu Kaggle notebook (cell 50). KHAC BIET so voi ban goc: model duoc
LAZY-LOAD (chi tai khi lan dau goi extract_r3d_embedding), khong tai ngay
luc import module - tranh viec chi "import" file nay cung phai tai weights
qua mang.
"""

import cv2
import numpy as np
import torch
import torchvision.models.video as video_models

KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

_model = None
_device = None


def _get_model():
    """Lazy-load model - chi tai pretrained weights o lan goi dau tien."""
    global _model, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)
        model.fc = torch.nn.Identity()  # bo lop classification cuoi, chi lay embedding
        model.eval().to(_device)
        _model = model
    return _model, _device


def extract_r3d_embedding(
    video_path: str,
    n_frames: int = 16,
    resize_dim: tuple[int, int] = (112, 112),
) -> np.ndarray | None:
    """Doc n_frames deu nhau tu video, dua qua r3d_18 pretrained, lay embedding 512 chieu."""
    try:
        model, device = _get_model()

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            return None

        indices = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((resize_dim[1], resize_dim[0], 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, resize_dim)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        clip = np.stack(frames).astype(np.float32) / 255.0
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)
        clip = (clip - KINETICS_MEAN) / KINETICS_STD
        clip = clip.to(device)

        with torch.no_grad():
            embedding = model(clip)

        return embedding.cpu().numpy().flatten()

    except Exception:
        return None