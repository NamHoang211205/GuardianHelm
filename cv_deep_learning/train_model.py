"""
train_model.py — Fine-tune ResNet-Mixed-Convolution (kien truc mc3_18) tren
data/normal/ va data/fall/ da chuan bi tu organize_egofalls.py.

QUAN TRONG: Script nay CHAY TREN GPU (Thunder Compute), KHONG PHAI QCS8550.
QCS8550 chi dung o buoc SAU (deploy_to_qcs8550.py) de CHAY model da train
xong, khong dung de train. Checkpoint (.pt) sinh ra tu script nay chinh la
input cho deploy_to_qcs8550.py.

CACH CHAY:
    python train_model.py --data_dir ./data --epochs 15 --output checkpoint.pt

Neu may khong co GPU / khong co mang de tai pretrained weights (vi du test
o moi truong sandbox), dung co --no_pretrained de bo qua buoc tai weights
Kinetics-400, chi de test luong code chay dung khong loi.
"""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

NUM_FRAMES = 16          # so frame/clip - chuan dau vao cua mc3_18
FRAME_SIZE = 112          # do phan giai H=W - chuan cua mc3_18


class VideoClipDataset(Dataset):
    """
    Doc video tu data/normal/ (label=0) va data/fall/ (label=1), lay mau
    NUM_FRAMES frame cach deu nhau trong moi video, resize ve FRAME_SIZE.
    """

    def __init__(self, video_label_pairs: list[tuple[Path, int]]):
        self.samples = video_label_pairs

    def __len__(self):
        return len(self.samples)

    def _load_clip(self, path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return torch.zeros(3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)

        indices = np.linspace(0, max(total_frames - 1, 0), NUM_FRAMES).astype(int)
        idx_set = set(indices.tolist())
        current = 0
        collected = {}

        while len(collected) < len(idx_set):
            ret, frame = cap.read()
            if not ret:
                break
            if current in idx_set:
                frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                collected[current] = frame
            current += 1
        cap.release()

        ordered = [collected.get(i, collected[max(collected.keys())] if collected else
                                   np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))
                   for i in indices]

        clip = np.stack(ordered, axis=0).astype(np.float32) / 255.0
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2)
        return clip

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = self._load_clip(path)
        return clip, label


def build_dataset(data_dir: str) -> list[tuple[Path, int]]:
    normal_videos = list((Path(data_dir) / "normal").glob("*.mp4"))
    fall_videos = list((Path(data_dir) / "fall").glob("*.mp4"))

    print(f"Tim thay {len(normal_videos)} video normal, {len(fall_videos)} video fall")
    if len(normal_videos) == 0 or len(fall_videos) == 0:
        raise ValueError("Thieu du lieu o mot trong 2 lop (normal/fall) - kiem tra lai --data_dir")

    pairs = [(p, 0) for p in normal_videos] + [(p, 1) for p in fall_videos]
    random.shuffle(pairs)
    return pairs


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    from torchvision.models.video import mc3_18, MC3_18_Weights

    weights = MC3_18_Weights.KINETICS400_V1 if pretrained else None
    model = mc3_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(clips)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * clips.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-Mixed-Convolution (mc3_18) cho fall detection")
    parser.add_argument("--data_dir", default="./data", help="Thu muc chua data/normal/ va data/fall/")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--output", default="checkpoint.pt")
    parser.add_argument("--no_pretrained", action="store_true",
                         help="Bo qua tai pretrained Kinetics-400 (dung khi test offline, khong co mang)")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang dung device: {device}")
    if device.type == "cpu":
        print("[CANH BAO] khong tim thay GPU - fine-tune tren CPU se rat cham. "
              "Kiem tra lai driver CUDA neu dang chay tren Thunder Compute.")

    pairs = build_dataset(args.data_dir)
    dataset = VideoClipDataset(pairs)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print(f"Chia du lieu: {train_size} train / {val_size} validation")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    print(f"Dang khoi tao model mc3_18 (pretrained={not args.no_pretrained})...")
    model = build_model(num_classes=2, pretrained=not args.no_pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    print("\nBat dau training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> luu checkpoint moi (val_acc={val_acc:.3f}) vao {args.output}")

    print(f"\nHoan tat training. Checkpoint tot nhat (val_acc={best_val_acc:.3f}) da luu tai: {args.output}")
    print("Buoc tiep theo: dung deploy_to_qcs8550.py de bien dich + quantize checkpoint nay len QCS8550.")


if __name__ == "__main__":
    main()