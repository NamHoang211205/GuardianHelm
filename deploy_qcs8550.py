"""
deploy_qcs8550.py — Bien dich + quantize checkpoint TU TRAIN (khong phai
checkpoint co san cua Qualcomm) len QCS8550.

LY DO CAN SCRIPT RIENG NAY: CLI `qai-hub-models export resnet_mixed` CHI
nhan checkpoint co san (kinetics400_v1) qua flag --weights, KHONG nhan file
checkpoint tu train qua duong dan tuy y. De dung checkpoint tu fine-tune,
phai goi thang Python API (qai_hub.submit_compile_and_quantize_jobs).

CACH CHAY:
    python deploy_to_qcs8550.py --ckpt checkpoint.pt --calib_dir data/normal
"""

import argparse

import numpy as np
import torch
import qai_hub as hub


def load_finetuned_model(ckpt_path: str, num_classes: int = 2):
    """
    Load model mc3_18 (kien truc goc cua ResNet-Mixed-Convolution) tu
    torchvision, thay lop cuoi cho dung so class, roi nap trong so da
    fine-tune tu checkpoint.pt (file nay tao ra boi train_model.py).
    """
    from torchvision.models.video import mc3_18

    model = mc3_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_calibration_samples(calib_dir: str, num_samples: int, input_shape: tuple) -> list:
    """
    Doc mot so video mau tu data/normal (hoac data/fall) de lam calibration
    data cho buoc quantize - AI Hub can vai chuc mau de tinh khoang gia tri
    (min/max) khi anh xa FP32 -> INT8. Day KHONG phai buoc "chuyen dataset
    sang INT8" nhu da giai thich - chi la du lieu mau de tinh toan thong so
    quantize, ban than cac mau nay van la du lieu binh thuong.
    """
    import cv2
    from pathlib import Path

    videos = list(Path(calib_dir).glob("*.mp4"))[:num_samples]
    samples = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        frames = []
        while len(frames) < input_shape[2]:  # input_shape = (B, C, T, H, W)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (input_shape[4], input_shape[3]))
            frames.append(frame)
        cap.release()
        if len(frames) == input_shape[2]:
            clip = np.stack(frames, axis=0)  # (T, H, W, C)
            clip = clip.transpose(3, 0, 1, 2)  # (C, T, H, W)
            clip = clip[np.newaxis, ...].astype(np.float32) / 255.0  # (1, C, T, H, W)
            samples.append(clip)

    print(f"Da doc {len(samples)} mau calibration tu {calib_dir}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Bien dich + quantize checkpoint tu train len QCS8550")
    parser.add_argument("--ckpt", required=True, help="Duong dan checkpoint.pt tu train_model.py")
    parser.add_argument("--calib_dir", default="data/normal", help="Thu muc video dung lam calibration data")
    parser.add_argument("--num_calibration_samples", type=int, default=20)
    parser.add_argument("--device", default="QCS8550 (Proxy)", help="Ten thiet bi dich tren AI Hub")
    args = parser.parse_args()

    input_shape = (1, 3, 16, 112, 112)  # (batch, channel, so_frame, H, W) - chuan cua mc3_18

    print("Buoc 1/4: Load model + checkpoint da fine-tune...")
    model = load_finetuned_model(args.ckpt)

    print("Buoc 2/4: Trace model sang TorchScript...")
    example_input = torch.rand(input_shape)
    traced_model = torch.jit.trace(model, example_input)

    print("Buoc 3/4: Doc du lieu calibration...")
    calibration_samples = load_calibration_samples(args.calib_dir, args.num_calibration_samples, input_shape)
    calibration_data = {"video": calibration_samples}

    print("Buoc 4/4: Gui job bien dich + quantize len Qualcomm AI Hub (co the mat vai phut)...")
    job = hub.submit_compile_and_quantize_jobs(
        traced_model,
        device=hub.Device(args.device),
        calibration_data=calibration_data,
        input_specs={"video": (input_shape, "float32")},
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name="helmetguard_cv_model",
    )

    print(f"\nDa gui job thanh cong. Theo doi tien do tai: {job.url}")
    print("Sau khi xong, co the tai model da bien dich ve va chay profiling/benchmark tren thiet bi that.")


if __name__ == "__main__":
    main()