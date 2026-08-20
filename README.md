# GuardianFall

Phát hiện té/ngã & hỗ trợ khẩn cấp tự động, xử lý on-device trên Qualcomm QCS8550 (Snapdragon 8550).

## Bối cảnh & Vấn đề

Ngã có thể xảy ra ở nhiều bối cảnh — trong nhà, ngoài đường, khu vực vắng người — và nạn nhân (người đi phượt một mình, người già, trẻ nhỏ...) có thể không thể tự gọi trợ giúp do bất tỉnh, đau đớn hoặc mất khả năng phản ứng. GuardianHelm rút ngắn thời gian phát hiện và ứng cứu bằng cách chạy hoàn toàn on-device, không phụ thuộc kết nối mạng.

## Kiến trúc hệ thống

Chia làm 2 giai đoạn: **huấn luyện** (Kaggle) và **triển khai** (QCS8550).

```
EGOFALLS (egofalls-1 + egofalls-v2)
      │
      ▼
┌─────────────────────────────────────────────┐
│   KAGGLE NOTEBOOK — TRAINING                  │
│   GPU T4/P100                                  │
│                                               │
│   [Audio pipeline]        [R3D-18 pipeline]    │
│    MFCC, spectral,         Pretrained embedding │
│    RMS, ZCR                512-d                │
│         │                       │              │
│    RandomForest (RF1)     RandomForest (RF2)    │
│         └──────────┬────────────┘              │
│                     ▼                          │
│           Hybrid model fusion                   │
└─────────────────────────────────────────────┘
      │  model .pkl (audio + R3D classifier)
      ▼
┌─────────────────────────────────────────────┐
│   QCS8550 — ON-DEVICE RUNTIME (systemd)        │
│                                               │
│   Microphone Array ──▶ [guardian-voice]         │
│   Camera Interface ──▶ [guardian-cv-model]       │
│                         (R3D-18 qua QNN/NPU)     │
│                    [guardian-fall-model]         │
│                    (fusion quyết định cuối)       │
│              ┌─────────────┴─────────────┐      │
│              ▼                           ▼      │
│     Notification/Alarm         HTTP :8080/healthz│
└─────────────────────────────────────────────┘
```

## Cấu trúc repo

```
GuardianHelm/
├── data/
│   ├── __init__.py
│   └── manifest.py          # Duyệt thư mục EGOFALLS, sinh manifest CSV có nhãn
├── features/
│   ├── __init__.py
│   ├── audio_features.py     # MFCC, spectral, RMS, ZCR (ffmpeg + librosa)
│   ├── r3d_embedding.py      # Embedding 512-d từ R3D-18 pretrained (Kinetics-400)
│   ├── optical_flow_v1.py    # [Thử nghiệm — không dùng trong model cuối]
│   ├── optical_flow_v2.py    # [Thử nghiệm — không dùng trong model cuối]
│   └── point_tracking.py     # [Thử nghiệm — không dùng trong model cuối]
├── train_audio_model.py      # Train + đánh giá + lưu model nhánh Audio
├── train_r3d_model.py        # Train + đánh giá + lưu model nhánh R3D
├── predict.py                 # Chạy cả 2 model trên 1 video, in kết quả song song
├── requirement.txt
├── .gitignore
└── README.md
```

`data/` (dataset thật, video) không commit lên Git — chỉ có `data/manifest.py` là code, xem mục Dataset bên dưới.

## Tech stack

| Phần | Công nghệ |
|---|---|
| Audio pipeline | `librosa` + `ffmpeg` (MFCC, spectral centroid, RMS, ZCR), `scikit-learn` (RandomForest) |
| CV pipeline | `torch`/`torchvision` (R3D-18, pretrain Kinetics-400, dùng làm feature extractor) + RandomForest |
| Deploy QCS8550 | Qualcomm AI Hub (`qai_hub`), QNN runtime (R3D-18 chạy trên Hexagon NPU) |
| Runtime on-device | systemd (`guardian-voice.service`), HTTP health check (`:8080/healthz`) |

## Dataset

**EGOFALLS** (egofalls-1 + egofalls-v2) — Đại học Groningen, ICPR 2024. Camera egocentric gắn cổ/thắt lưng, 4 loại ngã (back/downside/front/lateral falls) + 8 hoạt động sinh hoạt bình thường (Bending, Lying, Rising, SittingDown, SittingStatic, SquattingDown, Stumbling, Walking), cả indoor và outdoor.

License **CC BY-NC 4.0** — chỉ phi thương mại.

## Bắt đầu

### 1. Cài môi trường

```bash
pip install -r requirement.txt
```

Cần cài thêm `ffmpeg` ở cấp hệ điều hành cho phần audio:
```bash
sudo apt install ffmpeg   # Ubuntu/Debian
brew install ffmpeg        # macOS
```

### 2. Xây dựng manifest từ dataset

```bash
python data/manifest.py \
    --roots egofalls-1=/path/to/egofalls-1 egofalls-v2=/path/to/egofalls-v2 \
    --output full_manifest.csv
```

### 3. Train 2 nhánh model

```bash
python train_audio_model.py --manifest full_manifest.csv --output_dir ./models
python train_r3d_model.py --manifest full_manifest.csv --output_dir ./models
```

R3D chậm hơn Audio nhiều — có thể test nhanh trên tập nhỏ trước bằng `--sample_size 600`.

### 4. Dự đoán thử trên 1 video

```bash
python predict.py --video path/to/clip.mp4 --models_dir ./models
```

### 5. Deploy lên QCS8550

> Script compile/quantize R3D-18 qua Qualcomm AI Hub (QNN) chưa có trong repo — cần bổ sung `deploy_qcs8550.py` trước khi thực hiện bước này.

## Quy tắc làm việc chung

- Không commit dataset (`.mp4`, `.zip` lớn), checkpoint (`.pkl`), hay API token lên Git — đã có trong `.gitignore`.
- Đánh giá dùng `GroupKFold` theo `subject`, không phải random split, để tránh rò rỉ dữ liệu.
- Model R3D-18 benchmark trên QCS8550 dùng Qualcomm AI Hub, ưu tiên runtime **QNN** thay vì TFLite.

## License

Dự án phục vụ SoICT Summer School Edge AI 2026, tài trợ bởi Qualcomm. Dataset EGOFALLS dùng license CC BY-NC 4.0 — chỉ phi thương mại.