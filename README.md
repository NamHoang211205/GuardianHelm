
# GuardianHelm

Phát hiện tai nạn & cảnh báo khẩn cấp cho người đi xe máy, xử lý hoàn toàn on-device trên Qualcomm QCS8550.

## Bối cảnh & Vấn đề

Người đi phượt hoặc người đi xe máy đường dài (thường qua vùng núi, đèo, khu vực hẻo lánh) đối mặt với rủi ro đặc thù: nếu xảy ra tai nạn ở nơi vắng người, không có ai chứng kiến, và bản thân người lái bất tỉnh hoặc không thể tự gọi cứu hộ, thời gian phát hiện và ứng cứu có thể kéo dài hàng giờ. GuardianHelm rút ngắn khoảng thời gian đó bằng cách tự động phát hiện tai nạn và kích hoạt quy trình cứu hộ mà không cần con người chủ động thao tác.

## Ý tưởng cốt lõi

Thiết bị nhỏ gọn gắn trên mũ bảo hiểm (camera + microphone + loa, chạy trên QCS8550), liên tục theo dõi trạng thái người lái. Khi nghi ngờ có tai nạn, hệ thống **không gọi cấp cứu ngay** mà hỏi thăm bằng giọng nói trước, chỉ escalate khi không có phản hồi hợp lệ. Toàn bộ xử lý — CV, ASR, TTS — chạy **offline, không phụ thuộc mạng**, vì vùng núi/hẻo lánh (đúng nơi sản phẩm cần hoạt động) thường mất sóng.

```
Camera (egocentric)
      │
      ▼
[Module 1: CV] ── phát hiện bất thường (affine RANSAC + adaptive baseline + impact→stillness)
      │ (nếu suspect)
      ▼
[Module 2: Audio] ── hỏi "Bạn có ổn không?" → phân loại phản hồi (offline, PhoWhisper)
      │
      ▼
[Module 3: Decision] ── rule-based fusion → cancel / escalate
      │ (nếu escalate)
      ▼
[Module 4: Alert] ── SMS/gọi kèm GPS (Twilio, có SQLite retry queue khi mất mạng)
```

## Cấu trúc repo

```
GuardianHelm/
├── interface.py              # Hợp đồng input/output giữa các module — ĐỌC TRƯỚC KHI SỬA CODE
├── main.py                   # Ghép pipeline end-to-end, entrypoint chạy thật
├── audio_module.py           # Hỏi thăm qua TTS + ASR (PhoWhisper) + phân loại phản hồi
├── decision_module.py        # Rule-based fusion + gửi cảnh báo (Twilio/SQLite queue)
├── cv_heuristic/
│   └── cv_module.py           # Cách 1 — heuristic nhẹ, không cần train, chạy CPU trực tiếp
├── cv_deep_learning/
│   ├── organize_egofalls.py   # Sắp xếp dataset EGOFALLS đã tải về data/normal, data/fall
│   ├── validate_dataset.py    # Lọc video hỏng trước khi train
│   ├── preprocessing.py       # Tiền xử lý clip cho mc3_18
│   ├── train_model.py         # Fine-tune mc3_18 (chạy trên GPU)
│   ├── inference.py           # Load checkpoint đã train, dùng trong cv_hybrid
│   └── deploy_qcs8550.py      # Compile + quantize checkpoint lên QCS8550 qua Qualcomm AI Hub
├── cv_hybrid/
│   └── hybrid_detector.py     # Cách 2 — cascade: heuristic gate → deep model xác nhận
├── requirement.txt
└── README.md
```

`data/` (dataset) không commit lên Git — xem mục Dataset bên dưới.

## Tech stack

| Phần | Công nghệ |
|---|---|
| CV — Cách 1 (heuristic) | `opencv-python` (sparse optical flow Lucas-Kanade + affine RANSAC), `numpy` — không cần train, chạy CPU, latency ~4-6ms/frame |
| CV — Cách 2 (deep learning) | `torch`/`torchvision` (mc3_18, pretrain Kinetics-400), fine-tune trên EGOFALLS |
| Audio | PhoWhisper qua `transformers` (ASR offline), `pyttsx3` (TTS offline), `sounddevice` (thu âm), keyword-matching thuần Python — không dùng API cloud để tránh phụ thuộc mạng |
| Decision & Alert | Python state machine, `twilio` (SMS/call), `sqlite3` (store-and-forward khi mất mạng) |
| Deploy QCS8550 | Qualcomm AI Hub (`qai_hub`, `qai_hub_models`), runtime QNN |

## Bắt đầu

### 1. Cài môi trường

```bash
pip install -r requirement.txt
```

### 2. Chạy thử ngay (không cần dataset/checkpoint gì cả)

Cách 1 (heuristic) là mặc định, đã kiểm chứng trên 2449 clip EGOFALLS thật (recall 98.3% / false-positive 41.7%) — chạy được ngay:

```bash
# webcam thật
python main.py --video 0

# hoặc test bằng file video có sẵn, không cần webcam (vd server không có camera)
python main.py --video path/to/clip.mp4
```

Muốn test riêng từng module:

```bash
python cv_heuristic/cv_module.py --video 0          # chỉ CV, in anomaly score + latency
python audio_module.py                               # chỉ audio, cần mic + loa thật
python decision_module.py                             # chỉ decision, chạy 3 kịch bản demo
```

Cấu hình cảnh báo thật (không bắt buộc — thiếu thì chạy ở chế độ demo, chỉ log console):

```bash
export EMERGENCY_CONTACTS="+84900000000,+84911111111"
export TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=... TWILIO_FROM_NUMBER=...
```

### 3. (Tùy chọn) Train Cách 2 — deep learning

Chỉ cần nếu muốn dùng `--cv_mode hybrid` (chưa có checkpoint nào được train sẵn trong repo này).

Tải dataset EGOFALLS (egocentric fall dataset, 10.948 clip, indoor+outdoor) từ [dataverse.nl/HO5GE3](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/HO5GE3), giải nén, rồi:

```bash
python cv_deep_learning/organize_egofalls.py --source EGOFALLS --dest data
python cv_deep_learning/validate_dataset.py --data_dir ./data
python cv_deep_learning/train_model.py --data_dir ./data --epochs 15 --output checkpoint.pt
python main.py --video 0 --cv_mode hybrid --checkpoint checkpoint.pt
```

Deploy lên QCS8550 thật qua Qualcomm AI Hub:

```bash
python cv_deep_learning/deploy_qcs8550.py --ckpt checkpoint.pt --calib_dir data/normal
```

## Quy tắc làm việc chung

- **Đọc `interface.py` trước khi sửa module nào** — hợp đồng input/output đã chốt, mỗi module implement đúng chữ ký hàm để ghép được mà không cần sửa lại chỗ khác.
- Không commit dataset (`.mp4`, `.zip` lớn), `checkpoint.pt`, hay API token lên Git — đã có trong `.gitignore`.
- Model benchmark trên QCS8550 dùng Qualcomm AI Hub, ưu tiên runtime **QNN** thay vì TFLITE (nhanh hơn đáng kể theo số liệu đã đo).

## License

Dự án phục vụ SoICT Summer School Edge AI 2026, tài trợ bởi Qualcomm. Lưu ý dataset EGOFALLS dùng license CC BY-NC 4.0 — chỉ phi thương mại.
