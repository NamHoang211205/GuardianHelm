# GuardianHelm

Phát hiện tai nạn & cảnh báo khẩn cấp cho người đi xe máy, xử lý hoàn toàn on-device trên Qualcomm QCS8550.

## Bối cảnh & Vấn đề

Người đi phượt hoặc người đi xe máy đường dài (thường qua vùng núi, đèo, khu vực hẻo lánh) đối mặt với rủi ro đặc thù: nếu xảy ra tai nạn ở nơi vắng người, không có ai chứng kiến, và bản thân người lái bất tỉnh hoặc không thể tự gọi cứu hộ, thời gian phát hiện và ứng cứu có thể kéo dài hàng giờ. HelmetGuard rút ngắn khoảng thời gian đó bằng cách tự động phát hiện tai nạn và kích hoạt quy trình cứu hộ mà không cần con người chủ động thao tác.

## Ý tưởng cốt lõi

Thiết bị nhỏ gọn gắn trên mũ bảo hiểm (camera + microphone + loa, chạy trên QCS8550), liên tục theo dõi trạng thái người lái. Khi nghi ngờ có tai nạn, hệ thống **không gọi cấp cứu ngay** mà hỏi thăm bằng giọng nói trước, chỉ escalate khi không có phản hồi hợp lệ.

```
Camera (egocentric)
      │
      ▼
[Module 1: CV] ── phát hiện bất thường (optical flow + horizon detection)
      │ (nếu suspect)
      ▼
[Module 2: Audio] ── hỏi "Bạn có ổn không?" → phân loại phản hồi
      │
      ▼
[Module 3: Decision] ── rule-based fusion → cancel / escalate
      │ (nếu escalate)
      ▼
[Module 4: Alert] ── SMS/gọi kèm GPS đến người thân/cứu hộ
```

## Cấu trúc repo

```
HelmetGuard/
├── interfaces.py        # Hợp đồng input/output giữa 4 module — ĐỌC FILE NÀY TRƯỚC KHI CODE
├── cv_module.py          # Track 1 — optical flow + horizon detection
├── audio_module.py       # Track 3 — TTS hỏi thăm + ASR/keyword phân loại phản hồi
├── decision_module.py    # Track 4 — rule-based fusion + gửi cảnh báo (SMS/GPS)
├── hardware/              # Track 5 — script deploy/benchmark trên QCS8550 qua Qualcomm AI Hub
├── data/                  # Dataset (không commit dataset lớn lên Git — xem mục Dataset)
├── main.py                # Track 6 — ghép 4 module thành pipeline chạy end-to-end
├── requirements.txt
└── README.md
```

## Tech stack theo từng track

| Track | Việc chính | Công nghệ |
|---|---|---|
| 1 — CV Baseline | Phát hiện bất thường từ video | Python, `opencv-python` (optical flow, horizon detection), `numpy` |
| 2 — Data & Calibration | Thu thập & gán nhãn video | Kaggle & Thunder Compute |
| 3 — Audio | Hỏi thăm & nhận diện phản hồi | PhoWhisper (ASR), FPT.AI/Zalo TTS API, keyword-spotting bằng `re` |
| 4 — Decision & Alert | Ra quyết định + gửi cảnh báo | Python state machine, Twilio API (SMS/call), `sqlite3` (store-and-forward) |
| 5 — Hardware & Deploy | Benchmark/deploy lên QCS8550 | Qualcomm AI Hub (`qai_hub_models`), QNN runtime |
| 6 — Integration | Ghép pipeline end-to-end | Python, Git |

## Bắt đầu

### 1. Clone repo (trên server Thunder Compute dùng chung)

```bash
git clone https://github.com/<owner>/HelmetGuard.git
cd HelmetGuard
git config --global user.name "Ten cua ban"
git config --global user.email "email_gan_voi_github@example.com"
```

### 2. Cài môi trường

```bash
pip install -r requirements.txt
```

### 3. Tải dataset

Dataset lưu trên Kaggle (không commit lên GitHub — file lớn, nên nằm ngoài `.gitignore`):

```bash
pip install kaggle
# copy kaggle.json vào ~/.kaggle/ trước (xem hướng dẫn lấy API token trong Kaggle Settings)
kaggle datasets download -d <username>/<ten-dataset> -p ./data --unzip
```

Nguồn dataset đang dùng:
- **EGOFALLS** — egocentric fall dataset (indoor + outdoor), 10.948 clip — [dataverse.nl/HO5GE3](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/HO5GE3)
- **CCD (Car Crash Dataset)** — dashcam, có tình huống đang di chuyển → va chạm — [github.com/Cogito2012/CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset)
- Data tự quay: normal riding + mô phỏng ngã có kiểm soát (xem thư mục `data/`, log điều kiện quay trong Google Sheets chung)

### 4. Chạy thử pipeline (sau khi các module đã implement theo `interfaces.py`)

```bash
python main.py
```

## Quy tắc làm việc chung

- **Đọc `interfaces.py` trước khi code bất kỳ module nào** — đây là hợp đồng input/output đã chốt, mỗi track implement đúng chữ ký hàm để Track 6 ghép được mà không cần sửa lại.
- Mỗi track code trong file riêng, commit/push thường xuyên — tránh sửa trực tiếp file của track khác.
- Không commit dataset (`.mp4`, `.zip` lớn) hoặc `kaggle.json`/API token lên Git — đã có trong `.gitignore`.
- Model benchmark trên QCS8550 dùng Qualcomm AI Hub, ưu tiên runtime **QNN** thay vì TFLITE (nhanh hơn đáng kể theo số liệu đã đo).

## Model tham khảo (Qualcomm AI Hub)

Nếu baseline heuristic (optical flow/horizon) chưa đủ chính xác, fine-tune tiếp:
- **ResNet-Mixed-Convolution** — nhẹ nhất (11.7M tham số, 11.5MB sau quantize), pretrain Kinetics-400, hỗ trợ chính thức QCS8550 — [aihub.qualcomm.com/models/resnet_mixed](https://aihub.qualcomm.com/models/resnet_mixed)
- **ResNet-2Plus1D** — chính xác hơn nhưng nặng hơn ~3 lần, cân nhắc nếu baseline nhẹ không đủ tốt

## License

Dự án phục vụ SoICT Summer School Edge AI 2026, tài trợ bởi Qualcomm. Lưu ý dataset EGOFALLS dùng license CC BY-NC 4.0 — chỉ phi thương mại.