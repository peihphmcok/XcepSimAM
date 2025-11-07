## XcepMamba: Kiến trúc Lai CNN-SSM để Phát hiện Deepfake

XcepMamba là một kiến trúc lai (hybrid) mới kết hợp sức mạnh của mạng nơ-ron tích chập (CNN) với các Mô hình Không gian Trạng thái hiện đại (State-Space Models - SSM) nhằm phát hiện giả mạo khuôn mặt (deepfake). Mục tiêu của dự án là vượt qua những hạn chế của CNN truyền thống bằng cách bổ sung khả năng học biểu diễn toàn cục và mối quan hệ ngữ cảnh trên khuôn mặt.

### Các cải tiến cốt lõi
- **Backbone Xception**: Sử dụng XceptionNet làm bộ trích xuất đặc trưng cục bộ mạnh mẽ, nhạy với các vết tích (artifacts) ở cấp độ pixel.
- **Attention SOTA & Middle Flow phân cấp**: Thay thế các module attention cũ (như CBAM) bằng SimAM (không tham số, hiệu quả) và triển khai Middle Flow "Hierarchical Parallel" để học biểu diễn phong phú.
- **Mamba Encoder Head**: Loại bỏ Exit Flow truyền thống của Xception, thay thế bằng Mamba Encoder giúp mô hình nắm bắt mối quan hệ toàn cục giữa các vùng trên khuôn mặt với độ phức tạp tuyến tính `O(N)`.

Dự án được tổ chức để phục vụ các thí nghiệm bóc tách (ablation study) chi tiết, chứng minh đóng góp của từng thành phần.

## Cấu trúc Thư mục
```
XcepMamba_Deepfake_Detection/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── real_videos/
│   │   └── fake_videos/
│   │
│   ├── processed/
│   │   ├── video_lists/
│   │   │   ├── master_video_list.csv
│   │   │   ├── train_videos.csv
│   │   │   └── ...
│   │   └── face_images/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   └── final_csvs/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   ├── preprocessing/
│   │   ├── 01_create_file_list.py
│   │   ├── 02_split_dataset.py
│   │   ├── 03_extract_faces.py
│   │   └── 04_create_final_csvs.py
│   │
│   ├── models/
│   │   └── master_model.py
│   │
│   ├── utils/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   ├── losses.py
│   │   └── ...
│   │
│   ├── main.py
│   └── train.py
│
├── models/
│   └── ablation_study/
│       ├── FINAL_XcepMamba/
│       │   └── best_model.pth
│       └── ...
│
└── results/
    ├── ablation_study/
    │   ├── FINAL_XcepMamba/
    │   │   ├── confusion_matrix.png
    │   │   └── results.json
    │   └── ...
    │
    └── final_ablation_summary.csv
```

## Cài đặt

Clone repository:
```bash
git clone https://your-repo-url/XcepMamba_Deepfake_Detection.git
cd XcepMamba_Deepfake_Detection
```

Cài đặt thư viện bắt buộc (khuyến khích tạo môi trường ảo):
```bash
pip install -r requirements.txt
```

Cài đặt các thư viện chuyên biệt:
```bash
pip install mamba-ssm
```

## Quy trình Tiền xử lý (4 Bước)

### Bước 0: Chuẩn bị dữ liệu thô
- Tải các bộ dữ liệu (FaceForensics++, WildDeepfake, ...), đặt vào `data/raw/` theo cấu trúc:
  - `data/raw/real_videos/`
  - `data/raw/fake_videos/`
- Để so sánh SOTA công bằng, xử lý từng bộ dữ liệu riêng biệt. Chạy Bước 1–4 cho từng bộ, xoá `data/processed` trước khi chuyển sang bộ khác.

### Bước 1: Tạo danh sách video
```bash
cd src/preprocessing/
python 01_create_file_list.py
```
- Sinh `master_video_list.csv` liệt kê đường dẫn video và nhãn.

### Bước 2: Chia dữ liệu chống rò rỉ (data leakage)
```bash
python 02_split_dataset.py
```
- Chia cấp độ video thành `train_videos.csv`, `val_videos.csv`, `test_videos.csv`.

### Bước 3: Trích xuất khuôn mặt
```bash
python 03_extract_faces.py
```
- Đọc danh sách video, trích khung hình (OpenCV/ffmpeg), phát hiện khuôn mặt bằng MTCNN hoặc RetinaFace, mở rộng bounding box 20–30%, cắt và lưu vào `data/processed/face_images/`.
- Bước này tốn thời gian và tài nguyên; nên dùng GPU.

### Bước 4: Tạo CSV cuối cùng
```bash
python 04_create_final_csvs.py
```
- Quét thư mục `face_images`, tạo `train.csv`, `val.csv`, `test.csv` trong `data/final_csvs/`. Đây là input chính cho `src/main.py`.
- Đảm bảo cấu hình trong `src/main.py` (hoặc file config) trỏ tới các CSV này.

## Huấn luyện & Thí nghiệm

Quay lại thư mục gốc dự án:
```bash
cd ../..
cd src/
```

### Cấu hình thí nghiệm
- Mở `src/main.py`, cập nhật danh sách `EXPERIMENT_CONFIGS`. Đây là nơi định nghĩa các thí nghiệm bóc tách.

Ví dụ:
```python
EXPERIMENT_CONFIGS = [
    {
        "id": "FINAL_XcepMamba",
        "middle_flow": "hierarchical",
        "attention": "simam",
        "encoder": "mamba"
    },
    {
        "id": "Ablation_T4_GAP_Encoder",
        "middle_flow": "hierarchical",
        "attention": "simam",
        "encoder": "gap"
    }
]
```

### Chạy huấn luyện
```bash
python main.py
```
- Script sẽ lặp qua từng cấu hình, huấn luyện, đánh giá và lưu mô hình (`models/ablation_study/...`) cùng kết quả (`results/ablation_study/...`).

### Phân tích kết quả
- Kiểm tra `results/ablation_study/` để xem biểu đồ, ma trận nhầm lẫn, JSON thống kê.
- File tổng kết `results/final_ablation_summary.csv` chứa bảng so sánh cuối cùng (định dạng Markdown) sẵn sàng đưa vào báo cáo.

## Ghi chú & Khuyến nghị
- Chỉ đưa ảnh khuôn mặt đã cắt vào mô hình; feed full-frame sẽ làm giảm hiệu suất.
- Theo dõi chất lượng phát hiện khuôn mặt, loại bỏ hoặc gắn cờ các mẫu sai.
- Cân bằng số lượng ảnh thật và giả ở từng tập để ổn định quá trình học.
- Lưu đệm các ảnh mặt đã cắt để tiết kiệm thời gian khi chạy lại thí nghiệm.


