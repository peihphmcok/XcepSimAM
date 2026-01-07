\## Cấu trúc dự án



Để code chạy chính xác, sắp xếp thư mục như sau:



XcepSimAM/

├── src/

│   ├── data/

│   │   └── ff\_frame/           # Chứa các thư mục ảnh đã trích xuất

│   │       ├── fake\_Deepfakes\_000\_003/

│   │       ├── youtube\_.../

│   │       └── ...

│   ├── models/

│   │   ├── \_\_init\_\_.py         # (File rỗng)

│   │   └── XcepSimAM.py        # Code kiến trúc mô hình

│   ├── preprocessing/

│   │   └── splits/             # Chứa file CSV nhãn dữ liệu

│   │       ├── train.csv

│   │       ├── val.csv

│   │       └── test.csv

│   ├── runs/                   # Tự động tạo khi chạy (lưu log, model checkpoint)

│   ├── main.py                 # File huấn luyện chính

│   └── requirements.txt        # Các thư viện cần thiết







\## Thay đổi main



Thay đổi batchsize lên 64 hoặc 128 nếu máy khỏe



Thay file main.py ở bên ngoài vào file main.py trong file zip và chạy



Path da được cập nhật tự động, chỉ cần dat đúng vị trí thư mục là ok

