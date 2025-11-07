# Lưu với tên: src/preprocessing/04_create_final_csvs.py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

# --- CẤU HÌNH ---
# Thư mục chứa các khuôn mặt đã cắt
FACES_DIR = Path("data/processed/face_images")
# Thư mục lưu file CSV cuối cùng (Nơi main.py sẽ đọc)
FINAL_CSV_DIR = Path("data/final_csvs")


# --- KẾT THÚC CẤU HÌNH ---

def create_final_csvs():
    FINAL_CSV_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = FACES_DIR / split
        if not split_dir.exists():
            print(f"Không tìm thấy thư mục {split_dir}, bỏ qua.")
            continue

        print(f"Đang quét {split_dir}...")
        image_files = []

        # Quét 'real'
        for img_path in tqdm(split_dir.rglob("real/*.png"), desc=f"{split} (real)"):
            image_files.append({
                "image_path": str(img_path.resolve()),
                "label": 0
            })

        # Quét 'fake'
        for img_path in tqdm(split_dir.rglob("fake/*.png"), desc=f"{split} (fake)"):
            image_files.append({
                "image_path": str(img_path.resolve()),
                "label": 1
            })

        if not image_files:
            print(f"Không tìm thấy ảnh nào trong {split_dir}.")
            continue

        # Tạo DataFrame và lưu
        df = pd.DataFrame(image_files)
        df = df.sample(frac=1).reset_index(drop=True)  # Xáo trộn

        output_path = FINAL_CSV_DIR / f"{split}.csv"
        df.to_csv(output_path, index=False)

        print(f"Đã tạo {split}.csv: {len(df)} ảnh ({len(df[df.label == 1])} fake, {len(df[df.label == 0])} real)")

    print("\nHoàn tất! Dữ liệu đã sẵn sàng cho main.py.")


if __name__ == "__main__":
    create_final_csvs()