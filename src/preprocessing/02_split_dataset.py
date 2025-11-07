# Lưu với tên: src/preprocessing/02_split_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# --- CẤU HÌNH ---
INPUT_DIR = Path("data/processed/video_lists")
MASTER_FILE = INPUT_DIR / "master_video_list.csv"
TRAIN_RATIO = 0.8  # 80% cho training
VAL_RATIO = 0.1  # 10% cho validation
TEST_RATIO = 0.1  # 10% cho testing


# --- KẾT THÚC CẤU HÌNH ---

def split_dataset():
    if not MASTER_FILE.exists():
        print(f"Lỗi: Không tìm thấy file {MASTER_FILE}.")
        print("Vui lòng chạy script 01_create_file_list.py trước.")
        return

    df = pd.read_csv(MASTER_FILE)

    if df.empty:
        print(f"File {MASTER_FILE} bị rỗng. Không thể chia.")
        return

    # Lấy nhãn để chia (stratify)
    labels = df['label']

    # Chia lần 1: Tách train (80%) và temp (20%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_RATIO),
        stratify=labels,
        random_state=42
    )

    # Chia lần 2: Chia temp (20%) thành val (10%) và test (10%)
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_df['label'],
        random_state=42
    )

    # Lưu 3 file CSV
    train_df.to_csv(INPUT_DIR / "train_videos.csv", index=False)
    val_df.to_csv(INPUT_DIR / "val_videos.csv", index=False)
    test_df.to_csv(INPUT_DIR / "test_videos.csv", index=False)

    print("Hoàn tất chia dữ liệu (video-level)!")
    print(
        f"Train: {len(train_df)} video ({len(train_df[train_df.label == 1])} fake, {len(train_df[train_df.label == 0])} real)")
    print(f"Val:   {len(val_df)} video ({len(val_df[val_df.label == 1])} fake, {len(val_df[val_df.label == 0])} real)")
    print(
        f"Test:  {len(test_df)} video ({len(test_df[test_df.label == 1])} fake, {len(test_df[test_df.label == 0])} real)")


if __name__ == "__main__":
    split_dataset()