# Lưu với tên: src/preprocessing/01_create_file_list.py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

# --- CẤU HÌNH ---
# !! THAY ĐỔI ĐƯỜNG DẪN NÀY
RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/video_lists")
VIDEO_FORMATS = ['.mp4', '.avi', '.mov']


# --- KẾT THÚC CẤU HÌNH ---

def create_master_list():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_files = []

    # Quét thư mục 'real_videos'
    real_dir = RAW_DATA_DIR / "real_videos"
    if not real_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục {real_dir}")
    else:
        print(f"Scanning {real_dir}...")
        for ext in VIDEO_FORMATS:
            for video_path in tqdm(real_dir.rglob(f"*{ext}"), desc=f"Real ({ext})"):
                video_files.append({
                    "video_path": str(video_path.resolve()),
                    "label": 0  # 0 cho Real
                })

    # Quét thư mục 'fake_videos'
    fake_dir = RAW_DATA_DIR / "fake_videos"
    if not fake_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục {fake_dir}")
    else:
        print(f"Scanning {fake_dir}...")
        for ext in VIDEO_FORMATS:
            for video_path in tqdm(fake_dir.rglob(f"*{ext}"), desc=f"Fake ({ext})"):
                video_files.append({
                    "video_path": str(video_path.resolve()),
                    "label": 1  # 1 cho Fake
                })

    if not video_files:
        print("Không tìm thấy video nào. Vui lòng kiểm tra lại đường dẫn RAW_DATA_DIR.")
        return

    # Tạo DataFrame và lưu
    df = pd.DataFrame(video_files)
    df = df.sample(frac=1).reset_index(drop=True)  # Xáo trộn ngẫu nhiên

    output_path = OUTPUT_DIR / "master_video_list.csv"
    df.to_csv(output_path, index=False)

    print(f"\nHoàn tất! Đã tìm thấy {len(df)} video.")
    print(f"Real: {len(df[df.label == 0])}, Fake: {len(df[df.label == 1])}")
    print(f"Đã lưu master list vào: {output_path}")


if __name__ == "__main__":
    create_master_list()