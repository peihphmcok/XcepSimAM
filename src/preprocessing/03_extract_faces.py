# Lưu với tên: src/preprocessing/03_extract_faces.py
import torch
import cv2
import pandas as pd
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

# --- CẤU HÌNH ---
INPUT_DIR = Path("data/processed/video_lists")
OUTPUT_DIR = Path("data/processed/face_images")

FRAMES_PER_VIDEO = 20  # Lấy 20 frame từ mỗi video
IMAGE_SIZE = 256  # Lưu mặt người ở kích thước 256x256
CROP_MARGIN = 0.3  # Mở rộng 30% bounding box


# --- KẾT THÚC CẤU HÌNH ---

def get_padded_box(box, margin, frame_width, frame_height):
    """Mở rộng bounding box với lề (margin)."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    new_w = w + w * margin
    new_h = h + h * margin

    new_x1 = max(0, x1 - (new_w - w) / 2)
    new_y1 = max(0, y1 - (new_h - h) / 2)
    new_x2 = min(frame_width, x2 + (new_w - w) / 2)
    new_y2 = min(frame_height, y2 + (new_h - h) / 2)

    return [int(v) for v in [new_x1, new_y1, new_x2, new_y2]]


def process_video(video_path_str, label, output_dir_split, mtcnn):
    """Xử lý 1 video: trích xuất frame, phát hiện mặt, lưu."""
    try:
        video_path = Path(video_path_str)
        if not video_path.exists():
            print(f"Warning: File không tồn tại {video_path_str}, bỏ qua.")
            return

        video_name = video_path.stem
        label_str = "real" if label == 0 else "fake"

        # Tạo thư mục con (real/fake)
        final_output_dir = output_dir_split / label_str
        final_output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path_str)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"Warning: Không thể đọc {video_name} (0 frames), bỏ qua.")
            cap.release()
            return

        # Chọn N frame cách đều nhau
        frame_indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)

        faces_saved = 0
        for i, frame_index in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_height, frame_width = frame.shape[:2]

            # Chuyển BGR (OpenCV) sang RGB (MTCNN)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Phát hiện khuôn mặt
            boxes, _ = mtcnn.detect(frame_rgb)

            if boxes is not None:
                # Chỉ lấy mặt lớn nhất
                box = boxes[0]

                # Mở rộng bounding box
                box_padded = get_padded_box(box, CROP_MARGIN, frame_width, frame_height)
                x1, y1, x2, y2 = box_padded

                # Cắt khuôn mặt
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                # Resize về kích thước chuẩn
                face_resized = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

                # Lưu file
                output_filename = final_output_dir / f"{video_name}_frame{i:03d}.png"
                cv2.imwrite(str(output_filename), face_resized)
                faces_saved += 1

        cap.release()

    except Exception as e:
        print(f"Lỗi khi xử lý {video_path_str}: {e}")


def extract_faces():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # Khởi tạo MTCNN (sẽ tải model nếu là lần đầu chạy)
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device
    )

    for split in ["train", "val", "test"]:
        csv_path = INPUT_DIR / f"{split}_videos.csv"
        if not csv_path.exists():
            print(f"Không tìm thấy {csv_path}, bỏ qua.")
            continue

        df = pd.read_csv(csv_path)
        output_dir_split = OUTPUT_DIR / split

        print(f"\nĐang xử lý {split} set: {len(df)} videos...")

        # Dùng tqdm để xem tiến độ
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split} faces"):
            process_video(row['video_path'], row['label'], output_dir_split, mtcnn)

    print("\nHoàn tất trích xuất khuôn mặt!")


if __name__ == "__main__":
    extract_faces()