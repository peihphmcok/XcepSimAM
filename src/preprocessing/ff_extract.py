import cv2
import os
import torch
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm
import warnings

# Tắt cảnh báo
warnings.filterwarnings("ignore")

BASE_DIR = r"C:\Personal\XcepSimAM_data\ff_deepfakes"
OUTPUT_DIR = r"C:\Personal\XcepSimAM\src\data\ff_frame"
TARGET_SIZE = (224, 224)

BATCH_SIZE = 32

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.benchmark = True

def get_largest_face(boxes, probs):
    if boxes is None or len(boxes) == 0:
        return None, None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    max_idx = np.argmax(areas)
    return boxes[max_idx], probs[max_idx]


def detect_and_save_batch(frames, indices, save_dir, detector):
    count = 0
    if not frames:
        return 0

    try:
        boxes_list, probs_list = detector.detect(frames)

        for i, boxes in enumerate(boxes_list):
            if boxes is None: continue

            frame_original = frames[i]
            frame_idx = indices[i]
            probs = probs_list[i]

            box, prob = get_largest_face(boxes, probs)

            if box is None: continue
            if prob < 0.95: continue

            x1, y1, x2, y2 = [int(b) for b in box]
            h, w, _ = frame_original.shape

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue

            face_img = frame_original[y1:y2, x1:x2]

            if face_img.size > 0:
                face_img = cv2.resize(face_img, TARGET_SIZE)
                face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                filename = f"frame_{frame_idx:04d}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, face_img_bgr)
                count += 1

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"Hết VRAM. Hãy giảm {BATCH_SIZE} xuống !!!")
            torch.cuda.empty_cache()
        else:
            print(f"Error processing batch: {e}")
    except Exception as e:
        print(f"Error: {e}")

    return count


def process_video(video_path, output_dir, label, source_name, detector):
    video_name = os.path.basename(video_path).split('.')[0]
    save_dir = os.path.join(output_dir, f"{label}_{source_name}_{video_name}")

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 10:
        return

    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frames_buffer = []
    indices_buffer = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR -> RGB ngay để đưa vào list
        # (MTCNN cần RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames_buffer.append(frame_rgb)
        indices_buffer.append(frame_idx)
        frame_idx += 1

        # Khi buffer đầy -> Đẩy vào GPU
        if len(frames_buffer) >= BATCH_SIZE:
            saved_count += detect_and_save_batch(frames_buffer, indices_buffer, save_dir, detector)
            # Reset buffer để giải phóng RAM
            frames_buffer = []
            indices_buffer = []

    # Xử lý phần dư còn lại
    if len(frames_buffer) > 0:
        saved_count += detect_and_save_batch(frames_buffer, indices_buffer, save_dir, detector)

    cap.release()


def main():
    print(f"--- Processing on {DEVICE.upper()} with BATCH_SIZE={BATCH_SIZE} ---")

    detector = MTCNN(
        margin=0,
        min_face_size=140,
        thresholds=[0.8, 0.9, 0.95],
        factor=0.709,
        post_process=False,
        keep_all=True,
        device=DEVICE
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Process REAL videos
    real_dir = os.path.join(BASE_DIR, "original_sequences", "youtube", "c40", "videos")
    if os.path.exists(real_dir):
        videos = [f for f in os.listdir(real_dir) if f.endswith('.mp4')]
        print(f"Found {len(videos)} REAL videos.")
        for vid in tqdm(videos, desc="Real Videos"):
            process_video(os.path.join(real_dir, vid), OUTPUT_DIR, "real", "youtube", detector)

    # 2. Process FAKE videos
    # fake_dir = os.path.join(BASE_DIR, "manipulated_sequences", "Deepfakes", "c40", "videos")
    # if os.path.exists(fake_dir):
    #     videos = [f for f in os.listdir(fake_dir) if f.endswith('.mp4')]
    #     print(f"Found {len(videos)} FAKE videos.")
    #     for vid in tqdm(videos, desc="Fake Videos"):
    #         process_video(os.path.join(fake_dir, vid), OUTPUT_DIR, "fake", "Deepfakes", detector)

    print("All Done!")


if __name__ == "__main__":
    main()