import cv2
import torch
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

# --- IMPORT MODEL ---
# Giả sử file model nằm cùng thư mục hoặc đã được setup
try:
    from XcepSimAM import XceptionSimAM
except ImportError:
    # Fallback nếu chạy demo mà không có file model gốc (để tránh crash IDE)
    print("Warning: XcepSimAM.py not found. Make sure class exists.")
    pass

# --- CONFIG ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


def draw_stats_board(frame, stats, frame_idx, total_frames):
    """
    Hàm vẽ bảng thống kê lên góc video
    """
    total_faces = stats['real'] + stats['fake']

    # Tính toán ratio và kết luận hiện tại
    if total_faces > 0:
        fake_ratio = stats['fake'] / total_faces
        # Threshold 0.2 (20%) như logic của bạn
        current_status = "FAKE VIDEO" if fake_ratio > 0.2 else "REAL VIDEO"
        status_color = (0, 0, 255) if fake_ratio > 0.2 else (0, 255, 0)  # Đỏ nếu Fake, Xanh nếu Real
    else:
        fake_ratio = 0.0
        current_status = "ANALYZING..."
        status_color = (255, 255, 255)

    # Nội dung hiển thị
    lines = [
        f"Frame: {frame_idx}/{total_frames}",
        f"Faces Analyzed: {total_faces}",
        f"Real: {stats['real']} | Fake: {stats['fake']}",
        f"Fake Ratio: {fake_ratio:.2%}",
        f"RESULT: {current_status}"
    ]

    # Cấu hình font và box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    line_height = 25

    # Tính toán kích thước box nền
    max_width = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        if w > max_width: max_width = w

    box_w = max_width + 2 * margin
    box_h = len(lines) * line_height + margin

    # Vẽ box nền đen (semi-transparent nếu muốn, ở đây để đặc cho rõ)
    # Tọa độ góc trên trái (10, 10)
    top_left_x, top_left_y = 10, 10
    cv2.rectangle(frame,
                  (top_left_x, top_left_y),
                  (top_left_x + box_w, top_left_y + box_h),
                  (0, 0, 0), -1)  # -1 là fill đặc

    # Vẽ viền trắng cho box
    cv2.rectangle(frame,
                  (top_left_x, top_left_y),
                  (top_left_x + box_w, top_left_y + box_h),
                  (255, 255, 255), 1)

    for i, line in enumerate(lines):
        y_pos = top_left_y + margin + (i + 1) * line_height - 5
        color = status_color if i == len(lines) - 1 else (255, 255, 255)
        cv2.putText(frame, line, (top_left_x + margin, y_pos), font, font_scale, color, 1)


def run_inference(input_path, model_path, output_path):
    print(f"Running on device: {DEVICE}")

    try:
        model = XceptionSimAM(num_classes=2).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()

    mtcnn = MTCNN(
        keep_all=True,
        device=DEVICE,
        select_largest=False,
        post_process=False,
        thresholds=[0.8, 0.85, 0.9],
        min_face_size=40
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return

    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Statistics
    stats = {'real': 0, 'fake': 0}
    frame_idx = 0

    print(f"Processing video... ({total_frames} frames)")

    # 2. Main Loop
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Convert to RGB for Detection
            img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            try:
                boxes, _ = mtcnn.detect(img_rgb)
            except:
                boxes = None

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]

                    # Validate box size
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue

                    # Clamp coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    # Crop & Predict
                    face_crop = img_rgb.crop((x1, y1, x2, y2))
                    face_tensor = TRANSFORM(face_crop).unsqueeze(0).to(DEVICE)

                    fake_prob = torch.softmax(model(face_tensor), dim=1)[0, 1].item()

                    # Classification Logic
                    is_fake = fake_prob > 0.5

                    if is_fake:
                        stats['fake'] += 1
                        label = f"FAKE: {fake_prob:.2f}"
                        color = (0, 0, 255)  # Red
                    else:
                        stats['real'] += 1
                        label = f"REAL: {1 - fake_prob:.2f}"
                        color = (0, 255, 0)  # Green

                    # Draw Result Box around Face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            draw_stats_board(frame, stats, frame_idx, total_frames)

            writer.write(frame)

            if frame_idx % 10 == 0:
                print(f"Processed: {frame_idx}/{total_frames}", end='\r')

    # Cleanup
    cap.release()
    writer.release()

    total_faces = stats['fake'] + stats['real']
    print("\n\n--- ANALYSIS REPORT ---")
    if total_faces == 0:
        print("Result: No faces detected.")
    else:
        fake_ratio = stats['fake'] / total_faces
        print(f"Total faces analyzed: {total_faces}")
        print(f"Real faces count: {stats['real']}")
        print(f"Fake faces count: {stats['fake']}")
        print(f"Fake ratio: {fake_ratio:.2%}")
        if fake_ratio > 0.2:
            print("CONCLUSION: FAKE VIDEO DETECTED")
        else:
            print("CONCLUSION: REAL VIDEO")
    print("-----------------------")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    # VIDEO_PATH = r"C:\Personal\XcepSimAM_data\ff_deepfakes\original_sequences\youtube\c40\videos\075.mp4"
    VIDEO_PATH = r"C:\Personal\XcepSimAM_data\ff_deepfakes\manipulated_sequences\Deepfakes\c40\videos\001_870.mp4"
    MODEL_PATH = r"C:\Personal\XcepSimAM\src\best_model-no-augment.pth"
    OUTPUT_PATH = r"C:\Personal\XcepSimAM\src\test\28-12\001_870-no.mp4"

    run_inference(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH)