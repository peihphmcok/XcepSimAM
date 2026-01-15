import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa các folder video (dựa trên ảnh bạn gửi)
DATA_ROOT = 'data/ff_frame'

# Ngưỡng tối thiểu số lượng ảnh. Nếu folder có ít hơn số này sẽ bị coi là lỗi.
# Đặt = 1 nếu chỉ muốn tìm folder rỗng hoàn toàn.
MIN_FRAMES_THRESHOLD = 1


def check_and_clean_dataset():
    root_dir = Path(os.getcwd()) / DATA_ROOT

    if not root_dir.exists():
        print(f"❌ Không tìm thấy đường dẫn: {root_dir}")
        print("Hãy kiểm tra lại biến DATA_ROOT hoặc vị trí file script.")
        return

    print(f"Đang quét dữ liệu tại: {root_dir} ...")

    # Lấy danh sách tất cả các folder con
    video_folders = [f for f in root_dir.iterdir() if f.is_dir()]

    empty_folders = []
    valid_folders = 0

    # Duyệt qua từng folder để kiểm tra
    for folder in tqdm(video_folders, desc="Checking folders"):
        # Lấy danh sách file ảnh (jpg, png)
        images = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + list(folder.glob('*.jpeg'))

        if len(images) < MIN_FRAMES_THRESHOLD:
            empty_folders.append(folder)
        else:
            valid_folders += 1

    # --- BÁO CÁO KẾT QUẢ ---
    print("\n" + "=" * 40)
    print("KẾT QUẢ KIỂM TRA")
    print("=" * 40)
    print(f"✅ Số folder hợp lệ: {valid_folders}")
    print(f"⚠️  Số folder RỖNG hoặc LỖI (< {MIN_FRAMES_THRESHOLD} frames): {len(empty_folders)}")

    if len(empty_folders) > 0:
        print("\nDanh sách các folder rỗng (ví dụ 10 cái đầu):")
        for f in empty_folders[:10]:
            print(f" - {f.name}")

        if len(empty_folders) > 10:
            print(f" ... và {len(empty_folders) - 10} folder khác.")

        # --- TÙY CHỌN XÓA ---
        print("\n" + "-" * 40)
        confirm = input("Bạn có muốn XÓA VĨNH VIỄN các folder rỗng này không? (yes/no): ").strip().lower()

        if confirm == 'yes':
            print("Đang xóa...")
            count = 0
            for folder in tqdm(empty_folders, desc="Deleting"):
                try:
                    shutil.rmtree(folder)  # Xóa folder và nội dung bên trong
                    count += 1
                except Exception as e:
                    print(f"Lỗi khi xóa {folder.name}: {e}")
            print(f"✅ Đã xóa thành công {count} folder rỗng.")
        else:
            print("Đã hủy thao tác xóa. Các folder rỗng vẫn còn đó.")
    else:
        print("🎉 Tuyệt vời! Dataset sạch sẽ, không có folder rỗng.")


if __name__ == "__main__":
    check_and_clean_dataset()