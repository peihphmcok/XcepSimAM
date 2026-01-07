import os
import pandas as pd

IMG_DIR = r"C:\Personal\XcepSimAM\src\data\ff_frame"
CSV_OUT = r"C:\Personal\XcepSimAM\src\preprocessing\splits\ff_labels.csv"


def main():
    data = []
    print(f"Scanning: {IMG_DIR}...")

    for root, _, files in os.walk(IMG_DIR):
        folder = os.path.basename(root)
        if not files or '_' not in folder: continue

        # Parse folder name: {label}_{source}_{video_name}
        try:
            parts = folder.split('_')
            label_str = parts[0]  # 'real' or 'fake'
            source = parts[1]  # 'youtube', 'Deepfakes'

            # 0 for Real, 1 for Fake
            label = 0 if label_str == 'real' else 1

            # Unique Video ID to prevent leakage
            video_id = "_".join(parts[:3])
        except:
            continue

        for f in files:
            if f.endswith('.jpg'):
                data.append([os.path.join(root, f), label, video_id, source])

    # Save
    if data:
        df = pd.DataFrame(data, columns=["path", "label", "video_id", "source"])
        os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
        df.to_csv(CSV_OUT, index=False)
        print(f"Done. Saved {len(df)} frames to {CSV_OUT}")
        print(f"Distribution: {df['label'].value_counts().to_dict()} (0=Real, 1=Fake)")
    else:
        print("No images found.")


if __name__ == "__main__":
    main()