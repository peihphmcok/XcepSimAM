import os
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_IN = r"C:\Personal\XcepSimAM\src\preprocessing\splits\ff_labels.csv"
SPLIT_DIR = r"C:\Personal\XcepSimAM\src\preprocessing\splits"


def main():
    if not os.path.exists(CSV_IN): return print("CSV file not found.")

    df = pd.read_csv(CSV_IN)

    # 1. Group by Video ID (avoid leakage)
    videos = df[['video_id', 'label', 'source']].drop_duplicates()
    stratify_col = videos['label'].astype(str) + videos['source']

    # 2. Split: 72:14:14
    train_vids, temp_vids = train_test_split(
        videos, test_size=0.28, stratify=stratify_col, random_state=42
    )

    temp_stratify = temp_vids['label'].astype(str) + temp_vids['source']
    val_vids, test_vids = train_test_split(
        temp_vids, test_size=0.5, stratify=temp_stratify, random_state=42
    )

    # 4. Map back to frames and Save
    os.makedirs(SPLIT_DIR, exist_ok=True)

    splits = {
        "train.csv": train_vids,
        "val.csv": val_vids,
        "test.csv": test_vids
    }

    print("Splitting Videos:")
    for name, vids_df in splits.items():
        split_df = df[df['video_id'].isin(vids_df['video_id'])]
        split_df.to_csv(os.path.join(SPLIT_DIR, name), index=False)
        print(f"  {name}: {len(vids_df)} videos -> {len(split_df)} frames")

if __name__ == "__main__":
    main()