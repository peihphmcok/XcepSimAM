# Lưu với tên: src/main.py
import json
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

# --- NHẬP TỪ CÁC FILE KHÁC TRONG `src/` ---
from models.master_model import ConfigurableXcepMamba


# (Giả sử bạn đã tạo các file này trong src/utils/)
# from utils.dataset import DeepfakeDataset
# from utils.transforms import train_transform, val_test_transform
# from utils.losses import FocalLoss
# from utils.metrics import calculate_metrics
# from utils.visualize import save_plots

# --- BẮT ĐẦU ĐỊNH NGHĨA CÁC HÀM UTILS (Tạm thời để ở đây cho dễ chạy) ---
# (Bạn nên tách các class/hàm này ra file riêng trong src/utils/)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha;
        self.gamma = gamma;
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[0] * (1 - targets) + self.alpha[1] * targets
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'fake_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'real_detection_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'confusion_matrix': cm,
    }


def save_plots(history, test_metrics, y_true, y_pred_prob, output_path, model_name):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # ... (Copy hàm save_plots đầy đủ của bạn vào đây) ...
    print(f"Plots saved to {output_path}")


# (Bạn cần class DeepfakeDataset và transforms của bạn ở đây)
# Ví dụ (Placeholder - HÃY THAY BẰNG CODE CỦA BẠN):
from torchvision import transforms


class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = row['label']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Không thể tải {img_path}, {e}. Dùng ảnh rỗng.")
            image = Image.new('RGB', (256, 256))
            label = 0

        if self.transform:
            image = self.transform(image)
        return image, label, img_path


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# --- KẾT THÚC PHẦN UTILS ---


# --- CONFIG CHUNG ---
CONFIG = {
    'data_path': "data/final_csvs",  # Đường dẫn chuẩn
    'base_model_path': "models/ablation_study",  # Đường dẫn chuẩn
    'base_output_path': "results/ablation_study",  # Đường dẫn chuẩn
    'batch_size': 16,
    'num_workers': 4,
    'learning_rate': 2e-4,
    'weight_decay': 1e-4,
    'alpha': 0.82,
    'gamma': 2.0,
    'patience': 5,
    'total_epochs': 25,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- DANH SÁCH THÍ NGHIỆM ĐỂ CHẠY ---
EXPERIMENT_CONFIGS = [
    # 1. MÔ HÌNH CUỐI CÙNG (Cốt lõi)
    {
        "id": "FINAL_XcepMamba (Hierarchical+SimAM+Mamba)",
        "middle_flow": "hierarchical",
        "attention": "simam",
        "encoder": "mamba"
    },

    # 2. BẢNG 2: BÓC TÁCH MIDDLE FLOW
    {
        "id": "Ablation_T2_Sequential_Flow",
        "middle_flow": "sequential",
        "attention": "simam",
        "encoder": "mamba"
    },
    {
        "id": "Ablation_T2_Parallel3_Flow",
        "middle_flow": "parallel_3_branch",
        "attention": "simam",
        "encoder": "mamba"
    },

    # 3. BẢNG 3: BÓC TÁCH ATTENTION
    {
        "id": "Ablation_T3_No_Attention",
        "middle_flow": "hierarchical",
        "attention": "none",
        "encoder": "mamba"
    },

    # 4. BẢNG 4: BÓC TÁCH ENCODER
    {
        "id": "Ablation_T4_GAP_Encoder",
        "middle_flow": "hierarchical",
        "attention": "simam",
        "encoder": "gap"
    }
]


# --- LOGIC HUẤN LUYỆN (NÊN TÁCH RA src/train.py) ---
def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels, _ in progress_bar:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    progress_bar = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics, np.array(all_labels), np.array(all_preds)


def train_experiment(exp_config, train_loader, val_loader, test_loader):
    model_key = exp_config['id']
    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU THÍ NGHIỆM: {model_key}")
    print(f"Cấu hình: {exp_config}")
    print("=" * 80)

    # 1. Tạo Model
    model = ConfigurableXcepMamba(
        num_classes=1,
        middle_flow_type=exp_config['middle_flow'],
        attention_type=exp_config['attention'],
        encoder_type=exp_config['encoder']
    ).to(CONFIG['device'])

    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters: {params:.2f}M")

    # 2. Setup thư mục
    model_save_path = Path(CONFIG['base_model_path']) / model_key
    output_save_path = Path(CONFIG['base_output_path']) / model_key
    model_save_path.mkdir(parents=True, exist_ok=True)
    output_save_path.mkdir(parents=True, exist_ok=True)

    # 3. Setup huấn luyện
    criterion = FocalLoss(alpha=torch.tensor([CONFIG['alpha'], 1 - CONFIG['alpha']]), gamma=CONFIG['gamma'])
    scaler = GradScaler()
    history = {k: [] for k in
               ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1', 'train_auc', 'val_auc']}
    best_val_loss = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=CONFIG['patience'])

    print("Phase: Training From Scratch...")
    for epoch in range(CONFIG['total_epochs']):
        print(f"  Epoch {epoch + 1}/{CONFIG['total_epochs']}")
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, CONFIG['device'])
        val_loss, val_metrics, _, _ = validate_epoch(model, val_loader, criterion, CONFIG['device'])
        scheduler.step(val_loss)

        # Lưu history
        history['train_loss'].append(train_loss);
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy']);
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1']);
        history['val_f1'].append(val_metrics['f1'])
        history['train_auc'].append(train_metrics['auc']);
        history['val_auc'].append(val_metrics['auc'])

        print(
            f"    Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, model_save_path / "best_model.pth")

    # 4. Test (Dùng model tốt nhất)
    print("\nLoading best model for testing...")
    checkpoint = torch.load(model_save_path / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics, y_true, y_pred_prob = validate_epoch(model, test_loader, criterion, CONFIG['device'])

    print(f"\nTest Results for {model_key}:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")

    # 5. Lưu kết quả
    save_plots(history, test_metrics, y_true, y_pred_prob, output_save_path, model_key)
    # (Bạn có thể gọi hàm analyze_misclassifications ở đây)

    results = {
        'model_id': model_key,
        'test_loss': test_loss,
        'accuracy': test_metrics['accuracy'],
        'f1': test_metrics['f1'],
        'auc': test_metrics['auc'],
        'mcc': test_metrics['mcc'],
        'middle_flow': exp_config['middle_flow'],
        'attention': exp_config['attention'],
        'encoder': exp_config['encoder'],
        'parameters_M': params
    }

    with open(output_save_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    torch.cuda.empty_cache()
    return results


# --- HÀM MAIN (ĐIỀU KHIỂN) ---
def main():
    if not Path(CONFIG['data_path']).exists():
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu {CONFIG['data_path']}")
        print("Vui lòng chạy 4 script tiền xử lý trong 'src/preprocessing/' trước.")
        return

    # Load data (Chỉ load 1 lần)
    try:
        train_dataset = DeepfakeDataset(Path(CONFIG['data_path']) / "train.csv", transform=train_transform)
        val_dataset = DeepfakeDataset(Path(CONFIG['data_path']) / "val.csv", transform=val_test_transform)
        test_dataset = DeepfakeDataset(Path(CONFIG['data_path']) / "test.csv", transform=val_test_transform)
    except Exception as e:
        print(f"Lỗi khi tải dataset: {e}")
        print("Đảm bảo file train.csv, val.csv, test.csv tồn tại trong", CONFIG['data_path'])
        return

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'], pin_memory=True)
    print(f"Data loaded: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    all_results = []

    # Lặp qua tất cả các thí nghiệm đã định nghĩa
    for exp_config in EXPERIMENT_CONFIGS:
        try:
            results = train_experiment(exp_config, train_loader, val_loader, test_loader)
            all_results.append(results)
        except Exception as e:
            print(f"LỖI nặng ở thí nghiệm {exp_config['id']}: {e}")
            torch.cuda.empty_cache()

    # 6. In Bảng Tổng kết
    print("\n" + "=" * 80)
    print("TỔNG KẾT TẤT CẢ THÍ NGHIỆM")
    print("=" * 80)

    if not all_results:
        print("Không có kết quả nào để tổng kết.")
        return

    df = pd.DataFrame(all_results)
    df = df.set_index('model_id')

    columns_order = [
        'accuracy', 'f1', 'auc', 'mcc',
        'parameters_M', 'middle_flow', 'attention', 'encoder'
    ]
    # Lọc các cột có tồn tại trong df
    final_columns = [col for col in columns_order if col in df.columns]
    df = df[final_columns]

    print(df.to_markdown(floatfmt=".4f"))

    # Lưu bảng tổng kết
    summary_path = Path(CONFIG['base_output_path']) / "final_ablation_summary.csv"
    df.to_csv(summary_path)
    print(f"\nBảng tổng kết đã được lưu vào: {summary_path}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()