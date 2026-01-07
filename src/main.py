import os
import json
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from models.XcepSimAM import XceptionSimAM

from models.xception import xception

warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_current_time_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

BASE_CONFIG = {
    'project_name': 'XcepSimAM_Scientific',
    'data_path': 'preprocessing/splits',
    'root_data_dir': os.getcwd(),
    'model_params': {
        'num_classes': 2,
        'dropout': 0.5
    },
    'max_videos': 50, # Set to None to use full dataset
    'img_size': 224,
    'batch_size': 64,
    'lr': 0.0001,
    'epochs': 20,
    'lr_decay_step': 5,
    'lr_decay_gamma': 0.5,
    'seed': 42,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class ResearchUtils:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config):
        with open(self.run_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, default=str)

    def plot_history(self, history):
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        axs = axs.flatten()
        metrics_map = [
            ('loss', 'Loss', axs[0]), ('acc', 'Accuracy', axs[1]),
            ('auc', 'AUC Score', axs[2]), ('f1', 'F1 Score', axs[3]),
            ('precision', 'Precision', axs[4]), ('recall', 'Recall', axs[5])
        ]
        for key, title, ax in metrics_map:
            if f'train_{key}' in history:
                ax.plot(epochs, history[f'train_{key}'], 'b-o', label=f'Train {title}')
                ax.plot(epochs, history[f'val_{key}'], 'r-o', label=f'Val {title}')
                ax.set_title(title); ax.legend(); ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'training_charts.png')

class GradCAM:
    """Class for generating Heatmap visualizations"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        return cam

def generate_visualizations(model, loader, device, run_dir, num_samples=5):
    print("\n[Info] Generating Grad-CAM visualizations...")
    # Target the last convolutional layer in Exit Flow
    target_layer = model.exit_sep_conv_2 
    grad_cam = GradCAM(model, target_layer)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    found_real, found_fake = 0, 0
    images_to_show = [] 
    
    for imgs, lbls in loader:
        if found_real >= num_samples and found_fake >= num_samples:
            break
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        
        for i in range(imgs.size(0)):
            label = lbls[i].item()
            pred = preds[i].item()
            # Only visualize correctly classified samples
            if label == pred: 
                if label == 0 and found_real < num_samples:
                    images_to_show.append((imgs[i], label, "Real"))
                    found_real += 1
                elif label == 1 and found_fake < num_samples:
                    images_to_show.append((imgs[i], label, "Fake"))
                    found_fake += 1

    if not images_to_show:
        print("[Warn] No correctly classified images found to visualize.")
        return

    rows = len(images_to_show)
    fig, axs = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1: axs = axs.reshape(1, -1)
    
    fig.suptitle(f'Grad-CAM Analysis ({rows} samples)', fontsize=16)
    
    for idx, (img_tensor, label, tag) in enumerate(images_to_show):
        input_tensor = img_tensor.unsqueeze(0)
        mask = grad_cam(input_tensor, class_idx=label)
        
        # Denormalize
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Heatmap & Overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        overlay = heatmap + img_np
        overlay = overlay / np.max(overlay)
        
        axs[idx, 0].imshow(img_np); axs[idx, 0].set_title(f"Original ({tag})"); axs[idx, 0].axis('off')
        axs[idx, 1].imshow(heatmap); axs[idx, 1].set_title("Attention Map"); axs[idx, 1].axis('off')
        axs[idx, 2].imshow(overlay); axs[idx, 2].set_title("Overlay"); axs[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(run_dir / 'visualization_results.png')
    print(f"[Info] Visualizations saved to: {run_dir / 'visualization_results.png'}")

def get_transforms(img_size):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_ops = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])

    val_ops = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    return train_ops, val_ops

def transfer_weights(custom_model, pretrained_model):
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = custom_model.state_dict()
    transfer_layers = []
    
    for name, param in pretrained_dict.items():
        if name in custom_dict:
            if param.shape == custom_dict[name].shape:
                custom_dict[name].copy_(param)
                transfer_layers.append(name)
    
    custom_model.load_state_dict(custom_dict)
    print(f"[Info] Weights transferred successfully for {len(transfer_layers)} layers (Entry Flow).")
    return custom_model

class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_videos=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

        # --- Helper for Robust Path Extraction ---
        def get_video_name(path_str):
            clean_path = str(path_str).replace('\\', '/') # Fix Windows path issue
            parts = clean_path.split('/')
            return parts[-2] if len(parts) >= 2 else "unknown_video"

        self.data['video_name'] = self.data['path'].apply(get_video_name)

        if max_videos is not None:
            unique_video_df = self.data[['video_name', 'label']].drop_duplicates(subset='video_name')
            real_videos = unique_video_df[unique_video_df['label'] == 0]['video_name'].tolist()
            fake_videos = unique_video_df[unique_video_df['label'] == 1]['video_name'].tolist()
            
            rng = np.random.RandomState(42)
            rng.shuffle(real_videos)
            rng.shuffle(fake_videos)
            
            # Balance selection
            half_quota = max_videos // 2
            selected_real = real_videos[:half_quota]
            quota_fake = max_videos - len(selected_real) # Take remainder
            selected_fake = fake_videos[:quota_fake]
            
            selected_videos = set(selected_real + selected_fake)
            self.data = self.data[self.data['video_name'].isin(selected_videos)].reset_index(drop=True)
            print(f"[Dataset] {os.path.basename(str(csv_file))}: Selected {len(selected_videos)} videos.")

        # --- Statistics Reporting ---
        label_counts = self.data['label'].value_counts()
        num_real = label_counts.get(0, 0)
        num_fake = label_counts.get(1, 0)
        print(f"   -> Frames: {len(self.data)} | Real: {num_real} | Fake: {num_fake}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- FIX: Safe Path Handling ---
        path_str = str(row['path']).replace('\\', '/')
        parts = path_str.split('/')
        
        if len(parts) >= 2:
            video_name = parts[-2]
            filename = parts[-1]
        else:
            # Fallback for paths without directory structure
            video_name = "unknown" 
            filename = parts[-1]
            
        image_path = os.path.join(self.root_dir, 'data', 'ff_frame', video_name, filename)
        # -------------------------------
        
        image = Image.new('RGB', (224, 224)) # Placeholder black image
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
        except Exception:
            pass
            
        if self.transform:
            image = self.transform(image)
        return image, int(row['label'])

class Trainer:
    def __init__(self, model, loaders, config, save_dir):
        self.model = model.to(config['device'])
        self.loaders = loaders
        self.config = config
        self.save_dir = save_dir
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_decay_step'], gamma=0.5)

        self.history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_auc', 'val_auc', 'train_f1', 'val_f1', 'train_precision', 'val_precision', 'train_recall', 'val_recall']}
        self.best_acc = 0.0
        self.best_epoch = -1

    def compute_metrics(self, loss, labels, preds, probs):
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.5
        return {'loss': loss, 'acc': acc, 'auc': auc, 'f1': f1, 'precision': precision, 'recall': recall}

    def train_epoch(self):
        self.model.train()
        total_loss = 0; all_lbl = []; all_pred = []; all_prob = []
        
        for imgs, lbls in tqdm(self.loaders['train'], desc="Train", leave=False):
            imgs, lbls = imgs.to(self.config['device']), lbls.to(self.config['device'])
            self.optimizer.zero_grad()
            out = self.model(imgs)
            loss = self.criterion(out, lbls)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(out, dim=1).detach().cpu().numpy()
            all_lbl.extend(lbls.cpu().numpy()); all_pred.extend(preds); all_prob.extend(probs)
            
        return self.compute_metrics(total_loss / len(self.loaders['train'].dataset), all_lbl, all_pred, all_prob)

    @torch.no_grad()
    def evaluate(self, phase='val'):
        self.model.eval()
        total_loss = 0; all_lbl = []; all_pred = []; all_prob = []
        
        for imgs, lbls in tqdm(self.loaders[phase], desc=phase, leave=False):
            imgs, lbls = imgs.to(self.config['device']), lbls.to(self.config['device'])
            out = self.model(imgs)
            loss = self.criterion(out, lbls)
            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_lbl.extend(lbls.cpu().numpy()); all_pred.extend(preds); all_prob.extend(probs)
            
        return self.compute_metrics(total_loss / len(self.loaders[phase].dataset), all_lbl, all_pred, all_prob)

    def run(self):
        print(f"Device: {self.config['device']}")
        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            train_res = self.train_epoch()
            val_res = self.evaluate('val')
            self.scheduler.step()

            for k in ['loss', 'acc', 'auc', 'f1', 'precision', 'recall']:
                self.history[f'train_{k}'].append(train_res[k])
                self.history[f'val_{k}'].append(val_res[k])

            print(f"Train | Loss: {train_res['loss']:.4f} | Acc: {train_res['acc']:.4f} | AUC: {train_res['auc']:.4f}")
            print(f"Val   | Loss: {val_res['loss']:.4f} | Acc: {val_res['acc']:.4f} | AUC: {val_res['auc']:.4f}")

            if val_res['acc'] > self.best_acc:
                self.best_acc = val_res['acc']
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
                print(f"Best Model Saved (Epoch {self.best_epoch})")
        
        torch.save(self.model.state_dict(), self.save_dir / 'final_model.pth')
        print("Final Model Saved.")

def main():
    run_dir = Path(f"runs/{get_current_time_str()}_{BASE_CONFIG['project_name']}")
    utils = ResearchUtils(run_dir)
    utils.save_config(BASE_CONFIG)
    train_ops, val_ops = get_transforms(BASE_CONFIG['img_size'])
    csv_dir = Path(BASE_CONFIG['data_path'])
    
    limit = BASE_CONFIG.get('max_videos', None)
    train_set = DeepfakeDataset(csv_dir / 'train.csv', BASE_CONFIG['root_data_dir'], train_ops, limit)
    val_set = DeepfakeDataset(csv_dir / 'val.csv', BASE_CONFIG['root_data_dir'], val_ops, limit)
    test_set = DeepfakeDataset(csv_dir / 'test.csv', BASE_CONFIG['root_data_dir'], val_ops, limit)

    loaders = {
        'train': DataLoader(train_set, batch_size=BASE_CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_set, batch_size=BASE_CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(test_set, batch_size=BASE_CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    }

    print("Initializing XcepSimAM...")
    model = XceptionSimAM(num_classes=BASE_CONFIG['model_params']['num_classes'])
    
    try:
        print("Transferring ImageNet Weights...")
        std_model = xception(pretrained='imagenet')
        model = transfer_weights(model, std_model)
        del std_model
    except Exception as e:
        print(f"Weight transfer failed ({e}). Training from scratch.")

    trainer = Trainer(model, loaders, BASE_CONFIG, run_dir)
    trainer.run()
    
    utils.plot_history(trainer.history)
    
    if (run_dir / 'best_model.pth').exists():
        print(f"Loading Best Model (Epoch {trainer.best_epoch})...")
        model.load_state_dict(torch.load(run_dir / 'best_model.pth'))
        
        test_res = trainer.evaluate('test')
        print("="*30)
        print("Final Test Results (Best Model):")
        print(f"Acc: {test_res['acc']:.4f} | AUC: {test_res['auc']:.4f} | F1: {test_res['f1']:.4f} | Prec: {test_res['precision']:.4f} | Rec: {test_res['recall']:.4f}")
        
        # Generate Visualization
        generate_visualizations(model, loaders['test'], BASE_CONFIG['device'], run_dir)

if __name__ == "__main__":
    main()