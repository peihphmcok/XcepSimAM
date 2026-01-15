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

from models.xception import xception

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")

def get_current_time_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

BASE_CONFIG = {
    'project_name': 'Xception_Baseline', 
    'data_path': 'preprocessing/splits',  
    'root_data_dir': os.getcwd(),

    'model_params': {
        'num_classes': 2,
        'dropout': 0.5
    },

    'max_videos': None,        
    'frames_per_video': 20,   
    
    'img_size': 224, 
    'batch_size': 32,         
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
        self.log_file = self.run_dir / 'result.txt'
        
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log for {BASE_CONFIG['project_name']}\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Config: {json.dumps(BASE_CONFIG, indent=2)}\n")
            f.write("="*50 + "\n")

    def log(self, message):
        """Prints to console and appends to result.txt"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

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
        self.log(f"[Graph] Saved training history to {self.run_dir / 'training_charts.png'}")

class GradCAM:
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
    try:
        target_layer = model.conv4 
    except AttributeError:
        try:
            target_layer = model.exit_flow.conv 
        except:
            print("[Warn] Could not find target layer (conv4) for GradCAM.")
            return

    grad_cam = GradCAM(model, target_layer)
    mean = np.array([0.5, 0.5, 0.5]) 
    std = np.array([0.5, 0.5, 0.5])
    
    found_real, found_fake = 0, 0
    images_to_show = [] 
    
    for imgs, lbls in loader:
        if found_real >= num_samples and found_fake >= num_samples:
            break
        imgs, lbls = imgs.to(device), lbls.to(device)
        imgs.requires_grad = True 
        
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        
        for i in range(imgs.size(0)):
            label = lbls[i].item()
            pred = preds[i].item()
            
            if label == pred: 
                if label == 0 and found_real < num_samples:
                    images_to_show.append((imgs[i].detach(), label, "Real"))
                    found_real += 1
                elif label == 1 and found_fake < num_samples:
                    images_to_show.append((imgs[i].detach(), label, "Fake"))
                    found_fake += 1

    if not images_to_show:
        print("[Warn] No correctly classified images found to visualize.")
        return

    rows = len(images_to_show)
    fig, axs = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1: axs = axs.reshape(1, -1)
    
    fig.suptitle(f'Grad-CAM Analysis ({rows} samples)', fontsize=16)
    
    for idx, (img_tensor, label, tag) in enumerate(images_to_show):
        input_tensor = img_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        mask = grad_cam(input_tensor, class_idx=label)
        
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
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
    train_ops = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_ops = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return train_ops, val_ops


class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_videos=None, frames_per_video=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

        def get_video_name(path_str):
            clean_path = str(path_str).replace('\\', '/')
            if ':' in clean_path:
                clean_path = clean_path.split(':', 1)[1]
            parts = clean_path.strip('/').split('/')
            return parts[-2] if len(parts) >= 2 else "unknown"

        self.data['video_name'] = self.data['path'].apply(get_video_name)

        if max_videos is not None:
            unique_video_df = self.data[['video_name', 'label']].drop_duplicates(subset='video_name')
            real_videos = unique_video_df[unique_video_df['label'] == 0]['video_name'].tolist()
            fake_videos = unique_video_df[unique_video_df['label'] == 1]['video_name'].tolist()

            rng = np.random.RandomState(42)
            rng.shuffle(real_videos)
            rng.shuffle(fake_videos)
            
            half_quota = max_videos // 2
            quota_real = min(len(real_videos), half_quota)
            quota_fake = min(len(fake_videos), max_videos - quota_real)

            selected_videos = set(real_videos[:quota_real] + fake_videos[:quota_fake])
            self.data = self.data[self.data['video_name'].isin(selected_videos)].reset_index(drop=True)
            print(f"[Dataset] Video Selection: {len(selected_videos)} videos selected.")

        if frames_per_video is not None:
            self.data = self.data.groupby('video_name').apply(
                lambda x: x.sample(n=min(len(x), frames_per_video), random_state=42)
            ).reset_index(drop=True)
            print(f"[Dataset] Frame Selection: Limited to {frames_per_video} frames per video.")

        label_counts = self.data['label'].value_counts()
        print(f"Dataset Stats ({os.path.basename(str(csv_file))}):")
        print(f"Total: {len(self.data)} | Real: {label_counts.get(0,0)} | Fake: {label_counts.get(1,0)}")
        print("-" * 30)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        clean_path = str(row['path']).replace('\\', '/')
        if ':' in clean_path:
            clean_path = clean_path.split(':', 1)[1]

        filename = os.path.basename(clean_path)
        parts = clean_path.split('/')

        image_path = None
        if len(parts) >= 2:
            image_path = os.path.join(self.root_dir, 'data', 'ff_frame', parts[-2], filename)

        image = Image.new('RGB', (224, 224))
        try:
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
        except Exception:
            pass

        if self.transform:
            image = self.transform(image)

        return image, int(row['label'])


class Trainer:
    def __init__(self, model, loaders, config, save_dir, utils):
        self.model = model.to(config['device'])
        self.loaders = loaders
        self.config = config
        self.save_dir = save_dir
        self.utils = utils 
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
        except ValueError:
            auc = 0.5
        return {'loss': loss, 'acc': acc, 'auc': auc, 'f1': f1, 'precision': precision, 'recall': recall}

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_lbl, all_pred, all_prob = [], [], []

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

        mean_loss = total_loss / len(self.loaders['train'].dataset)
        return self.compute_metrics(mean_loss, all_lbl, all_pred, all_prob)

    @torch.no_grad()
    def evaluate(self, phase='val'):
        self.model.eval()
        total_loss = 0
        all_lbl, all_pred, all_prob = [], [], []

        for imgs, lbls in tqdm(self.loaders[phase], desc=phase, leave=False):
            imgs, lbls = imgs.to(self.config['device']), lbls.to(self.config['device'])
            out = self.model(imgs)
            loss = self.criterion(out, lbls)
            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_lbl.extend(lbls.cpu().numpy()); all_pred.extend(preds); all_prob.extend(probs)

        mean_loss = total_loss / len(self.loaders[phase].dataset)
        return self.compute_metrics(mean_loss, all_lbl, all_pred, all_prob)

    def run(self):
        self.utils.log(f"Device: {self.config['device']}")
        self.utils.log(f"Start training with Max Videos: {self.config['max_videos']}, Frames/Video: {self.config['frames_per_video']}")
        
        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            train_res = self.train_epoch()
            val_res = self.evaluate('val')
            self.scheduler.step()

            for k in ['loss', 'acc', 'auc', 'f1', 'precision', 'recall']:
                self.history[f'train_{k}'].append(train_res[k])
                self.history[f'val_{k}'].append(val_res[k])

            log_msg = (f"Epoch {epoch+1}:\n"
                       f"   Train | Loss: {train_res['loss']:.4f} | Acc: {train_res['acc']:.4f} | AUC: {train_res['auc']:.4f} | F1: {train_res['f1']:.4f} | Pre: {train_res['precision']:.4f} | Rec: {train_res['recall']:.4f}\n"
                       f"   Val   | Loss: {val_res['loss']:.4f} | Acc: {val_res['acc']:.4f} | AUC: {val_res['auc']:.4f} | F1: {val_res['f1']:.4f} | Pre: {val_res['precision']:.4f} | Rec: {val_res['recall']:.4f}")
            self.utils.log(log_msg)

            if val_res['acc'] > self.best_acc:
                self.best_acc = val_res['acc']
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
                self.utils.log(f"   >>> Best Model Saved (Epoch {self.best_epoch})")
            
            self.utils.log("-" * 60)
        
        torch.save(self.model.state_dict(), self.save_dir / 'final_model.pth')
        self.utils.log("Final Model Saved.")


def main():
    run_dir = Path(f"runs/{get_current_time_str()}_{BASE_CONFIG['project_name']}")
    utils = ResearchUtils(run_dir)
    utils.save_config(BASE_CONFIG)

    train_ops, val_ops = get_transforms(BASE_CONFIG['img_size'])
    csv_dir = Path(BASE_CONFIG['data_path'])

    if not (csv_dir / 'train.csv').exists():
        print(f"not found train.csv in {csv_dir}")
        return

    limit_videos = BASE_CONFIG.get('max_videos', None)
    limit_frames = BASE_CONFIG.get('frames_per_video', None)

    train = DeepfakeDataset(csv_dir / 'train.csv', BASE_CONFIG['root_data_dir'], train_ops, 
                            max_videos=limit_videos, frames_per_video=limit_frames)
    val = DeepfakeDataset(csv_dir / 'val.csv', BASE_CONFIG['root_data_dir'], val_ops, 
                          max_videos=limit_videos, frames_per_video=limit_frames)
    test = DeepfakeDataset(csv_dir / 'test.csv', BASE_CONFIG['root_data_dir'], val_ops, 
                           max_videos=limit_videos, frames_per_video=limit_frames)

    use_pin_memory = (BASE_CONFIG['device'] == 'cuda')

    loaders = {
        'train': DataLoader(train, batch_size=BASE_CONFIG['batch_size'], shuffle=True, num_workers=BASE_CONFIG['num_workers'], pin_memory=use_pin_memory),
        'val': DataLoader(val, batch_size=BASE_CONFIG['batch_size'], shuffle=False, num_workers=BASE_CONFIG['num_workers'], pin_memory=use_pin_memory),
        'test': DataLoader(test, batch_size=BASE_CONFIG['batch_size'], shuffle=False, num_workers=BASE_CONFIG['num_workers'], pin_memory=use_pin_memory)
    }

    print("Initializing Standard Xception Model...")
    
    model = xception(pretrained=None) 

    possible_paths = [
        'pretrained/xception-43020ad28.pth', 
        os.path.expanduser('~/.cache/torch/hub/checkpoints/xception-43020ad28.pth'), 
        'xception-43020ad28.pth' 
    ]
    
    weights_loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found local weights at: {path}")
            try:
                state_dict = torch.load(path)
                model.load_state_dict(state_dict)
                weights_loaded = True
                print("Weights loaded successfully!")
                break
            except Exception as e:
                print(f"Error loading {path}: {e}")

    if not weights_loaded:
        print("\n[WARNING] Could not find local weights!")
        print("Server seems offline, so we CANNOT download from internet.")
        print("Model will initialize with RANDOM weights (Performance will be poor initially).")
        print(f"Please upload 'xception-43020ad28.pth' to: {os.getcwd()}/pretrained/\n")
    
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Sequential(
        nn.Dropout(p=BASE_CONFIG['model_params']['dropout']),
        nn.Linear(num_ftrs, BASE_CONFIG['model_params']['num_classes'])
    )

    trainer = Trainer(model, loaders, BASE_CONFIG, run_dir, utils)
    trainer.run()

    utils.plot_history(trainer.history)

    print("\n" + "=" * 30)
    utils.log("Evaluating on TEST Set (Best Model)...")
    print("=" * 30)

    best_model_path = run_dir / 'best_model.pth'

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        model.to(BASE_CONFIG['device'])

        test_res = trainer.evaluate('test')

        final_msg = (f"FINAL TEST RESULTS (Best Epoch {trainer.best_epoch}):\n"
                     f"Accuracy : {test_res['acc']:.4f}\n"
                     f"AUC      : {test_res['auc']:.4f}\n"
                     f"F1       : {test_res['f1']:.4f}\n"
                     f"Precision: {test_res['precision']:.4f}\n"
                     f"Recall   : {test_res['recall']:.4f}\n"
                     f"Loss     : {test_res['loss']:.4f}")
        utils.log(final_msg)
        
        try:
            generate_visualizations(model, loaders['test'], BASE_CONFIG['device'], run_dir)
        except Exception as e:
            utils.log(f"Error generating visualization: {e}")
            
    else:
        utils.log("Not found best_model.pth. Skipping Test evaluation.")

if __name__ == "__main__":
    main()