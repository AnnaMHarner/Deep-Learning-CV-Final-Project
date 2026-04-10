"""
Assignment 2 — CPSC 542
NIH Chest X-Ray Multi-Label Classification

Authors:
  - Anna Harner      : Data pipeline, preprocessing, dataset class, run_epoch
  - Marissa Estramonte: Transfer learning, build_optimizer, evaluate_metrics, visualization
  - C. Gillette      : Custom architecture (ChestXRayCNN), GradCAM, best/worst predictions,
                       scenario (c) fix, scenario (d) 224x224 fix, metrics expansion

Run from inside Docker container:
  python3 /app/rundir/Deep-Learning-CV-Final-Project/assignment2.py \
    >& /app/rundir/training_log.txt &
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import csv
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score)
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights, ViT_B_16_Weights
from tqdm import tqdm

# ----------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------
DATA_DIR       = '/app/rundir/Chest_XRay+Data'
CHECKPOINT_DIR = '/app/rundir/Deep-Learning-CV-Final-Project/checkpoints'
OUTPUT_DIR     = '/app/rundir/results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------
# LOAD METADATA — Anna Harner
# ----------------------------------------------------------------
MetaData = pd.read_csv(f'{DATA_DIR}/Data_Entry_2017.csv')

image_path_dict = {os.path.basename(x): x for x in
                   Path(DATA_DIR).glob('images_*/images/*.png')}
MetaData['path'] = MetaData['Image Index'].map(image_path_dict)

missing_count = MetaData['path'].isnull().sum()
print(f"Missing images: {missing_count}")

MetaData['Finding Labels'] = MetaData['Finding Labels'].map(lambda x: x.split('|'))
mlb         = MultiLabelBinarizer()
label_masks = mlb.fit_transform(MetaData['Finding Labels'])
label_names = mlb.classes_

MetaData_Final = pd.concat([MetaData, pd.DataFrame(label_masks, columns=label_names)], axis=1)
print(f"Dataset ready with {len(label_names)} classes.")

# ----------------------------------------------------------------
# TRAIN/VAL/TEST SPLIT — Anna Harner
# Use 20% of patients — change subset_size to 1.0 for full training
# ----------------------------------------------------------------
with open(f'{DATA_DIR}/train_val_list.txt', 'r') as f:
    train_val_list = [line.strip() for line in f]
with open(f'{DATA_DIR}/test_list.txt', 'r') as f:
    test_list = [line.strip() for line in f]

train_val_df_raw = MetaData_Final[MetaData_Final['Image Index'].isin(train_val_list)]
test_df          = MetaData_Final[MetaData_Final['Image Index'].isin(test_list)]

unique_patients = train_val_df_raw['Patient ID'].unique()
np.random.seed(42)
np.random.shuffle(unique_patients)

# ---- Change 0.20 to 1.0 to train on full dataset ----
subset_size     = int(1.0 * len(unique_patients))
unique_patients = unique_patients[:subset_size]

split_idx = int(0.8 * len(unique_patients))
train_pts = unique_patients[:split_idx]
val_pts   = unique_patients[split_idx:]

train_df = train_val_df_raw[train_val_df_raw['Patient ID'].isin(train_pts)].reset_index(drop=True)
val_df   = train_val_df_raw[train_val_df_raw['Patient ID'].isin(val_pts)].reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Training images   : {len(train_df)}")
print(f"Validation images : {len(val_df)}")
print(f"Testing images    : {len(test_df)}")

# ----------------------------------------------------------------
# DATASET CLASS — Anna Harner
# ----------------------------------------------------------------
TARGET_SIZE = 512

class NIHChestXrayDataset(Dataset):
    def __init__(self, dataframe, target_size=512, split="train"):
        self.df          = dataframe.copy()
        self.target_size = target_size
        self.split       = split
        self.labels      = self.df.iloc[:, -15:].values
        self.df['adjusted_height'] = target_size
        self.df['adjusted_width']  = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = str(self.df.iloc[idx]['path'])
        image    = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"OpenCV could not find: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_size, self.target_size),
                           interpolation=cv2.INTER_LANCZOS4)
        if self.split == "train" and np.random.random() > 0.5:
            image = np.fliplr(image).copy()
        image = image.astype(np.float32) / 255.0
        mean  = np.array([0.485, 0.456, 0.406])
        std   = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# ----------------------------------------------------------------
# DATALOADERS (512x512 for CNN scenarios)
# ----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = NIHChestXrayDataset(train_df, target_size=TARGET_SIZE, split="train")
val_dataset   = NIHChestXrayDataset(val_df,   target_size=TARGET_SIZE, split="val")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                          num_workers=8, pin_memory=True, prefetch_factor=2)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,
                          num_workers=8, pin_memory=True, prefetch_factor=2)

# ----------------------------------------------------------------
# SCENARIO (a): CUSTOM ARCHITECTURE — ChestXRayCNN
# Lightweight CNN with residual skip connections designed for
# multi-label chest X-ray classification.
# ----------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity)


class ChestXRayCNN(nn.Module):
    """
    Custom CNN for chest X-ray multi-label classification.
    - 7x7 stem kernel for broad radiological feature capture
    - 4 residual stages with progressive channel widening
    - Global average pooling to reduce overfitting
    - Dropout for regularization
    """
    def __init__(self, num_classes=15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.stage1 = nn.Sequential(ResidualBlock(32,  64,  2), ResidualBlock(64,  64,  1))
        self.stage2 = nn.Sequential(ResidualBlock(64,  128, 2), ResidualBlock(128, 128, 1))
        self.stage3 = nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256, 1))
        self.stage4 = nn.Sequential(ResidualBlock(256, 512, 2), ResidualBlock(512, 512, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout          = nn.Dropout(p=0.4)
        self.classifier       = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


# ----------------------------------------------------------------
# SCENARIO (b): PRE-EXISTING ARCHITECTURES FROM SCRATCH — Anna Harner
# ----------------------------------------------------------------
def get_medical_model(architecture='cnn', num_classes=15):
    if architecture == 'cnn':
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif architecture == 'vit':
        model = models.vit_b_16(weights=None)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    return model


# ----------------------------------------------------------------
# TRANSFER LEARNING MODEL — Marissa Estramonte
# ----------------------------------------------------------------
def get_transfer_model(architecture='cnn', num_classes=15, phase=1):
    if architecture == 'cnn':
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        if phase == 1:
            for param in model.features.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
    elif architecture == 'vit':
        model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        if phase == 1:
            for name, param in model.named_parameters():
                if 'heads' not in name:
                    param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[{architecture.upper()} Phase {phase}] Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")
    return model


# ----------------------------------------------------------------
# TRAINING HELPERS — Anna Harner & Marissa Estramonte
# ----------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, phase='train'):
    model.train() if phase == 'train' else model.eval()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"{phase.capitalize()} Phase")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        pbar.set_postfix({'loss': loss.item()})
    return running_loss / len(loader.dataset)


def evaluate_metrics(model, loader, device, label_names):
    """
    Full evaluation: AUC-ROC + macro precision, recall, F1.
    Returns per_class_auc dict and macro_auc float.
    — Marissa Estramonte
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            outputs = model(inputs.to(device))
            probs   = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    all_preds  = (all_probs >= 0.5).astype(int)

    # Per-class AUC
    per_class_auc = {}
    for i, name in enumerate(label_names):
        if len(np.unique(all_labels[:, i])) < 2:
            per_class_auc[name] = float('nan')
        else:
            per_class_auc[name] = roc_auc_score(all_labels[:, i], all_probs[:, i])
    valid_aucs = [v for v in per_class_auc.values() if not np.isnan(v)]
    macro_auc  = float(np.mean(valid_aucs))

    # Macro precision, recall, F1
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\n--- Per-Class AUC-ROC ---")
    for name, auc in per_class_auc.items():
        print(f"  {name:<22}: {'N/A' if np.isnan(auc) else f'{auc:.4f}'}")
    print(f"\n  Macro AUC       : {macro_auc:.4f}")
    print(f"  Macro Precision : {precision:.4f}")
    print(f"  Macro Recall    : {recall:.4f}")
    print(f"  Macro F1        : {f1:.4f}")

    return per_class_auc, macro_auc, precision, recall, f1


def build_optimizer(model, architecture, phase, head_lr=1e-4):
    """Differential LR optimizer — Marissa Estramonte."""
    backbone_lr = head_lr / 10
    if architecture == 'cnn':
        head_params     = list(model.classifier.parameters())
        backbone_params = list(model.features.parameters())
    elif architecture == 'vit':
        head_params     = list(model.heads.parameters())
        backbone_params = [p for n, p in model.named_parameters() if 'heads' not in n]
    if phase == 1:
        print(f"Phase 1 optimizer: head only  |  lr={head_lr}")
        return torch.optim.Adam(head_params, lr=head_lr)
    else:
        print(f"Phase 2 optimizer: backbone lr={backbone_lr}  |  head lr={head_lr}")
        return torch.optim.Adam([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params,     'lr': head_lr},
        ])


# ----------------------------------------------------------------
# CSV LOGGER — saves per-epoch results so nothing is lost on crash
# ----------------------------------------------------------------
def init_csv_logger(scenario_name):
    path = os.path.join(OUTPUT_DIR, f'{scenario_name}_epoch_log.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss',
                         'macro_auc', 'precision', 'recall', 'f1', 'is_best'])
    return path


def log_epoch_csv(path, epoch, train_loss, val_loss,
                  macro_auc, precision, recall, f1, is_best):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}',
                         f'{macro_auc:.4f}', f'{precision:.4f}',
                         f'{recall:.4f}', f'{f1:.4f}', is_best])


# ----------------------------------------------------------------
# BEST / WORST PREDICTIONS
# Saves 3 images the model was most confident about (correct)
# and 3 it was least confident about (most wrong)
# ----------------------------------------------------------------
def save_best_worst_predictions(model, dataset, device, label_names,
                                scenario_name, n=3):
    """
    Saves the n best and n worst predicted images as PNGs.
    Best  = highest max probability where prediction matches ground truth
    Worst = highest max probability where prediction does NOT match ground truth
    """
    model.eval()
    records = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(loader, desc="Finding best/worst")):
            outputs = model(inputs.to(device))
            probs   = torch.sigmoid(outputs).cpu().numpy()[0]
            labels  = labels.numpy()[0]
            preds   = (probs >= 0.5).astype(int)

            max_prob   = float(probs.max())
            pred_class = int(probs.argmax())
            correct    = bool(preds[pred_class] == labels[pred_class])

            records.append({
                'idx':        idx,
                'max_prob':   max_prob,
                'correct':    correct,
                'pred_class': pred_class,
                'probs':      probs,
                'labels':     labels,
            })

    # Sort by confidence
    correct_records   = sorted([r for r in records if r['correct']],
                               key=lambda x: x['max_prob'], reverse=True)
    incorrect_records = sorted([r for r in records if not r['correct']],
                               key=lambda x: x['max_prob'], reverse=True)

    best_records  = correct_records[:n]
    worst_records = incorrect_records[:n]

    out_dir = os.path.join(OUTPUT_DIR, scenario_name)
    os.makedirs(out_dir, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for tag, group in [('best', best_records), ('worst', worst_records)]:
        for rank, rec in enumerate(group, 1):
            img_tensor, _ = dataset[rec['idx']]
            # Un-normalize for display
            img = img_tensor.permute(1, 2, 0).numpy()
            img = (img * std + mean)
            img = np.clip(img, 0, 1)

            pred_label = label_names[rec['pred_class']]
            true_labels = [label_names[i] for i, v in enumerate(rec['labels']) if v == 1]

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(
                f"{tag.upper()} #{rank}\n"
                f"Predicted: {pred_label} ({rec['max_prob']:.2f})\n"
                f"True: {', '.join(true_labels) if true_labels else 'No Finding'}",
                fontsize=9
            )
            save_path = os.path.join(out_dir, f'{tag}_{rank}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  Saved {save_path}")


# ----------------------------------------------------------------
# GRADCAM — for CNN-based models only
# ----------------------------------------------------------------
class GradCAM:
    """
    GradCAM implementation for visualizing which regions of the
    X-ray the model focuses on when making predictions.
    Works with CNN models that have a final convolutional layer.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights      = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam          = (weights * self.activations).sum(dim=1, keepdim=True)
        cam          = F.relu(cam)
        cam          = cam.squeeze().cpu().numpy()
        cam          = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam          = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def save_gradcam(model, dataset, device, label_names,
                 scenario_name, target_layer, n=5):
    """
    Generate and save GradCAM overlays for n sample images.
    """
    gradcam = GradCAM(model, target_layer)
    out_dir = os.path.join(OUTPUT_DIR, f'{scenario_name}_gradcam')
    os.makedirs(out_dir, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)

    for rank, idx in enumerate(indices, 1):
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        cam, class_idx = gradcam.generate(input_tensor)

        # Un-normalize image
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img * std + mean)
        img = np.clip(img, 0, 1)

        # Overlay heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.6 * img + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)

        true_labels = [label_names[i] for i, v in enumerate(label.numpy()) if v == 1]
        pred_label  = label_names[class_idx]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img)
        axes[0].set_title('Original X-Ray')
        axes[0].axis('off')
        axes[1].imshow(overlay)
        axes[1].set_title(f'GradCAM — Focus: {pred_label}\nTrue: {", ".join(true_labels) if true_labels else "No Finding"}')
        axes[1].axis('off')

        save_path = os.path.join(out_dir, f'gradcam_{rank}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved {save_path}")


# ----------------------------------------------------------------
# SCENARIO (a): CUSTOM ARCHITECTURE TRAINING
# ----------------------------------------------------------------
def train_scenario_a(train_loader, val_loader, device, label_names,
                     num_epochs=20, lr=1e-4):
    print("\nSCENARIO (a): Custom ChestXRayCNN — Trained From Scratch")
    csv_path  = init_csv_logger('scenario_a')
    model     = ChestXRayCNN(num_classes=15).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")

    losses, aucs = [], []
    best_val_loss, best_epoch, patience_counter = float('inf'), 0, 0
    early_stop_patience = 5
    best_ckpt = os.path.join(CHECKPOINT_DIR, 'custom_cnn_scratch_best.pth')

    print(f"\n{'='*55}")
    print(f"  FROM SCRATCH | ChestXRayCNN | up to {num_epochs} epochs")
    print(f"{'='*55}")

    for epoch in range(1, num_epochs + 1):
        train_loss                        = run_epoch(model, train_loader, criterion, optimizer, device, 'train')
        val_loss                          = run_epoch(model, val_loader,   criterion, optimizer, device, 'val')
        _, macro_auc, precision, recall, f1 = evaluate_metrics(model, val_loader, device, label_names)
        scheduler.step(val_loss)
        losses.append({'train': train_loss, 'val': val_loss})
        aucs.append(macro_auc)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            torch.save(model.state_dict(), best_ckpt)
            marker = "  <-- best"
        else:
            patience_counter += 1
            marker = f"  (patience {patience_counter}/{early_stop_patience})"

        log_epoch_csv(csv_path, epoch, train_loss, val_loss,
                      macro_auc, precision, recall, f1, is_best)

        print(f"  Epoch {epoch}/{num_epochs} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | AUC: {macro_auc:.4f} | "
              f"P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}{marker}")

        if patience_counter >= early_stop_patience:
            print(f"\n  Early stop triggered.")
            break

    print(f"\n  Best Val Loss : {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Best AUC      : {aucs[best_epoch - 1]:.4f}")

    # GradCAM on best checkpoint
    print("\n  Generating GradCAM visualizations...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    save_gradcam(model, val_dataset, device, label_names,
                 'scenario_a', target_layer=model.stage4[-1].conv2)

    # Best/worst predictions
    print("\n  Saving best/worst predictions...")
    save_best_worst_predictions(model, val_dataset, device, label_names, 'scenario_a')

    return {'losses': losses, 'aucs': aucs, 'best_val_loss': best_val_loss,
            'best_macro_auc': aucs[best_epoch - 1], 'best_epoch': best_epoch,
            'checkpoint': best_ckpt}


# ----------------------------------------------------------------
# SCENARIO (b): PRE-EXISTING ARCHITECTURE FROM SCRATCH — Anna Harner
# ----------------------------------------------------------------
def _run_scratch_training(architecture, train_loader, val_loader, device, label_names,
                          num_epochs=20, lr=1e-4, run_gradcam=False):
    scenario_name = f'scenario_b_{architecture}'
    csv_path  = init_csv_logger(scenario_name)
    model     = get_medical_model(architecture=architecture, num_classes=15).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    losses, aucs = [], []
    best_val_loss, best_epoch, patience_counter = float('inf'), 0, 0
    early_stop_patience = 5
    best_ckpt = os.path.join(CHECKPOINT_DIR, f'{architecture}_scratch_best.pth')

    print(f"\n{'='*55}")
    print(f"  FROM SCRATCH | {architecture.upper()} | up to {num_epochs} epochs")
    print(f"{'='*55}")

    for epoch in range(1, num_epochs + 1):
        train_loss                        = run_epoch(model, train_loader, criterion, optimizer, device, 'train')
        val_loss                          = run_epoch(model, val_loader,   criterion, optimizer, device, 'val')
        _, macro_auc, precision, recall, f1 = evaluate_metrics(model, val_loader, device, label_names)
        scheduler.step(val_loss)
        losses.append({'train': train_loss, 'val': val_loss})
        aucs.append(macro_auc)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            torch.save(model.state_dict(), best_ckpt)
            marker = "  <-- best"
        else:
            patience_counter += 1
            marker = f"  (patience {patience_counter}/{early_stop_patience})"

        log_epoch_csv(csv_path, epoch, train_loss, val_loss,
                      macro_auc, precision, recall, f1, is_best)

        print(f"  Epoch {epoch}/{num_epochs} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | AUC: {macro_auc:.4f} | "
              f"P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}{marker}")

        if patience_counter >= early_stop_patience:
            print(f"\n  Early stop triggered.")
            break

    print(f"\n  Best Val Loss : {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Best AUC      : {aucs[best_epoch - 1]:.4f}")

    # GradCAM for DenseNet (CNN only)
    if run_gradcam and architecture == 'cnn':
        print("\n  Generating GradCAM visualizations...")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        target_layer = model.features.denseblock4.denselayer16.conv2
        save_gradcam(model, val_dataset, device, label_names,
                     f'scenario_b', target_layer=target_layer)

    # Best/worst predictions
    print("\n  Saving best/worst predictions...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    save_best_worst_predictions(model, val_dataset, device, label_names, 'scenario_b')

    return {'losses': losses, 'aucs': aucs, 'best_val_loss': best_val_loss,
            'best_macro_auc': aucs[best_epoch - 1], 'best_epoch': best_epoch,
            'checkpoint': best_ckpt}


def train_scenario_b(train_loader, val_loader, device, label_names, **kwargs):
    print("\nSCENARIO (b): DenseNet-121 From Scratch")
    return _run_scratch_training('cnn', train_loader, val_loader, device,
                                 label_names, run_gradcam=True, **kwargs)


# ----------------------------------------------------------------
# SCENARIO (c): PRETRAINED FROZEN — NO FINE-TUNING — Marissa Estramonte (fixed)
# Backbone stays frozen the ENTIRE time — distinct from (d)
# ----------------------------------------------------------------
def train_scenario_c(train_loader, val_loader, device, label_names,
                     num_epochs=5, head_lr=1e-4):
    print("\nSCENARIO (c): DenseNet-121 Pretrained — Frozen Backbone (No Fine-Tuning)")
    csv_path  = init_csv_logger('scenario_c')
    model     = get_transfer_model('cnn', num_classes=15, phase=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, 'cnn', phase=1, head_lr=head_lr)

    losses, aucs = [], []
    best_val_loss, best_epoch, patience_counter = float('inf'), 0, 0
    early_stop_patience = 5
    best_ckpt = os.path.join(CHECKPOINT_DIR, 'cnn_scenario_c_best.pth')

    print(f"\n{'='*55}")
    print(f"  FROZEN BACKBONE | DenseNet-121 | up to {num_epochs} epochs")
    print(f"{'='*55}")

    for epoch in range(1, num_epochs + 1):
        train_loss                        = run_epoch(model, train_loader, criterion, optimizer, device, 'train')
        val_loss                          = run_epoch(model, val_loader,   criterion, optimizer, device, 'val')
        _, macro_auc, precision, recall, f1 = evaluate_metrics(model, val_loader, device, label_names)
        losses.append({'train': train_loss, 'val': val_loss})
        aucs.append(macro_auc)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            torch.save(model.state_dict(), best_ckpt)
            marker = "  <-- best"
        else:
            patience_counter += 1
            marker = f"  (patience {patience_counter}/{early_stop_patience})"

        log_epoch_csv(csv_path, epoch, train_loss, val_loss,
                      macro_auc, precision, recall, f1, is_best)

        print(f"  Epoch {epoch}/{num_epochs} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | AUC: {macro_auc:.4f} | "
              f"P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}{marker}")

        if patience_counter >= early_stop_patience:
            print(f"\n  Early stop triggered.")
            break

    print(f"\n  Best Val Loss : {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Best AUC      : {aucs[best_epoch - 1]:.4f}")

    return {'losses': losses, 'aucs': aucs, 'best_val_loss': best_val_loss,
            'best_macro_auc': aucs[best_epoch - 1], 'best_epoch': best_epoch,
            'checkpoint': best_ckpt}


# ----------------------------------------------------------------
# SCENARIO (d): PRETRAINED + FINE-TUNED (TWO-PHASE) — Marissa Estramonte
# Uses 224x224 images as required by ViT-B/16
# ----------------------------------------------------------------
def _run_transfer_training(architecture, train_loader, val_loader, device, label_names,
                           phase1_epochs=5, phase2_epochs=20, head_lr=1e-4):
    csv_path_p1 = init_csv_logger(f'scenario_d_{architecture}_phase1')
    csv_path_p2 = init_csv_logger(f'scenario_d_{architecture}_phase2')
    criterion   = nn.BCEWithLogitsLoss()
    results     = {}

    # --- PHASE 1 ---
    print(f"\n{'='*55}")
    print(f"  PHASE 1 | {architecture.upper()} | Frozen Backbone | {phase1_epochs} epochs")
    print(f"{'='*55}")
    model     = get_transfer_model(architecture, num_classes=15, phase=1).to(device)
    optimizer = build_optimizer(model, architecture, phase=1, head_lr=head_lr)
    p1_losses, p1_aucs = [], []

    for epoch in range(1, phase1_epochs + 1):
        train_loss                        = run_epoch(model, train_loader, criterion, optimizer, device, 'train')
        val_loss                          = run_epoch(model, val_loader,   criterion, optimizer, device, 'val')
        _, macro_auc, precision, recall, f1 = evaluate_metrics(model, val_loader, device, label_names)
        p1_losses.append({'train': train_loss, 'val': val_loss})
        p1_aucs.append(macro_auc)
        log_epoch_csv(csv_path_p1, epoch, train_loss, val_loss,
                      macro_auc, precision, recall, f1, False)
        print(f"  Epoch {epoch}/{phase1_epochs} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | AUC: {macro_auc:.4f}")

    p1_ckpt = os.path.join(CHECKPOINT_DIR, f'{architecture}_phase1.pth')
    torch.save(model.state_dict(), p1_ckpt)
    print(f"\n  Phase 1 checkpoint saved -> {p1_ckpt}")
    results['phase1'] = {'losses': p1_losses, 'aucs': p1_aucs,
                         'final_macro_auc': p1_aucs[-1], 'checkpoint': p1_ckpt}

    # --- PHASE 2 ---
    print(f"\n{'='*55}")
    print(f"  PHASE 2 | {architecture.upper()} | Full Fine-Tune | up to {phase2_epochs} epochs")
    print(f"{'='*55}")
    model = get_transfer_model(architecture, num_classes=15, phase=2).to(device)
    model.load_state_dict(torch.load(p1_ckpt, map_location=device))
    optimizer = build_optimizer(model, architecture, phase=2, head_lr=head_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5)

    p2_losses, p2_aucs = [], []
    best_auc, best_epoch, patience_counter = 0.0, 0, 0
    early_stop_patience = 7
    p2_ckpt = os.path.join(CHECKPOINT_DIR, f'{architecture}_phase2_best.pth')

    for epoch in range(1, phase2_epochs + 1):
        train_loss                        = run_epoch(model, train_loader, criterion, optimizer, device, 'train')
        val_loss                          = run_epoch(model, val_loader,   criterion, optimizer, device, 'val')
        _, macro_auc, precision, recall, f1 = evaluate_metrics(model, val_loader, device, label_names)
        scheduler.step(macro_auc)
        p2_losses.append({'train': train_loss, 'val': val_loss})
        p2_aucs.append(macro_auc)

        is_best = macro_auc > best_auc
        if is_best:
            best_auc, best_epoch, patience_counter = macro_auc, epoch, 0
            torch.save(model.state_dict(), p2_ckpt)
            marker = "  <-- best"
        else:
            patience_counter += 1
            marker = f"  (patience {patience_counter}/{early_stop_patience})"

        log_epoch_csv(csv_path_p2, epoch, train_loss, val_loss,
                      macro_auc, precision, recall, f1, is_best)

        print(f"  Epoch {epoch}/{phase2_epochs} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | AUC: {macro_auc:.4f} | "
              f"P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}{marker}")

        if patience_counter >= early_stop_patience:
            print(f"\n  Early stop triggered.")
            break

    results['phase2'] = {'losses': p2_losses, 'aucs': p2_aucs,
                         'best_macro_auc': best_auc, 'best_epoch': best_epoch,
                         'checkpoint': p2_ckpt}

    print(f"\n  Phase 1 Final AUC : {results['phase1']['final_macro_auc']:.4f}")
    print(f"  Phase 2 Best AUC  : {best_auc:.4f} (epoch {best_epoch})")
    return results


def train_scenario_d(train_loader, val_loader, device, label_names, **kwargs):
    """
    Scenario (d): ViT-B/16 with ImageNet pretraining, two-phase fine-tuning.
    Creates its own 224x224 dataloaders as required by ViT-B/16 architecture.
    """
    print("\nSCENARIO (d): ViT-B/16 Pretrained — Two-Phase Fine-Tuning")
    vit_train = NIHChestXrayDataset(train_df, target_size=224, split="train")
    vit_val   = NIHChestXrayDataset(val_df,   target_size=224, split="val")
    vit_train_loader = DataLoader(vit_train, batch_size=16, shuffle=True,
                                  num_workers=8, pin_memory=True, prefetch_factor=2)
    vit_val_loader   = DataLoader(vit_val,   batch_size=16, shuffle=False,
                                  num_workers=8, pin_memory=True, prefetch_factor=2)
    return _run_transfer_training('vit', vit_train_loader, vit_val_loader,
                                  device, label_names, **kwargs)


# ----------------------------------------------------------------
# VISUALIZATION — Marissa Estramonte
# ----------------------------------------------------------------
def _normalize(results, scenario):
    if scenario in ('a', 'b', 'c'):
        return {
            'losses':         results['losses'],
            'aucs':           results['aucs'],
            'best_epoch':     results['best_epoch'],
            'best_macro_auc': results['best_macro_auc'],
            'best_val_loss':  results['best_val_loss'],
            'phase1_auc':     None,
            'phase1_len':     0,
        }
    else:
        p1, p2 = results['phase1'], results['phase2']
        return {
            'losses':         p1['losses'] + p2['losses'],
            'aucs':           p1['aucs']   + p2['aucs'],
            'best_epoch':     len(p1['losses']) + p2['best_epoch'],
            'best_macro_auc': p2['best_macro_auc'],
            'best_val_loss':  min(l['val'] for l in p2['losses']),
            'phase1_auc':     p1['final_macro_auc'],
            'phase1_len':     len(p1['losses']),
        }


def visualize_all(results_a, results_b, results_c, results_d):
    labels    = ['(a) Custom ChestXRayCNN', '(b) DenseNet-121 Scratch',
                 '(c) DenseNet-121 Frozen', '(d) ViT-B/16 Fine-Tuned']
    raw       = [results_a, results_b, results_c, results_d]
    scenarios = ['a', 'b', 'c', 'd']
    normed    = [_normalize(r, s) for r, s in zip(raw, scenarios)]

    # --- Comparison table ---
    rows = []
    for label, n in zip(labels, normed):
        rows.append({
            'Scenario':       label,
            'Best Epoch':     n['best_epoch'],
            'Best Val Loss':  f"{n['best_val_loss']:.4f}",
            'Phase 1 AUC':    f"{n['phase1_auc']:.4f}" if n['phase1_auc'] else '—',
            'Best Macro AUC': f"{n['best_macro_auc']:.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("  RESULTS COMPARISON — ALL SCENARIOS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    fig, ax = plt.subplots(figsize=(13, 2.2))
    ax.axis('off')
    tbl = ax.table(cellText=df.values.tolist(), colLabels=list(df.columns),
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    best_idx = max(range(len(normed)), key=lambda i: normed[i]['best_macro_auc'])
    for j in range(len(df.columns)):
        tbl[best_idx + 1, j].set_facecolor('#d5f5e3')
    plt.title('Scenario Comparison — NIH Chest X-ray', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {OUTPUT_DIR}/comparison_table.png")

    # --- Loss curves ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, label, n in zip(axes, labels, normed):
        epochs     = list(range(1, len(n['losses']) + 1))
        train_loss = [l['train'] for l in n['losses']]
        val_loss   = [l['val']   for l in n['losses']]
        ax.plot(epochs, train_loss, label='Train Loss', color='#2980b9', linewidth=1.8)
        ax.plot(epochs, val_loss,   label='Val Loss',   color='#e74c3c', linewidth=1.8)
        if n['phase1_len'] > 0:
            ax.axvspan(1, n['phase1_len'], alpha=0.12, color='orange',
                       label=f"Phase 1 ({n['phase1_len']} epochs)")
            ax.axvline(x=n['phase1_len'], color='orange', linestyle='--', linewidth=1.2)
        ax.axvline(x=n['best_epoch'], color='#27ae60', linestyle=':',
                   linewidth=1.5, label=f"Best (ep {n['best_epoch']})")
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BCE Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Training & Validation Loss — All Scenarios', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {OUTPUT_DIR}/loss_curves.png")

    # --- AUC bar chart ---
    final_aucs = [n['best_macro_auc'] for n in normed]
    bar_colors = ['#2980b9', '#c0392b', '#1a5276', '#7b241c']
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(4), final_aucs, color=bar_colors, width=0.5, zorder=3)
    for i, (n, color) in enumerate(zip(normed, bar_colors)):
        if n['phase1_auc'] is not None:
            ax.bar(i, n['phase1_auc'], color=color, width=0.5,
                   alpha=0.35, hatch='//', zorder=4)
    for bar, auc in zip(bars, final_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{auc:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    best_idx = final_aucs.index(max(final_aucs))
    ax.annotate('Best', xy=(best_idx, final_aucs[best_idx]),
                xytext=(best_idx, final_aucs[best_idx] + 0.025),
                ha='center', fontsize=10, color='#27ae60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27ae60'))
    ax.set_xticks(range(4))
    ax.set_xticklabels(['(a)\nCustom CNN', '(b)\nDenseNet\nScratch',
                         '(c)\nDenseNet\nFrozen', '(d)\nViT\nFine-Tuned'], fontsize=10)
    ax.set_ylabel('Macro AUC-ROC', fontsize=11)
    ax.set_title('Macro AUC-ROC Comparison — All Scenarios', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    cnn_patch = mpatches.Patch(color='#2980b9', label='Custom CNN')
    dn_patch  = mpatches.Patch(color='#c0392b', label='DenseNet-121')
    vit_patch = mpatches.Patch(color='#7b241c', label='ViT-B/16')
    p1_patch  = mpatches.Patch(facecolor='grey', alpha=0.4, hatch='//', label='Phase 1 AUC')
    ax.legend(handles=[cnn_patch, dn_patch, vit_patch, p1_patch], loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'auc_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {OUTPUT_DIR}/auc_comparison.png")


# ----------------------------------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------------------------------
def predict(image_path, model, device, label_names, target_size=512, threshold=0.5):
    """
    Run inference on a single chest X-ray image.

    Args:
        image_path:  path to PNG image
        model:       trained model
        device:      torch device
        label_names: list of class name strings
        target_size: must match training size (512 for CNN, 224 for ViT)
        threshold:   probability threshold for positive prediction

    Returns:
        predictions: dict {class_name: probability}
        positive:    list of predicted positive class names
    """
    model.eval()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image  = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    image  = image.astype(np.float32) / 255.0
    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    image  = (image - mean) / std
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.sigmoid(output).cpu().numpy()[0]

    predictions = {name: float(prob) for name, prob in zip(label_names, probs)}
    positive    = [name for name, prob in predictions.items() if prob >= threshold]

    print(f"\n--- Predictions for {os.path.basename(image_path)} ---")
    for name, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- POSITIVE" if prob >= threshold else ""
        print(f"  {name:<22}: {prob:.4f}{marker}")
    print(f"\nPredicted: {positive if positive else ['No Finding']}")

    return predictions, positive


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
if __name__ == '__main__':
    results_a = train_scenario_a(train_loader, val_loader, device, label_names=list(label_names))
    results_b = train_scenario_b(train_loader, val_loader, device, label_names=list(label_names))
    results_c = train_scenario_c(train_loader, val_loader, device, label_names=list(label_names))
    results_d = train_scenario_d(train_loader, val_loader, device, label_names=list(label_names))
    visualize_all(results_a, results_b, results_c, results_d)
