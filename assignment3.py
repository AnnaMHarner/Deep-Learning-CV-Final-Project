"""
Assignment 3 — CPSC 542
VinBigData Chest X-Ray Object Detection

Authors:
  - Anna Harner      : Data pipeline, dataset class, augmentation, WBF consolidation
  - Marissa Estramonte: Model definitions, training infrastructure, visualization
  - Chaz Gillette     : GradCAM, best/worst predictions, transparent bounding boxes,
                       predict function, CSV logging, NaN handling, .py conversion

Scenarios:
  (a) Custom CNN backbone + Faster R-CNN, from scratch
  (b) RT-DETR (PekingU/rtdetr_r50vd), from scratch
  (c) Faster R-CNN pretrained, backbone frozen throughout
  (d) DETR pretrained, two-phase fine-tuning

Run from inside Docker container:
  python3 /app/rundir/Deep-Learning-CV-Final-Project/assignment3.py \
    >& /app/rundir/a3_training_log.txt &
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import csv
import json
import time
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.tv_tensors import BoundingBoxes
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from transformers import AutoModelForObjectDetection, AutoConfig
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion
import cv2

# ----------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------
DATA_DIR       = '/app/rundir/vinbigdata_cache/datasets/awsaf49/vinbigdata-1024-image-dataset/versions/1/vinbigdata'
IMAGE_DIR      = f'{DATA_DIR}/train'
CHECKPOINT_DIR = '/app/rundir/Deep-Learning-CV-Final-Project/checkpoints_a3'
OUTPUT_DIR     = '/app/rundir/results_a3'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------
# CLASS NAMES
# ----------------------------------------------------------------
CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis",
]

FRCNN_NUM_CLASSES = 15   # 14 pathology classes + background
DETR_NUM_LABELS   = 14   # DETR uses no background class
BATCH_SIZE        = 8
NUM_WORKERS       = 2
PATIENCE          = 5

# ----------------------------------------------------------------
# REPRODUCIBILITY
# ----------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------
# LOAD AND CONSOLIDATE DATA — Anna Harner
# ----------------------------------------------------------------
print("Loading data...")
records = pd.read_csv(f'{DATA_DIR}/train.csv')

def consolidate_dataset(df, iou_thr=0.5, skip_box_thr=0.0001):
    """Weighted Box Fusion to merge annotations from multiple radiologists."""
    findings_df = df[df['class_id'] != 14].copy()
    class_map   = findings_df[['class_id', 'class_name']].drop_duplicates().set_index('class_id')['class_name'].to_dict()
    dim_map     = df[['image_id', 'width', 'height']].drop_duplicates().set_index('image_id').to_dict('index')
    findings_df['x_min'] /= findings_df['width']
    findings_df['y_min'] /= findings_df['height']
    findings_df['x_max'] /= findings_df['width']
    findings_df['y_max'] /= findings_df['height']
    new_rows  = []
    image_ids = findings_df['image_id'].unique()
    for img_id in tqdm(image_ids, desc="Fusing Boxes"):
        img_group   = findings_df[findings_df['image_id'] == img_id]
        boxes_list  = [img_group[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()]
        scores_list = [[1.0] * len(img_group)]
        labels_list = [img_group['class_id'].values.tolist()]
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        w = dim_map[img_id]['width']
        h = dim_map[img_id]['height']
        for i in range(len(boxes)):
            cls_id = int(labels[i])
            new_rows.append({
                'image_id': img_id, 'class_id': cls_id,
                'class_name': class_map[cls_id],
                'width': w, 'height': h,
                'x_min': boxes[i][0], 'y_min': boxes[i][1],
                'x_max': boxes[i][2], 'y_max': boxes[i][3],
            })
    return pd.DataFrame(new_rows)

consolidated_records = consolidate_dataset(records)

# Re-introduce No Finding images with a 1x1 pixel dummy box (Kaggle convention)
no_findings_unique = records[records['class_id'] == 14].drop_duplicates(subset='image_id').copy()
no_findings_unique[['x_min', 'y_min', 'x_max', 'y_max']] = [0, 0, 1, 1]
final_records = pd.concat([consolidated_records, no_findings_unique], ignore_index=True)
final_records = final_records.drop(columns=['rad_id'], errors='ignore')
print(f"Total records: {len(final_records)}, Unique images: {final_records['image_id'].nunique()}")

# ----------------------------------------------------------------
# SPLITS — Anna Harner
# ----------------------------------------------------------------
files_on_disk          = [f.replace('.png', '') for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
final_records_filtered = final_records[final_records['image_id'].isin(files_on_disk)].copy()
unique_images          = final_records_filtered['image_id'].unique()
train_val_ids, test_ids = train_test_split(unique_images, test_size=0.10, random_state=42)
train_ids, val_ids      = train_test_split(train_val_ids, test_size=0.11, random_state=42)
train_df = final_records_filtered[final_records_filtered['image_id'].isin(train_ids)]
val_df   = final_records_filtered[final_records_filtered['image_id'].isin(val_ids)]
test_df  = final_records_filtered[final_records_filtered['image_id'].isin(test_ids)]
print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

# ----------------------------------------------------------------
# TRANSFORMS — Anna Harner
# ----------------------------------------------------------------
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=3),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=10),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.Resize(size=(512, 512), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=3),
    v2.Resize(size=(512, 512), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----------------------------------------------------------------
# DATASET — Anna Harner
# ----------------------------------------------------------------
class VinBigDataDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df; self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = df['image_id'].unique()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id      = self.img_ids[idx]
        img_path    = f"{self.img_dir}/{img_id}.png"
        img         = read_image(img_path)
        img_records = self.df[self.df['image_id'] == img_id]
        labels      = torch.tensor(img_records['class_id'].values, dtype=torch.int64)

        # No Finding images get empty boxes — model needs to see healthy X-rays
        # without the dummy 1x1 box causing training instability
        if (labels == 14).all():
            bboxes = BoundingBoxes(torch.zeros((0, 4), dtype=torch.float32),
                                   format="XYXY", canvas_size=img.shape[-2:])
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            h, w   = img.shape[-2], img.shape[-1]
            bboxes = img_records[['x_min', 'y_min', 'x_max', 'y_max']].values * [w, h, w, h]
            bboxes = BoundingBoxes(bboxes, format="XYXY", canvas_size=img.shape[-2:])
        if self.transforms:
            img, bboxes, labels = self.transforms(img, bboxes, labels)
        return img, {"boxes": bboxes, "labels": labels, "image_id": torch.tensor([idx])}

train_dataset = VinBigDataDataset(train_df, IMAGE_DIR, transforms=train_transforms)
val_dataset   = VinBigDataDataset(val_df,   IMAGE_DIR, transforms=val_transforms)
test_dataset  = VinBigDataDataset(test_df,  IMAGE_DIR, transforms=val_transforms)

# ----------------------------------------------------------------
# COLLATE FUNCTIONS — Marissa Estramonte
# ----------------------------------------------------------------
def collate_fn_fasterrcnn(batch):
    images, targets = zip(*batch)
    new_targets = []
    for img, t in zip(images, targets):
        h, w  = img.shape[-2], img.shape[-1]
        boxes = torch.as_tensor(t["boxes"], dtype=torch.float32)
        if boxes.numel() > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
            # Drop degenerate boxes with zero width or height
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes          = boxes[keep]
            labels_shifted = t["labels"].long()[keep] + 1  # shift 0-13 -> 1-14, background=0
        else:
            labels_shifted = t["labels"].long() + 1
        new_targets.append({"boxes": boxes, "labels": labels_shifted, "image_id": t["image_id"]})
    return list(images), new_targets


def collate_fn_detr(batch):
    images, targets = zip(*batch)
    new_targets = []
    for img, t in zip(images, targets):
        h, w   = img.shape[-2], img.shape[-1]
        boxes  = torch.as_tensor(t["boxes"], dtype=torch.float32)
        labels = t["labels"].long()
        if boxes.numel() > 0:
            bx = boxes.clone()
            bx[:, [0, 2]] = bx[:, [0, 2]].clamp(0, w)
            bx[:, [1, 3]] = bx[:, [1, 3]].clamp(0, h)
            cx = (bx[:, 0] + bx[:, 2]) / 2 / w
            cy = (bx[:, 1] + bx[:, 3]) / 2 / h
            bw = (bx[:, 2] - bx[:, 0]) / w
            bh = (bx[:, 3] - bx[:, 1]) / h
            boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=1).clamp(0, 1)
            # Drop any boxes that produced NaN coordinates
            valid        = ~torch.isnan(boxes_cxcywh).any(dim=1)
            boxes_cxcywh = boxes_cxcywh[valid]
            labels       = labels[valid]
            bx           = bx[valid]
            boxes_xyxy   = bx
        else:
            boxes_xyxy   = torch.zeros((0, 4), dtype=torch.float32)
            boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)
        new_targets.append({
            "boxes": boxes_cxcywh, "boxes_xyxy": boxes_xyxy,
            "labels": labels, "image_id": t["image_id"],
        })
    return list(images), new_targets


train_loader_frcnn = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn_fasterrcnn, pin_memory=True)
val_loader_frcnn   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn_fasterrcnn, pin_memory=True)
test_loader_frcnn  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn_fasterrcnn, pin_memory=True)
train_loader_detr  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn_detr, pin_memory=True)
val_loader_detr    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn_detr, pin_memory=True)
test_loader_detr   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, collate_fn=collate_fn_detr, pin_memory=True)

# ----------------------------------------------------------------
# CUSTOM CNN BACKBONE — Marissa Estramonte
# ----------------------------------------------------------------
class CustomCNNBackbone(nn.Module):
    """4-stage CNN backbone; strides 4/8/16/32, channels 64/128/256/512."""
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,   64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  64,  3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),           nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),           nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),           nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s1 = self.stage1(x); s2 = self.stage2(s1)
        s3 = self.stage3(s2); s4 = self.stage4(s3)
        return {"stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4}


def _wrap_custom_backbone_with_fpn(out_channels=256):
    backbone = CustomCNNBackbone()
    return BackboneWithFPN(
        backbone,
        return_layers={"stage1": "0", "stage2": "1", "stage3": "2", "stage4": "3"},
        in_channels_list=[64, 128, 256, 512],
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )

# ----------------------------------------------------------------
# MODEL DEFINITIONS — Marissa Estramonte
# ----------------------------------------------------------------
def get_fasterrcnn_model(num_classes=FRCNN_NUM_CLASSES, pretrained=False, custom_backbone=False):
    if custom_backbone:
        model = FasterRCNN(_wrap_custom_backbone_with_fpn(), num_classes=num_classes)
    elif pretrained:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_f  = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    # Disable internal normalization since we normalize in transforms
    model.transform.image_mean = [0.0, 0.0, 0.0]
    model.transform.image_std  = [1.0, 1.0, 1.0]
    return model


def get_detr_model(num_labels=DETR_NUM_LABELS, pretrained=False):
    if pretrained:
        return AutoModelForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", num_labels=num_labels, ignore_mismatched_sizes=True)
    cfg = AutoConfig.from_pretrained("facebook/detr-resnet-50", num_labels=num_labels)
    return AutoModelForObjectDetection.from_config(cfg)

# ----------------------------------------------------------------
# OPTIMIZER — Marissa Estramonte
# ----------------------------------------------------------------
def build_detection_optimizer(model, phase, model_type="fasterrcnn", head_lr=1e-3):
    """
    phase 0: all params, single LR (from-scratch scenarios)
    phase 1: backbone frozen, head only
    phase 2: backbone unfrozen with 10x lower LR than head
    """
    wd = 1e-4
    get_bb = (lambda: list(model.backbone.parameters())) if model_type == "fasterrcnn" \
             else (lambda: list(model.model.backbone.parameters()))

    if phase == 0:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=head_lr, weight_decay=wd)
    elif phase == 1:
        for p in get_bb(): p.requires_grad_(False)
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=head_lr, weight_decay=wd)
    else:
        for p in get_bb(): p.requires_grad_(True)
        bp     = get_bb()
        bp_set = set(bp)
        hp     = [p for p in model.parameters() if p not in bp_set]
        return torch.optim.AdamW([{"params": bp, "lr": head_lr / 10},
                                   {"params": hp, "lr": head_lr}], weight_decay=wd)

# ----------------------------------------------------------------
# TRAIN / EVAL LOOP — Marissa Estramonte
# ----------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, scaler, train=True, model_type="fasterrcnn"):
    model.train() if train else model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    for images, targets in tqdm(loader, desc="Train" if train else "Eval"):
        images = [img.to(device) for img in images]

        if model_type == "fasterrcnn":
            td = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in t.items()} for t in targets]
            if train:
                with torch.cuda.amp.autocast():
                    loss = sum(model(images, td).values())
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                total_loss += loss.item()
            else:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    preds = model(images)
                for pred, tgt in zip(preds, td):
                    keep = pred["labels"] != 0
                    all_preds.append({"boxes":  pred["boxes"][keep].cpu(),
                                      "labels": pred["labels"][keep].cpu(),
                                      "scores": pred["scores"][keep].cpu()})
                    all_targets.append({"boxes":  tgt["boxes"].cpu(),
                                        "labels": tgt["labels"].cpu()})
        else:
            pv   = torch.stack(images).to(device)
            h, w = pv.shape[-2], pv.shape[-1]
            if train:
                hf = [{"class_labels": t["labels"].to(device),
                        "boxes":        t["boxes"].to(device)} for t in targets]
                with torch.cuda.amp.autocast():
                    loss = model(pixel_values=pv, labels=hf).loss
                # Skip batches with numerically unstable loss
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(); scaler.update(); continue
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Skip batches where gradients went NaN
                has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                              for p in model.parameters())
                if has_nan:
                    optimizer.zero_grad(); scaler.update(); continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer); scaler.update()
                total_loss += loss.item()
            else:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    out = model(pixel_values=pv)
                for i, tgt in enumerate(targets):
                    sa     = out.logits[i].softmax(-1)[:, :-1]
                    sc, lb = sa.max(-1)
                    bx     = out.pred_boxes[i]
                    if len(sc) > 100:
                        ti = sc.topk(100).indices
                        sc = sc[ti]; lb = lb[ti]; bx = bx[ti]
                    cx, cy, bw, bh = bx[:,0]*w, bx[:,1]*h, bx[:,2]*w, bx[:,3]*h
                    xyxy = torch.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], dim=1)
                    all_preds.append({"boxes":  xyxy.cpu(),
                                      "labels": lb.cpu(),
                                      "scores": sc.cpu()})
                    all_targets.append({"boxes":  tgt["boxes_xyxy"].cpu(),
                                        "labels": tgt["labels"].cpu()})

    if train:
        return total_loss / max(len(loader), 1)
    return all_preds, all_targets

# ----------------------------------------------------------------
# METRICS — Marissa Estramonte
# ----------------------------------------------------------------
def evaluate_metrics(preds, targets, num_classes):
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    metric.update(preds, targets)
    result = metric.compute()
    return result["map_50"].item(), result.get("map_per_class", None)

# ----------------------------------------------------------------
# CSV EPOCH LOGGER — Chaz Gillette
# ----------------------------------------------------------------
def init_csv_logger(name):
    path = os.path.join(OUTPUT_DIR, f'{name}_epoch_log.csv')
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(['phase', 'epoch', 'train_loss', 'val_map50', 'is_best'])
    return path


def log_epoch_csv(path, phase, epoch, loss, map50, is_best):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([phase, epoch,
                                 f'{loss:.4f}' if loss else '',
                                 f'{map50:.4f}', is_best])

# ----------------------------------------------------------------
# GRADCAM — Chaz Gillette
# Visualizes which regions of the X-ray drive predictions.
# Implemented for the custom CNN backbone (scenario a) only.
# ----------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.gradients = None; self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, input_tensor, class_idx=0):
        self.model.train()
        features = self.model.backbone.body(input_tensor)
        stage4   = list(features.values())[-1]
        pooled   = stage4.mean(dim=(2, 3))
        self.model.zero_grad()
        pooled[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = cam.squeeze().cpu().numpy()
        cam     = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def save_gradcam_a3(model, dataset, device, scenario_name, n=5):
    out_dir      = os.path.join(OUTPUT_DIR, f'{scenario_name}_gradcam')
    os.makedirs(out_dir, exist_ok=True)
    target_layer = model.backbone.body.stage4[-1]
    gradcam      = GradCAM(model, target_layer)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    for rank, idx in enumerate(np.random.choice(len(dataset), min(n, len(dataset)), replace=False), 1):
        img_tensor, target = dataset[idx]
        try:
            cam = gradcam.generate(img_tensor.unsqueeze(0).to(device))
        except Exception as e:
            print(f"  GradCAM failed: {e}"); continue
        img     = np.clip(img_tensor.permute(1, 2, 0).numpy() * std + mean, 0, 1)
        heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
                               cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.6 * img + 0.4 * heatmap, 0, 1)
        gt_names = [CLASS_NAMES[int(l) % 14] for l in target["labels"].numpy()] or ["No Finding"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img); axes[0].set_title('Original X-Ray'); axes[0].axis('off')
        axes[1].imshow(overlay)
        axes[1].set_title(f'GradCAM\nTrue: {", ".join(gt_names[:3])}')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'gradcam_{rank}.png'), dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved gradcam_{rank}.png")

# ----------------------------------------------------------------
# BEST / WORST PREDICTIONS — Chaz Gillette
# ----------------------------------------------------------------
def save_best_worst_predictions_a3(model, dataset, device, model_type, scenario_name, n=3):
    """Save the n highest and lowest confidence predictions as PNGs."""
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    records_list = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=collate_fn_fasterrcnn if model_type == "fasterrcnn"
                        else collate_fn_detr)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(loader, desc="Finding best/worst")):
            if idx >= 200: break
            imgs_dev = [img.to(device) for img in images]
            if model_type == "fasterrcnn":
                pred = model(imgs_dev)[0]
                keep = (pred["labels"] != 0) & (pred["scores"] >= 0.3)
                sc   = pred["scores"][keep].cpu()
            else:
                pv = torch.stack(imgs_dev)
                sa = model(pixel_values=pv).logits[0].softmax(-1)[:, :-1]
                sc, _ = sa.max(-1)
            records_list.append({'idx': idx, 'max_score': sc.max().item() if len(sc) > 0 else 0.0})

    records_list.sort(key=lambda x: x['max_score'], reverse=True)
    out_dir = os.path.join(OUTPUT_DIR, scenario_name)
    os.makedirs(out_dir, exist_ok=True)

    for tag, group in [('best', records_list[:n]), ('worst', records_list[-n:])]:
        for rank, rec in enumerate(group, 1):
            img_tensor, target = dataset[rec['idx']]
            img = np.clip(img_tensor.permute(1, 2, 0).numpy() * std + mean, 0, 1)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap='gray')
            for box, lbl in zip(torch.as_tensor(target["boxes"], dtype=torch.float32),
                                 torch.as_tensor(target["labels"], dtype=torch.long)):
                x1, y1, x2, y2 = box.tolist()
                ax.add_patch(patches.FancyBboxPatch(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='lime', facecolor=(0, 1, 0, 0.15)))
                ax.text(x1, y1 - 4, CLASS_NAMES[int(lbl) % 14], fontsize=7, color='lime',
                        bbox=dict(facecolor='black', alpha=0.4, pad=1))
            ax.axis('off')
            ax.set_title(f"{tag.upper()} #{rank}\nConfidence: {rec['max_score']:.2f}", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{tag}_{rank}.png'), dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  Saved {tag}_{rank}.png")

# ----------------------------------------------------------------
# VISUALIZATION — Marissa Estramonte
# ----------------------------------------------------------------
def visualize_predictions(model, dataset, device, model_type="fasterrcnn",
                           n=4, score_thresh=0.3, title="", save_path=None):
    """Plot ground truth (green) and predictions (red) with transparent fills."""
    model.eval()
    mean    = np.array([0.485, 0.456, 0.406])
    std     = np.array([0.229, 0.224, 0.225])
    indices = random.sample(range(len(dataset)), k=n)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]

    for ax, idx in zip(axes, indices):
        img, target = dataset[idx]
        img_t = img.unsqueeze(0).to(device)
        with torch.no_grad():
            if model_type == "fasterrcnn":
                pred        = model(img_t)[0]
                keep        = (pred["labels"] != 0) & (pred["scores"] >= score_thresh)
                pred_boxes  = pred["boxes"][keep].cpu()
                pred_labels = (pred["labels"][keep] - 1).cpu()
                pred_scores = pred["scores"][keep].cpu()
            else:
                out        = model(pixel_values=img_t)
                h, w       = img_t.shape[-2], img_t.shape[-1]
                sa         = out.logits[0].softmax(-1)[:, :-1]
                sc, lb     = sa.max(-1)
                bx         = out.pred_boxes[0]
                if len(sc) > 100:
                    ti = sc.topk(100).indices
                    sc = sc[ti]; lb = lb[ti]; bx = bx[ti]
                keep        = sc >= score_thresh
                sc = sc[keep]; lb = lb[keep]; bx = bx[keep]
                cx, cy, bw, bh = bx[:,0]*w, bx[:,1]*h, bx[:,2]*w, bx[:,3]*h
                pred_boxes  = torch.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], dim=1).cpu()
                pred_labels = lb.cpu()
                pred_scores = sc.cpu()

        img_np = np.clip(img.permute(1, 2, 0).numpy() * std + mean, 0, 1)
        ax.imshow(img_np, cmap='gray')

        for box, lbl in zip(torch.as_tensor(target["boxes"],  dtype=torch.float32),
                             torch.as_tensor(target["labels"], dtype=torch.long)):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(patches.FancyBboxPatch(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1.5, edgecolor='lime', facecolor=(0, 1, 0, 0.15)))
            ax.text(x1, y1-2, CLASS_NAMES[int(lbl) % 14], fontsize=7, color='lime',
                    bbox=dict(facecolor='black', alpha=0.4, pad=1))

        for box, lbl, sc in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(patches.FancyBboxPatch(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1.5, edgecolor='red', facecolor=(1, 0, 0, 0.15)))
            ax.text(x1, y2+8, f"{CLASS_NAMES[int(lbl) % 14]} {sc:.2f}", fontsize=7, color='red',
                    bbox=dict(facecolor='black', alpha=0.4, pad=1))

        ax.axis('off'); ax.set_title(f"img {idx}")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close()


def plot_scenario_comparison(results_dict, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    colors = {0: "steelblue", 1: "darkorange", 2: "seagreen"}
    for ax, (sc, history) in zip(axes, results_dict.items()):
        by_phase = {}
        for row in history:
            by_phase.setdefault(row["phase"], []).append(row["val_map50"])
        offset = 0
        for ph, vals in sorted(by_phase.items()):
            xs = range(offset + 1, offset + len(vals) + 1)
            ax.plot(list(xs), vals, label=f"phase {ph}",
                    color=colors.get(ph, "gray"), marker="o", ms=3)
            offset += len(vals)
        ax.set_title(f"Scenario ({sc})")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mAP@0.5")
    plt.suptitle("Validation mAP@0.5 — All Scenarios", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close()

# ----------------------------------------------------------------
# STANDALONE PREDICT — Chaz Gillette
# ----------------------------------------------------------------
def predict(image_path, model, device, model_type="fasterrcnn", score_thresh=0.3):
    """
    Run detection inference on a single chest X-ray image.

    Args:
        image_path:   path to PNG image
        model:        trained detection model
        device:       torch device
        model_type:   'fasterrcnn' or 'detr'
        score_thresh: confidence threshold

    Returns:
        list of dicts with keys 'class', 'score', 'box'
    """
    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    image  = cv2.imread(image_path)
    if image is None: raise FileNotFoundError(f"Could not load: {image_path}")
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image  = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    image  = (image.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        if model_type == "fasterrcnn":
            pred   = model(tensor)[0]
            keep   = (pred["labels"] != 0) & (pred["scores"] >= score_thresh)
            boxes  = pred["boxes"][keep].cpu()
            labels = (pred["labels"][keep] - 1).cpu()
            scores = pred["scores"][keep].cpu()
        else:
            out        = model(pixel_values=tensor)
            h, w       = tensor.shape[-2], tensor.shape[-1]
            sa         = out.logits[0].softmax(-1)[:, :-1]
            sc, lb     = sa.max(-1)
            bx         = out.pred_boxes[0]
            keep       = sc >= score_thresh
            sc = sc[keep]; lb = lb[keep]; bx = bx[keep]
            cx, cy, bw, bh = bx[:,0]*w, bx[:,1]*h, bx[:,2]*w, bx[:,3]*h
            boxes  = torch.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], dim=1).cpu()
            labels = lb.cpu(); scores = sc.cpu()

    detections = []
    print(f"\n--- Detections for {os.path.basename(image_path)} ---")
    for box, lbl, sc in zip(boxes, labels, scores):
        name = CLASS_NAMES[int(lbl) % 14]
        detections.append({'class': name, 'score': float(sc), 'box': box.tolist()})
        print(f"  {name:<25}: {sc:.4f}")
    if not detections:
        print("  No findings detected above threshold.")
    return detections

# ----------------------------------------------------------------
# SCENARIO TRAINING — Marissa Estramonte
# ----------------------------------------------------------------
def train_scenario(scenario, device, epochs_p0=20, epochs_p1=15, epochs_p2=25, head_lr=1e-3):
    set_seed(42)
    t0       = time.time()
    csv_path = init_csv_logger(f'scenario_{scenario}')

    cfgs = {
        "a": dict(model_type="fasterrcnn", pretrained=False, custom_backbone=True),
        "b": dict(model_type="detr",       pretrained=False, custom_backbone=False),
        "c": dict(model_type="fasterrcnn", pretrained=True,  custom_backbone=False),
        "d": dict(model_type="detr",       pretrained=True,  custom_backbone=False),
    }
    cfg        = cfgs[scenario]
    model_type = cfg["model_type"]

    if model_type == "fasterrcnn":
        model = get_fasterrcnn_model(FRCNN_NUM_CLASSES, cfg["pretrained"], cfg["custom_backbone"])
        tl, vl, n_cls = train_loader_frcnn, val_loader_frcnn, FRCNN_NUM_CLASSES
    else:
        model = get_detr_model(DETR_NUM_LABELS, cfg["pretrained"])
        tl, vl, n_cls = train_loader_detr, val_loader_detr, DETR_NUM_LABELS

    model  = model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    metrics_history = []
    best_map = 0.0; best_per_cls = None; best_state = None

    def _phase(phase_id, n_epochs, local_track=False):
        nonlocal best_map, best_per_cls, best_state
        opt   = build_detection_optimizer(model, phase_id, model_type, head_lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", 0.5, 3) \
                if phase_id == 0 else None
        no_improve = 0; local_best = 0.0

        for epoch in range(n_epochs):
            loss = run_epoch(model, tl, opt, device, scaler, True,  model_type)
            p, t = run_epoch(model, vl, opt, device, scaler, False, model_type)
            map50, per_cls = evaluate_metrics(p, t, n_cls)
            if sched: sched.step(map50)

            is_best = map50 > (local_best if local_track else best_map)
            metrics_history.append({"phase": phase_id, "epoch": epoch + 1,
                                     "train_loss": loss, "val_map50": map50})
            log_epoch_csv(csv_path, phase_id, epoch + 1, loss, map50, is_best)
            print(f"[{scenario}] p{phase_id} ep{epoch+1}/{n_epochs} "
                  f"loss={loss:.4f} mAP@0.5={map50:.4f}" +
                  ("  <-- best" if is_best else f"  (patience {no_improve+1}/{PATIENCE})"))

            if is_best:
                if local_track: local_best = map50
                best_map     = max(best_map, map50)
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_per_cls = per_cls.tolist() if per_cls is not None else None
                no_improve   = 0
                # Save checkpoint immediately on each improvement
                _ckpt_dir = os.path.join(OUTPUT_DIR, f'scenario_{scenario}')
                os.makedirs(_ckpt_dir, exist_ok=True)
                torch.save(best_state, os.path.join(_ckpt_dir, 'best_checkpoint.pt'))
                print(f"  Checkpoint saved (mAP={map50:.4f})")
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop epoch {epoch+1}"); break

    if scenario in ("a", "b"):
        _phase(0, epochs_p0)
    elif scenario == "c":
        _phase(1, epochs_p1)
        assert all(not p.requires_grad for p in model.backbone.parameters()), \
            "backbone parameters should remain frozen throughout scenario (c)"
    else:
        # Phase 1: frozen backbone, head only
        _phase(1, epochs_p1)
        # Phase 2: unfreeze all layers, differential LRs
        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        _phase(2, epochs_p2, local_track=True)

    out_dir = os.path.join(OUTPUT_DIR, f'scenario_{scenario}')
    os.makedirs(out_dir, exist_ok=True)
    if best_state:
        torch.save(best_state, os.path.join(out_dir, 'best_checkpoint.pt'))

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({"scenario": scenario, "model_type": model_type,
                   "pretrained": cfg["pretrained"], "best_map50": best_map,
                   "wall_clock_seconds": round(time.time() - t0, 2),
                   "best_per_cls_map50": best_per_cls}, f, indent=2)
    print(f"\nScenario ({scenario}) done. Best mAP@0.5: {best_map:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if scenario == "a":
        print("\n  GradCAM...")
        save_gradcam_a3(model, val_dataset, device, f'scenario_{scenario}')
    print("\n  Best/worst predictions...")
    save_best_worst_predictions_a3(model, val_dataset, device, model_type, f'scenario_{scenario}')
    print("\n  Prediction visualization...")
    visualize_predictions(model, val_dataset, device, model_type=model_type,
                          n=4, score_thresh=0.3, title=f"Scenario ({scenario})",
                          save_path=os.path.join(out_dir, 'val_predictions.png'))
    return model, metrics_history, best_map

# ----------------------------------------------------------------
# DEPENDENCY CHECK — Chaz Gillette
# ----------------------------------------------------------------
def preflight_check():
    import importlib
    required = ['cv2', 'torch', 'torchvision', 'transformers',
                'torchmetrics', 'ensemble_boxes', 'pycocotools']
    for pkg in required:
        try:
            importlib.import_module(pkg)
            print(f"  ok  {pkg}")
        except ImportError:
            raise ImportError(f"Missing: {pkg} -- run: pip install {pkg}")
    assert os.path.exists(IMAGE_DIR), f"Image directory not found: {IMAGE_DIR}"
    assert os.path.exists(f"{DATA_DIR}/train.csv"), f"train.csv not found: {DATA_DIR}"
    assert torch.cuda.is_available(), "No GPU detected"
    print(f"  ok  GPU: {torch.cuda.get_device_name(0)}")
    print("Preflight passed.\n")

# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
if __name__ == '__main__':
    preflight_check()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    results_all = {}

    for s in ["a", "b", "c", "d"]:
        print(f"\n{'='*60}\nScenario ({s})\n{'='*60}")
        _model, _history, _best = train_scenario(s, device)
        results_all[s] = _history
        del _model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_scenario_comparison(
        results_all,
        save_path=os.path.join(OUTPUT_DIR, 'scenario_comparison.png'))

    print("\n--- Test Set Evaluation ---")
    meta = {
        "a": ("fasterrcnn", False, True),
        "b": ("detr",       False, False),
        "c": ("fasterrcnn", True,  False),
        "d": ("detr",       True,  False),
    }
    print(f"{'Scenario':<12}  {'Test mAP@0.5':>13}")
    print("-" * 27)
    for s, (mtype, pretrained, custom_bb) in meta.items():
        ckpt = os.path.join(OUTPUT_DIR, f'scenario_{s}', 'best_checkpoint.pt')
        if not os.path.exists(ckpt):
            print(f"({s})  no checkpoint"); continue
        _tm = (get_fasterrcnn_model(pretrained=pretrained, custom_backbone=custom_bb)
               if mtype == "fasterrcnn" else get_detr_model(pretrained=pretrained)).to(device)
        _tl   = test_loader_frcnn if mtype == "fasterrcnn" else test_loader_detr
        n_cls = FRCNN_NUM_CLASSES  if mtype == "fasterrcnn" else DETR_NUM_LABELS
        _tm.load_state_dict(torch.load(ckpt, map_location=device))
        p, t = run_epoch(_tm, _tl, None, device, None, train=False, model_type=mtype)
        test_map50, test_per_cls = evaluate_metrics(p, t, n_cls)
        with open(os.path.join(OUTPUT_DIR, f'scenario_{s}', 'test_results.json'), 'w') as f:
            json.dump({"scenario": s, "test_map50": test_map50,
                       "test_per_cls": test_per_cls.tolist()
                       if test_per_cls is not None else None}, f, indent=2)
        print(f"({s}){'':<10}  {test_map50:.4f}")
        del _tm

    print("\nAll done! Results saved to", OUTPUT_DIR)
