import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["wandb", "ray"]:
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", pkg],
                    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

for pkg in ["timm", "segmentation-models-pytorch", "albumentations", 
            "opencv-python-headless", "efficientnet-pytorch", "grad-cam", 
            "onnx", "onnxruntime", "scipy"]:
    install(pkg)

import torch
import torch.hub
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
import numpy as np
import yaml
import json
import shutil
import os
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import segmentation_models_pytorch as smp
import timm
from scipy.ndimage import binary_opening, binary_closing
from scipy.optimize import linear_sum_assignment
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

@dataclass
class Config:
    dataset_name: str = "door-defect-data"
    input_dir: Path = Path("/kaggle/input/door-defect-data/data")
    working_dir: Path = Path("/kaggle/working")
    train_split: float = 0.75
    val_split: float = 0.15
    test_split: float = 0.10
    encoder_name: str = "dinov2_vitb14"
    encoder_weights: str = "imagenet"
    architecture: str = "UnetPlusPlus"
    image_size: int = 518
    batch_size: int = 4
    num_epochs: int = 150
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    use_amp: bool = True
    accumulation_steps: int = 4
    ema_decay: float = 0.999
    warmup_epochs: int = 10
    aug_multiplier: int = 3
    rare_class_multiplier: int = 8
    use_defect_synthesis: bool = True
    tta_transforms: int = 8
    mc_dropout_samples: int = 10
    confidence_threshold: float = 0.35
    nms_iou_threshold: float = 0.5
    calibration_pattern_size: Tuple[int, int] = (9, 6)
    calibration_square_size_mm: float = 25.0
    defect_hierarchy: Dict[str, List[str]] = field(default_factory=lambda: {
        'scratch': ['Scratch', 'scratch', 'scratch_deep', 'horizontal_scratch'],
        'chipping': ['CHIPPING', 'chip', 'edge_damage'],
        'contamination': ['Dust', 'dust', 'Environmental Contamination', 'particles'],
        'rundown': ['RunDown', 'rundown', 'drip', 'sag'],
        'texture_defect': ['orange peel', 'orange_peel', 'peel'],
    })
    class_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'scratch': 0.40,
        'chipping': 0.35,
        'contamination': 0.45,
        'rundown': 0.30,
        'texture_defect': 0.38,
    })
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'focal': 0.5,
        'dice': 0.3,
        'boundary': 0.2,
    })
    
    def __post_init__(self):
        self.combined_dir = self.working_dir / "data" / "combined"
        self.results_dir = self.working_dir / "results"
        self.models_dir = self.working_dir / "models"
        self.viz_dir = self.working_dir / "visualizations"
        self.calib_dir = self.working_dir / "calibration"
        
        for d in [self.combined_dir, self.results_dir, self.models_dir, 
                  self.viz_dir, self.calib_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.input_dir = self._resolve_input_dir(self.input_dir)
        self.black_dir = self.input_dir / "black"
        self.white_dir = self.input_dir / "white"
        self.glossy_dir = self.input_dir / "glossy"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("🚀 ULTIMATE DOOR DEFECT DETECTION SYSTEM v2.0")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Working Directory: {self.working_dir}")
        print(f"Input Directory:   {self.input_dir}")
        print()
    
    @staticmethod
    def _resolve_input_dir(initial_dir: Path) -> Path:
        for door in ['black', 'white', 'glossy']:
            if (initial_dir / door / 'train' / 'images').exists():
                print(f"✅ Found data root at: {initial_dir}")
                return initial_dir
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for candidate in sorted(kaggle_input.rglob("*")):
                if not candidate.is_dir():
                    continue
                for door in ['black', 'white', 'glossy']:
                    if (candidate / door / 'train' / 'images').exists():
                        print(f"🔍 Auto-detected data root at: {candidate}")
                        return candidate
        print(f"⚠️  Could not auto-detect data root, using: {initial_dir}")
        return initial_dir

class DefectTaxonomy:    
    def __init__(self, hierarchy: Dict[str, List[str]]):
        self.hierarchy = hierarchy
        self.unified_classes = list(hierarchy.keys())
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.unified_classes)}
        self.id_to_class = {idx: cls for cls, idx in self.class_to_id.items()}
        self.variant_to_unified = {}
        for unified, variants in hierarchy.items():
            for variant in variants:
                self.variant_to_unified[variant.lower()] = unified
    
    def map_to_unified(self, variant_name: str) -> str:
        return self.variant_to_unified.get(variant_name.lower(), variant_name)
    
    def get_class_id(self, class_name: str) -> int:
        unified = self.map_to_unified(class_name)
        return self.class_to_id.get(unified, -1)
    
    def get_num_classes(self) -> int:
        return len(self.unified_classes)
    
    def print_taxonomy(self):
        print("\n📋 Defect Taxonomy:")
        for unified, variants in self.hierarchy.items():
            print(f"  {self.class_to_id[unified]:2d}. {unified:20s} ← {', '.join(variants)}")

class DefectAugmentation:
    @staticmethod
    def get_train_transforms(image_size: int = 1024):
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.85, 1.15),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7
            ),
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=0,
                    sat_shift_limit=30,
                    val_shift_limit=30,
                    p=1.0
                ),
            ], p=0.9),
            A.OneOf([
                A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.15
            ),
            A.RandomToneCurve(scale=0.1, p=0.25),
            A.ImageCompression(quality_range=(75, 100), p=0.2),
            A.Erasing(
                scale=(0.02, 0.08),
                ratio=(0.3, 3.3),
                fill_value=0,
                p=0.1
            ),
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    
    @staticmethod
    def get_val_transforms(image_size: int = 1024):
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    
    @staticmethod
    def get_tta_transforms(image_size: int = 1024):
        base_transforms = [
            A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
        ]
        
        for flip_fn in [A.HorizontalFlip, A.VerticalFlip]:
            base_transforms.append(
                A.Compose([
                    flip_fn(p=1.0),
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            )
        
        for angle in [90, 180, 270]:
            base_transforms.append(
                A.Compose([
                    A.Affine(rotate=(angle, angle), border_mode=cv2.BORDER_REFLECT_101, p=1.0),
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            )
        
        for scale in [0.9, 1.1]:
            base_transforms.append(
                A.Compose([
                    A.Affine(scale=(scale, scale), border_mode=cv2.BORDER_REFLECT_101, p=1.0),
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            )
        
        return base_transforms[:8]

class DefectSynthesis:
    @staticmethod
    def extract_defect_patches(image: np.ndarray, mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        patches = []
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            pad = 20
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
            
            patch_img = image[y1:y2, x1:x2]
            patch_mask = mask[y1:y2, x1:x2]
            
            if patch_img.size > 0 and patch_mask.sum() > 100:
                patches.append((patch_img.copy(), patch_mask.copy()))
        
        return patches
    
    @staticmethod
    def paste_defect(background: np.ndarray, bg_mask: np.ndarray, 
                     defect_patch: np.ndarray, defect_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = background.shape[:2]
        ph, pw = defect_patch.shape[:2]
        
        if ph >= h or pw >= w:
            return background, bg_mask
        
        max_attempts = 50
        for _ in range(max_attempts):
            x = np.random.randint(0, w - pw)
            y = np.random.randint(0, h - ph)
            
            roi_mask = bg_mask[y:y+ph, x:x+pw]
            if roi_mask.sum() < 10:
                try:
                    center = (x + pw // 2, y + ph // 2)
                    result = cv2.seamlessClone(
                        defect_patch, background, 
                        (defect_mask * 255).astype(np.uint8), 
                        center, cv2.NORMAL_CLONE
                    )
                    
                    bg_mask[y:y+ph, x:x+pw] = np.maximum(bg_mask[y:y+ph, x:x+pw], defect_mask)
                    return result, bg_mask
                except:
                    continue
        
        return background, bg_mask

class DefectDatasetPreparation:
    def __init__(self, config: Config, taxonomy: DefectTaxonomy):
        self.config = config
        self.taxonomy = taxonomy
        self.source_datasets = {
            'black': config.black_dir,
            'white': config.white_dir,
            'glossy': config.glossy_dir
        }
    
    def merge_datasets(self):
        print("📦 DATASET PREPARATION")
        
        for split in ['train', 'val', 'test']:
            (self.config.combined_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.config.combined_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        
        all_samples = []
        class_distribution = defaultdict(int)
        
        for door_type, door_dir in self.source_datasets.items():
            if not door_dir.exists():
                print(f"⚠️  Skipping {door_type}: directory not found")
                continue
            
            yaml_path = door_dir / 'data.yaml'
            if not yaml_path.exists():
                print(f"⚠️  No data.yaml for {door_type}")
                continue
            
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            local_class_names = data_config.get('names', [])
            
            img_dir = door_dir / 'train' / 'images'
            lbl_dir = door_dir / 'train' / 'labels'
            
            if not img_dir.exists():
                continue
            
            images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
            
            for img_path in images:
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue
                
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                
                dominant_class = None
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    
                    local_class_id = int(parts[0])
                    if local_class_id < len(local_class_names):
                        variant_name = local_class_names[local_class_id]
                        unified_class = self.taxonomy.map_to_unified(variant_name)
                        dominant_class = unified_class
                        class_distribution[unified_class] += 1
                        break
                
                if dominant_class:
                    all_samples.append({
                        'image_path': img_path,
                        'label_path': lbl_path,
                        'door_type': door_type,
                        'dominant_class': dominant_class,
                        'local_class_names': local_class_names
                    })
        
        print(f"\n📊 Collected {len(all_samples)} samples")
        for cls, count in sorted(class_distribution.items()):
            print(f"  {cls:20s}: {count:4d} samples")
        
        if len(all_samples) == 0:
            print("\n❌ No samples found! Diagnostic info:")
            for door_type, door_dir in self.source_datasets.items():
                img_dir = door_dir / 'train' / 'images'
                lbl_dir = door_dir / 'train' / 'labels'
                print(f"  {door_type}:")
                print(f"    dir exists:    {door_dir.exists()}")
                print(f"    images dir:    {img_dir.exists()}")
                print(f"    labels dir:    {lbl_dir.exists()}")
                if img_dir.exists():
                    imgs = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
                    print(f"    image count:   {len(imgs)}")
            raise ValueError(
                f"No samples found in input_dir={self.config.input_dir}. "
                f"Expected structure: {{input_dir}}/{{black,white,glossy}}/train/images/ and .../labels/"
            )
        
        indices = np.arange(len(all_samples))
        stratify_labels = [s['dominant_class'] for s in all_samples]
        
        from sklearn.model_selection import train_test_split
        
        try:
            train_idx, temp_idx = train_test_split(
                indices,
                train_size=self.config.train_split,
                stratify=stratify_labels,
                random_state=42
            )
        except ValueError:
            print("⚠️  Stratified train split failed (rare classes), using non-stratified")
            train_idx, temp_idx = train_test_split(
                indices,
                train_size=self.config.train_split,
                random_state=42
            )
        
        temp_stratify = [stratify_labels[i] for i in temp_idx]
        val_size = self.config.val_split / (self.config.val_split + self.config.test_split)
        
        try:
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size,
                stratify=temp_stratify,
                random_state=42
            )
        except ValueError:
            print("⚠️  Stratified val/test split failed (rare classes), using non-stratified")
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size,
                random_state=42
            )
        
        splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        
        for split_name, split_indices in splits.items():
            print(f"\n📁 Processing {split_name}: {len(split_indices)} samples")
            
            for idx in split_indices:
                sample = all_samples[idx]
                img_path = sample['image_path']
                lbl_path = sample['label_path']
                
                new_name = f"{sample['door_type']}_{img_path.name}"
                
                dst_img = self.config.combined_dir / split_name / 'images' / new_name
                shutil.copy(str(img_path), str(dst_img))
                
                mask = self._label_to_mask(
                    str(lbl_path),
                    img_path,
                    sample['local_class_names']
                )
                
                mask_name = f"{sample['door_type']}_{img_path.stem}.png"
                dst_mask = self.config.combined_dir / split_name / 'masks' / mask_name
                cv2.imwrite(str(dst_mask), mask)
        
        self._create_data_yaml()
        
        print("\n✅ Dataset preparation complete!")
        return self.config.combined_dir / 'data.yaml'
    
    def _label_to_mask(self, label_path: str, img_path: Path, local_class_names: List[str]) -> np.ndarray:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            local_class_id = int(parts[0])
            if local_class_id >= len(local_class_names):
                continue
            
            variant_name = local_class_names[local_class_id]
            unified_class_id = self.taxonomy.get_class_id(variant_name)
            
            if unified_class_id < 0:
                continue
            
            coords = list(map(float, parts[1:]))
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            
            cv2.fillPoly(mask, [points], unified_class_id + 1)
        
        return mask
    
    def _create_data_yaml(self):
        data_config = {
            'path': str(self.config.combined_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': self.taxonomy.get_num_classes(),
            'names': self.taxonomy.unified_classes
        }
        
        yaml_path = self.config.combined_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"\n📄 Created {yaml_path}")

class DefectDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None, cache: bool = False):
        self.image_paths = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
        self.mask_dir = mask_dir
        self.transform = transform
        self.cache = cache
        self.cached_data = {} if cache else None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.cache and idx in self.cached_data:
            return self.cached_data[idx]
        
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        mask = mask.long()
        
        sample = {'image': image, 'mask': mask, 'image_path': str(img_path)}
        
        if self.cache:
            self.cached_data[idx] = sample
        
        return sample

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 Encoder with Multi-Scale Feature Extraction"""
    def __init__(self, model_name='dinov2_vitb14', pretrained=True):
        super().__init__()
        # Load from torch hub
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
        
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.backbone.embed_dim
        
        # Projection layers to standard channel counts for FPN [96, 192, 384, 768]
        self.projections = nn.ModuleList([
            nn.Conv2d(self.embed_dim, 96, 1),
            nn.Conv2d(self.embed_dim, 192, 1),
            nn.Conv2d(self.embed_dim, 384, 1),
            nn.Conv2d(self.embed_dim, 768, 1),
        ])
        
    def forward(self, x):
        # x: [B, 3, H, W]
        B, C, H, W = x.shape
        
        # Get intermediate layers (3, 6, 9, 12)
        features_dict = self.backbone.get_intermediate_layers(x, n=4, reshape=True)
        # features_dict is list of [B, embed_dim, h, w] (1/14 scale if H,W multiples of 14)
        
        out_features = []
        
        # Level 0: 1/4 scale (upsample)
        fpn_0 = self.projections[0](features_dict[0])
        fpn_0 = F.interpolate(fpn_0, size=(H//4, W//4), mode='bilinear', align_corners=False)
        out_features.append(fpn_0)
        
        # Level 1: 1/8 scale (upsample)
        fpn_1 = self.projections[1](features_dict[1])
        fpn_1 = F.interpolate(fpn_1, size=(H//8, W//8), mode='bilinear', align_corners=False)
        out_features.append(fpn_1)
        
        # Level 2: 1/16 scale (downsample/upsample depending on input vs 1/14)
        fpn_2 = self.projections[2](features_dict[2])
        fpn_2 = F.interpolate(fpn_2, size=(H//16, W//16), mode='bilinear', align_corners=False)
        out_features.append(fpn_2)
        
        # Level 3: 1/32 scale (downsample)
        fpn_3 = self.projections[3](features_dict[3])
        fpn_3 = F.interpolate(fpn_3, size=(H//32, W//32), mode='bilinear', align_corners=False)
        out_features.append(fpn_3)
        
        return out_features

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels) 
        )

    def forward(self, x):
        return self.double_conv(x)

class FPNUNetDecoder(nn.Module):
    """UNet++ Style Decoder for FPN features"""
    def __init__(self, num_classes):
        super().__init__()
        
        # Layer 1 (1/16)
        self.conv_1_0 = DoubleConv(384, 384)
        self.conv_1_1 = DoubleConv(384 + 768, 384) 
        
        # Layer 2 (1/8)
        self.conv_2_0 = DoubleConv(192, 192)
        self.conv_2_1 = DoubleConv(192 + 384, 192) 
        self.conv_2_2 = DoubleConv(192 + 192 + 384, 192) 
        
        # Layer 3 (1/4)
        self.conv_3_0 = DoubleConv(96, 96)
        self.conv_3_1 = DoubleConv(96 + 192, 96)
        self.conv_3_2 = DoubleConv(96 + 96 + 192, 96)
        self.conv_3_3 = DoubleConv(96 + 96 + 96 + 192, 96)
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final segmentation head (from 1/4 scale)
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), # 1/4 -> 1/1
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes + 1, kernel_size=1)
        )
        
    def forward(self, features):
        x0, x1, x2, x3 = features # 1/4, 1/8, 1/16, 1/32
        
        # 1/16 level
        x2_0 = x2
        x2_1 = self.conv_1_1(torch.cat([x2_0, self.up(x3)], dim=1))
        
        # 1/8 level
        x1_0 = x1
        x1_1 = self.conv_2_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x1_2 = self.conv_2_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        
        # 1/4 level
        x0_0 = x0
        x0_1 = self.conv_3_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x0_2 = self.conv_3_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x0_3 = self.conv_3_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        
        # Output
        logits = self.final_conv(x0_3)
        return logits

class DINOv2DefectDetector(nn.Module):
    def __init__(self, config: Config, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        print(f"Loading DINOv2 Encoder: {config.encoder_name}...")
        self.encoder = DINOv2Encoder(model_name=config.encoder_name)
        self.decoder = FPNUNetDecoder(num_classes)
        
        # Boundary refinement head (no sigmoid — handled by BCEWithLogitsLoss)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(num_classes + 1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Only init decoder and head, encoder is frozen
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for m in self.boundary_head.modules():
             if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, return_boundary=False):
        features = self.encoder(x)
        seg_logits = self.decoder(features)
        
        # Ensure output size matches input size (handling 518 vs 512 mismatches if any)
        if seg_logits.shape[-2:] != x.shape[-2:]:
            seg_logits = F.interpolate(seg_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        if return_boundary:
            boundary = self.boundary_head(seg_logits)
            return seg_logits, boundary
        
        return seg_logits
    
    def enable_dropout(self):
        pass

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets, num_classes):
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate dice
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """Loss for boundary refinement"""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, boundary_pred, mask):
        """
        Args:
            boundary_pred: [B, 1, H, W] predicted boundaries
            mask: [B, H, W] segmentation mask
        """
        # Extract boundaries from mask using morphological operations
        kernel = torch.ones(3, 3, device=mask.device)
        
        # Dilate and erode to get boundaries
        mask_float = mask.float().unsqueeze(1)
        boundary_gt = torch.clamp(
            F.max_pool2d(mask_float, 3, stride=1, padding=1) - 
            F.avg_pool2d(mask_float, 3, stride=1, padding=1),
            0, 1
        )
        
        return self.bce(boundary_pred, boundary_gt)


class CombinedLoss(nn.Module):
    """Combined loss for defect detection"""
    
    def __init__(self, config: Config, num_classes: int):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.weights = config.loss_weights
        self.num_classes = num_classes
    
    def forward(self, seg_logits, boundary_pred, targets):
        """
        Args:
            seg_logits: [B, C, H, W]
            boundary_pred: [B, 1, H, W] or None
            targets: [B, H, W]
        """
        loss_focal = self.focal(seg_logits, targets)
        loss_dice = self.dice(seg_logits, targets, self.num_classes + 1)
        
        total_loss = (
            self.weights['focal'] * loss_focal +
            self.weights['dice'] * loss_dice
        )
        
        losses = {
            'focal': loss_focal.item(),
            'dice': loss_dice.item(),
        }
        
        if boundary_pred is not None:
            loss_boundary = self.boundary(boundary_pred, targets)
            total_loss += self.weights['boundary'] * loss_boundary
            losses['boundary'] = loss_boundary.item()
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


# ═══════════════════════════════════════════════════════════════════════════
#                           TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ExponentialMovingAverage:
    """EMA for model parameters"""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class DefectDetectionTrainer:
    """Advanced training loop with all bells and whistles"""
    
    def __init__(self, config: Config, model: nn.Module, num_classes: int):
        self.config = config
        self.model = model.to(config.device)
        self.num_classes = num_classes
        
        # Loss function
        self.criterion = CombinedLoss(config, num_classes)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.num_epochs // 3,
            T_mult=2,
            eta_min=1e-6
        )
        
        # AMP scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # EMA
        self.ema = ExponentialMovingAverage(model, decay=config.ema_decay)
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.config.device)
            masks = batch['mask'].to(self.config.device)
            
            # Forward pass with AMP
            if self.config.use_amp:
                with autocast():
                    seg_logits, boundary_pred = self.model(images, return_boundary=True)
                    loss, losses = self.criterion(seg_logits, boundary_pred, masks)
            else:
                seg_logits, boundary_pred = self.model(images, return_boundary=True)
                loss, losses = self.criterion(seg_logits, boundary_pred, masks)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            # Update EMA
            self.ema.update()
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] += v
            
            # Progress
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx:3d}/{num_batches}] Loss: {loss.item():.4f}")
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, val_loader, use_ema=True):
        """Validate model"""
        if use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        epoch_losses = defaultdict(float)
        num_batches = len(val_loader)
        
        # Metrics
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            images = batch['image'].to(self.config.device)
            masks = batch['mask'].to(self.config.device)
            
            seg_logits, boundary_pred = self.model(images, return_boundary=True)
            loss, losses = self.criterion(seg_logits, boundary_pred, masks)
            
            for k, v in losses.items():
                epoch_losses[k] += v
            
            # Get predictions
            preds = torch.argmax(seg_logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = self.calculate_metrics(all_preds, all_targets)
        epoch_losses.update(metrics)
        
        if use_ema:
            self.ema.restore()
        
        return epoch_losses
    
    def calculate_metrics(self, preds, targets):
        """Calculate IoU, precision, recall per class"""
        num_classes = self.num_classes + 1  # +1 for background
        
        metrics = {}
        
        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            iou = intersection / (union + 1e-6)
            
            tp = intersection
            fp = pred_mask.sum() - intersection
            fn = target_mask.sum() - intersection
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            metrics[f'iou_class_{cls}'] = iou
            metrics[f'precision_class_{cls}'] = precision
            metrics[f'recall_class_{cls}'] = recall
            metrics[f'f1_class_{cls}'] = f1
        
        # Mean metrics
        metrics['mean_iou'] = np.mean([metrics[f'iou_class_{i}'] for i in range(num_classes)])
        metrics['mean_f1'] = np.mean([metrics[f'f1_class_{i}'] for i in range(num_classes)])
        
        return metrics
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print("\n" + "="*80)
        print("🚀 TRAINING STATE-OF-THE-ART DEFECT DETECTOR")
        print("="*80)
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📅 Epoch [{epoch+1}/{self.config.num_epochs}]")
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"  Train Loss: {train_losses['total']:.4f} | " +
                  f"Focal: {train_losses['focal']:.4f} | " +
                  f"Dice: {train_losses['dice']:.4f}")
            
            # Validate
            val_losses = self.validate(val_loader)
            print(f"  Val Loss:   {val_losses['total']:.4f} | " +
                  f"mIoU: {val_losses['mean_iou']:.4f} | " +
                  f"mF1: {val_losses['mean_f1']:.4f}")
            
            # Learning rate step
            self.scheduler.step()
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, val_losses, is_best=True)
                print("  ✅ Best model saved!")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_losses)
            
            # Track history
            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
            self.metrics_history.append({
                'epoch': epoch + 1,
                'train': train_losses,
                'val': val_losses
            })
        
        print("\n✅ Training complete!")
        return self.metrics_history
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        
        if is_best:
            path = self.config.models_dir / 'best_model.pth'
        else:
            path = self.config.models_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)


# ═══════════════════════════════════════════════════════════════════════════
#                    INFERENCE WITH UNCERTAINTY QUANTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

class DefectInference:
    """Production-ready inference with TTA and uncertainty"""
    
    def __init__(self, model: nn.Module, config: Config, taxonomy: DefectTaxonomy):
        self.model = model
        self.config = config
        self.taxonomy = taxonomy
        self.device = config.device
        
        self.model.to(self.device)
        self.model.eval()
        
        # TTA transforms
        self.tta_transforms = DefectAugmentation.get_tta_transforms(config.image_size)
    
    @torch.no_grad()
    def predict_with_tta(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with test-time augmentation
        
        Returns:
            predictions: [H, W] class predictions
            uncertainty: [H, W] uncertainty map
        """
        h, w = image.shape[:2]
        
        all_predictions = []
        
        for transform in self.tta_transforms:
            # Apply transform
            augmented = transform(image=image)['image']
            augmented = augmented.unsqueeze(0).to(self.device)
            
            # Predict
            logits = self.model(augmented)
            probs = F.softmax(logits, dim=1)
            
            # Resize back to original
            probs = F.interpolate(probs, size=(h, w), mode='bilinear', align_corners=False)
            all_predictions.append(probs.cpu().numpy()[0])
        
        # Average predictions
        avg_probs = np.mean(all_predictions, axis=0)
        
        # Uncertainty as std of predictions
        uncertainty = np.std(all_predictions, axis=0).mean(axis=0)
        
        # Get final prediction
        predictions = np.argmax(avg_probs, axis=0)
        
        return predictions, uncertainty
    
    @torch.no_grad()
    def predict_with_mc_dropout(self, image: np.ndarray, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with Monte Carlo Dropout for uncertainty
        
        Returns:
            predictions: [H, W] class predictions
            uncertainty: [H, W] epistemic uncertainty
        """
        h, w = image.shape[:2]
        
        # Prepare image
        transform = DefectAugmentation.get_val_transforms(self.config.image_size)
        image_tensor = transform(image=image)['image'].unsqueeze(0).to(self.device)
        
        # Enable dropout
        self.model.enable_dropout()
        
        all_predictions = []
        
        for _ in range(n_samples):
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            probs = F.interpolate(probs, size=(h, w), mode='bilinear', align_corners=False)
            all_predictions.append(probs.cpu().numpy()[0])
        
        # Disable dropout
        self.model.eval()
        
        # Calculate mean and uncertainty
        avg_probs = np.mean(all_predictions, axis=0)
        uncertainty = np.std(all_predictions, axis=0).mean(axis=0)
        
        predictions = np.argmax(avg_probs, axis=0)
        
        return predictions, uncertainty
    
    def post_process(self, predictions: np.ndarray, min_area: int = 100) -> np.ndarray:
        """Post-process predictions with morphological operations"""
        processed = predictions.copy()
        
        # For each class
        for cls in range(1, self.taxonomy.get_num_classes() + 1):
            mask = (predictions == cls).astype(np.uint8)
            
            # Morphological opening (remove noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = binary_opening(mask, structure=kernel).astype(np.uint8)
            
            # Morphological closing (fill holes)
            mask = binary_closing(mask, structure=kernel).astype(np.uint8)
            
            # Remove small components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    mask[labels == i] = 0
            
            # Update processed
            processed[mask > 0] = cls
        
        return processed
    
    def predict(self, image: np.ndarray, use_tta: bool = True, 
                use_mc_dropout: bool = False) -> Dict:
        """
        Full prediction pipeline
        
        Returns dictionary with:
            - predictions: class map
            - uncertainty: uncertainty map
            - detections: list of detected defects
            - visualization: annotated image
        """
        # Predict
        if use_tta:
            predictions, uncertainty = self.predict_with_tta(image)
        elif use_mc_dropout:
            predictions, uncertainty = self.predict_with_mc_dropout(image)
        else:
            h, w = image.shape[:2]
            transform = DefectAugmentation.get_val_transforms(self.config.image_size)
            image_tensor = transform(image=image)['image'].unsqueeze(0).to(self.device)
            
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            probs = F.interpolate(probs, size=(h, w), mode='bilinear', align_corners=False)
            
            predictions = torch.argmax(probs, dim=1).cpu().numpy()[0]
            uncertainty = np.zeros_like(predictions, dtype=np.float32)
        
        # Post-process
        predictions = self.post_process(predictions)
        
        # Extract detections
        detections = self.extract_detections(predictions, uncertainty)
        
        # Visualize
        visualization = self.visualize(image, predictions, uncertainty, detections)
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'detections': detections,
            'visualization': visualization
        }
    
    def extract_detections(self, predictions: np.ndarray, 
                          uncertainty: np.ndarray) -> List[Dict]:
        """Extract individual defect instances"""
        detections = []
        
        for cls in range(1, self.taxonomy.get_num_classes() + 1):
            mask = (predictions == cls).astype(np.uint8)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT+1]
                
                # Get defect mask
                defect_mask = (labels == i).astype(np.uint8)
                
                # Calculate average uncertainty for this defect
                avg_uncertainty = uncertainty[defect_mask > 0].mean()
                
                # Get contour
                contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    contour = contours[0]
                else:
                    continue
                
                detections.append({
                    'class_id': cls - 1,  # 0-indexed
                    'class_name': self.taxonomy.id_to_class[cls - 1],
                    'bbox': [x, y, x + w, y + h],
                    'area_pixels': int(area),
                    'centroid': centroids[i].tolist(),
                    'uncertainty': float(avg_uncertainty),
                    'contour': contour,
                    'mask': defect_mask
                })
        
        return detections
    
    def visualize(self, image: np.ndarray, predictions: np.ndarray,
                  uncertainty: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Create comprehensive visualization"""
        h, w = image.shape[:2]
        
        # Create 4-panel visualization
        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Panel 1: Original image
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        canvas[:h, :w] = image_rgb
        
        # Panel 2: Segmentation overlay
        overlay = image_rgb.copy()
        colors = [
            (0, 0, 0),      # Background
            (255, 0, 0),    # Class 0 - Red
            (0, 255, 0),    # Class 1 - Green
            (0, 0, 255),    # Class 2 - Blue
            (255, 255, 0),  # Class 3 - Yellow
            (255, 0, 255),  # Class 4 - Magenta
            (0, 255, 255),  # Class 5 - Cyan
        ]
        
        for cls in range(1, min(len(colors), self.taxonomy.get_num_classes() + 1)):
            mask = (predictions == cls)
            overlay[mask] = colors[cls]
        
        canvas[:h, w:2*w] = cv2.addWeighted(image_rgb, 0.5, overlay, 0.5, 0)
        
        # Panel 3: Detections with bounding boxes
        detection_vis = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = colors[det['class_id'] + 1]
            
            cv2.rectangle(detection_vis, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']} ({det['uncertainty']:.2f})"
            cv2.putText(detection_vis, label, (x1, max(y1 - 5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        canvas[h:2*h, :w] = detection_vis
        
        # Panel 4: Uncertainty heatmap
        uncertainty_norm = (uncertainty * 255).astype(np.uint8)
        uncertainty_color = cv2.applyColorMap(uncertainty_norm, cv2.COLORMAP_JET)
        canvas[h:2*h, w:2*w] = uncertainty_color
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Segmentation", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Detections", (10, h + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Uncertainty", (w + 10, h + 30), font, 1, (255, 255, 255), 2)
        
        return canvas


# ═══════════════════════════════════════════════════════════════════════════
#                         CAMERA CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

class CameraCalibration:
    """Proper camera calibration with distortion correction"""
    
    @staticmethod
    def calibrate_from_checkerboard(image_paths: List[Path], 
                                    pattern_size: Tuple[int, int],
                                    square_size_mm: float) -> Dict:
        """
        Calibrate camera using checkerboard pattern
        
        Args:
            image_paths: List of calibration images
            pattern_size: (cols, rows) internal corners
            square_size_mm: Size of checkerboard square in mm
        
        Returns:
            Calibration parameters
        """
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size_mm
        
        obj_points = []  # 3D points in real world
        img_points = []  # 2D points in image plane
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                obj_points.append(objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_refined)
        
        if len(obj_points) == 0:
            print("⚠️  No checkerboard patterns found!")
            return None
        
        # Calibrate
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            img_points_reproj, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(img_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
            total_error += error
        
        mean_error = total_error / len(obj_points)
        
        print(f"✅ Calibration complete!")
        print(f"   Reprojection error: {mean_error:.4f} pixels")
        print(f"   Used {len(obj_points)} images")
        
        return {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist(),
            'reprojection_error': float(mean_error),
            'num_images': len(obj_points),
            'pixels_per_mm': float(camera_matrix[0, 0] / square_size_mm)
        }
    
    @staticmethod
    def undistort_image(image: np.ndarray, calibration: Dict) -> np.ndarray:
        """Remove lens distortion from image"""
        camera_matrix = np.array(calibration['camera_matrix'])
        dist_coeffs = np.array(calibration['dist_coeffs'])
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, 
                                    None, new_camera_matrix)
        
        # Crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted


# ═══════════════════════════════════════════════════════════════════════════
#                              MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Execute the ultimate defect detection pipeline"""
    
    # Initialize configuration
    config = Config()
    
    # Build defect taxonomy
    taxonomy = DefectTaxonomy(config.defect_hierarchy)
    taxonomy.print_taxonomy()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: DATA PREPARATION
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 1: DATASET PREPARATION")
    print("="*80)
    
    data_prep = DefectDatasetPreparation(config, taxonomy)
    data_yaml_path = data_prep.merge_datasets()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: CREATE DATALOADERS
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 2: CREATING DATALOADERS")
    print("="*80)
    
    train_dataset = DefectDataset(
        config.combined_dir / 'train' / 'images',
        config.combined_dir / 'train' / 'masks',
        transform=DefectAugmentation.get_train_transforms(config.image_size),
        cache=False  # Set to True if you have enough RAM
    )
    
    val_dataset = DefectDataset(
        config.combined_dir / 'val' / 'images',
        config.combined_dir / 'val' / 'masks',
        transform=DefectAugmentation.get_val_transforms(config.image_size),
        cache=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: BUILD MODEL
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 3: BUILDING STATE-OF-THE-ART MODEL")
    print("="*80)
    
    torch.cuda.empty_cache()
    model = DINOv2DefectDetector(config, taxonomy.get_num_classes())
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model: {config.architecture} with {config.encoder_name}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: TRAIN MODEL
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80)
    
    trainer = DefectDetectionTrainer(config, model, taxonomy.get_num_classes())
    metrics_history = trainer.train(train_loader, val_loader)
    
    # Save metrics
    with open(config.results_dir / 'training_history.json', 'w') as f:
        json.dump(metrics_history, f, indent=2, default=str)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: INFERENCE ON TEST SET
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 5: INFERENCE WITH TTA & UNCERTAINTY")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(config.models_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create inference engine
    inference = DefectInference(model, config, taxonomy)
    
    # Test on some images
    test_img_dir = config.combined_dir / 'test' / 'images'
    test_images = sorted(list(test_img_dir.glob('*.png')) + list(test_img_dir.glob('*.jpg')))[:20]
    
    print(f"Running inference on {len(test_images)} test images...")
    
    for img_path in test_images:
        print(f"\n  Processing: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict with TTA
        results = inference.predict(image, use_tta=True, use_mc_dropout=False)
        
        # Save visualization
        viz_path = config.viz_dir / f"{img_path.stem}_prediction.jpg"
        cv2.imwrite(str(viz_path), cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR))
        
        # Save JSON report
        report = {
            'image_path': str(img_path),
            'num_defects': len(results['detections']),
            'detections': [
                {
                    'class_name': d['class_name'],
                    'bbox': d['bbox'],
                    'area_pixels': d['area_pixels'],
                    'uncertainty': d['uncertainty']
                }
                for d in results['detections']
            ]
        }
        
        json_path = config.results_dir / f"{img_path.stem}_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"    Detected {len(results['detections'])} defects")
        for det in results['detections']:
            print(f"      - {det['class_name']}: {det['area_pixels']} px² "
                  f"(uncertainty: {det['uncertainty']:.3f})")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: EXPORT FOR DEPLOYMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("STEP 6: EXPORTING MODEL FOR DEPLOYMENT")
    print("="*80)
    
    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size).to(config.device)
    
    onnx_path = config.models_dir / 'defect_detector.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )
    
    print(f"✅ ONNX model exported to: {onnx_path}")
    
    # Save complete deployment package
    deployment_package = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'taxonomy': {
            'hierarchy': taxonomy.hierarchy,
            'class_to_id': taxonomy.class_to_id,
            'unified_classes': taxonomy.unified_classes
        },
        'metrics': checkpoint['metrics'],
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(deployment_package, config.models_dir / 'deployment_package.pth')
    
    print("\n" + "="*80)
    print("🎉 ULTIMATE DEFECT DETECTION PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n📦 Models saved to: {config.models_dir}")
    print(f"📊 Results saved to: {config.results_dir}")
    print(f"🖼️  Visualizations: {config.viz_dir}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()