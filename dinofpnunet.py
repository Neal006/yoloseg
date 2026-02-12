import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PIXEL_TO_MM = 0.05

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["wandb", "ray"]:
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", pkg],
                    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
print("📦 Installing dependencies...")
for pkg in ["torch", "torchvision", "albumentations", "opencv-python-headless", 
            "scipy", "einops", "timm"]:
    try:
        install(pkg)
    except:
        pass

import torch
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
from scipy.ndimage import binary_opening, binary_closing
from einops import rearrange, repeat
import math

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

# ═══════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DINOv2Config:
    """Complete configuration following technical brief specifications"""
    
    # Dataset paths
    dataset_name: str = "data-bwg"
    input_dir: Path = Path(f"/kaggle/input/data-bwg")
    working_dir: Path = Path("/kaggle/working")
    
    # Model architecture (exactly as per technical brief)
    encoder_name: str = "dinov2_vitb14"  # DINOv2 ViT-B/14
    encoder_frozen: bool = False  # Configurable - can freeze encoder
    skip_layers: List[int] = field(default_factory=lambda: [3, 7, 11])  # Multi-scale features
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64])  # Decoder channel progression
    
    # Input specifications
    image_size: int = 518  # 518 = 14 × 37 (exact divisibility by patch size)
    patch_size: int = 14  # DINOv2 ViT-B/14 patch size
    embed_dim: int = 768  # DINOv2 ViT-B embedding dimension
    num_heads: int = 12  # Multi-head attention heads
    num_layers: int = 12  # Transformer blocks (0-11)
    
    # Data splits
    train_split: float = 0.60
    val_split: float = 0.20
    test_split: float = 0.20
    
    # Training hyperparameters (as per technical brief)
    batch_size: int = 4  # With frozen encoder
    num_epochs: int = 100
    learning_rate: float = 1e-5  # 0.00001 as specified
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_epochs: int = 10
    unfreeze_epoch: int = 20  # Epoch to unfreeze encoder
    
    # Advanced training
    use_amp: bool = True  # Automatic Mixed Precision
    accumulation_steps: int = 1  # Gradient accumulation
    ema_decay: float = 0.999  # Exponential Moving Average
    
    # Loss function weights (as per technical brief)
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: List[float] = field(default_factory=lambda: [0.1, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Inference
    tta_transforms: int = 8  # Test-time augmentation
    mc_dropout_samples: int = 10  # Monte Carlo dropout
    confidence_threshold: float = 0.25
    min_defect_area: int = 50  # Minimum defect size in pixels
    pixel_to_mm: float = PIXEL_TO_MM
    backup_frequency: int = 5  # Every N epochs, backup best model to Drive
    
    # Output backup (auto-detected: Colab Drive or Kaggle Output)
    drive_results_dir: Path = field(init=False)
    
    # Defect taxonomy (hierarchical mapping)
    defect_hierarchy: Dict[str, List[str]] = field(default_factory=lambda: {
        'scratch': ['Scratch', 'scratch', 'scratch_deep', 'horizontal_scratch'],
        'chipping': ['CHIPPING', 'chip', 'edge_damage'],
        'contamination': ['Dust', 'dust', 'Environmental Contamination', 'particles'],
        'rundown': ['RunDown', 'rundown', 'drip', 'sag'],
        'texture_defect': ['orange peel', 'orange_peel', 'peel'],
    })
    
    def __post_init__(self):
        # Auto-resolve input directory
        self.input_dir = self._resolve_input_dir(self.input_dir)
        print(f"✅ Found data root at: {self.input_dir}")

        # Auto-detect environment based on actual mount status
        if Path('/content/drive').exists():
            self.is_colab = True
            self.drive_results_dir = Path("/content/drive/MyDrive/dinov2_results")
        else:
            self.is_colab = False
            self.drive_results_dir = Path("/kaggle/working/deploy")
            
        print(f"✅ Environment: {'Colab (Drive Mounted)' if self.is_colab else 'Kaggle/Local (Notebook Output)'}")
        print(f"📂 Saving results to: {self.drive_results_dir}")

        # Create directories
        self.combined_dir = self.working_dir / "data" / "combined"
        self.results_dir = self.working_dir / "results"
        self.models_dir = self.working_dir / "models"
        self.viz_dir = self.working_dir / "visualizations"
        
        for d in [self.combined_dir, self.results_dir, self.models_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.drive_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Source directories
        self.black_dir = self.input_dir / "black"
        self.white_dir = self.input_dir / "white"
        self.glossy_dir = self.input_dir / "glossy"
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 37 × 37 = 1369
        self.spatial_size = self.image_size // self.patch_size  # 37
        
        print("═" * 80)
        print("🚀 DINOV2-BASED DEFECT DETECTION SYSTEM v3.0")
        print("═" * 80)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Image Size: {self.image_size}×{self.image_size}")
        print(f"Patches: {self.num_patches} ({self.spatial_size}×{self.spatial_size})")
        print(f"Encoder: {self.encoder_name} ({'frozen' if self.encoder_frozen else 'trainable'})")
        print()

    @staticmethod
    def _resolve_input_dir(base_path: Path) -> Path:
        """
        Auto-detects the correct data root by searching for 'black/train/images'
        or 'white/glossy' folders, handling potential Kaggle nesting.
        """
        if not base_path.exists():
            # Try /kaggle/input/dataset_name/data/
            # This handles cases where the zip was extracted into a subdir
            candidates = list(base_path.parent.glob("*/data")) + list(base_path.parent.glob("*"))
            for cand in candidates:
                if (cand / "black").exists() or (cand / "white").exists():
                    return cand
        
        # Recursively search for a 'black' directory
        found = list(base_path.rglob("black"))
        if found:
            # Return the parent of the first 'black' folder found
            return found[0].parent
            
        return base_path


# ═══════════════════════════════════════════════════════════════════════════
#                         DEFECT TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════

class DefectTaxonomy:
    """Hierarchical defect classification"""
    
    def __init__(self, hierarchy: Dict[str, List[str]]):
        self.hierarchy = hierarchy
        self.unified_classes = list(hierarchy.keys())
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.unified_classes)}
        self.id_to_class = {idx: cls for cls, idx in self.class_to_id.items()}
        
        # Build reverse mapping
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


# ═══════════════════════════════════════════════════════════════════════════
#                      DINOV2 ENCODER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

class DINOv2Encoder(nn.Module):
    """
    DINOv2 ViT-B/14 Encoder with Multi-Scale Feature Extraction
    
    Following technical brief specifications:
    - Extracts features from layers [3, 7, 11]
    - Reshapes 1D sequence to 2D feature maps
    - Supports frozen/trainable modes
    """
    
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.config = config
        
        # Load DINOv2 model from torch hub
        print(f"📥 Loading DINOv2 ViT-B/14...")
        try:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        except Exception as e:
            print(f"⚠️  Failed to load from torch hub: {e}")
            print("   Attempting alternative loading method...")
            # Fallback: try loading with different syntax
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', model='dinov2_vitb14')
        
        # Freeze encoder if specified
        if config.encoder_frozen:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            print("❄️  Encoder frozen")
        else:
            print("🔥 Encoder trainable")
        
        # Register hooks for intermediate features
        self.features = {}
        self.hooks = []
        
        for layer_idx in config.skip_layers:
            def make_hook(name):
                def hook(module, input, output):
                    self.features[name] = output
                return hook
            
            # DINOv2 uses blocks[i] for transformer layers
            handle = self.dinov2.blocks[layer_idx].register_forward_hook(
                make_hook(f'layer_{layer_idx}')
            )
            self.hooks.append(handle)
        
        print(f"✅ DINOv2 encoder loaded with skip layers: {config.skip_layers}")
    
    def forward(self, x):
        """
        Forward pass with multi-scale feature extraction
        
        Args:
            x: Input tensor [B, 3, 518, 518]
        
        Returns:
            features: Dict of feature maps at different scales
                - layer_3: [B, 768, 37, 37] (shallow features)
                - layer_7: [B, 768, 37, 37] (mid-level features)
                - layer_11: [B, 768, 37, 37] (deep features)
        """
        B = x.shape[0]
        H, W = self.config.spatial_size, self.config.spatial_size
        
        # Clear previous features
        self.features = {}
        
        # Forward through DINOv2
        # DINOv2 returns: [B, N+1, 768] where N = 1369 patches, +1 for CLS token
        _ = self.dinov2(x)
        
        # Process hooked features
        output_features = {}
        
        for layer_idx in self.config.skip_layers:
            key = f'layer_{layer_idx}'
            if key in self.features:
                feat = self.features[key]  # [B, 1370, 768]
                
                # Remove CLS token (first token)
                feat = feat[:, 1:, :]  # [B, 1369, 768]
                
                # Reshape to 2D spatial: [B, 1369, 768] -> [B, 768, 37, 37]
                feat = rearrange(feat, 'b (h w) c -> b c h w', h=H, w=W)
                
                output_features[key] = feat
        
        return output_features
    
    def remove_hooks(self):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()


# ═══════════════════════════════════════════════════════════════════════════
#                  MULTI-SCALE DECODER (UNet-FPN)
# ═══════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """
    Convolutional block with GroupNorm and GELU
    
    As per technical brief:
    - 3×3 convolution
    - GroupNorm (batch-independent)
    - GELU activation (smooth gradients)
    """
    
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.gelu1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        self.gelu2 = nn.GELU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.gelu1(x)
        
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.gelu2(x)
        
        return x


class DecoderStage(nn.Module):
    """
    Single decoder stage with skip connection
    
    Architecture:
    1. Upsample input
    2. Concatenate with skip connection (if provided)
    3. Convolutional block
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # Upsample 2x
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Convolutional block (processes upsampled + skip)
        total_channels = in_channels + skip_channels if skip_channels > 0 else in_channels
        self.conv_block = ConvBlock(total_channels, out_channels)
    
    def forward(self, x, skip=None):
        """
        Args:
            x: Input from previous stage
            skip: Skip connection from encoder
        """
        # Upsample
        x = self.upsample(x)
        
        # Concatenate skip connection
        if skip is not None:
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        # Process
        x = self.conv_block(x)
        
        return x


class MultiScaleUNetFPNDecoder(nn.Module):
    """
    Multi-Scale UNet-FPN Decoder
    
    Architecture (from technical brief):
    
    Stage 1: 768 → 256, upsample 2× (37×37 → 74×74)
    Stage 2: 256+768 → 128, upsample 2× (74×74 → 148×148) + skip from layer 7
    Stage 3: 128+768 → 64, upsample 2× (148×148 → 296×296) + skip from layer 3
    Stage 4: 64 → 64, upsample 2× (296×296 → 592×592) [refinement]
    Final: Interpolate to 518×518
    """
    
    def __init__(self, config: DINOv2Config, num_classes: int):
        super().__init__()
        self.config = config
        
        # Decoder channels: [256, 128, 64]
        dec_ch = config.decoder_channels
        embed_dim = config.embed_dim  # 768
        
        # Stage 1: Deep features (layer 11) -> 256
        self.stage1 = DecoderStage(
            in_channels=embed_dim,
            skip_channels=0,  # No skip
            out_channels=dec_ch[0]  # 256
        )
        
        # Stage 2: 256 + skip from layer 7 -> 128
        self.stage2 = DecoderStage(
            in_channels=dec_ch[0],  # 256
            skip_channels=embed_dim,  # 768 from layer 7
            out_channels=dec_ch[1]  # 128
        )
        
        # Stage 3: 128 + skip from layer 3 -> 64
        self.stage3 = DecoderStage(
            in_channels=dec_ch[1],  # 128
            skip_channels=embed_dim,  # 768 from layer 3
            out_channels=dec_ch[2]  # 64
        )
        
        # Stage 4: Refinement (64 -> 64)
        self.stage4 = DecoderStage(
            in_channels=dec_ch[2],  # 64
            skip_channels=0,  # No skip
            out_channels=dec_ch[2]  # 64
        )
        
        # Segmentation head (1×1 conv)
        self.seg_head = nn.Conv2d(dec_ch[2], num_classes + 1, 1)  # +1 for background
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: Dict with keys 'layer_3', 'layer_7', 'layer_11'
        
        Returns:
            Segmentation logits [B, num_classes+1, 518, 518]
        """
        # Extract features
        shallow = encoder_features['layer_3']  # [B, 768, 37, 37]
        mid = encoder_features['layer_7']      # [B, 768, 37, 37]
        deep = encoder_features['layer_11']    # [B, 768, 37, 37]
        
        # Stage 1: 37×37 -> 74×74
        x = self.stage1(deep, skip=None)  # [B, 256, 74, 74]
        
        # Stage 2: 74×74 -> 148×148 + skip from layer 7
        x = self.stage2(x, skip=mid)  # [B, 128, 148, 148]
        
        # Stage 3: 148×148 -> 296×296 + skip from layer 3
        x = self.stage3(x, skip=shallow)  # [B, 64, 296, 296]
        
        # Stage 4: 296×296 -> 592×592 (refinement)
        x = self.stage4(x, skip=None)  # [B, 64, 592, 592]
        
        # Segmentation head
        x = self.seg_head(x)  # [B, num_classes+1, 592, 592]
        
        # Interpolate to target size (518×518)
        x = F.interpolate(x, size=(self.config.image_size, self.config.image_size),
                         mode='bilinear', align_corners=False)
        
        return x


# ═══════════════════════════════════════════════════════════════════════════
#                    COMPLETE DINOV2 SEGMENTATION MODEL
# ═══════════════════════════════════════════════════════════════════════════

class DINOv2DefectDetector(nn.Module):
    """
    Complete DINOv2-based defect detection model
    
    Components:
    1. DINOv2 ViT-B/14 encoder (frozen/trainable)
    2. Multi-Scale UNet-FPN decoder
    3. Multi-scale skip connections from layers [3, 7, 11]
    """
    
    def __init__(self, config: DINOv2Config, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = DINOv2Encoder(config)
        
        # Decoder
        self.decoder = MultiScaleUNetFPNDecoder(config, num_classes)
        
        # Initialize decoder weights
        self._init_decoder_weights()
        
        print(f"✅ Model initialized: {num_classes} defect classes")
    
    def _init_decoder_weights(self):
        """Initialize decoder with He initialization"""
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, 518, 518]
        
        Returns:
            Segmentation logits [B, num_classes+1, 518, 518]
        """
        # Encoder: Extract multi-scale features
        encoder_features = self.encoder(x)
        
        # Decoder: Progressive upsampling with skip connections
        seg_logits = self.decoder(encoder_features)
        
        return seg_logits
    
    def enable_dropout(self):
        """Enable dropout for MC Dropout"""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
    
    def count_parameters(self):
        """Count trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder_params,
            'decoder': decoder_params
        }


# ═══════════════════════════════════════════════════════════════════════════
#                        LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    
    Formula (from technical brief):
    Dice_c = 2|P_c ∩ G_c| / (|P_c| + |G_c|)
    """
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets, num_classes):
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        # Softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice per class
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        cardinality = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Return 1 - mean dice
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Formula (from technical brief):
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        # Cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        
        # Get probabilities
        p_t = torch.exp(-ce_loss)
        
        # Focal term
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss
    
    As per technical brief:
    L_total = λ_dice * L_dice + λ_focal * L_focal
    """
    
    def __init__(self, config: DINOv2Config, num_classes: int):
        super().__init__()
        
        # Prepare class weights
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32)
        
        self.dice = DiceLoss()
        self.focal = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            class_weights=class_weights.to(config.device)
        )
        
        self.dice_weight = config.dice_weight
        self.focal_weight = config.focal_weight
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] logits
            targets: [B, H, W] class indices
        """
        loss_dice = self.dice(inputs, targets, self.num_classes + 1)
        loss_focal = self.focal(inputs, targets)
        
        total_loss = self.dice_weight * loss_dice + self.focal_weight * loss_focal
        
        return total_loss, {
            'dice': loss_dice.item(),
            'focal': loss_focal.item(),
            'total': total_loss.item()
        }


# ═══════════════════════════════════════════════════════════════════════════
#                        DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

class DINOv2Augmentation:
    """Augmentation pipeline following technical brief"""
    
    @staticmethod
    def get_train_transforms(image_size: int = 518):
        """Training augmentation with ImageNet normalization"""
        return A.Compose([
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
            
            # Photometric
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
            
            # Noise and blur
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10, 30), p=0.3),
            
            # Spatial transforms
            A.ElasticTransform(alpha=50, sigma=5, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            
            # Resize to exact size
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            
            # ImageNet normalization (critical for DINOv2)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})
    
    @staticmethod
    def get_val_transforms(image_size: int = 518):
        """Validation transforms (no augmentation)"""
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
    def get_tta_transforms(image_size: int = 518):
        """Test-time augmentation (8 variants)"""
        transforms = []
        
        # Original
        transforms.append(DINOv2Augmentation.get_val_transforms(image_size))
        
        # Horizontal flip
        transforms.append(A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))
        
        # Vertical flip
        transforms.append(A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))
        
        # Rotations
        for angle in [90, 180, 270]:
            transforms.append(A.Compose([
                A.Rotate(limit=(angle, angle), p=1.0, border_mode=cv2.BORDER_REFLECT_101),
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]))
        
        # Scale variants
        for scale in [0.9, 1.1]:
            transforms.append(A.Compose([
                A.ShiftScaleRotate(shift_limit=0, scale_limit=(scale-1, scale-1),
                                  rotate_limit=0, p=1.0),
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]))
        
        return transforms[:8]  # Return exactly 8


# ═══════════════════════════════════════════════════════════════════════════
#                         DATASET
# ═══════════════════════════════════════════════════════════════════════════

class DefectDataset(Dataset):
    """PyTorch dataset for defect detection"""
    
    def __init__(self, image_dir: Path, mask_dir: Path, transform=None):
        self.image_paths = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
        self.mask_dir = mask_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        mask = mask.long()
        
        return {'image': image, 'mask': mask, 'image_path': str(img_path)}


# ═══════════════════════════════════════════════════════════════════════════
#                   DATASET PREPARATION (SAME AS BEFORE)
# ═══════════════════════════════════════════════════════════════════════════

class DefectDatasetPreparation:
    """Dataset merging with semantic class unification"""
    
    def __init__(self, config: DINOv2Config, taxonomy: DefectTaxonomy):
        self.config = config
        self.taxonomy = taxonomy
        self.source_datasets = {
            'black': config.black_dir,
            'white': config.white_dir,
            'glossy': config.glossy_dir
        }
    
    def merge_datasets(self):
        print("\n" + "="*80)
        print("📦 DATASET PREPARATION")
        print("="*80)
        
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
        print("\n📈 Class Distribution:")
        for cls, count in sorted(class_distribution.items()):
            print(f"  {cls:20s}: {count:4d} samples")
        
        # Stratified split
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
            
            temp_stratify = [stratify_labels[i] for i in temp_idx]
            val_size = self.config.val_split / (self.config.val_split + self.config.test_split)
            
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size,
                stratify=temp_stratify,
                random_state=42
            )
        except ValueError as e:
            print(f"⚠️  Stratified split failed (rare classes): {e}")
            print("   Falling back to non-stratified split...")
            
            train_idx, temp_idx = train_test_split(
                indices,
                train_size=self.config.train_split,
                random_state=42
            )
            
            val_size = self.config.val_split / (self.config.val_split + self.config.test_split)
            
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size,
                random_state=42
            )
        
        splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        
        for split_name, split_indices in splits.items():
            print(f"\n📁 Processing {split_name}: {len(split_indices)} samples")
            
            for idx in split_indices:
                sample = all_samples[idx]
                img_path = sample['image_path']
                lbl_path = sample['label_path']
                
                new_name = f"{sample['door_type']}_{img_path.name}"
                
                dst_img = self.config.combined_dir / split_name / 'images' / new_name
                shutil.copy(str(img_path), str(dst_img))
                
                mask = self._label_to_mask(str(lbl_path), img_path, sample['local_class_names'])
                
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


# ═══════════════════════════════════════════════════════════════════════════
#                           TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average"""
    
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
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class DINOv2Trainer:
    """Training engine for DINOv2 defect detector"""
    
    def __init__(self, config: DINOv2Config, model: nn.Module, num_classes: int):
        self.config = config
        self.model = model.to(config.device)
        self.num_classes = num_classes
        
        self.criterion = CombinedLoss(config, num_classes)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Warmup + Cosine scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.num_epochs // 3,
            T_mult=2,
            eta_min=1e-7
        )
        
        self.scaler = GradScaler() if config.use_amp else None
        self.ema = EMA(model, decay=config.ema_decay)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.config.device)
            masks = batch['mask'].to(self.config.device)
            
            if self.config.use_amp:
                with autocast():
                    seg_logits = self.model(images)
                    loss, losses = self.criterion(seg_logits, masks)
            else:
                seg_logits = self.model(images)
                loss, losses = self.criterion(seg_logits, masks)
            
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            self.ema.update()
            
            for k, v in losses.items():
                epoch_losses[k] += v
            
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx:3d}/{num_batches}] Loss: {loss.item():.4f}")
        
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, val_loader, use_ema=True):
        if use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        epoch_losses = defaultdict(float)
        num_batches = len(val_loader)
        
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            images = batch['image'].to(self.config.device)
            masks = batch['mask'].to(self.config.device)
            
            seg_logits = self.model(images)
            loss, losses = self.criterion(seg_logits, masks)
            
            for k, v in losses.items():
                epoch_losses[k] += v
            
            preds = torch.argmax(seg_logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
        
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = self.calculate_metrics(all_preds, all_targets)
        epoch_losses.update(metrics)
        
        if use_ema:
            self.ema.restore()
        
        return epoch_losses
    
    def calculate_metrics(self, preds, targets):
        num_classes = self.num_classes + 1
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
            metrics[f'f1_class_{cls}'] = f1
        
        metrics['mean_iou'] = np.mean([metrics[f'iou_class_{i}'] for i in range(num_classes)])
        metrics['mean_f1'] = np.mean([metrics[f'f1_class_{i}'] for i in range(num_classes)])
        
        return metrics
    
    def train(self, train_loader, val_loader):
        print("\n" + "="*80)
        print("🚀 TRAINING DINOV2 DEFECT DETECTOR")
        print("="*80)
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📅 Epoch [{epoch+1}/{self.config.num_epochs}]")
            
            if (epoch + 1) == self.config.unfreeze_epoch:
                print(f"\n❄️➡️🔥 Unfreezing Encoder at Epoch {epoch+1}!")
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                
                # Re-initialize optimizer to include encoder parameters
                # Use a smaller learning rate for the encoder typically, 
                # but for simplicity here we restart with base LR or a fraction
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate * 0.1, # Lower LR for fine-tuning
                    weight_decay=self.config.weight_decay,
                    betas=(0.9, 0.999)
                )
                print("   Optimizer updated with learning rate scaled by 0.1")
            
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"  Train Loss: {train_losses['total']:.4f} | " +
                  f"Dice: {train_losses['dice']:.4f} | " +
                  f"Focal: {train_losses['focal']:.4f}")
            
            val_losses = self.validate(val_loader)
            print(f"  Val Loss:   {val_losses['total']:.4f} | " +
                  f"mIoU: {val_losses['mean_iou']:.4f} | " +
                  f"mF1: {val_losses['mean_f1']:.4f}")
            
            self.scheduler.step()
            
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, val_losses, is_best=True)
                print("  ✅ Best model saved!")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_losses)
            
            # Periodic backup of best model to Drive/Output
            if (epoch + 1) % self.config.backup_frequency == 0:
                best_path = self.config.models_dir / 'best_dinov2_model.pth'
                if best_path.exists():
                    print(f"  ☁️  Syncing best model to persistent storage (Epoch {epoch+1})...")
                    try:
                        shutil.copy(str(best_path), str(self.config.drive_results_dir / 'best_dinov2_model.pth'))
                    except Exception as e:
                        print(f"  ⚠️  Backup failed: {e}")

            self.train_losses.append(train_losses)
            self.val_losses.append(val_losses)
        
        print("\n✅ Training complete!")
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        
        if is_best:
            path = self.config.models_dir / 'best_dinov2_model.pth'
        else:
            path = self.config.models_dir / f'dinov2_checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)


# ═══════════════════════════════════════════════════════════════════════════
#                      INFERENCE WITH TTA
# ═══════════════════════════════════════════════════════════════════════════

class DINOv2Inference:
    
    def __init__(self, model: nn.Module, config: DINOv2Config, taxonomy: DefectTaxonomy):
        self.model = model
        self.config = config
        self.taxonomy = taxonomy
        self.device = config.device
        self.px_to_mm = config.pixel_to_mm
        
        self.model.to(self.device)
        self.model.eval()
        
        self.tta_transforms = DINOv2Augmentation.get_tta_transforms(config.image_size)
    
    @torch.no_grad()
    def predict_with_tta(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        all_predictions = []
        
        for transform in self.tta_transforms:
            augmented = transform(image=image)['image']
            augmented = augmented.unsqueeze(0).to(self.device)
            
            logits = self.model(augmented)
            probs = F.softmax(logits, dim=1)
            probs = F.interpolate(probs, size=(h, w), mode='bilinear', align_corners=False)
            all_predictions.append(probs.cpu().numpy()[0])
        
        avg_probs = np.mean(all_predictions, axis=0)
        uncertainty = np.std(all_predictions, axis=0).mean(axis=0)
        predictions = np.argmax(avg_probs, axis=0)
        
        return predictions, uncertainty
    
    def post_process(self, predictions: np.ndarray) -> np.ndarray:
        processed = predictions.copy()
        
        for cls in range(1, self.taxonomy.get_num_classes() + 1):
            mask = (predictions == cls).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = binary_opening(mask, structure=kernel).astype(np.uint8)
            mask = binary_closing(mask, structure=kernel).astype(np.uint8)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < self.config.min_defect_area:
                    mask[labels == i] = 0
            
            processed[mask > 0] = cls
        
        return processed
    
    def _extract(self, predictions, uncertainty):
        mm2_per_px = self.px_to_mm ** 2
        dets = []
        for cls in range(1, self.taxonomy.get_num_classes() + 1):
            m = (predictions == cls).astype(np.uint8)
            nl, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, nl):
                area_px = int(stats[i, cv2.CC_STAT_AREA])
                x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT + 1]
                dm = (labels == i).astype(np.uint8)
                au = float(uncertainty[dm > 0].mean()) if dm.sum() > 0 else 0.0
                contours, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    dets.append({
                        'class_id': cls - 1,
                        'class_name': self.taxonomy.id_to_class[cls - 1],
                        'bbox': [int(x), int(y), int(x + bw), int(y + bh)],
                        'area_pixels': area_px,
                        'area_mm2': round(area_px * mm2_per_px, 4),
                        'num_pixels': area_px,
                        'centroid': [round(c, 2) for c in cents[i].tolist()],
                        'uncertainty': round(au, 4),
                        'contour': contours[0],
                    })
        return dets

    def _class_analysis(self, detections, img_h, img_w):
        total_img_px = img_h * img_w
        mm2_per_px   = self.px_to_mm ** 2
        total_defect_px = sum(d['area_pixels'] for d in detections)

        analysis = {}
        for cls_id in range(self.taxonomy.get_num_classes()):
            cls_name = self.taxonomy.id_to_class[cls_id]
            cls_dets = [d for d in detections if d['class_id'] == cls_id]

            if not cls_dets:
                analysis[cls_name] = {
                    'count': 0, 'total_area_pixels': 0, 'total_area_mm2': 0.0,
                    'area_percentage_of_image': 0.0,
                    'area_percentage_of_all_defects': 0.0, 'instances': [],
                }
                continue

            cls_area_px = sum(d['area_pixels'] for d in cls_dets)
            analysis[cls_name] = {
                'count': len(cls_dets),
                'total_area_pixels': cls_area_px,
                'total_area_mm2': round(cls_area_px * mm2_per_px, 4),
                'area_percentage_of_image': round(cls_area_px / total_img_px * 100, 4),
                'area_percentage_of_all_defects': round(
                    cls_area_px / total_defect_px * 100, 4) if total_defect_px > 0 else 0.0,
                'instances': [
                    {
                        'instance_id': idx + 1,
                        'bbox': d['bbox'],
                        'area_pixels': d['area_pixels'],
                        'area_mm2': d['area_mm2'],
                        'centroid': d['centroid'],
                        'uncertainty': d['uncertainty'],
                    }
                    for idx, d in enumerate(cls_dets)
                ],
            }
        return analysis

    def _summary(self, detections, img_h, img_w):
        total_img_px   = img_h * img_w
        mm2_per_px     = self.px_to_mm ** 2
        total_def_px   = sum(d['area_pixels'] for d in detections)
        return {
            'total_defects': len(detections),
            'total_defect_area_pixels': total_def_px,
            'total_defect_area_mm2': round(total_def_px * mm2_per_px, 4),
            'defect_area_percentage': round(total_def_px / total_img_px * 100, 4) if total_img_px > 0 else 0.0,
        }

    def _generate_heatmap(self, uncertainty):
        unc_norm = np.clip(uncertainty / (uncertainty.max() + 1e-8), 0, 1)
        unc_u8   = (unc_norm * 255).astype(np.uint8)
        heatmap  = cv2.applyColorMap(unc_u8, cv2.COLORMAP_JET)
        return heatmap

    def predict(self, image: np.ndarray, use_tta: bool = True) -> Dict:
        h, w = image.shape[:2]
        if use_tta:
            predictions, uncertainty = self.predict_with_tta(image)
        else:
            transform = DINOv2Augmentation.get_val_transforms(self.config.image_size)
            image_tensor = transform(image=image)['image'].unsqueeze(0).to(self.device)
            
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            probs = F.interpolate(probs, size=(h, w), mode='bilinear', align_corners=False)
            
            predictions = torch.argmax(probs, dim=1).cpu().numpy()[0]
            uncertainty = np.zeros_like(predictions, dtype=np.float32)
        
        predictions    = self.post_process(predictions)
        detections     = self._extract(predictions, uncertainty)
        class_analysis = self._class_analysis(detections, h, w)
        summary        = self._summary(detections, h, w)
        heatmap        = self._generate_heatmap(uncertainty)
        viz            = self._visualize(image, predictions, uncertainty, detections)

        return {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'detections': detections,
            'class_analysis': class_analysis,
            'summary': summary,
            'heatmap': heatmap,
            'visualization': viz,
            'image_height': h,
            'image_width': w,
        }
    
    def _visualize(self, image: np.ndarray, predictions: np.ndarray,
                  uncertainty: np.ndarray, detections: List[Dict]) -> np.ndarray:
        h, w = image.shape[:2]
        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        
        canvas[:h, :w] = image_rgb
        
        overlay = image_rgb.copy()
        colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for cls in range(1, min(len(colors), self.taxonomy.get_num_classes() + 1)):
            mask = (predictions == cls)
            overlay[mask] = colors[cls]
        
        canvas[:h, w:2*w] = cv2.addWeighted(image_rgb, 0.5, overlay, 0.5, 0)
        
        detection_vis = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = colors[det['class_id'] + 1]
            cv2.rectangle(detection_vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} ({det['area_mm2']:.2f}mm2)"
            cv2.putText(detection_vis, label, (x1, max(y1 - 5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        canvas[h:2*h, :w] = detection_vis
        
        uncertainty_norm = (uncertainty * 255).astype(np.uint8)
        uncertainty_color = cv2.applyColorMap(uncertainty_norm, cv2.COLORMAP_JET)
        canvas[h:2*h, w:2*w] = uncertainty_color
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Segmentation", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Detections", (10, h + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(canvas, "Uncertainty", (w + 10, h + 30), font, 1, (255, 255, 255), 2)
        
        return canvas


# ═══════════════════════════════════════════════════════════════════════════
#                              MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Execute DINOv2 defect detection pipeline"""
    
    # Try mounting Drive if on Colab (with robust error handling)
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
    except Exception as e:
        print(f"⚠️ Drive mount skipped: {e}")
        print("   Continuing with local/Kaggle storage.")
    
    config = DINOv2Config()
    taxonomy = DefectTaxonomy(config.defect_hierarchy)
    taxonomy.print_taxonomy()
    
    # Dataset preparation
    print("\n" + "="*80)
    print("STEP 1: DATASET PREPARATION")
    print("="*80)
    
    data_prep = DefectDatasetPreparation(config, taxonomy)
    data_yaml_path = data_prep.merge_datasets()
    
    # Create dataloaders
    print("\n" + "="*80)
    print("STEP 2: CREATING DATALOADERS")
    print("="*80)
    
    train_dataset = DefectDataset(
        config.combined_dir / 'train' / 'images',
        config.combined_dir / 'train' / 'masks',
        transform=DINOv2Augmentation.get_train_transforms(config.image_size)
    )
    
    val_dataset = DefectDataset(
        config.combined_dir / 'val' / 'images',
        config.combined_dir / 'val' / 'masks',
        transform=DINOv2Augmentation.get_val_transforms(config.image_size)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Build model
    print("\n" + "="*80)
    print("STEP 3: BUILDING DINOV2 MODEL")
    print("="*80)
    
    model = DINOv2DefectDetector(config, taxonomy.get_num_classes())
    param_counts = model.count_parameters()
    
    print(f"✅ Model built:")
    print(f"   Total params: {param_counts['total']:,}")
    print(f"   Trainable: {param_counts['trainable']:,}")
    print(f"   Encoder: {param_counts['encoder']:,}")
    print(f"   Decoder: {param_counts['decoder']:,}")
    
    # Train
    print("\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80)
    
    trainer = DINOv2Trainer(config, model, taxonomy.get_num_classes())
    trainer.train(train_loader, val_loader)
    
    # Inference
    print("\n" + "="*80)
    print("STEP 5: INFERENCE")
    print("="*80)
    
    checkpoint = torch.load(config.models_dir / 'best_dinov2_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    inference = DINOv2Inference(model, config, taxonomy)
    
    test_img_dir = config.combined_dir / 'test' / 'images'
    test_images = sorted(list(test_img_dir.glob('*.png')) + list(test_img_dir.glob('*.jpg')))[:20]
    
    for img_path in test_images:
        print(f"\n  Processing: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = inference.predict(image, use_tta=True)
        
        cv2.imwrite(str(config.viz_dir / f"{img_path.stem}_dinov2.jpg"),
                    cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR))
        
        cv2.imwrite(str(config.viz_dir / f"{img_path.stem}_heatmap.jpg"),
                    results['heatmap'])
        
        h = results['image_height']
        w = results['image_width']
        total_px   = h * w
        mm2_per_px = config.pixel_to_mm ** 2
        
        report = {
            'image_path': str(img_path),
            'image_dimensions': {'width': w, 'height': h},
            'total_image_area_pixels': total_px,
            'total_image_area_mm2': round(total_px * mm2_per_px, 4),
            'pixel_to_mm': config.pixel_to_mm,
            'summary': results['summary'],
            'class_analysis': results['class_analysis'],
        }
        
        json_path = config.results_dir / f"{img_path.stem}_dinov2_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        s = results['summary']
        print(f"    {s['total_defects']} defects | "
              f"{s['total_defect_area_pixels']} px | "
              f"{s['total_defect_area_mm2']} mm² | "
              f"{s['defect_area_percentage']:.2f}% of image")
    
    # Deployment bundle (TorchScript — hides architecture)
    print("\n" + "="*80)
    print("STEP 6: DEPLOYMENT BUNDLE")
    print("="*80)
    
    model.eval()
    dummy = torch.randn(1, 3, config.image_size, config.image_size).to(config.device)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    meta = json.dumps({
        'class_names': taxonomy.unified_classes,
        'num_classes': taxonomy.get_num_classes(),
        'image_size': config.image_size,
        'pixel_to_mm': config.pixel_to_mm,
        'min_defect_area': config.min_defect_area,
    })

    deploy_path = config.models_dir / 'model.pt'
    torch.jit.save(traced, str(deploy_path), _extra_files={'config.json': meta})
    print(f"✅ Deployment model saved: {deploy_path}")
    
    # Copy key outputs to /kaggle/working/deploy/ for easy download
    print("\n📦 Copying outputs to Kaggle output directory...")
    shutil.copy(str(deploy_path), str(config.drive_results_dir / 'model.pt'))
    
    # Copy best checkpoint too
    best_ckpt = config.models_dir / 'best_dinov2_model.pth'
    if best_ckpt.exists():
        shutil.copy(str(best_ckpt), str(config.drive_results_dir / 'best_dinov2_model.pth'))
    
    # Copy all JSON reports
    for jf in config.results_dir.glob('*.json'):
        shutil.copy(str(jf), str(config.drive_results_dir / jf.name))
    
    # Copy visualizations
    viz_deploy = config.drive_results_dir / 'visualizations'
    viz_deploy.mkdir(parents=True, exist_ok=True)
    for vf in config.viz_dir.glob('*.jpg'):
        shutil.copy(str(vf), str(viz_deploy / vf.name))
    
    print(f"✅ All outputs copied to: {config.drive_results_dir}")
    print(f"   Download from Kaggle Output after notebook finishes.\n")
    
    print("=" * 80)
    print("🏁 ALL DONE!")
    print(f"   📦 Deploy  : {config.drive_results_dir / 'model.pt'}")
    print(f"   📊 Results : {config.results_dir}")
    print(f"   🖼️  Viz    : {config.viz_dir}")
    print(f"   💾 Output  : {config.drive_results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
