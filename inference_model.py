import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
from scipy.ndimage import binary_opening, binary_closing
from einops import rearrange

# ═══════════════════════════════════════════════════════════════════════════
#                        INFERENCE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class InferenceConfig:
    """Minimal configuration for inference"""
    model_name: str = "dinov2_vitb14"
    image_size: int = 518
    num_classes: int = 5  # Excluding background
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Post-processing
    confidence_threshold: float = 0.35
    min_defect_area: int = 100
    
    # Class mapping (0 is background)
    id_to_class: Dict[int, str] = None
    
    def __post_init__(self):
        if self.id_to_class is None:
            self.id_to_class = {
                0: "scratch",
                1: "chipping", 
                2: "contamination",
                3: "rundown",
                4: "texture_defect"
            }

# ═══════════════════════════════════════════════════════════════════════════
#                        MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

class DINOv2Encoder(nn.Module):
    def __init__(self, model_name: str = 'dinov2_vitb14'):
        super().__init__()
        # Load DINOv2 model from torch hub (internet required for first run)
        try:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        except:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', model=model_name)
            
        self.spatial_size = 518 // 14  # 37
        self.skip_layers = [3, 7, 11]
        
        # Register hooks
        self.features = {}
        for layer_idx in self.skip_layers:
            self.dinov2.blocks[layer_idx].register_forward_hook(
                lambda module, input, output, name=f'layer_{layer_idx}': 
                self.features.update({name: output})
            )
            
    def forward(self, x):
        self.features = {}
        _ = self.dinov2(x)
        
        output_features = {}
        H, W = self.spatial_size, self.spatial_size
        
        for layer_idx in self.skip_layers:
            key = f'layer_{layer_idx}'
            feat = self.features[key][:, 1:, :] # Remove CLS token
            feat = rearrange(feat, 'b (h w) c -> b c h w', h=H, w=W)
            output_features[key] = feat
            
        return output_features

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)

class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MultiScaleUNetFPNDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 768 is DINOv2 embedding dim
        self.s1 = DecoderStage(768, 0, 256)
        self.s2 = DecoderStage(256, 768, 128)
        self.s3 = DecoderStage(128, 768, 64)
        self.s4 = DecoderStage(64, 0, 64)
        self.head = nn.Conv2d(64, num_classes + 1, 1) # +1 for background
        
    def forward(self, features):
        x = self.s1(features['layer_11'])
        x = self.s2(x, features['layer_7'])
        x = self.s3(x, features['layer_3'])
        x = self.s4(x)
        x = self.head(x)
        return F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)

class DINOv2DefectDetector(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.encoder = DINOv2Encoder()
        self.decoder = MultiScaleUNetFPNDecoder(num_classes)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ═══════════════════════════════════════════════════════════════════════════
#                        INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class DefectPredictor:
    def __init__(self, model_path: str, config: InferenceConfig = InferenceConfig()):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load Model
        print(f"Loading model from {model_path}...")
        self.model = DINOv2DefectDetector(num_classes=config.num_classes)
        
        # Load Weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint) # Support raw state dict
            
        self.model.to(self.device)
        self.model.eval()
        
        # Transforms
        self.transform = A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    def predict_image(self, image_path: str) -> Dict:
        """
        Runs inference on a single image file.
        Returns dictionary with detections and result visualization.
        """
        # Load & Preprocess
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        h, w = original_image.shape[:2]
        
        transformed = self.transform(image=original_image)['image']
        tensor = transformed.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            
            # Upsample to original size if needed (though we resize input to 518 usually)
            probs = F.interpolate(probs, size=(h, w), mode='bilinear', align_corners=False)
            preds = torch.argmax(probs, dim=1).cpu().numpy()[0] # [H, W]
            
        # Post-Processing
        detections = self._extract_detections(preds, probs.cpu().numpy()[0])
        vis_image = self._visualize(original_image, preds, detections)
        
        return {
            "detections": detections,
            "segmentation_mask": preds,
            "visualization": vis_image
        }

    def _extract_detections(self, prediction_mask, probability_map) -> List[Dict]:
        detections = []
        for cls_id in range(1, self.config.num_classes + 1):
            mask = (prediction_mask == cls_id).astype(np.uint8)
            
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = binary_opening(mask, structure=kernel).astype(np.uint8)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.config.min_defect_area:
                    continue
                
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT+1]
                confidence = probability_map[cls_id, labels == i].mean()
                
                # Only keep high confidence
                if confidence < self.config.confidence_threshold:
                    continue
                    
                detections.append({
                    "class_id": cls_id - 1, # 0-indexed for user
                    "class_name": self.config.id_to_class[cls_id - 1],
                    "bbox": [int(x), int(y), int(x+w), int(y+h)],
                    "area": int(area),
                    "confidence": float(confidence),
                    "centroid": centroids[i].tolist()
                })
        return detections

    def _visualize(self, image, mask, detections):
        vis = image.copy()
        
        # Colors for 5 classes
        colors = [
            (255, 0, 0),   # Scratch - Red
            (0, 255, 0),   # Chipping - Green
            (0, 0, 255),   # Contamination - Blue
            (255, 255, 0), # Rundown - Cyan
            (255, 0, 255)  # Texture - Magenta
        ]
        
        # Draw Masks
        overlay = vis.copy()
        for det in detections:
            cls_idx = det['class_id']
            color = colors[cls_idx % len(colors)]
            
            x1, y1, x2, y2 = det['bbox']
            # Draw bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det['class_name']} ({det['confidence']:.2f})"
            cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis

# Example Use:
# predictor = DefectPredictor("path/to/best_model.pth")
# result = predictor.predict_image("test_image.jpg")
# cv2.imwrite("output.jpg", cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR))
