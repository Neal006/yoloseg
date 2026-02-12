"""
==========================================================================
Door Defect Detection & Segmentation — Kaggle Notebook Script
==========================================================================
Made with Love by Neal Daftary

SETUP INSTRUCTIONS (see KAGGLE_SETUP.md for full details):
  1. Upload your 3 dataset folders (black, glossy, white) as a single
     Kaggle dataset named "door-defect-data".
  2. Create a new Kaggle Notebook, add the dataset, enable GPU.
  3. Paste this entire file into a single code cell and run.
==========================================================================
"""

# ── 0. Install Dependencies ──────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("ultralytics")
install("albumentations")

# ── 1. Imports ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import json
import shutil
import os
from ultralytics import YOLO
import albumentations as A
from sklearn.model_selection import train_test_split

# ── 2. Kaggle Path Configuration ─────────────────────────────────────────
# Adjust DATASET_NAME to match YOUR Kaggle dataset slug
DATASET_NAME = "door-defect-data"

INPUT_DIR   = Path(f"/kaggle/input/{DATASET_NAME}")
WORKING_DIR = Path("/kaggle/working")
COMBINED_DIR = WORKING_DIR / "data" / "combined"
RESULTS_DIR  = WORKING_DIR / "results"
MODELS_DIR   = WORKING_DIR / "models"

# Source dataset directories (inside the uploaded Kaggle dataset)
BLACK_DIR  = INPUT_DIR / "black"
WHITE_DIR  = INPUT_DIR / "white"
GLOSSY_DIR = INPUT_DIR / "glossy"

# Create working directories
for d in [COMBINED_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DOOR DEFECT DETECTION — KAGGLE NOTEBOOK")
print("=" * 70)
print(f"  Input dir  : {INPUT_DIR}")
print(f"  Working dir: {WORKING_DIR}")
print(f"  GPU        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU name   : {torch.cuda.get_device_name(0)}")
print()

# Verify dataset exists
for name, path in [("black", BLACK_DIR), ("white", WHITE_DIR), ("glossy", GLOSSY_DIR)]:
    exists = path.exists()
    print(f"  ✅ {name}: {path}" if exists else f"  ❌ {name}: {path} NOT FOUND")
    if not exists:
        # Try flat structure (dataset might be one level up)
        alt = INPUT_DIR / name
        if not alt.exists():
            raise FileNotFoundError(
                f"Dataset folder '{name}' not found. "
                f"Checked: {path} and {alt}\n"
                f"Contents of {INPUT_DIR}: {list(INPUT_DIR.iterdir()) if INPUT_DIR.exists() else 'DIR NOT FOUND'}"
            )
print()


# ==========================================================================
# 3. DATA PREPARATION — Merge 3 datasets into unified structure
# ==========================================================================

class DoorDefectDataPreparation:
    """Combines 3 separate door datasets into unified training structure"""

    def __init__(self,
                 black_dir: str,
                 white_dir: str,
                 glossy_dir: str,
                 output_dir: str = "data/combined"):

        self.black_dir = Path(black_dir)
        self.white_dir = Path(white_dir)
        self.glossy_dir = Path(glossy_dir)
        self.output_dir = Path(output_dir)

        self.source_datasets = {
            'black': self.black_dir,
            'white': self.white_dir,
            'glossy': self.glossy_dir
        }

        self.class_mapping, self.remap_tables = self._build_unified_class_mapping()

        self.defect_colors = {
            0: (0, 255, 0),
            1: (255, 0, 0),
            2: (0, 255, 255),
            3: (0, 128, 255),
            4: (255, 0, 255),
            5: (0, 0, 255)
        }

    def _build_unified_class_mapping(self):
        unified_names = []
        remap_tables = {}

        for ds_name, ds_dir in self.source_datasets.items():
            yaml_path = ds_dir / 'data.yaml'
            if not yaml_path.exists():
                print(f"   ⚠️ No data.yaml in {ds_dir}, skipping")
                continue

            with open(yaml_path, 'r') as f:
                ds_config = yaml.safe_load(f)

            local_names = ds_config.get('names', [])
            remap = {}

            for local_id, name in enumerate(local_names):
                found = False
                for uid, uname in enumerate(unified_names):
                    if uname.lower() == name.lower():
                        remap[local_id] = uid
                        found = True
                        break
                if not found:
                    new_uid = len(unified_names)
                    unified_names.append(name)
                    remap[local_id] = new_uid

            remap_tables[ds_name] = remap
            print(f"   📂 {ds_name}: {local_names} → remap {remap}")

        class_mapping = {name: idx for idx, name in enumerate(unified_names)}
        print(f"   🏷️  Unified classes ({len(class_mapping)}): {unified_names}")
        return class_mapping, remap_tables

    def _remap_label_file(self, src: Path, dst: Path, remap: dict):
        with open(src, 'r') as f:
            lines = f.readlines()
        out = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            cid = remap.get(int(parts[0]), int(parts[0]))
            out.append(f"{cid} {' '.join(parts[1:])}\n")
        with open(dst, 'w') as f:
            f.writelines(out)

    def merge_datasets(self, train_split=0.75, val_split=0.15):
        print("🔄 Merging datasets from 3 door types...")

        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        all_images, all_labels = [], []

        for door_type, door_dir in self.source_datasets.items():
            img_dir = door_dir / 'train' / 'images'
            lbl_dir = door_dir / 'train' / 'labels'

            if img_dir.exists():
                images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
                for img_path in images:
                    lbl_path = lbl_dir / f"{img_path.stem}.txt"
                    if lbl_path.exists():
                        all_images.append((img_path, door_type))
                        all_labels.append(lbl_path)
                print(f"   📂 {door_type}: {len(images)} images")
            else:
                print(f"   ⚠️ {img_dir} does not exist, skipping")

        print(f"   Total images: {len(all_images)}")

        indices = np.arange(len(all_images))
        door_types = [x[1] for x in all_images]

        train_idx, temp_idx = train_test_split(
            indices, train_size=train_split, stratify=door_types, random_state=42
        )
        val_size = val_split / (1 - train_split)
        temp_types = [door_types[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size, stratify=temp_types, random_state=42
        )

        for split_name, split_indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            print(f"   {split_name}: {len(split_indices)} images")
            for idx in split_indices:
                img_path, dtype = all_images[idx]
                lbl_path = all_labels[idx]

                pfx_img = f"{dtype}_{img_path.name}"
                pfx_lbl = f"{dtype}_{lbl_path.name}"

                shutil.copy(img_path, self.output_dir / split_name / 'images' / pfx_img)
                self._remap_label_file(
                    lbl_path,
                    self.output_dir / split_name / 'labels' / pfx_lbl,
                    self.remap_tables.get(dtype, {})
                )

        self._create_data_yaml()
        print("✅ Dataset merging complete!\n")

    def _create_data_yaml(self):
        sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
        class_names = [name for name, _ in sorted_classes]

        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"   Created {yaml_path}")
        print(f"   Classes: {class_names}")


# ==========================================================================
# 4. GRAYSCALE AUGMENTATION
# ==========================================================================

class GrayscaleAugmentation:
    @staticmethod
    def get_augmentation_pipeline(image_size: int = 640):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10,
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1,
                           num_shadows_upper=2, shadow_dimension=5, p=0.3),
            A.Resize(image_size, image_size, p=1.0),
        ])

    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return image


# ==========================================================================
# 5. UNIFIED MODEL WITH AREA CALCULATION + VISUALIZATION + JSON REPORT
# ==========================================================================

class UnifiedDefectDetectionModel(nn.Module):
    """YOLOv8-Seg with calibration for mm² area calculation"""

    def __init__(self,
                 yolo_weights: str,
                 calibration_config: Optional[Dict] = None,
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45):

        super().__init__()
        self.model = YOLO(yolo_weights)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        if calibration_config is None:
            calibration_config = {
                'pixels_per_mm': 2.0,
                'calibration_method': 'default',
                'camera_distance_mm': 500,
                'focal_length_px': 1000
            }
        self.calibration = calibration_config

        self.class_names = [
            'CHIPPING', 'Dust', 'RunDown', 'Scratch',
            'orange peel', 'Environmental Contamination'
        ]

        self.colors = {
            0: (0, 255, 0),
            1: (255, 0, 0),
            2: (0, 255, 255),
            3: (0, 128, 255),
            4: (255, 0, 255),
            5: (0, 0, 255)
        }

    def calculate_area_mm2(self, mask: np.ndarray) -> float:
        pixel_count = np.sum(mask > 0)
        return pixel_count / (self.calibration['pixels_per_mm'] ** 2)

    # ─── Forward ──────────────────────────────────────────────────────
    def forward(self, image: np.ndarray, return_visualization: bool = True,
                gt_label_path: Optional[str] = None):

        if len(image.shape) == 3:
            gray_image = GrayscaleAugmentation.convert_to_grayscale(image)
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = gray_image.shape[:2]
        total_image_pixels = h * w

        results = self.model.predict(
            gray_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]

        detections = []
        total_defect_area_mm2 = 0.0
        total_defect_pixels = 0

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.data.cpu().numpy()

            for idx, (mask, box) in enumerate(zip(masks, boxes)):
                class_id = int(box[5])
                confidence = float(box[4])

                mask_resized = cv2.resize(mask, (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                pixel_count = int(np.sum(mask_binary > 0))
                area_mm2 = self.calculate_area_mm2(mask_binary)
                area_pct = (pixel_count / total_image_pixels) * 100.0

                total_defect_area_mm2 += area_mm2
                total_defect_pixels += pixel_count

                detections.append({
                    'defect_id': idx,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                    'confidence': confidence,
                    'bbox': box[:4].tolist(),
                    'mask': mask_binary,
                    'area_mm2': area_mm2,
                    'area_pixels': pixel_count,
                    'area_percentage': round(area_pct, 4)
                })

        total_defect_pct = (total_defect_pixels / total_image_pixels) * 100.0
        heatmap = self.generate_heatmap(gray_image, detections)

        output = {
            'detections': detections,
            'num_defects': len(detections),
            'total_defect_area_mm2': total_defect_area_mm2,
            'total_defect_area_pixels': total_defect_pixels,
            'total_defect_percentage': round(total_defect_pct, 4),
            'total_image_area_pixels': total_image_pixels,
            'image_shape': list(gray_image.shape),
            'heatmap': heatmap,
            'calibration': self.calibration
        }

        if return_visualization:
            output['visualization'] = self.visualize_results(
                gray_image, detections, gt_label_path=gt_label_path, heatmap=heatmap
            )

        return output

    # ─── Heatmap ──────────────────────────────────────────────────────
    def generate_heatmap(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        h, w = image.shape[:2]
        density = np.zeros((h, w), dtype=np.float32)

        for det in detections:
            mask = det['mask']
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h))
            density += mask.astype(np.float32)

        if np.max(density) > 0:
            density = cv2.GaussianBlur(density, (31, 31), 0)
            density = (density / density.max() * 255).astype(np.uint8)
        else:
            density = density.astype(np.uint8)

        return cv2.applyColorMap(density, cv2.COLORMAP_JET)

    # ─── 4-Panel Visualization ────────────────────────────────────────
    def visualize_results(self, image: np.ndarray, detections: List[Dict],
                          gt_label_path: Optional[str] = None,
                          heatmap: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = image.shape[:2]
        canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)

        # Panel 1: Input
        canvas[:, :w] = image

        # Panel 2: Ground Truth
        gt_panel = image.copy()
        if gt_label_path and Path(gt_label_path).exists():
            gt_overlay = gt_panel.copy()
            with open(gt_label_path, 'r') as f:
                gt_lines = f.readlines()
            for line in gt_lines:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cid = int(parts[0])
                coords = list(map(float, parts[1:]))
                pts = np.array([[int(coords[i]*w), int(coords[i+1]*h)]
                                for i in range(0, len(coords), 2)], dtype=np.int32)
                color = self.colors.get(cid, (255, 255, 255))
                cv2.fillPoly(gt_overlay, [pts], color)
                cv2.polylines(gt_panel, [pts], True, color, 2)
                if len(pts) > 0:
                    cx, cy = pts.mean(axis=0).astype(int)
                    label = self.class_names[cid] if cid < len(self.class_names) else str(cid)
                    cv2.putText(gt_panel, label, (cx, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            gt_panel = cv2.addWeighted(gt_panel, 0.6, gt_overlay, 0.4, 0)
        else:
            cv2.putText(gt_panel, "No GT Available", (w//4, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        canvas[:, w:2*w] = gt_panel

        # Panel 3: Predictions
        pred_panel = image.copy()
        overlay = pred_panel.copy()
        for det in detections:
            mask = det['mask']
            cid = det['class_id']
            color = self.colors.get(cid, (255, 255, 255))
            if mask.shape[:2] != (h, w):
                mr = cv2.resize(mask.astype(np.float32), (w, h))
            else:
                mr = mask.astype(np.float32)
            overlay[mr > 0.5] = color
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(pred_panel, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']}: {det['confidence']:.2f} | {det['area_mm2']:.1f}mm2 ({det['area_percentage']:.2f}%)"
            cv2.putText(pred_panel, label, (x1, max(y1-5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        pred_panel = cv2.addWeighted(pred_panel, 0.6, overlay, 0.4, 0)
        canvas[:, 2*w:3*w] = pred_panel

        # Panel 4: Heatmap
        if heatmap is None:
            heatmap = self.generate_heatmap(image, detections)
        hp = heatmap if heatmap.shape[:2] == (h, w) else cv2.resize(heatmap, (w, h))
        canvas[:, 3*w:] = cv2.addWeighted(image, 0.3, hp, 0.7, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, title in enumerate(["Input Image", "Ground Truth", "Prediction", "Heatmap"]):
            cv2.putText(canvas, title, (i*w+10, 30), font, 1, (255, 255, 255), 2)

        return canvas

    # ─── JSON Report ──────────────────────────────────────────────────
    def generate_json_report(self, output: Dict, image_path: str,
                              output_dir: str = "results",
                              heatmap_path: Optional[str] = None) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        image_stem = Path(image_path).stem

        defect_list = []
        for det in output['detections']:
            defect_list.append({
                'defect_id': det['defect_id'],
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': round(det['confidence'], 4),
                'bbox': [round(v, 2) for v in det['bbox']],
                'area_mm2': round(det['area_mm2'], 4),
                'area_pixels': det['area_pixels'],
                'area_percentage': det['area_percentage']
            })

        report = {
            'image_path': str(image_path),
            'image_shape': output['image_shape'],
            'total_image_area_pixels': output['total_image_area_pixels'],
            'num_defects': output['num_defects'],
            'defects': defect_list,
            'summary': {
                'total_defect_area_mm2': round(output['total_defect_area_mm2'], 4),
                'total_defect_area_pixels': output['total_defect_area_pixels'],
                'total_defect_percentage': output['total_defect_percentage'],
            },
            'heatmap_path': heatmap_path or '',
            'calibration': output['calibration']
        }

        json_path = str(Path(output_dir) / f"{image_stem}_report.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"   📄 JSON report → {json_path}")
        return json_path


# ==========================================================================
# 6. TRAINING PIPELINE
# ==========================================================================

class DefectDetectionTrainer:
    """Training pipeline optimized for Kaggle GPU"""

    def __init__(self,
                 data_yaml: str,
                 model_size: str = 'n',
                 image_size: int = 640,
                 batch_size: int = 16,
                 epochs: int = 200,
                 device: str = 'auto'):

        self.data_yaml = data_yaml
        self.model_size = model_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        # Auto-detect device
        self.device = device if device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(f'yolov8{model_size}-seg.pt')

    def train(self, output_dir: str = '/kaggle/working/runs/segment'):
        print(f"🚀 Training on {self.device} for {self.epochs} epochs...")

        results = self.model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            device=self.device,

            patience=50,
            save=True,
            save_period=10,

            # Augmentation
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=10.0, translate=0.1, scale=0.5,
            shear=0.0, perspective=0.0,
            flipud=0.3, fliplr=0.5,
            mosaic=1.0, mixup=0.1,
            copy_paste=0.3,

            # Regularization
            dropout=0.1, weight_decay=0.0005,

            # Loss weights
            box=7.5, cls=0.5, dfl=1.5,

            # Learning rate
            lr0=0.001, lrf=0.01,
            momentum=0.937,
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,

            val=True, plots=True,

            project=output_dir,
            name='door_defect_detection',
            exist_ok=True,

            workers=2,        # Kaggle has limited CPUs
            cache='ram'
        )

        print("✅ Training complete!")
        return results

    def validate(self, weights: str):
        model = YOLO(weights)
        results = model.val(
            data=self.data_yaml,
            split='test',
            imgsz=self.image_size,
            batch=self.batch_size,
            device=self.device
        )
        print(f"\n📊 Validation Results:")
        print(f"   mAP50:    {results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"   mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
        return results


# ==========================================================================
# 7. DEPLOYMENT MODEL CREATOR
# ==========================================================================

class DeploymentModelCreator:
    @staticmethod
    def create_deployment_model(trained_weights: str,
                                 calibration_config: Dict,
                                 output_path: str = '/kaggle/working/models/door_defect_detector.pth',
                                 confidence_threshold: float = 0.25,
                                 iou_threshold: float = 0.45):
        print("📦 Creating deployment model...")
        yolo_model = YOLO(trained_weights)
        unified_model = UnifiedDefectDetectionModel(
            yolo_weights=trained_weights,
            calibration_config=calibration_config,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )

        deployment_package = {
            'model_state_dict': yolo_model.model.state_dict(),
            'model_architecture': 'YOLOv8-Seg',
            'trained_weights': trained_weights,
            'calibration': calibration_config,
            'class_names': unified_model.class_names,
            'colors': unified_model.colors,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'metadata': {
                'version': '1.0',
                'training_date': '2026-02-12',
                'input_format': 'grayscale_3channel',
                'output_format': 'detection_segmentation_area'
            }
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(deployment_package, output_path)

        size_mb = Path(output_path).stat().st_size / (1024**2)
        print(f"✅ Saved to {output_path} ({size_mb:.2f} MB)")
        return output_path


# ==========================================================================
# 8. CAMERA CALIBRATION
# ==========================================================================

class CameraCalibration:
    @staticmethod
    def calibrate_with_reference_object(reference_width_mm: float,
                                         reference_width_pixels: int) -> Dict:
        """
        Made with Love by Neal Daftary
        """
        pixels_per_mm = reference_width_pixels / reference_width_mm
        return {
            'pixels_per_mm': float(pixels_per_mm),
            'calibration_method': 'reference_object',
            'reference_width_mm': reference_width_mm,
            'reference_width_pixels': reference_width_pixels
        }


# ==========================================================================
# 9. INFERENCE ON TEST IMAGES (AFTER TRAINING)
# ==========================================================================

def run_inference_on_test_set(weights_path: str,
                               data_yaml_path: str,
                               calibration_config: Dict,
                               output_dir: str = "/kaggle/working/results",
                               max_images: int = 10):
    """
    Run inference on test images and save visualization + JSON reports.
    """
    print(f"\n{'='*60}")
    print("RUNNING INFERENCE ON TEST SET")
    print(f"{'='*60}")

    model = UnifiedDefectDetectionModel(
        yolo_weights=weights_path,
        calibration_config=calibration_config
    )

    # Read test image paths from data.yaml
    with open(data_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    test_img_dir = Path(cfg['path']) / cfg['test']
    test_images = sorted(list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png')))

    print(f"   Found {len(test_images)} test images")
    test_images = test_images[:max_images]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for img_path in test_images:
        print(f"\n   Processing: {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"   ⚠️ Could not load {img_path}")
            continue

        # Look for matching GT label
        lbl_dir = test_img_dir.parent / 'labels'
        gt_lbl = lbl_dir / f"{img_path.stem}.txt"
        gt_path = str(gt_lbl) if gt_lbl.exists() else None

        output = model.forward(image, return_visualization=True, gt_label_path=gt_path)

        # Save visualization
        vis_path = str(Path(output_dir) / f"{img_path.stem}_visualization.jpg")
        cv2.imwrite(vis_path, output['visualization'])

        # Save heatmap
        hm_path = str(Path(output_dir) / f"{img_path.stem}_heatmap.jpg")
        cv2.imwrite(hm_path, output['heatmap'])

        # Save JSON report
        model.generate_json_report(output, str(img_path), output_dir, hm_path)

        print(f"   → {output['num_defects']} defects | "
              f"{output['total_defect_area_mm2']:.2f} mm² | "
              f"{output['total_defect_percentage']:.4f}%")

    print(f"\n✅ Inference complete! Results saved to {output_dir}")


# ==========================================================================
# 10. MAIN PIPELINE — EXECUTE THIS ON KAGGLE
# ==========================================================================

def main():
    """
    Complete end-to-end pipeline for Kaggle.
    """

    # ── Step 1: Merge datasets ────────────────────────────────────────
    print("\n[STEP 1/5] Data Preparation")
    data_prep = DoorDefectDataPreparation(
        black_dir=str(BLACK_DIR),
        white_dir=str(WHITE_DIR),
        glossy_dir=str(GLOSSY_DIR),
        output_dir=str(COMBINED_DIR)
    )
    data_prep.merge_datasets(train_split=0.75, val_split=0.15)

    data_yaml_path = str(COMBINED_DIR / 'data.yaml')

    # ── Step 2: Calibration ───────────────────────────────────────────
    print("\n[STEP 2/5] Camera Calibration")
    calibration_config = CameraCalibration.calibrate_with_reference_object(
        reference_width_mm=100.0,
        reference_width_pixels=200
    )
    print(f"   pixels/mm: {calibration_config['pixels_per_mm']:.2f}")

    calib_path = str(WORKING_DIR / "calibration_config.json")
    with open(calib_path, 'w') as f:
        json.dump(calibration_config, f, indent=2)
    print(f"   Saved to {calib_path}")

    # ── Step 3: Train ─────────────────────────────────────────────────
    print("\n[STEP 3/5] Model Training")
    trainer = DefectDetectionTrainer(
        data_yaml=data_yaml_path,
        model_size='n',      # 'n' for speed, 's' or 'm' for accuracy
        image_size=640,
        batch_size=16,
        epochs=200,
        device='auto'
    )
    results = trainer.train()

    # ── Step 4: Validate ──────────────────────────────────────────────
    print("\n[STEP 4/5] Validation")
    best_weights = '/kaggle/working/runs/segment/door_defect_detection/weights/best.pt'
    if Path(best_weights).exists():
        trainer.validate(best_weights)
    else:
        print(f"   ⚠️ Best weights not found at {best_weights}")
        # Try to find them
        weight_files = list(Path('/kaggle/working/runs').rglob('best.pt'))
        if weight_files:
            best_weights = str(weight_files[0])
            print(f"   Found weights at: {best_weights}")
            trainer.validate(best_weights)

    # ── Step 5: Deploy + Inference ────────────────────────────────────
    print("\n[STEP 5/5] Deployment & Inference")
    deploy_path = DeploymentModelCreator.create_deployment_model(
        trained_weights=best_weights,
        calibration_config=calibration_config,
        output_path=str(MODELS_DIR / 'door_defect_detector.pth')
    )

    # Run inference on test images
    run_inference_on_test_set(
        weights_path=best_weights,
        data_yaml_path=data_yaml_path,
        calibration_config=calibration_config,
        output_dir=str(RESULTS_DIR),
        max_images=20
    )

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n📦 Deployment model : {deploy_path}")
    print(f"📊 Results          : {RESULTS_DIR}")
    print(f"📋 Training logs    : /kaggle/working/runs/")
    print("\nDownload models/ and results/ from Kaggle Output tab.")


"""
Made with Love by Neal Daftary
"""

# ── Run the pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()
else:
    # When pasted into a Kaggle cell, just call main()
    main()
