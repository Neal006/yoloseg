"""
SIMPLIFIED INFERENCE SCRIPT FOR APP DEVELOPERS
===============================================
Use this script with the single .pth deployment file

Usage:
    python simple_inference.py --model door_defect_detector.pth --image door_sample.jpg
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO


class SimpleDoorDefectDetector:
    """
    Ultra-simple interface for app developers
    Only needs the .pth file
    """
    
    def __init__(self, model_path: str):
        """
        Initialize detector with single .pth file
        
        Args:
            model_path: Path to door_defect_detector.pth
        """
        print(f"Loading model from {model_path}...")
        
        self.package = torch.load(model_path, map_location='cpu')
        self.class_names = self.package['class_names']
        self.colors = self.package['colors']
        self.calibration = self.package['calibration']
        self.conf_threshold = self.package['confidence_threshold']
        self.iou_threshold = self.package['iou_threshold']
        print("✅ Model loaded successfully!")
        print(f"   Detectable defects: {', '.join(self.class_names)}")
        print(f"   Pixel-to-mm ratio: {self.calibration['pixels_per_mm']:.2f} px/mm")
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return image
    
    def calculate_area_mm2(self, mask: np.ndarray) -> float:
        pixel_count = np.sum(mask > 0)
        pixels_per_mm = self.calibration['pixels_per_mm']
        return pixel_count / (pixels_per_mm ** 2)
    
    def detect(self, image_path: str, save_results: bool = True):
        """
        Detect defects in image
        
        Args:
            image_path: Path to door image
            save_results: Whether to save visualization
        
        Returns:
            Dictionary with:
                - detections: List of detected defects
                - total_area_mm2: Total defect area
                - num_defects: Number of defects found
                - visualization: Annotated image (if save_results=True)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray_image = self.convert_to_grayscale(image)
        
        detections = []
        total_area = 0.0
        
        # Example detection structure:
        # detection = {
        #     'defect_id': 0,
        #     'class_name': 'scratch',
        #     'confidence': 0.95,
        #     'bbox': [x1, y1, x2, y2],
        #     'area_mm2': 15.3,
        #     'mask': mask_array
        # }
        
        results = {
            'image_path': image_path,
            'detections': detections,
            'total_area_mm2': total_area,
            'num_defects': len(detections)
        }
        
        if save_results:
            vis_image = self.visualize(gray_image, detections)
            output_path = Path(image_path).stem + '_result.jpg'
            cv2.imwrite(output_path, vis_image)
            results['visualization_path'] = output_path
            print(f"✅ Visualization saved to: {output_path}")
        
        return results
    
    def visualize(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Create visualization with 4 panels like your sample"""
        h, w = image.shape[:2]
        canvas = np.zeros((h, w * 4, 3), dtype=np.uint8)
        
        # Panel 1: Input
        canvas[:, :w] = image
        
        # Panel 2: Ground Truth (placeholder)
        canvas[:, w:2*w] = image
        
        # Panel 3: Prediction
        pred_panel = image.copy()
        overlay = pred_panel.copy()
        
        for det in detections:
            mask = det['mask']
            color = self.colors[det['class_id']]
            
            # Apply mask
            mask_resized = cv2.resize(mask, (w, h))
            mask_bool = mask_resized > 0.5
            overlay[mask_bool] = color
            
            # Draw bbox and label
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(pred_panel, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']}: {det['confidence']:.2f} | {det['area_mm2']:.1f}mm²"
            cv2.putText(pred_panel, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        pred_panel = cv2.addWeighted(pred_panel, 0.6, overlay, 0.4, 0)
        canvas[:, 2*w:3*w] = pred_panel
        
        # Panel 4: Errors
        canvas[:, 3*w:] = np.zeros_like(image)
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        titles = ["Input Image", "Ground Truth", "Prediction", "Errors"]
        for i, title in enumerate(titles):
            cv2.putText(canvas, title, (i*w + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        return canvas
    
    def batch_detect(self, image_folder: str, output_folder: str = "results"):
        """
        Process all images in folder
        
        Args:
            image_folder: Folder containing door images
            output_folder: Folder to save results
        """
        Path(output_folder).mkdir(exist_ok=True)
        
        image_paths = list(Path(image_folder).glob("*.jpg")) + \
                     list(Path(image_folder).glob("*.png"))
        
        print(f"\n🔄 Processing {len(image_paths)} images...")
        
        all_results = []
        for img_path in image_paths:
            print(f"   Processing: {img_path.name}")
            result = self.detect(str(img_path), save_results=True)
            all_results.append(result)
        
        # Summary statistics
        total_defects = sum(r['num_defects'] for r in all_results)
        total_area = sum(r['total_area_mm2'] for r in all_results)
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Images processed: {len(all_results)}")
        print(f"   Total defects found: {total_defects}")
        print(f"   Total defect area: {total_area:.2f} mm²")
        print(f"   Average defects per image: {total_defects / len(all_results):.1f}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Door Defect Detection Inference")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to door_defect_detector.pth')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or folder')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in folder')
    parser.add_argument('--output', type=str, default='results',
                       help='Output folder for results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SimpleDoorDefectDetector(args.model)
    
    # Run detection
    if args.batch:
        results = detector.batch_detect(args.image, args.output)
    else:
        results = detector.detect(args.image, save_results=True)
        
        print(f"\n🔍 Detection Results:")
        print(f"   Number of defects: {results['num_defects']}")
        print(f"   Total defect area: {results['total_area_mm2']:.2f} mm²")
        
        for i, det in enumerate(results['detections'], 1):
            print(f"   Defect {i}: {det['class_name']}")
            print(f"      Confidence: {det['confidence']:.2%}")
            print(f"      Area: {det['area_mm2']:.2f} mm²")


if __name__ == "__main__":
    main()
