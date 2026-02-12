# Door Defect Detection System - Requirements & Setup

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB free space for models and data
- **Camera**: Industrial camera with minimum 1920x1080 resolution for production

### Software
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 12+
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)
- **cuDNN**: 8.6+ (matching CUDA version)

## Installation Steps

### 1. Create Python Environment

```bash
# Using conda (recommended)
conda create -n door_defect python=3.10
conda activate door_defect

# Or using venv
python3.10 -m venv door_defect_env
source door_defect_env/bin/activate  # Linux/Mac
# door_defect_env\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
# For CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended for training)
pip install torch==2.1.0 torchvision==0.16.0

# Install other requirements
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully')"
```

## Package Requirements

### requirements.txt
```
# Core dependencies
ultralytics==8.1.0              # YOLOv8 framework
opencv-python==4.8.1.78         # Computer vision
opencv-contrib-python==4.8.1.78 # Additional CV modules
numpy==1.24.3                   # Numerical computing
scipy==1.11.3                   # Scientific computing

# Data augmentation
albumentations==1.3.1           # Advanced augmentations
imgaug==0.4.0                   # Additional augmentations

# ML utilities
scikit-learn==1.3.2            # Train/val split, metrics
scikit-image==0.22.0           # Image processing
pandas==2.1.3                  # Data manipulation
matplotlib==3.8.2              # Visualization
seaborn==0.13.0                # Statistical visualization
Pillow==10.1.0                 # Image I/O

# Progress and utilities
tqdm==4.66.1                   # Progress bars
pyyaml==6.0.1                  # Configuration files
tensorboard==2.15.1            # Training monitoring
wandb==0.16.1                  # Experiment tracking (optional)

# Deployment
onnx==1.15.0                   # Model export
onnxruntime-gpu==1.16.3        # ONNX inference (GPU)
# onnxruntime==1.16.3          # ONNX inference (CPU)

# Optional: Model optimization
# openvino-dev==2023.2.0       # Intel optimization
# tensorrt==8.6.1              # NVIDIA optimization
```

### Additional Tools (Optional)

```bash
# For advanced monitoring
pip install aim==3.17.5

# For model profiling
pip install torchinfo==1.8.0
pip install pytorch-model-summary==0.1.2

# For distributed training (multi-GPU)
pip install accelerate==0.25.0
```

## Project Structure Setup

```bash
# Create project structure
mkdir -p door_defect_project/{data,models,calibration,runs,utils,inference}
cd door_defect_project

# Download pretrained YOLOv8 weights
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov8s-seg.pt')"

# Organize your data
# Place your 3 door datasets in data/
mv /path/to/black data/
mv /path/to/white data/
mv /path/to/glossy_white data/
```

## Directory Structure

```
door_defect_project/
├── data/
│   ├── black/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── data.yaml
│   ├── white/
│   │   └── [same structure]
│   ├── glossy_white/
│   │   └── [same structure]
│   └── combined/          # Created during training
│       ├── train/
│       ├── val/
│       ├── test/
│       └── data.yaml
├── models/
│   ├── yolov8n-seg.pt     # Pretrained
│   ├── yolov8s-seg.pt     # Pretrained
│   └── door_defect_detector.pth  # Trained model
├── calibration/
│   └── calibration_config.json
├── runs/
│   └── segment/
│       └── door_defect_detection/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.png
│           └── confusion_matrix.png
├── door_defect_detection_architecture.py
├── simple_inference.py
├── requirements.txt
└── README.md
```

## Quick Start Guide

### Training

```bash
# Activate environment
conda activate door_defect

# Run complete training pipeline
python door_defect_detection_architecture.py

# This will:
# 1. Merge 3 door datasets
# 2. Perform calibration
# 3. Train YOLOv8-Seg model
# 4. Validate on test set
# 5. Create deployment .pth file
```

### Inference (App Developer)

```bash
# Single image
python simple_inference.py \
    --model models/door_defect_detector.pth \
    --image test_door.jpg

# Batch processing
python simple_inference.py \
    --model models/door_defect_detector.pth \
    --image test_images/ \
    --batch \
    --output results/
```

## Expected Training Time

With the given dataset (382 images):
- **GPU (RTX 3060)**: ~2-3 hours for 200 epochs
- **GPU (RTX 4090)**: ~1-1.5 hours for 200 epochs
- **CPU**: Not recommended (12+ hours)

## Expected Performance Metrics

Based on similar industrial defect detection projects:

```
Model: YOLOv8n-seg (Nano - fastest)
- mAP50: 90-93%
- mAP50-95: 75-80%
- Inference: ~50ms per image (GPU)
- Model size: ~6MB

Model: YOLOv8s-seg (Small - recommended)
- mAP50: 92-95%
- mAP50-95: 78-83%
- Inference: ~80ms per image (GPU)
- Model size: ~22MB

Model: YOLOv8m-seg (Medium - high accuracy)
- mAP50: 93-96%
- mAP50-95: 80-85%
- Inference: ~150ms per image (GPU)
- Model size: ~50MB
```

## Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Reduce batch size in training
batch_size = 8  # Instead of 16

# Or use gradient accumulation
accumulate = 2  # Effective batch = 8 * 2 = 16
```

### Issue: Low Accuracy (<92%)
```
Solutions:
1. Increase model size: yolov8n → yolov8s → yolov8m
2. Train longer: epochs = 300
3. Enhance augmentation
4. Collect more data (target: 500-1000 images)
5. Check annotation quality
```

### Issue: Poor Segmentation
```
Solutions:
1. Increase copy_paste augmentation
2. Use mosaic augmentation
3. Adjust mask loss weight
4. Review mask annotations for accuracy
```

### Issue: Incorrect Area Measurements
```
Solutions:
1. Re-calibrate camera using checkerboard
2. Verify pixels_per_mm ratio
3. Use consistent camera distance
4. Account for lens distortion
```

## Camera Calibration Guide

### Method 1: Checkerboard Pattern

1. **Print checkerboard**: 9x6 internal corners, 25mm squares
2. **Capture images**: 10-20 images at different angles
3. **Run calibration**:
```python
from door_defect_detection_architecture import CameraCalibration

calibration = CameraCalibration.calibrate_with_checkerboard(
    calibration_images=['calib1.jpg', 'calib2.jpg', ...],
    checkerboard_size=(9, 6),
    square_size_mm=25.0
)
```

### Method 2: Reference Object

1. **Use known object**: Place ruler or object with known dimensions
2. **Measure in pixels**: Use image editor to measure pixel width
3. **Calculate ratio**:
```python
calibration = CameraCalibration.calibrate_with_reference_object(
    image_path='reference.jpg',
    reference_width_mm=100.0,   # Known width in mm
    reference_width_pixels=200  # Measured in image
)
# Result: 2 pixels = 1mm
```

## Model Export for Production

### Export to ONNX (Cross-platform)
```python
from ultralytics import YOLO

model = YOLO('runs/segment/door_defect_detection/weights/best.pt')
model.export(format='onnx', imgsz=640, simplify=True)
```

### Export to TensorRT (NVIDIA)
```python
model.export(format='engine', imgsz=640, device=0)
```

### Export to OpenVINO (Intel)
```python
model.export(format='openvino', imgsz=640)
```

## Integration with App

### Python Integration
```python
from simple_inference import SimpleDoorDefectDetector

# Initialize once
detector = SimpleDoorDefectDetector('door_defect_detector.pth')

# Process images
result = detector.detect('door_image.jpg')

# Access results
for defect in result['detections']:
    print(f"Found {defect['class_name']}: {defect['area_mm2']:.2f} mm²")
```

### REST API Wrapper (Flask)
```python
from flask import Flask, request, jsonify
from simple_inference import SimpleDoorDefectDetector

app = Flask(__name__)
detector = SimpleDoorDefectDetector('door_defect_detector.pth')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    file.save('temp.jpg')
    results = detector.detect('temp.jpg')
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Performance Optimization Tips

1. **Batch Processing**: Process multiple images together
2. **GPU Utilization**: Use CUDA for inference
3. **Model Quantization**: Int8 quantization for 4x speedup
4. **Image Preprocessing**: Cache preprocessed images
5. **Multi-threading**: Process images in parallel

## Support & Maintenance

- **Model Retraining**: Recommended every 3-6 months with new data
- **Calibration Check**: Verify monthly or after camera changes
- **Performance Monitoring**: Track accuracy on validation set
- **Data Collection**: Continuously collect edge cases

## License & Credits

- **YOLOv8**: AGPL-3.0 (Ultralytics)
- **PyTorch**: BSD License
- **OpenCV**: Apache 2.0

---

For questions or issues, contact: [your-email@example.com]
