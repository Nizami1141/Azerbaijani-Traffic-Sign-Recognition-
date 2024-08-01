import os
from ultralytics import YOLO
import yaml
import torch

# Allow OpenMP to load multiple times (use with caution)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Specify the GPU ID (e.g., 0 for the first GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load the configuration file and print its contents
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
print("Configuration:", config)

# Load a model
model = YOLO("yolov8n.yaml")  # specify the model you want to use

# Check the dataset paths in the config
train_path = config.get('train', None)
val_path = config.get('val', None)
print("Training data path:", train_path)
print("Validation data path:", val_path)

# Verify that the dataset paths are correct and contain data
if train_path is None or not os.path.exists(train_path):
    raise ValueError(f"Training data path '{train_path}' is invalid.")
if val_path is None or not os.path.exists(val_path):
    raise ValueError(f"Validation data path '{val_path}' is invalid.")

# Add a check for empty datasets
train_path = 'C:/Users/nizam/Downloads/data/images/train'
val_path = 'C:/Users/nizam/Downloads/data/images/train'
def is_dataset_empty(path):
    if not os.path.exists(path):
        return True
    # Check if the dataset directory is empty or has no valid data
    for root, dirs, files in os.walk(path):
        if any(file.endswith(('.jpg', '.jpeg', '.png', '.bmp')) for file in files):
            return False
    return True

if is_dataset_empty(train_path):
    raise ValueError(f"Training dataset at '{train_path}' is empty or invalid.")
if is_dataset_empty(val_path):
    raise ValueError(f"Validation dataset at '{val_path}' is empty or invalid.")

# Train the model
try:
    print("Starting training...")
    results = model.train(data="config.yaml", epochs=1)
    print("Training completed.")
except Exception as e:
    print(f"Error during training: {e}")

# Validate the model
try:
    print("Starting validation...")
    metrics = model.val()
    print("Validation metrics:", metrics)
    if metrics is None or len(metrics) == 0:
        raise ValueError("Validation failed: metrics are empty.")
except Exception as e:
    print(f"Error during validation: {e}")

# Export the model
try:
    model.export(format="onnx")
    print("Model exported.")
except Exception as e:
    print(f"Error during model export: {e}")
