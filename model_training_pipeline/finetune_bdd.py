import os
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path 

# --- Configuration ---

# Path to the pre-trained model file you want to fine-tune
MODEL_TO_FINETUNE = 'yolov8n.pt'

DATA_YAML_PATH = 'bdd_data.yaml' 

# Training parameters
EPOCHS = 10     
IMAGE_SIZE = 640 
BATCH_SIZE = 16
PROJECT_NAME = 'finetune_output' # Main directory name to save results
RUN_NAME = f'train_{Path(MODEL_TO_FINETUNE).stem}_bdd_epochs{EPOCHS}_img{IMAGE_SIZE}' # Sub-dir name for this specific run
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
WORKERS = 4       
PATIENCE = 10 

# --- Main Fine-Tuning ---

def main():
    print(f"Using device: {DEVICE}")
    print(f"Starting fine-tuning for {MODEL_TO_FINETUNE}")
    print(f"Using dataset config: {DATA_YAML_PATH}")

    # Check if YAML file exists
    if not os.path.exists(DATA_YAML_PATH):
        print(f"\nError: Dataset YAML file not found at '{DATA_YAML_PATH}'")
        print("Please ensure the YAML file exists and the path is correct.")
        return

    # Try loading YAML to catch basic syntax errors and check paths
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            data_cfg = yaml.safe_load(f)
        # Check for required keys
        required_keys = ['train', 'val', 'nc', 'names']
        missing_keys = [key for key in required_keys if key not in data_cfg or not data_cfg[key]]
        if missing_keys:
             print(f"\nError: Missing or empty required keys in '{DATA_YAML_PATH}': {', '.join(missing_keys)}")
             return
        print(f"Found train path: {data_cfg['train']}")
        print(f"Found val path: {data_cfg['val']}")
        print(f"Found nc: {data_cfg['nc']}")
        print(f"Found {len(data_cfg['names'])} names.")
        if data_cfg['nc'] != 10 or len(data_cfg.get('names', [])) != 10:
             print("\nWarning: 'nc' or number of 'names' in YAML is not 10. Ensure this matches your BDD task.")

    except yaml.YAMLError as e:
        print(f"\nError parsing YAML file '{DATA_YAML_PATH}': {e}")
        return
    except Exception as e:
         print(f"\nError checking YAML file '{DATA_YAML_PATH}': {e}")
         return
    # --- End Checks ---


    # Load the YOLO model (starting from pre-trained weights)
    print(f"\nLoading base model '{MODEL_TO_FINETUNE}'...")
    try:
        model = YOLO(MODEL_TO_FINETUNE)
    except Exception as e:
         print(f"Error loading model '{MODEL_TO_FINETUNE}': {e}")
         return

    print(f"\nStarting training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Project: {PROJECT_NAME}")
    print(f"  Run Name: {RUN_NAME}")

    # Start fine-tuning
    try:
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=RUN_NAME,
            device=DEVICE,
            workers=WORKERS,
            patience=PATIENCE
        )
        print("\nTraining finished successfully.")

    except Exception as e:
        print(f"\nError during model.train(): {e}")
        print("Please check GPU memory, dataset paths/format, and dependencies.")
        return

    print("-" * 30)
    print("Training results, checkpoints, and weights are saved in:")
    final_output_dir = Path(PROJECT_NAME) / RUN_NAME
    print(f"  {final_output_dir}")
    print("The best model weights are typically saved as:")
    print(f"  {final_output_dir / 'weights' / 'best.pt'}")
    print("-" * 30)


if __name__ == '__main__':
    main()