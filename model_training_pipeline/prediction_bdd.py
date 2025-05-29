from ultralytics import YOLO
import os
import torch
import yaml # To load YAML
from pathlib import Path 

# --- Configuration ---
# !!! IMPORTANT: Update this path to your specific fine-tuned model !!!
MODEL_PATH = 'yolov8n.pt' 

# Path to your BDD dataset configuration YAML file
# !!! IMPORTANT: Ensure this YAML file exists and paths inside are correct and absolute !!!
DATA_YAML_PATH = 'bdd_data.yaml' # Assumes it's in the same directory as model_trainer_pipeline

# Prediction parameters (for model.predict to get raw outputs)
PREDICT_IMAGE_SIZE = 640 
PREDICT_BATCH_SIZE = 16
PREDICT_CONF_THRESHOLD = 0.25 
PREDICT_IOU_THRESHOLD = 0.6

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu') # Auto-select device

PREDICT_PROJECT_NAME = 'predictions' # Directory to save prediction results
PREDICT_RUN_NAME = f'predict_{Path(MODEL_PATH).stem}_img{PREDICT_IMAGE_SIZE}_conf{PREDICT_CONF_THRESHOLD}' # Sub-dir name for prediction run

# --- Prediction ---

def main():
    print(f"Using device: {DEVICE}")
    print(f"Using model: {MODEL_PATH}")
    print(f"Using dataset config: {DATA_YAML_PATH}")
    model_path_obj = Path(MODEL_PATH)
    data_yaml_path_obj = Path(DATA_YAML_PATH)


    # --- Pre-run Checks ---
    if not model_path_obj.exists():
        print(f"\nError: Model file not found at '{MODEL_PATH}'")
        return

    if not data_yaml_path_obj.exists():
        print(f"\nError: Dataset YAML file not found at '{DATA_YAML_PATH}'")
        return

    try:
        with open(data_yaml_path_obj, 'r') as f:
            data_cfg = yaml.safe_load(f)
        if 'val' not in data_cfg or not data_cfg['val']:
            print(f"\nError: 'val' path is missing or empty in '{DATA_YAML_PATH}'")
            return

        val_img_source = (data_yaml_path_obj.parent / Path(data_cfg['val'])).resolve()

        if not val_img_source.exists():
             print(f"\nWarning: Validation source path '{val_img_source}' derived from YAML does not exist.")

        print(f"Resolved validation image source from YAML: {val_img_source}")

        if 'names' not in data_cfg or len(data_cfg.get('names', [])) == 0:
             print(f"\nWarning: 'names' list missing or empty in '{DATA_YAML_PATH}'")

    except yaml.YAMLError as e:
        print(f"\nError parsing YAML file '{DATA_YAML_PATH}': {e}")
        return
    except Exception as e:
         print(f"\nError processing YAML file '{DATA_YAML_PATH}': {e}")
         return
    # --- End Checks ---

    # Load the YOLO model
    print(f"\nLoading YOLO model '{MODEL_PATH}'...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
         print(f"\nError loading model '{MODEL_PATH}': {e}")
         return

    print("\n" + "="*10 + " Starting Prediction (model.predict) " + "="*10)
    print("  (This generates raw detection files needed for detailed analysis)")
    print(f"  Image size: {PREDICT_IMAGE_SIZE}")
    print(f"  Confidence threshold: {PREDICT_CONF_THRESHOLD}")
    print(f"  IoU threshold: {PREDICT_IOU_THRESHOLD}")

    try:
        predict_results = model.predict(
            source=str(val_img_source), 
            imgsz=PREDICT_IMAGE_SIZE,
            conf=PREDICT_CONF_THRESHOLD,
            iou=PREDICT_IOU_THRESHOLD,
            device=DEVICE,
            project=PREDICT_PROJECT_NAME,
            name=PREDICT_RUN_NAME,
            save=False,          
            save_txt=True,       
            save_conf=True,      
            stream=False      
        )

        prediction_save_dir = Path(PREDICT_PROJECT_NAME) / PREDICT_RUN_NAME / 'labels'
        print(f"\nPrediction finished.")
        print(f"Raw prediction results (.txt files) should be saved in: {prediction_save_dir}")
        print("These files contain 'class cx cy w h conf' per detection.")

    except Exception as e:
        print(f"\nError during model.predict(): {e}")

    print("\nFull process complete.")
    print("-" * 30)
    return


if __name__ == '__main__':
    main()