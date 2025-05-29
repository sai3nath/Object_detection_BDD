# BDD100k Dataset Analysis and YOLOv8n Object Detection Pipeline

This repository contains the code and pipeline for a comprehensive analysis of the Berkeley DeepDrive 100k (BDD100k) dataset, along with the fine-tuning and evaluation of a YOLOv8n object detection model on this dataset.

The primary goals of this project are to:
* Perform an in-depth analysis of the BDD100k dataset's ground truth to understand its characteristics, class distributions, object properties (size, location), and environmental attributes.
* Prepare the BDD100k data for object detection model training (including subset creation and conversion to YOLO format).
* Fine-tune a YOLOv8n model on a subset of the BDD100k dataset.
* Generate predictions using the fine-tuned model on the validation set.
* Conduct a thorough quantitative and qualitative evaluation of the model's performance, including detailed metric breakdowns and visualizations.

This pipeline is orchestrated by a central script (`src/master_analyzer.py`) driven by a YAML configuration file (`configs/config.yaml`), allowing for modular execution of different analysis stages. Additionally, the ground truth data analysis task is containerized using Docker for ease of reproducibility, as per the assignment requirements.

The following sections detail the setup, data preparation, and execution steps for the various components of this project.

---

### 1. Create python virtual env, and activate it. 
```bash
python -m venv .your_env_name
source .your_env_name/bin/activate
```

### Download the data, which will have images and labels in separate folders, 
path to images are updated in the bdd_data.yaml accordingly , make sure you are updating the paths are absolute paths to the respective folder
path to labels, which are in json format are needs to updated in configs/config.yaml. 

### 2. update paths to train and val in bdd_data.yaml 
example: train: /absolute/path/to/your/train

### 3. update JSON paths for bdd_labels_train_json and bdd_labels_val_json in configs/config.yaml file. 
example: data_bdd/bdd100k/labels_original/bdd100k_labels_images_train.json


<!-- Can run the GT analysis only, now -->
### 4. Run Ground Truth Anlaysis, outputs of this step can be found at gt section of analysis_run_outputs folder (this is the main output folder)
In run_modules section of the configs/config.yaml, set, ground_truth_analysis to true.
run the below command, from the obd_for_packaging path.

```bash 
python src/master_analyzer.py --config configs/config.yaml
```

<!-- Lets run the prediction script -->
### 5. Run the prediction over images, to get the detections for each image which you will find at predictions/ folder.
This will generate txt file for each of the image, each txt will have the detections. 

```bash 
python model_training_pipeline/prediction_bdd.py
```

### 6. Run the evaluation pipeline, outputs of this step can be found at gt section of analysis_run_outputs folder (this is the main output folder)
in run_modules section of the configs/config.yaml, if you have already run the ground truth analysis, then we can set the ground_truth_analysis to False
else, set both ground_truth_analysis and evaluation_analysis to true.

```bash 
python src/master_analyzer.py --config configs/config.yaml
```


### 7. Run the finetuning
model_training_pipeline/finetune_bdd.py is the responsible file. 

In that file there are configurable parameters, like model path, make sure bdd_data.yaml path is correct. 

To run finetuning, run below command. This will create the finetune_output folder. weight will be stored at finetune_output/train_yolov8n_bdd_epochs10_img640/weights/

run the below command,

```bash 
python model_training_pipeline/finetune_bdd.py
```

