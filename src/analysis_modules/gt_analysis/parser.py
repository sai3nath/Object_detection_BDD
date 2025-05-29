import json
import pandas as pd
from pathlib import Path
import os
from typing import Tuple, Dict, Any, List
from tqdm import tqdm

IMG_COLS = [
    "image_id",
    "image_name",
    "split",
    "timestamp",
    "weather",
    "scene",
    "timeofday",
]

OBJ_COLS = [
    "object_id",
    "image_id",
    "category",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_cx",
    "bbox_cy",
    "occluded",
    "truncated",
    "traffic_light_color",
    "manual_shape",
    "manual_attributes",
]


class BDDAnnotationParser:
    def __init__(self, module_config: Dict[str, Any], main_output_dir: str):
        """
        Initializes the BDD Annotation Parser.

        Args:
            module_config (Dict[str, Any]): Specific configuration for this parsing module,
                                             e.g., from cfg['gt_analysis_config'].
            main_output_dir (str): The base directory where all outputs for the current
                                   master_analyzer.py run are saved.
        """
        self.module_config = module_config
        self.main_output_dir = Path(main_output_dir)

        intermediate_subdir_name = self.module_config.get('intermediate_data_subdir', '01_parsed_annotations')
        intermediate_subdir_name = os.path.join(intermediate_subdir_name, "gt_parsed_data")
        self.output_path_for_parquets = self.main_output_dir / intermediate_subdir_name
        os.makedirs(self.output_path_for_parquets, exist_ok=True)
        print(f"BDDAnnotationParser initialized. Parquet outputs will be saved to: {self.output_path_for_parquets}")

    def parse_bdd_json(self, json_path: str, split_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parses a BDD100K object detection JSON annotation file into two DataFrames,
        including calculated bounding box properties (width, height, area, center).

        Args:
            json_path: Path to the BDD JSON annotation file.
            split_name: Name of the dataset split ('train' or 'val').

        Returns:
            A tuple containing two pandas DataFrames:
            1. images_df: DataFrame with image-level information.
            2. objects_df: DataFrame with object-level information (annotations).
        """
        print(f"Parsing {split_name} data from: {json_path}...")
        image_data_list: List[Dict[str, Any]] = []
        object_data_list: List[Dict[str, Any]] = []

        try:
            with open(json_path, "r") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_path}")
            return pd.DataFrame(columns=IMG_COLS), pd.DataFrame(columns=OBJ_COLS)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_path}")
            return pd.DataFrame(columns=IMG_COLS), pd.DataFrame(columns=OBJ_COLS)

        for img_annotation in raw_data:
            image_name = img_annotation.get("name")
            if not image_name:
                print("Warning: Skipping image annotation with missing 'name'")
                continue

            # --- Extract Image-Level Data ---
            img_attributes = img_annotation.get("attributes", {})
            image_info = {
                "image_id": image_name,
                "image_name": image_name,
                "split": split_name,
                "timestamp": img_annotation.get("timestamp"),
                "weather": img_attributes.get("weather", "unknown"),
                "scene": img_attributes.get("scene", "unknown"),
                "timeofday": img_attributes.get("timeofday", "unknown"),
            }
            image_data_list.append(image_info)

            # --- Extract Object-Level Data ---
            labels = img_annotation.get("labels", [])
            for label in labels:
                box2d = label.get("box2d")
                category = label.get("category")
                object_id = label.get("id")

                if box2d and category and object_id is not None:
                    # Extract coordinates
                    x1 = box2d.get("x1")
                    y1 = box2d.get("y1")
                    x2 = box2d.get("x2")
                    y2 = box2d.get("y2")

                    # Basic validation and calculation of derived properties
                    bbox_width = 0.0
                    bbox_height = 0.0
                    bbox_area = 0.0
                    bbox_cx = None
                    bbox_cy = None

                    # Ensure coordinates are valid numbers before calculations
                    if all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1

                        # Calculate area only if width and height are positive
                        if bbox_width > 0 and bbox_height > 0:
                            bbox_area = bbox_width * bbox_height
                        else:
                            # Handle cases like single points or lines if they occur
                            bbox_area = 0.0

                        # Calculate center coordinates
                        bbox_cx = (x1 + x2) / 2.0
                        bbox_cy = (y1 + y2) / 2.0
                    else:
                        print(f"Warning: Invalid coordinates for object {object_id} in image {image_name}. Skipping derived calculations.")


                    label_attributes = label.get("attributes", {})
                    object_info = {
                        "object_id": object_id,
                        "image_id": image_name,
                        "category": category,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "bbox_width": bbox_width,   # Added
                        "bbox_height": bbox_height, # Added
                        "bbox_area": bbox_area,     # Added
                        "bbox_cx": bbox_cx,         # Added
                        "bbox_cy": bbox_cy,         # Added
                        "occluded": label_attributes.get("occluded"),
                        "truncated": label_attributes.get("truncated"),
                        "traffic_light_color": label_attributes.get(
                            "trafficLightColor", "none"
                        ),
                        "manual_shape": label.get("manualShape"),
                        "manual_attributes": label.get("manualAttributes"),
                    }
                    object_data_list.append(object_info)


        # Create DataFrames
        images_df = pd.DataFrame(image_data_list)
        objects_df = pd.DataFrame(object_data_list)

        if not images_df.empty:
            images_df = images_df.astype({
                "image_id": "string",
                "image_name": "string",
                "split": "category",
                "timestamp": "Int64",
                "weather": "category",
                "scene": "category",
                "timeofday": "category",
            })
        if not objects_df.empty:
            bool_cols = ['occluded', 'truncated', 'manual_shape', 'manual_attributes']
            for col in bool_cols:
                if col in objects_df.columns:
                    try:
                            objects_df[col] = objects_df[col].astype(pd.BooleanDtype())
                    except (TypeError, ValueError):
                        print(f"Warning: Column '{col}' could not be reliably converted to boolean. Check data.")

            # Define types for all columns, including new ones
            objects_df = objects_df.astype({
                "object_id": "int64",
                "image_id": "string",
                "category": "category",
                "bbox_x1": "float64",
                "bbox_y1": "float64",
                "bbox_x2": "float64",
                "bbox_y2": "float64",
                "bbox_width": "float64",      
                "bbox_height": "float64",     
                "bbox_area": "float64",       
                "bbox_cx": "float64",         
                "bbox_cy": "float64",         
                "traffic_light_color": "category",
            })


        print(f"Finished parsing {split_name}. Found {len(images_df)} images and {len(objects_df)} objects.")
        return images_df, objects_df

    def run(self, train_json_path_str: str, val_json_path_str: str) -> Dict[str, str]:
        """
        Main execution method for this parser.
        Parses train and validation JSONs and saves the resulting DataFrames as Parquet files.

        Args:
            train_json_path_str (str): Path to the BDD training JSON file.
            val_json_path_str (str): Path to the BDD validation JSON file.

        Returns:
            Dict[str, str]: A dictionary containing absolute paths to the saved parquet files.
        """
        print("BDDAnnotationParser: Starting parsing of train and validation annotations...")

        print(f"train_json_path_str is {train_json_path_str}")
        print(f"val_json_path_str is {val_json_path_str}")

        # Process Training Data
        train_images_df, train_objects_df = self.parse_bdd_json(
            json_path=train_json_path_str, split_name="train"
        )
        train_images_parquet_path = self.output_path_for_parquets / "bdd_train_images.parquet"
        train_objects_parquet_path = self.output_path_for_parquets / "bdd_train_objects.parquet"

        if not train_images_df.empty:
            train_images_df.to_parquet(train_images_parquet_path, index=False)
            print(f"Saved parsed training images to: {train_images_parquet_path.resolve()}")
        if not train_objects_df.empty:
            train_objects_df.to_parquet(train_objects_parquet_path, index=False)
            print(f"Saved parsed training objects to: {train_objects_parquet_path.resolve()}")

        # Process Validation Data
        val_images_df, val_objects_df = self.parse_bdd_json(
            json_path=val_json_path_str, split_name="val"
        )
        val_images_parquet_path = self.output_path_for_parquets / "bdd_val_images.parquet"
        val_objects_parquet_path = self.output_path_for_parquets / "bdd_val_objects.parquet"

        if not val_images_df.empty:
            val_images_df.to_parquet(val_images_parquet_path, index=False)
            print(f"Saved parsed validation images to: {val_images_parquet_path.resolve()}")
        if not val_objects_df.empty:
            val_objects_df.to_parquet(val_objects_parquet_path, index=False)
            print(f"Saved parsed validation objects to: {val_objects_parquet_path.resolve()}")

        print("BDDAnnotationParser: Parsing complete.")
        
        return {
            "train_images_path": str(train_images_parquet_path.resolve()),
            "train_objects_path": str(train_objects_parquet_path.resolve()),
            "val_images_path": str(val_images_parquet_path.resolve()),
            "val_objects_path": str(val_objects_parquet_path.resolve())
        }
