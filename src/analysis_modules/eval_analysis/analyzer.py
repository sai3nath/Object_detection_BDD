import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from tqdm import tqdm
import zipfile

CLASS_MAP = {
  0: "traffic light", 1: "traffic sign", 2: "car", 3: "person", 4: "bus",
  5: "truck", 6: "rider", 7: "bike", 8: "motor", 9: "train"
}

# 4. Image Dimensions
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# 5. IoU Threshold for matching
IOU_THRESHOLD = 0.5

class HelperForEval:
    # --- Helper Function: Calculate IoU ---
    def calculate_iou(self, boxA: List[float], boxB: List[float]) -> float:
        """Calculates Intersection over Union (IoU) between two boxes.

        Boxes are expected in [x1, y1, x2, y2] format.

        Args:
            boxA: First bounding box coordinates [x1, y1, x2, y2].
            boxB: Second bounding box coordinates [x1, y1, x2, y2].

        Returns:
            The IoU value as a float between 0.0 and 1.0.
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        # Use max(0, ...) to handle cases where boxes don't overlap
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # compute the union area
        unionArea = float(boxAArea + boxBArea - interArea)

        # compute the intersection over union
        # Handle edge case of zero union area
        iou = interArea / unionArea if unionArea > 0 else 0.0

        # Return the intersection over union value
        return iou


    # --- Helper Parsing Predictions ---
    def parse_yolo_predictions(self,
        predictions_dir: Path,
        class_map: Dict[int, str],
        image_width: int,
        image_height: int
        ) -> Optional[pd.DataFrame]:
        """
        Parses YOLO prediction .txt files into a pandas DataFrame.

        Assumes files are named {image_id}.txt and contain lines like:
        <class_index> <cx_norm> <cy_norm> <w_norm> <h_norm> <confidence>

        Args:
            predictions_dir: Path to the directory containing .txt files.
            class_map: Dictionary mapping class index (int) to class name (str).
            image_width: Width of the original images.
            image_height: Height of the original images.

        Returns:
            A pandas DataFrame containing parsed predictions or None if directory not found/empty.
            Columns: ['image_id', 'category', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence']
        """
        prediction_data = []
        print(f"Scanning for prediction files in: {predictions_dir}")

        if not predictions_dir.is_dir():
            print(f"Error: Prediction directory not found: {predictions_dir}")
            return pd.DataFrame()  # Return empty DataFrame

        prediction_files = list(predictions_dir.glob("*.txt"))
        print(f"Found {len(prediction_files)} prediction files.")

        if not prediction_files:
            print("Warning: No .txt files found in the directory.")
            return pd.DataFrame() # Return empty DataFrame if no files found

        for txt_file in tqdm(prediction_files, desc="Parsing Predictions"):
            image_id = txt_file.stem # Get filename without extension as image_id
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 6: # class_idx cx cy w h conf
                            try:
                                class_idx = int(parts[0])
                                cx_norm = float(parts[1])
                                cy_norm = float(parts[2])
                                w_norm = float(parts[3])
                                h_norm = float(parts[4])
                                confidence = float(parts[5])

                                if not (0 <= cx_norm <= 1 and 0 <= cy_norm <= 1 and 0 <= w_norm <= 1 and 0 <= h_norm <= 1):
                                    print(f"Warning: Invalid normalized coordinates in {txt_file}, line: '{line.strip()}'. Skipping line.")
                                    continue

                                category = class_map.get(class_idx)
                                if category is None:
                                    print(f"Warning: Class index {class_idx} not found in class_map in file {txt_file}. Skipping line.")
                                    continue

                                # Convert normalized cx,cy,w,h to absolute x1,y1,x2,y2
                                abs_cx = cx_norm * image_width
                                abs_cy = cy_norm * image_height
                                abs_w = w_norm * image_width
                                abs_h = h_norm * image_height

                                x1 = abs_cx - (abs_w / 2)
                                y1 = abs_cy - (abs_h / 2)
                                x2 = abs_cx + (abs_w / 2)
                                y2 = abs_cy + (abs_h / 2)

                                # Clip coordinates to image boundaries
                                x1 = max(0.0, x1)
                                y1 = max(0.0, y1)
                                x2 = min(float(image_width), x2)
                                y2 = min(float(image_height), y2)

                                # Ensure box has valid positive area after clipping
                                if x2 <= x1 or y2 <= y1:
                                    print(f"Warning: Skipping prediction with non-positive area after clipping in {txt_file}, line: '{line.strip()}'.")
                                    continue

                                prediction_data.append({
                                    'image_id': image_id,
                                    'category': category,
                                    'bbox_x1': x1,
                                    'bbox_y1': y1,
                                    'bbox_x2': x2,
                                    'bbox_y2': y2,
                                    'confidence': confidence
                                })
                            except ValueError:
                                print(f"Warning: Skipping invalid number format in {txt_file}, line: '{line.strip()}'")
                        else:
                            print(f"Warning: Skipping malformed line in {txt_file} (expected 6 parts): '{line.strip()}'")
            except Exception as e:
                print(f"Error processing file {txt_file}: {e}")

        print(f"Parsed {len(prediction_data)} total valid detections.")
        if not prediction_data:
            return pd.DataFrame() # Return empty if no detections parsed

        predictions_df = pd.DataFrame(prediction_data)

        # Set data types
        predictions_df = predictions_df.astype({
            'image_id': 'string',
            'category': 'category',
            'bbox_x1': 'float64',
            'bbox_y1': 'float64',
            'bbox_x2': 'float64',
            'bbox_y2': 'float64',
            'confidence': 'float64'
        })
        # Re-apply categories from class_map to ensure all potential classes are known levels
        predictions_df['category'] = predictions_df['category'].cat.set_categories(class_map.values())

        return predictions_df


    # --- Helper: Prepare Failure Data ---
    def prepare_failure_analysis_data(self, fp_df, fn_df, val_images_df):
        """Merges FP/FN data with image attributes and calculates needed fields."""
        print("\nPreparing data for failure analysis...")
        fp_enriched = fp_df.copy()
        fn_enriched = fn_df.copy()

        # Calculate area/center for FPs
        if not fp_enriched.empty and all(c in fp_enriched for c in ['bbox_x1','y1','x2','y2']):
            fp_enriched['bbox_width'] = fp_enriched['bbox_x2'] - fp_enriched['bbox_x1']
            fp_enriched['bbox_height'] = fp_enriched['bbox_y2'] - fp_enriched['bbox_y1']
            fp_enriched['bbox_area'] = fp_enriched['bbox_width'] * fp_enriched['bbox_height']
            fp_enriched['bbox_cx'] = (fp_enriched['bbox_x1'] + fp_enriched['bbox_x2']) / 2.0
            fp_enriched['bbox_cy'] = (fp_enriched['bbox_y1'] + fp_enriched['bbox_y2']) / 2.0

        # Merge with image attributes
        if val_images_df is not None and not val_images_df.empty:
            if 'image_id' not in val_images_df.columns:
                print("Warning: val_images_df missing 'image_id' for merge.")
            else:
                try:
                    # Ensure consistent string type for merge keys
                    fp_enriched['image_id'] = fp_enriched['image_id'].astype(str)
                    fn_enriched['image_id'] = fn_enriched['image_id'].astype(str)
                    val_images_df['image_id'] = val_images_df['image_id'].astype(str)

                    img_attr_cols = ['image_id', 'weather', 'timeofday', 'scene']
                    img_attrs_to_merge = val_images_df[[c for c in img_attr_cols if c in val_images_df.columns]]

                    fp_enriched = pd.merge(fp_enriched, img_attrs_to_merge, on='image_id', how='left')
                    fn_enriched = pd.merge(fn_enriched, img_attrs_to_merge, on='image_id', how='left')
                    print("Merged FP/FN data with image attributes.")
                except Exception as e:
                    print(f"Warning: Failed to merge with image attributes - {e}")
        else:
            print("Warning: val_images_df not provided, cannot add image attributes.")


        print("Failure analysis data preparation complete.")
        return fp_enriched, fn_enriched


class BDDEvaluator:
    """
    Handles matching predictions to ground truth and calculating
    evaluation metrics for BDD object detection.
    """
    def __init__(self, gt_df_path: Path, class_map: Dict[int, str], iou_threshold: float = 0.5):
        """
        Initializes the evaluator.

        Args:
            gt_df: DataFrame containing ground truth annotations (e.g., val_objects_df).
                   Must include 'image_id', 'category', 'bbox_x1', 'y1', 'x2', 'y2'.
                   'object_id' is recommended for unique identification.
            class_map: Dictionary mapping class index (int) to class name (str).
            iou_threshold: IoU threshold for matching.
        """
        gt_df = self._load_data(gt_df_path)
        self.gt_df = gt_df.copy()
        self.class_map = class_map
        self.iou_threshold = iou_threshold
        self.helper = HelperForEval()

        if 'object_id' not in self.gt_df.columns:
             print("Warning: 'object_id' column not found in ground truth. Adding from index.")
             if not self.gt_df.index.is_unique:
                  self.gt_df = self.gt_df.reset_index(drop=True) 
             self.gt_df['object_id'] = self.gt_df.index
        self.gt_df['category'] = self.gt_df['category'].astype('category').cat.set_categories(self.class_map.values())
        self.gt_df['is_matched'] = False 


    def _load_data(self, path: Path) -> pd.DataFrame:
        """Helper method to load parquet file with error handling."""
        print(f"Attempting to load data from {path}...")
        if not path.exists():
            print(f"Error: File not found at {path}. Returning empty DataFrame.")
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            print(f"Successfully loaded {len(df)} rows from {path}.")
            return df
        except Exception as e:
            print(f"Error loading {path}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()


    def match_predictions(self, predictions_df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Matches predictions to the stored ground truth based on IoU.

        Follows COCO-style matching logic: highest confidence predictions get
        first chance to match available ground truth boxes above IoU threshold.

        Args:
            predictions_df: DataFrame of predictions (output from parse_yolo_predictions).
                            Must include 'image_id', 'category', 'bbox_x1', 'y1', 'x2', 'y2', 'confidence'.

        Returns:
            Tuple: (matched_preds_df, updated_gt_df) with matching info added, or None on error.
                   'matched_preds_df' has new columns: 'status' (TP/FP), 'matched_gt_id', 'iou'.
                   'updated_gt_df' has updated 'is_matched' (True/False) column.
                   Returns copies of the original dataframes with added/updated columns.
        """
        helper_obj = HelperForEval()
        if predictions_df is None or predictions_df.empty:
             print("Error: Input predictions_df is empty or None.")
             return None

        # Ensure prediction categories are set consistently
        predictions_df['category'] = predictions_df['category'].astype('category').cat.set_categories(self.class_map.values())

        matched_preds_df = predictions_df.copy()
        # Reset match status on the stored GT df for this run
        self.gt_df['is_matched'] = False
        updated_gt_df = self.gt_df # We'll update this instance's GT df

        # Ensure consistent image_id format (no extensions) in both DFs before processing
        print("Standardizing image IDs (removing extensions)...")
        if matched_preds_df['image_id'].dtype == 'string' or matched_preds_df['image_id'].dtype == 'object':
            matched_preds_df['image_id'] = matched_preds_df['image_id'].str.split('.').str[0]
        if updated_gt_df['image_id'].dtype == 'string' or updated_gt_df['image_id'].dtype == 'object':
            self.gt_df['image_id'] = self.gt_df['image_id'].str.split('.').str[0]
            updated_gt_df = self.gt_df # Re-alias after modification if needed


        # Add columns to store matching results in predictions
        matched_preds_df['status'] = 'FP' # Default predictions to FP
        matched_preds_df['matched_gt_id'] = pd.NA # Use pandas NA for missing int/object ID
        matched_preds_df['iou'] = 0.0

        image_ids = pd.concat([matched_preds_df['image_id'], updated_gt_df['image_id']]).unique()
        print(f"Processing {len(image_ids)} images for matching...")

        # Convert GT bbox columns to numpy for faster access inside loop if df is large
        # Do this once outside the loops
        gt_coords = updated_gt_df[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values
        gt_image_ids = updated_gt_df['image_id'].values
        gt_categories = updated_gt_df['category'].values
        gt_object_ids = updated_gt_df['object_id'].values
        gt_original_indices = updated_gt_df.index # Keep track of original index

        # Convert prediction bbox columns to numpy
        pred_coords = matched_preds_df[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values
        pred_image_ids = matched_preds_df['image_id'].values
        pred_categories = matched_preds_df['category'].values
        pred_confidences = matched_preds_df['confidence'].values
        pred_original_indices = matched_preds_df.index

        for image_id in tqdm(image_ids, desc="Matching Images"):
            # Get indices for current image's predictions and GT using numpy boolean indexing
            img_pred_mask = (pred_image_ids == image_id)
            img_gt_mask = (gt_image_ids == image_id)

            current_pred_indices = pred_original_indices[img_pred_mask]
            current_gt_indices = gt_original_indices[img_gt_mask]

            # Further filter by class within the image
            current_pred_cats = pred_categories[img_pred_mask]
            current_gt_cats = gt_categories[img_gt_mask]
            unique_cats = np.unique(np.concatenate((current_pred_cats[pd.notna(current_pred_cats)],
                                                  current_gt_cats[pd.notna(current_gt_cats)])))


            for category in unique_cats:
                # Get masks for current class within the current image
                cat_pred_mask_img = img_pred_mask & (pred_categories == category)
                cat_gt_mask_img = img_gt_mask & (gt_categories == category)

                # Get original indices for preds/GT of this class in this image
                cat_pred_original_indices = pred_original_indices[cat_pred_mask_img]
                cat_gt_original_indices = gt_original_indices[cat_gt_mask_img]

                if len(cat_gt_original_indices) == 0 or len(cat_pred_original_indices) == 0:
                    continue # No GT or no Preds for this class in this image

                # Get data slices for this class/image
                # Important: Use original indices to get correct data from numpy arrays
                cat_pred_boxes = pred_coords[cat_pred_mask_img]
                cat_pred_confs = pred_confidences[cat_pred_mask_img]
                cat_gt_boxes = gt_coords[cat_gt_mask_img]
                cat_gt_obj_ids = gt_object_ids[cat_gt_mask_img]

                # Sort predictions by confidence (descending)
                sort_order = np.argsort(cat_pred_confs)[::-1]
                sorted_cat_pred_original_indices = cat_pred_original_indices[sort_order]
                sorted_cat_pred_boxes = cat_pred_boxes[sort_order]

                # Calculate IoU matrix (sorted_preds rows, gt columns)
                iou_matrix = np.zeros((len(sorted_cat_pred_boxes), len(cat_gt_boxes)))
                for p_idx, pred_box in enumerate(sorted_cat_pred_boxes):
                    for g_idx, gt_box in enumerate(cat_gt_boxes):
                         # Convert numpy row slices back to list/tuple for calculate_iou if needed
                         iou_matrix[p_idx, g_idx] = helper_obj.calculate_iou(list(pred_box), list(gt_box))

                # Keep track of matched GT indices (using 0-based index relative to cat_gt_*)
                gt_matched_map = {idx: False for idx in range(len(cat_gt_original_indices))}

                # Match greedily
                for p_idx in range(len(sorted_cat_pred_original_indices)):
                    pred_original_idx = sorted_cat_pred_original_indices[p_idx] # Index in main preds_df
                    pred_ious = iou_matrix[p_idx, :] # IoUs for this pred against all GTs

                    max_iou = 0.0
                    best_gt_local_idx = -1 # 0-based index within cat_gt_* arrays
                        
                    # Find best *available* GT match
                    for g_idx in range(len(cat_gt_original_indices)):
                        if not gt_matched_map[g_idx]: # If GT is available
                            if pred_ious[g_idx] > max_iou:
                                max_iou = pred_ious[g_idx]
                                best_gt_local_idx = g_idx


                    # If match found above threshold
                    if max_iou >= self.iou_threshold and best_gt_local_idx != -1:

                        # Update prediction status in the main DataFrame
                        matched_preds_df.loc[pred_original_idx, 'status'] = 'TP'
                        matched_preds_df.loc[pred_original_idx, 'matched_gt_id'] = cat_gt_obj_ids[best_gt_local_idx]
                        matched_preds_df.loc[pred_original_idx, 'iou'] = max_iou

                        # Update GT status in the main DataFrame
                        gt_original_idx_to_update = cat_gt_original_indices[best_gt_local_idx]
                        # Use .loc on the DataFrame stored in self.gt_df (aliased as updated_gt_df)
                        updated_gt_df.loc[gt_original_idx_to_update, 'is_matched'] = True
                        gt_matched_map[best_gt_local_idx] = True # Mark as used locally

        print("Matching complete.")
        # Return the modified predictions and the updated internal GT dataframe
        return matched_preds_df, updated_gt_df


    def calculate_recall_vs_size(
        self,
        updated_gt_df: Optional[pd.DataFrame] = None, # Allow passing it in
        bins: Optional[Dict[str, Tuple[float, float]]] = None,
        group_by_class: bool = False
        ) -> Optional[pd.DataFrame]:
        """
        Calculates recall based on object size bins using matched GT data.

        Uses standard COCO size bins by default:
            - Small: area <= 32^2 (1024)
            - Medium: 32^2 < area <= 96^2 (9216)
            - Large: area > 96^2

        Args:
            updated_gt_df: Optional. The ground truth DataFrame previously updated by
                           match_predictions (must contain 'bbox_area', 'is_matched').
                           If None, uses the internal self.gt_df.
            bins: Optional dictionary defining custom bins, e.g.,
                  {'Tiny': (0, 256), 'Small': (256, 1024), ...}.
                  Keys are bin names, values are (lower_bound, upper_bound].
                  Uses COCO bins if None.
            group_by_class: If True, calculates recall per class within each size bin.
                            If False (default), calculates overall recall per size bin.

        Returns:
            A DataFrame summarizing Recall and GT Count per size bin
            (and potentially per class), or None on error.
        """
        print(f"\nCalculating Recall vs. Size (Group by class: {group_by_class})...")

        target_gt_df = updated_gt_df if updated_gt_df is not None else self.gt_df
        if target_gt_df is None or target_gt_df.empty:
             print("Error: Ground truth DataFrame not available.")
             return None
        required_cols = ['bbox_area', 'is_matched']
        if group_by_class:
            required_cols.append('category')
        if not all(col in target_gt_df.columns for col in required_cols):
             print(f"Error: GT DataFrame missing required columns: {required_cols}. Run matching first?")
             return None

        # --- Define Bins ---
        if bins is None:
            # Default to COCO bins
            coco_bins = {
                'Small': (0, 1024),          # area <= 32^2
                'Medium': (1024, 9216),      # 32^2 < area <= 96^2
                'Large': (9216, np.inf)      # area > 96^2
            }
            bins_to_use = coco_bins
            print("Using standard COCO size bins (Small<=1024, Medium<=9216, Large>9216)")
        else:
            bins_to_use = bins
            print(f"Using custom size bins: {bins_to_use}")

        bin_labels = list(bins_to_use.keys())
        cut_bin_edges = [list(bins_to_use.values())[0][0]] + [b[1] for b in bins_to_use.values()]
        if cut_bin_edges[0] == 0:
             cut_bin_edges[0] = -0.001 # pd.cut needs lower bound < lowest value for include_lowest

        # --- Assign Size Bins ---
        try:
            binned_gt_df = target_gt_df.copy()
            binned_gt_df['size_bin'] = pd.cut(
                binned_gt_df['bbox_area'],
                bins=cut_bin_edges,
                labels=bin_labels,
                right=True,
                include_lowest=True 
            )
            if binned_gt_df['size_bin'].isnull().any():
                print("Warning: Some objects did not fall into defined size bins.")
                binned_gt_df.dropna(subset=['size_bin'], inplace=True) # Drop rows that couldn't be binned

        except Exception as e:
             print(f"Error assigning size bins: {e}")
             return None

        # --- Calculate Recall per Bin (and optionally Class) ---
        try:
            grouping_cols = ['size_bin']
            if group_by_class:
                grouping_cols.append('category')

            recall_stats = binned_gt_df.groupby(grouping_cols, observed=False).agg(
                Recall=('is_matched', 'mean'),
                GT_Count=('is_matched', 'size') # 'size' gives the count per group
            ).reset_index()

            recall_stats['Recall'] = (recall_stats['Recall'] * 100).round(2)

            print("Recall vs. Size calculation complete.")
            return recall_stats

        except Exception as e:
            print(f"Error calculating recall stats: {e}")
            return None


    def calculate_recall_vs_visibility(
        self,
        updated_gt_df: Optional[pd.DataFrame] = None,
        group_by_class: bool = False
        ) -> Optional[pd.DataFrame]:
        """
        Calculates recall based on object occlusion and truncation status.

        Args:
            updated_gt_df: Optional. The ground truth DataFrame previously updated by
                           match_predictions (must contain 'occluded', 'truncated',
                           'is_matched', and 'category' if group_by_class=True).
                           If None, uses the internal self.gt_df.
            group_by_class: If True, calculates recall per class within each
                            occlusion/truncation status. If False (default),
                            calculates overall recall per status.

        Returns:
            A DataFrame summarizing Recall (%) and GT Count per visibility status
            (Occluded True/False, Truncated True/False), potentially broken down by class,
            or None on error.
        """
        print(f"\nCalculating Recall vs. Visibility (Occlusion/Truncation, Group by class: {group_by_class})...")

        target_gt_df = updated_gt_df if updated_gt_df is not None else self.gt_df
        if target_gt_df is None or target_gt_df.empty:
             print("Error: Ground truth DataFrame not available.")
             return None

        required_cols = ['occluded', 'truncated', 'is_matched']
        if group_by_class:
            required_cols.append('category')
        if not all(col in target_gt_df.columns for col in required_cols):
             print(f"Error: GT DataFrame missing required columns: {required_cols}. Run matching first?")
             return None

        try:
             vis_df = target_gt_df.copy() # Work on a copy
             vis_df['occluded'] = vis_df['occluded'].astype(pd.BooleanDtype())
             vis_df['truncated'] = vis_df['truncated'].astype(pd.BooleanDtype())
        except Exception as e:
             print(f"Error converting visibility columns to boolean: {e}")
             return None

        # --- Calculate Recall vs Occlusion ---
        try:
            grouping_cols_occ = ['occluded']
            if group_by_class:
                grouping_cols_occ.append('category')

            occ_agg = vis_df.groupby(grouping_cols_occ, observed=False).agg(
                Recall=('is_matched', 'mean'),
                GT_Count=('is_matched', 'size')
            )
            occ_agg['Recall'] = (occ_agg['Recall'] * 100)

            if group_by_class:
                if 'occluded' in occ_agg.index.names:
                    occ_stats = occ_agg.unstack(level='occluded')
                    if not occ_stats.empty:
                        occ_stats.columns = [f"{stat}_{('NotOccluded' if not vis_flag else 'Occluded')}"
                                             for stat, vis_flag in occ_stats.columns]
                    else: 
                        occ_stats = pd.DataFrame() 
                else:
                    print("Warning: Could not unstack occlusion results properly.")
                    occ_stats = pd.DataFrame()
            else:
                occ_stats = pd.DataFrame(index=[0]) 
                if False in occ_agg.index:
                    occ_stats['NotOccluded_Recall'] = occ_agg.loc[False, 'Recall']
                    occ_stats['NotOccluded_Count'] = occ_agg.loc[False, 'GT_Count']
                else:
                    occ_stats['NotOccluded_Recall'] = 0.0
                    occ_stats['NotOccluded_Count'] = 0
                if True in occ_agg.index:
                    occ_stats['Occluded_Recall'] = occ_agg.loc[True, 'Recall']
                    occ_stats['Occluded_Count'] = occ_agg.loc[True, 'GT_Count']
                else:
                    occ_stats['Occluded_Recall'] = 0.0
                    occ_stats['Occluded_Count'] = 0

        except Exception as e:
            print(f"Error calculating occlusion recall stats: {e}")
            occ_stats = None

        # --- Calculate Recall vs Truncation ---
        try:
            grouping_cols_trunc = ['truncated']
            if group_by_class:
                grouping_cols_trunc.append('category')

            trunc_agg = vis_df.groupby(grouping_cols_trunc, observed=False).agg(
                Recall=('is_matched', 'mean'),
                GT_Count=('is_matched', 'size')
            )
            trunc_agg['Recall'] = (trunc_agg['Recall'] * 100)

            if group_by_class:

                if 'truncated' in trunc_agg.index.names:
                    trunc_stats = trunc_agg.unstack(level='truncated')
                    if not trunc_stats.empty:
                        trunc_stats.columns = [f"{stat}_{('NotTruncated' if not vis_flag else 'Truncated')}"
                                            for stat, vis_flag in trunc_stats.columns]
                    else:
                        trunc_stats = pd.DataFrame()
                else:
                    print("Warning: Could not unstack truncation results properly.")
                    trunc_stats = pd.DataFrame()
            else:
                trunc_stats = pd.DataFrame(index=[0])
                if False in trunc_agg.index:
                    trunc_stats['NotTruncated_Recall'] = trunc_agg.loc[False, 'Recall']
                    trunc_stats['NotTruncated_Count'] = trunc_agg.loc[False, 'GT_Count']
                else:
                    trunc_stats['NotTruncated_Recall'] = 0.0
                    trunc_stats['NotTruncated_Count'] = 0
                if True in trunc_agg.index:
                    trunc_stats['Truncated_Recall'] = trunc_agg.loc[True, 'Recall']
                    trunc_stats['Truncated_Count'] = trunc_agg.loc[True, 'GT_Count']
                else:
                    trunc_stats['Truncated_Recall'] = 0.0
                    trunc_stats['Truncated_Count'] = 0
            

        except Exception as e:
            print(f"Error calculating truncation recall stats: {e}")
            trunc_stats = None

        # --- Combine Results ---
        if occ_stats is None and trunc_stats is None:
            print("Error: Both occlusion and truncation calculations failed.")
            return None
        elif occ_stats is None:
            final_stats = trunc_stats
        elif trunc_stats is None:
            final_stats = occ_stats
        else:
            final_stats = pd.concat([occ_stats, trunc_stats], axis=1)

        # --- Final Formatting ---
        if group_by_class and final_stats is not None:
             final_stats = final_stats.reindex(self.class_map.values()).fillna(0)
             count_cols = [col for col in final_stats.columns if 'Count' in col]
             final_stats[count_cols] = final_stats[count_cols].astype(int)
             recall_cols = [col for col in final_stats.columns if 'Recall' in col]
             final_stats[recall_cols] = final_stats[recall_cols].round(2)
        elif final_stats is not None:
             count_cols = [col for col in final_stats.columns if 'Count' in col]
             final_stats[count_cols] = final_stats[count_cols].astype(int)
             recall_cols = [col for col in final_stats.columns if 'Recall' in col]
             final_stats[recall_cols] = final_stats[recall_cols].round(2)


        print("Recall vs. Visibility calculation complete.")
        return final_stats
    

    def calculate_recall_vs_image_attributes(
        self,
        val_images_df: pd.DataFrame,
        updated_gt_df: Optional[pd.DataFrame] = None,
        attributes_to_analyze: List[str] = ['timeofday', 'weather'], # Specify attributes
        group_by_class: bool = False
        ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Calculates recall based on image-level attributes (e.g., timeofday, weather).

        Args:
            val_images_df: DataFrame containing image attributes (must have 'image_id'
                           and columns listed in attributes_to_analyze).
            updated_gt_df: Optional. The ground truth DataFrame previously updated by
                           match_predictions (must contain 'image_id', 'is_matched',
                           and 'category' if group_by_class=True).
                           If None, uses the internal self.gt_df.
            attributes_to_analyze: A list of column names in val_images_df to analyze
                                   (e.g., ['timeofday', 'weather']).
            group_by_class: If True, calculates recall per class within each attribute group.
                            If False (default), calculates overall recall per attribute group.

        Returns:
            A dictionary where keys are attribute names and values are DataFrames
            summarizing Recall (%) and GT Count per attribute value (and potentially
            per class), or None on error. Returns recall rounded to 2 decimal places.
        """
        print(f"\nCalculating Recall vs. Image Attributes ({', '.join(attributes_to_analyze)}, Group by class: {group_by_class})...")

        target_gt_df = updated_gt_df if updated_gt_df is not None else self.gt_df
        if target_gt_df is None or target_gt_df.empty:
             print("Error: Ground truth DataFrame not available.")
             return None
        if val_images_df is None or val_images_df.empty:
             print("Error: Validation images DataFrame not available.")
             return None

        required_gt_cols = ['image_id', 'is_matched']
        required_img_cols = ['image_id'] + attributes_to_analyze
        if group_by_class:
            required_gt_cols.append('category')
        if not all(col in target_gt_df.columns for col in required_gt_cols):
             print(f"Error: GT DataFrame missing required columns: {required_gt_cols}. Run matching first?")
             return None
        if not all(col in val_images_df.columns for col in required_img_cols):
             print(f"Error: Images DataFrame missing required columns: {required_img_cols}.")
             return None

        # --- Merge GT with Image Attributes ---
        try:
            target_gt_df['image_id'] = target_gt_df['image_id'].astype(str)
            val_images_df['image_id'] = val_images_df['image_id'].astype(str)

            merged_df = pd.merge(
                target_gt_df[required_gt_cols], # Select only needed cols from GT
                val_images_df[['image_id'] + attributes_to_analyze],
                on='image_id',
                how='left'
            )
            merged_df['is_matched'] = merged_df['is_matched'].astype(bool)

            if merged_df[attributes_to_analyze].isnull().any().any():
                 missing_count = merged_df[attributes_to_analyze].isnull().any(axis=1).sum()
                 print(f"Warning: Could not find image attributes for {missing_count} GT objects after merge.")


        except Exception as e:
             print(f"Error merging GT and image attribute data: {e}")
             return None

        # --- Calculate Recall per Attribute ---
        results_dict = {}
        for attribute in attributes_to_analyze:
            print(f"  Calculating recall breakdown for attribute: '{attribute}'")
            try:
                grouping_cols = [attribute]
                if group_by_class:
                    grouping_cols.append('category')

                attr_stats = merged_df.groupby(grouping_cols, observed=False, dropna=False).agg( 
                    Recall=('is_matched', 'mean'),
                    GT_Count=('is_matched', 'size')
                )
                attr_stats['Recall'] = (attr_stats['Recall'] * 100).round(2)

                if group_by_class:
                    if 'category' in attr_stats.index.names:
                         idx_levels = [attr_stats.index.get_level_values(i) for i in range(attr_stats.index.nlevels)]
                         new_index = pd.MultiIndex.from_product(
                              [idx_levels[0].unique(), self.class_map.values()],
                              names=attr_stats.index.names
                         )
                         attr_stats = attr_stats.reindex(new_index).fillna({'Recall': 0.0, 'GT_Count': 0})
                         attr_stats['GT_Count'] = attr_stats['GT_Count'].astype(int)


                results_dict[attribute] = attr_stats.reset_index() 

            except Exception as e:
                 print(f"Error calculating recall stats for attribute '{attribute}': {e}")
                 results_dict[attribute] = None 

        print("Recall vs. Image Attributes calculation finished (or attempted).")
        return results_dict if any(df is not None for df in results_dict.values()) else None
    

    def identify_failures(
        self,
        matched_preds_df: Optional[pd.DataFrame] = None,
        updated_gt_df: Optional[pd.DataFrame] = None
        ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Identifies False Positive (FP) predictions and False Negative (FN)
        ground truth objects from the matched results.

        Args:
            matched_preds_df: DataFrame of predictions after running match_predictions.
                              Must contain the 'status' column ('TP' or 'FP').
            updated_gt_df: DataFrame of ground truth after running match_predictions.
                           Must contain the 'is_matched' column (True or False).
                           If None, uses the internal self.gt_df (assuming it was updated).

        Returns:
            A tuple containing two DataFrames: (fp_predictions_df, fn_ground_truth_df),
            or None if input data is invalid.
            - fp_predictions_df contains rows from matched_preds_df where status='FP'.
            - fn_ground_truth_df contains rows from updated_gt_df where is_matched=False.
        """
        print("\nIdentifying Failure Cases (FPs and FNs)...")

        # --- Validate Inputs ---
        if matched_preds_df is None or matched_preds_df.empty:
             print("Error: Matched predictions DataFrame is required and cannot be empty.")
             return None
        if 'status' not in matched_preds_df.columns:
             print("Error: 'status' column not found in matched_preds_df.")
             return None

        target_gt_df = updated_gt_df if updated_gt_df is not None else self.gt_df
        if target_gt_df is None or target_gt_df.empty:
             print("Error: Ground truth DataFrame is required and cannot be empty.")
             return None
        if 'is_matched' not in target_gt_df.columns:
             print("Error: 'is_matched' column not found in updated_gt_df. Run matching first?")
             return None
        # --- End Validation ---

        try:
            # Identify False Positives
            fp_df = matched_preds_df[matched_preds_df['status'] == 'FP'].copy()
            print(f"Identified {len(fp_df)} False Positive predictions.")

            # Identify False Negatives
            fn_df = target_gt_df[target_gt_df['is_matched'] == False].copy()
            print(f"Identified {len(fn_df)} False Negative ground truth objects.")

            # Select potentially useful columns for analysis (optional)
            fp_cols = ['image_id', 'category', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence']
            fn_cols = ['image_id', 'category', 'object_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                       'bbox_area', 'occluded', 'truncated']
            fp_df = fp_df[list(set(fp_cols) & set(fp_df.columns))]
            fn_df = fn_df[list(set(fn_cols) & set(fn_df.columns))]

            print("Failure identification complete.")
            return fp_df, fn_df

        except KeyError as e:
             print(f"Error accessing required columns during failure identification: {e}")
             return None
        except Exception as e:
            print(f"An unexpected error occurred during failure identification: {e}")
            return None
        

class EvalRunner: 

    def __init__(self, config, module_config: Dict[str, Any], 
                 project_config: Dict[str, Any],
                 main_output_dir: str):
        self.module_config = module_config
        self.main_output_dir = Path(main_output_dir)
        intermediate_subdir_name = self.module_config.get('intermediate_data_subdir', '02_eval_analysis_metrics')
        self.output_dir = self.main_output_dir / intermediate_subdir_name

        self.output_dir = Path(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.project_path_config = project_config
        print(f"EvalRunner initialized. Outputs will be saved to: {self.output_dir}")

        gt_files_dir_path = config.get("gt_analysis_config").get("intermediate_data_subdir")
        self.gt_val_objects_path = Path(main_output_dir) / Path(gt_files_dir_path) / "gt_parsed_data/bdd_val_objects.parquet"
        self.gt_val_images_path = Path(main_output_dir) / Path(gt_files_dir_path) / "gt_parsed_data/bdd_val_images.parquet"

        preds_labels_path = config.get("project_paths").get("model_preds_labels_txt_path")

        labels_path = Path(preds_labels_path) / "labels"
        print(f"LABELS PATH IS {labels_path}")
        if os.path.exists(labels_path):
            self.detected_labels_path = labels_path
        else:
            def unzip_labels(zip_path):
                """
                Unzips a labels.zip file and returns the path to the extracted labels directory.
                """
                try:
                    extract_dir = Path(zip_path).parent / "labels"
                    os.makedirs(extract_dir, exist_ok=True)

                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)

                    print(f"Successfully extracted labels to {extract_dir}")
                    return extract_dir

                except zipfile.BadZipFile:
                    print(f"Error: {zip_path} is not a valid zip file")
                    return None
                except Exception as e:
                    print(f"Error extracting zip file: {e}")
                    return None     

            labels_zip_path = labels_path / "labels.zip"        
            self.detected_labels_path = unzip_labels(labels_zip_path)


        self.helper = HelperForEval()
        self.evaluator = BDDEvaluator(
            gt_df_path=self.gt_val_objects_path,
            class_map=CLASS_MAP,
            iou_threshold=IOU_THRESHOLD
        )
        

    def _compute_match_predictions(self):
        print("\nPerforming prediction-to-ground-truth matching...")
        match_results = self.evaluator.match_predictions(
                            self.predictions_df
                            )

        if match_results is None:
            print("Error during matching process. Exiting.")
            exit()

        matched_preds_df, updated_gt_df = match_results
        print("Matching process completed.")

        print("\n--- Matching Results Summary ---")
        print("\nMatched Predictions Sample (with Status):")
        print(matched_preds_df[['image_id', 'category', 'confidence',
                                'status', 'iou', 'matched_gt_id']].head())
        print("\nPrediction Status Counts:")
        print(matched_preds_df['status'].value_counts())

        print("\nUpdated Ground Truth Sample (with Match Status):")
        # Show GT that were potentially matched or missed
        print(updated_gt_df[['image_id', 'category', 'object_id', 'is_matched']].head())
        # Show overall matched status
        print("\nGround Truth Match Status Counts:")
        print(updated_gt_df['is_matched'].value_counts())

        # Save the results (optional but recommended)
        try:
            print(f"\nSaving matched dataframes to {self.output_dir}...")
            matched_preds_path = self.output_dir / "matched_predictions.parquet"
            matched_gt_path = self.output_dir / "matched_ground_truth.parquet"
            matched_preds_df.to_parquet(matched_preds_path, index=False)
            updated_gt_df.to_parquet(matched_gt_path, index=None) 
            print("Saved successfully.")
            print("\nEvaluation matching finished. Next steps: Calculate performance breakdowns.")
            return matched_preds_df, updated_gt_df
        
        except Exception as e:
            print(f"Error saving matched dataframes: {e}")

    def _compute_recall_vs_size(self, updated_gt_df):
        # --- Step 5: Calculate Recall vs. Size ---
        print("\nCalculating performance vs. size...")
        # Calculate overall recall per size bin
        recall_by_size_overall = self.evaluator.calculate_recall_vs_size(
            updated_gt_df=updated_gt_df, 
            group_by_class=False
        )
        if recall_by_size_overall is not None:
            print("\n--- Recall vs. Size (Overall) ---")
            print(recall_by_size_overall)
            recall_by_size_overall.to_parquet(self.output_dir / "recall_vs_size_overall.parquet", index=False)
        else:
            print("Error during recall_vs_size computation process. Exiting.")
        return
    
    def _calulate_recall_vs_size_per_class(self, updated_gt_df):
        recall_by_size_per_class = self.evaluator.calculate_recall_vs_size(
            updated_gt_df=updated_gt_df,
            group_by_class=True 
        )
        if recall_by_size_per_class is not None:
            print("\n--- Recall vs. Size (Per Class) ---")
            print(recall_by_size_per_class.head(15)) 
            recall_by_size_per_class.to_parquet(self.output_dir / "recall_vs_size_per_class.parquet", index=False)
        else:
            print("Error during recall_vs_size per class computation process. Exiting.")
        return

    def _calculate_recall_vs_visibility(self, updated_gt_df):
        recall_by_vis_overall = self.evaluator.calculate_recall_vs_visibility(
        updated_gt_df=updated_gt_df,
        group_by_class=False
    )
        if recall_by_vis_overall is not None:
            print("\n--- Recall vs. Visibility (Overall) ---")
            print(recall_by_vis_overall)
            recall_by_vis_overall.to_parquet(self.output_dir / "recall_vs_visibility_overall.parquet")
        else:
            print("Error during recall_vs_visibility computation process. Exiting.")
        return

    def _calculate_recall_vs_visibility_per_class(self, updated_gt_df):
        recall_by_vis_per_class = self.evaluator.calculate_recall_vs_visibility(
        updated_gt_df=updated_gt_df,
        group_by_class=True
    )
        if recall_by_vis_per_class is not None:
            print("\n--- Recall vs. Visibility (Per Class) ---")

            print(recall_by_vis_per_class) 
            recall_by_vis_per_class.to_parquet(self.output_dir / "recall_vs_visibility_per_class.parquet", index=True)
        else:
            print("Error during recall_vs_visibility per class computation process. Exiting.")
        return

    def _calculate_recall_vs_image_attributes(self, val_images_df, updated_gt_df):
        val_images_df['image_id'] = val_images_df['image_id'].str.split('.').str[0]
        print(f"Loaded {len(val_images_df)} image attribute records.")

        attributes_to_analyze = ['timeofday', 'weather']

        recall_by_attr_overall = self.evaluator.calculate_recall_vs_image_attributes(
            val_images_df=val_images_df,
            updated_gt_df=updated_gt_df,
            attributes_to_analyze=attributes_to_analyze,
            group_by_class=False
        )
        if recall_by_attr_overall:
            print("\n--- Recall vs. Image Attributes (Overall) ---")
            for attribute, df in recall_by_attr_overall.items():
                if df is not None:
                    print(f"\nAttribute: {attribute}")
                    print(df.head())
                    df.to_parquet(self.output_dir / f"recall_vs_{attribute}_overall.parquet", index=False)
                else:
                    print(f"\nCalculation failed for attribute: {attribute}")
        else:
            print("Error during recall_vs_image_attributes computation process. Exiting.")
        return

    def _calculat_recall_vs_image_attributes_per_class(self, val_images_df, updated_gt_df):
        val_images_df['image_id'] = val_images_df['image_id'].str.split('.').str[0]
        print(f"Loaded {len(val_images_df)} image attribute records.")

        attributes_to_analyze = ['timeofday', 'weather']

        recall_by_attr_per_class = self.evaluator.calculate_recall_vs_image_attributes(
                val_images_df=val_images_df,
                updated_gt_df=updated_gt_df,
                attributes_to_analyze=attributes_to_analyze,
                group_by_class=True
            )
        if recall_by_attr_per_class:
            print("\n--- Recall vs. Image Attributes (Per Class) ---")
            for attribute, df in recall_by_attr_per_class.items():
                if df is not None:
                    print(f"\nAttribute: {attribute}")
                    print(df.head(15))
                    df.to_parquet(self.output_dir / f"recall_vs_{attribute}_per_class.parquet", index=False)
                else:
                    print(f"\nCalculation failed for attribute: {attribute}")
        else:
            print("Error during recall_vs_image_attributes per class computation process. Exiting.")
        return
    
    def _print_fn_failure_characs(self, fn_enriched_df):
        print("\n--- False Negative (FN) Analysis ---")

        # 1. FN Count per Class
        print("\nFN Counts per Class:")
        print(fn_enriched_df['category'].value_counts())

        # 2. FN Size Distribution (if bbox_area exists)
        if 'bbox_area' in fn_enriched_df.columns:
            print("\nFN Size (Area) Stats:")
            print(fn_enriched_df['bbox_area'].describe())

        # 3. FN Occlusion/Truncation Rates
        if 'occluded' in fn_enriched_df.columns:
            occ_rate_fn = (fn_enriched_df['occluded'].mean() * 100).round(2)
            print(f"\nFN Occlusion Rate: {occ_rate_fn}%")
        if 'truncated' in fn_enriched_df.columns:
            trunc_rate_fn = (fn_enriched_df['truncated'].mean() * 100).round(2)
            print(f"\nFN Truncation Rate: {trunc_rate_fn}%")

        # 4. FNs by TimeOfDay (if merged)
        if 'timeofday' in fn_enriched_df.columns:
            print("\nFN Counts per TimeOfDay:")
            print(fn_enriched_df['timeofday'].value_counts(normalize=True).round(3) * 100) # Show percentage

        # 5. FNs by Weather (if merged)
        if 'weather' in fn_enriched_df.columns:
            print("\nFN Counts per Weather:")
            print(fn_enriched_df['weather'].value_counts(normalize=True).round(3) * 100) # Show percentage
        
        return

    def _print_fp_failure_characs(self, fp_enriched_df):
        print("\n--- False Positive (FP) Analysis ---")
        # 1. FP Confidence Distribution
        if 'confidence' in fp_enriched_df.columns:
            print("\nFP Confidence Score Stats:")
            print(fp_enriched_df['confidence'].describe())

        # 2. FP Count per Class
        print("\nFP Counts per Predicted Class:")
        print(fp_enriched_df['category'].value_counts())

        # 3. FP Size Distribution (if calculated)
        if 'bbox_area' in fp_enriched_df.columns:
            print("\nFP Size (Area) Stats:")
            print(fp_enriched_df['bbox_area'].describe())

        # 4. FPs by TimeOfDay (if merged)
        if 'timeofday' in fp_enriched_df.columns:
            print("\nFP Counts per TimeOfDay:")
            print(fp_enriched_df['timeofday'].value_counts(normalize=True).round(3) * 100)

        # 5. FPs by Weather (if merged)
        if 'weather' in fp_enriched_df.columns:
            print("\nFP Counts per Weather:")
            print(fp_enriched_df['weather'].value_counts(normalize=True).round(3) * 100)
    
        return

    def _get_identify_failures(self, matched_preds_df, updated_gt_df, val_images_df):
        failure_results = self.evaluator.identify_failures(
        matched_preds_df=matched_preds_df,
        updated_gt_df=updated_gt_df 
    )

        if failure_results is not None:
            fp_df, fn_df = failure_results
            print(f"\n--- False Positives (Top 5 by Confidence) ---")
            print(fp_df.nlargest(5, 'confidence'))

            print(f"\n--- False Negatives (Sample) ---")
            print(fn_df.head())

            try:
                fp_save_path = self.output_dir / "false_positives.parquet"
                fn_save_path = self.output_dir / "false_negatives.parquet"
                fp_df.to_parquet(fp_save_path, index=False)
                fn_df.to_parquet(fn_save_path, index=False)
                print(f"Saved FP/FN dataframes to {self.output_dir}")
            except Exception as e:
                print(f"Error saving FP/FN dataframes: {e}")
        else:
            print("Could not identify failures.")
        
        val_images_df['image_id'] = val_images_df['image_id'].str.split('.').str[0]
        prep_results = self.helper.prepare_failure_analysis_data(fp_df, fn_df, val_images_df)

        if prep_results is None:
            print("Error preparing failure data. Skipping analysis.")
        else:
            fp_enriched_df, fn_enriched_df = prep_results
            print("Data prepared.")

            # --- Step 9: Analyze Failure Characteristics ---
            print("\n" + "="*10 + " Analyzing Failure Characteristics " + "="*10)

            if not fn_enriched_df.empty:
                out_path = self.output_dir / "fn_enriched_df.parquet"
                fn_enriched_df.to_parquet(out_path, index=False)
                self._print_fn_failure_characs(fn_enriched_df)
            else:
                print("\nNo False Negatives to analyze.")
            
            if not fp_enriched_df.empty:
                out_path =self.output_dir / "fp_enriched_df.parquet"
                fp_enriched_df.to_parquet(out_path, index=False)
                self._print_fp_failure_characs(fp_enriched_df)
            else:
                print("\nNo False Positives to analyze.")

    def __call__(self) -> Dict[str, str]:
        """
        Analyzes parsed GT data to calculate various statistics.

        Returns:
            Dict[str, str]: Dictionary of paths to the saved metrics/summary files.
        """
        print("GTMetricsAnalyzer: Starting ground truth metrics calculation...")
        
        
        val_images_df = pd.read_parquet(self.gt_val_images_path)

        self.predictions_df = self.helper.parse_yolo_predictions(
                                        predictions_dir=self.detected_labels_path,
                                        class_map=CLASS_MAP,
                                        image_width=IMAGE_WIDTH,
                                        image_height=IMAGE_HEIGHT
                                    )
        
        matched_preds_df, updated_gt_df = self._compute_match_predictions()
        self._compute_recall_vs_size(updated_gt_df)
        self._calulate_recall_vs_size_per_class(updated_gt_df)
        self._calculate_recall_vs_visibility(updated_gt_df)
        self._calculate_recall_vs_visibility_per_class(updated_gt_df)
        self._calculate_recall_vs_image_attributes(val_images_df, updated_gt_df)
        self._calculat_recall_vs_image_attributes_per_class(val_images_df, updated_gt_df)
        self._get_identify_failures(matched_preds_df, updated_gt_df, val_images_df)

        print("\nEvaluation matching finished. Next steps: Calculate performance breakdowns.")

        
        