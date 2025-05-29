import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Set

BDD_CLASSES = [
    "traffic light", "traffic sign", "car", "person", "bus",
    "truck", "rider", "bike", "motor", "train"
]

class BDDGroundTruthAnalyzer:
    """
    Analyzes the ground truth data from the BDD100K object detection dataset.

    Loads the parsed Parquet files and provides methods to calculate
    various statistics and insights required for dataset understanding.
    """

    def __init__(self, train_objects_path: Path, val_objects_path: Path,
                 train_images_path: Path, val_images_path: Path, 
                 output_dir_path: Path):
        """
        Initializes the analyzer by loading the datasets.

        Args:
            train_objects_path: Path to the training objects parquet file.
            val_objects_path: Path to the validation objects parquet file.
            train_images_path: Path to the training images parquet file.
            val_images_path: Path to the validation images parquet file.
        """
        self.train_objects_df = self._load_data(train_objects_path)
        self.val_objects_df = self._load_data(val_objects_path)
        self.train_images_df = self._load_data(train_images_path)
        self.val_images_df = self._load_data(val_images_path)

        self.bdd_classes = BDD_CLASSES

        self.train_merged_df = self._merge_data(self.train_objects_df, self.train_images_df)
        self.val_merged_df = self._merge_data(self.val_objects_df, self.val_images_df)

        print("Analyzer initialized and data loaded.")


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


    def _merge_data(self, objects_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
        """Merges object and image dataframes."""
        if objects_df.empty or images_df.empty:
            print("Warning: Cannot merge empty DataFrames.")
            return pd.DataFrame()
        try:
            if 'image_id' not in objects_df.columns or 'image_id' not in images_df.columns:
                print("Error: 'image_id' column missing in one or both DataFrames for merging.")
                return pd.DataFrame()
            merged_df = pd.merge(objects_df, images_df, on='image_id', how='left')
            print("Object and Image DataFrames merged successfully.")
            return merged_df
        except Exception as e:
            print(f"Error merging DataFrames: {e}")
            return pd.DataFrame()


    # --- Analysis Methods ---

    def get_instance_counts(self, include_total: bool = False) -> Optional[pd.DataFrame]:
        """
        Calculates and returns the instance counts per class for train and val sets.

        Args:
            include_total: If True, adds a 'total' row with sums.

        Returns:
            A DataFrame with classes as index and 'train_count', 'val_count' columns,
            or None if data is missing.
        """
        print("\nCalculating instance counts per class...")
        if self.train_objects_df.empty or self.val_objects_df.empty:
            print("Error: Object DataFrames are empty. Cannot calculate counts.")
            return None

        try:
            train_counts = self.train_objects_df['category'].value_counts()
            val_counts = self.val_objects_df['category'].value_counts()

            combined_counts_df = pd.DataFrame({
                'train_count': train_counts,
                'val_count': val_counts
            })
            combined_counts_df = combined_counts_df.reindex(self.bdd_classes, fill_value=0)
            combined_counts_df = combined_counts_df.fillna(0).astype(int) 
            combined_counts_df = combined_counts_df.sort_values(by='train_count', ascending=True)

            if include_total:
                 total_row = combined_counts_df.sum().rename('total')
                 combined_counts_df = pd.concat([combined_counts_df, pd.DataFrame(total_row).T.astype(int)])

            print("Instance counts calculation complete.")
            return combined_counts_df
        except KeyError as e:
            print(f"Error: Missing expected column '{e}' for instance count calculation.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during instance count calculation: {e}")
            return None


    def get_object_size_distribution(self, stats: List[str] = ['mean', 'median', 'min', 'max', 'std']) -> Optional[pd.DataFrame]:
        """
        Calculates descriptive statistics for object bounding box area per class.

        Args:
            stats: A list of strings representing pandas descriptive statistics
                   to calculate (e.g., 'mean', 'median', 'min', 'max', 'std',
                   'count', 'quantile').

        Returns:
            A DataFrame summarizing area statistics per class for train and val sets,
            with classes as index and multi-level columns (e.g., ('train', 'mean'),
            ('val', 'std')), or None if an error occurs. Returns stats rounded to
            sensible precision.
        """
        print(f"\nCalculating object size (area) distribution using stats: {stats}...")

        required_col = 'bbox_area'
        group_col = 'category'

        # --- Input Data Validation ---
        if self.train_objects_df.empty or self.val_objects_df.empty:
             print("Error: Object DataFrames are empty. Cannot calculate size distribution.")
             return None
        if required_col not in self.train_objects_df.columns or required_col not in self.val_objects_df.columns:
             print(f"Error: Required column '{required_col}' missing. Cannot calculate size distribution.")
             return None
        if group_col not in self.train_objects_df.columns or group_col not in self.val_objects_df.columns:
            print(f"Error: Grouping column '{group_col}' missing. Cannot calculate size distribution.")
            return None
        # --- End Validation ---

        try:
            train_size_stats = self.train_objects_df.groupby(group_col)[required_col].agg(stats)

            val_size_stats = self.val_objects_df.groupby(group_col)[required_col].agg(stats)


            combined_stats = pd.concat(
                {'train': train_size_stats, 'val': val_size_stats},
                axis=1 
            )

            combined_stats = combined_stats.reindex(self.bdd_classes).fillna(0) 

            combined_stats = combined_stats.round(2) 

            print("Object size distribution calculation complete.")
            return combined_stats

        except ValueError as e:
            print(f"Error: Invalid statistic requested in 'stats' list: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during size distribution calculation: {e}")
            return None


    def get_object_location_data(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
         """
         Extracts object center coordinates (cx, cy) and category for heatmap analysis.

         Returns:
             A tuple containing two pandas DataFrames:
             1. train_locations_df: DataFrame with 'category', 'bbox_cx', 'bbox_cy' for train set.
             2. val_locations_df: DataFrame with 'category', 'bbox_cx', 'bbox_cy' for val set.
             Returns None if an error occurs or data is missing.
         """
         print("\nExtracting object location data (category, cx, cy)...")

         cols_to_extract = ['category', 'bbox_cx', 'bbox_cy']

         if self.train_objects_df.empty or self.val_objects_df.empty:
             print("Error: Object DataFrames are empty. Cannot extract location data.")
             return None

         if not all(col in self.train_objects_df.columns for col in cols_to_extract) or \
            not all(col in self.val_objects_df.columns for col in cols_to_extract):
             print(f"Error: One or more required columns ({cols_to_extract}) missing for location data extraction.")
             return None

         try:
             train_locations_df = self.train_objects_df[cols_to_extract].copy()
             val_locations_df = self.val_objects_df[cols_to_extract].copy()

             train_locations_df.dropna(subset=['bbox_cx', 'bbox_cy'], inplace=True)
             val_locations_df.dropna(subset=['bbox_cx', 'bbox_cy'], inplace=True)


             print(f"Object location data extracted: {len(train_locations_df)} train points, {len(val_locations_df)} val points.")
             return train_locations_df, val_locations_df

         except KeyError as e:
             print(f"Error: Missing expected column '{e}' during location data extraction.")
             return None
         except Exception as e:
             print(f"An unexpected error occurred during location data extraction: {e}")
             return None


    def get_occlusion_truncation_rates(self) -> Optional[pd.DataFrame]:
         """
         Calculates the percentage of occluded and truncated objects per class
         for both the training and validation sets.

         Returns:
             A DataFrame with classes as index and columns for train/val
             occlusion/truncation percentages (e.g., 'train_occluded_pct',
             'val_truncated_pct'), or None if an error occurs.
             Returns percentages rounded to sensible precision.
         """
         print("\nCalculating occlusion and truncation rates per class...")

         cols_to_check = ['category', 'occluded', 'truncated']
         group_col = 'category'
         rate_cols = ['occluded', 'truncated']

         if self.train_objects_df.empty or self.val_objects_df.empty:
             print("Error: Object DataFrames are empty. Cannot calculate rates.")
             return None

         if not all(col in self.train_objects_df.columns for col in cols_to_check) or \
            not all(col in self.val_objects_df.columns for col in cols_to_check):
             print(f"Error: One or more required columns ({cols_to_check}) missing for rate calculation.")
             return None

         for df_name, df in [('Train', self.train_objects_df), ('Validation', self.val_objects_df)]:
             for col in rate_cols:
                 if not pd.api.types.is_bool_dtype(df[col]) and not pd.api.types.is_numeric_dtype(df[col]):
                      print(f"Warning: Column '{col}' in {df_name} DataFrame is not boolean or numeric.")
                      print("         Ensure it's parsed correctly (True/False or 1/0) for accurate percentage calculation.")
                      try:
                          df[col] = df[col].astype(pd.BooleanDtype())
                          print(f"         Attempted conversion of '{col}' to boolean.")
                      except (TypeError, ValueError):
                           print(f"Error: Could not convert column '{col}' to boolean. Cannot calculate rates accurately.")
                           return None


         try:
             train_rates = (self.train_objects_df.groupby(group_col)[rate_cols]
                            .mean()
                            .multiply(100)
                            .add_suffix('_pct') 
                            .add_prefix('train_'))

             val_rates = (self.val_objects_df.groupby(group_col)[rate_cols]
                          .mean()
                          .multiply(100)
                          .add_suffix('_pct')  
                          .add_prefix('val_'))

             combined_rates = pd.concat([train_rates, val_rates], axis=1)

             combined_rates = combined_rates.reindex(self.bdd_classes).fillna(0) 

             combined_rates = combined_rates.round(2) 


             print("Occlusion/truncation rate calculation complete.")
             return combined_rates

         except KeyError as e:
             print(f"Error: Missing expected column '{e}' during rate calculation.")
             return None
         except Exception as e:
             print(f"An unexpected error occurred during occlusion/truncation rate calculation: {e}")
             return None



    # --- Methods for Finding Interesting Samples ---

    def _get_filtered_image_ids(
        self,
        target_split: str = 'train',
        weather_conditions: Optional[List[str]] = None,
        time_conditions: Optional[List[str]] = None,
        scene_conditions: Optional[List[str]] = None
        ) -> Optional[Set[str]]:
        """Helper to get image IDs based on image-level attributes."""
        if target_split == 'train':
            img_df = self.train_images_df
        elif target_split == 'val':
            img_df = self.val_images_df
        else:
            print(f"Error: Invalid target_split '{target_split}'. Choose 'train' or 'val'.")
            return None

        if img_df.empty:
            print(f"Error: Image DataFrame for '{target_split}' split is empty.")
            return set()

        mask = pd.Series([True] * len(img_df), index=img_df.index)
        if weather_conditions and 'weather' in img_df.columns:
            mask &= img_df['weather'].isin(weather_conditions)
        if time_conditions and 'timeofday' in img_df.columns:
            mask &= img_df['timeofday'].isin(time_conditions)
        if scene_conditions and 'scene' in img_df.columns:
             mask &= img_df['scene'].isin(scene_conditions)

        return set(img_df[mask]['image_id'])


    def find_multi_condition_images(
        self,
        weather_conditions: List[str],
        time_condition: str,
        required_categories: List[str],
        target_split: str = 'train',
        max_samples: Optional[int] = 20
        ) -> Optional[List[str]]:
        """
        Finds images matching specific weather, time, AND containing certain objects..
        """
        print(f"\nFinding images: weather in {weather_conditions}, timeofday='{time_condition}', "
              f"must contain all {required_categories} (split: {target_split})...")

        # 1. Filter images by weather and time
        candidate_image_ids = self._get_filtered_image_ids(
            target_split=target_split,
            weather_conditions=weather_conditions,
            time_conditions=[time_condition] 
        )

        if candidate_image_ids is None or not candidate_image_ids:
            print("No images found matching the specified weather/time conditions.")
            return []

        # 2. Filter objects belonging to these candidate images
        if target_split == 'train':
            objects_df = self.train_objects_df
        else:
            objects_df = self.val_objects_df

        if objects_df.empty:
            print("Error: Objects DataFrame is empty.")
            return []
        if 'image_id' not in objects_df.columns or 'category' not in objects_df.columns:
            print("Error: Missing 'image_id' or 'category' column in objects DataFrame.")
            return []

        filtered_objects = objects_df[objects_df['image_id'].isin(candidate_image_ids)]

        if filtered_objects.empty:
            print("No objects found in images matching the specified weather/time conditions.")
            return []

        # 3. Check if each candidate image contains all required categories
        required_set = set(required_categories)
        image_groups = filtered_objects.groupby('image_id')['category'].apply(set)
        matching_image_ids = image_groups[image_groups.apply(lambda cats: required_set.issubset(cats))].index.tolist()

        # 4. Limit samples
        if max_samples is not None and len(matching_image_ids) > max_samples:
            final_list = matching_image_ids[:max_samples]
            print(f"Found {len(matching_image_ids)} matching images, returning first {max_samples}.")
        else:
            final_list = matching_image_ids
            print(f"Found {len(final_list)} matching images.")

        return final_list


    def find_images_with_small_objects(
        self,
        max_area: float,
        category: Optional[str] = None,
        target_split: str = 'train',
        max_samples: Optional[int] = 20
        ) -> Optional[List[str]]:
        """
        Finds images containing objects smaller than a given area threshold..
        """
        print(f"\nFinding images with objects (Category: {category or 'any'}) smaller than area {max_area} (split: {target_split})...")

        if target_split == 'train':
            objects_df = self.train_objects_df
        else:
            objects_df = self.val_objects_df

        if objects_df.empty:
            print("Error: Objects DataFrame is empty.")
            return []
        if 'bbox_area' not in objects_df.columns or 'image_id' not in objects_df.columns:
             print("Error: Missing 'bbox_area' or 'image_id'. Cannot find small objects.")
             return []

        mask = objects_df['bbox_area'] < max_area
        if category and 'category' in objects_df.columns:
            mask &= (objects_df['category'] == category)

        filtered_objects = objects_df[mask]

        if filtered_objects.empty:
             print("No objects found matching the small size criteria.")
             return []

        matching_image_ids = filtered_objects['image_id'].unique().tolist()

        if max_samples is not None and len(matching_image_ids) > max_samples:
            final_list = matching_image_ids[:max_samples]
            print(f"Found {len(matching_image_ids)} images with small objects, returning first {max_samples}.")
        else:
            final_list = matching_image_ids
            print(f"Found {len(final_list)} images with small objects.")

        return final_list


    def find_images_with_occluded_or_truncated_objects(
        self,
        category: Optional[str] = None,
        target_split: str = 'train',
        max_samples: Optional[int] = 20
        ) -> Optional[List[str]]:
        """
        Finds images containing objects marked as occluded OR truncated.
        """
        print(f"\nFinding images with occluded/truncated objects (Category: {category or 'any'}) (split: {target_split})...")

        if target_split == 'train':
            objects_df = self.train_objects_df
        else:
            objects_df = self.val_objects_df

        if objects_df.empty:
            print("Error: Objects DataFrame is empty.")
            return []

        cols_to_check = ['image_id', 'occluded', 'truncated']
        if not all(col in objects_df.columns for col in cols_to_check):
             print(f"Error: Missing one or more columns: {cols_to_check}. Cannot find occluded/truncated.")
             return []

        try:
            objects_df['occluded'] = objects_df['occluded'].astype(pd.BooleanDtype())
            objects_df['truncated'] = objects_df['truncated'].astype(pd.BooleanDtype())
            mask = (objects_df['occluded'] == True) | (objects_df['truncated'] == True)
        except Exception as e:
             print(f"Error processing occluded/truncated columns (ensure they are boolean): {e}")
             return []

        if category and 'category' in objects_df.columns:
            mask &= (objects_df['category'] == category)

        filtered_objects = objects_df[mask]

        if filtered_objects.empty:
             print("No occluded or truncated objects found matching the criteria.")
             return []

        matching_image_ids = filtered_objects['image_id'].unique().tolist()

        if max_samples is not None and len(matching_image_ids) > max_samples:
            final_list = matching_image_ids[:max_samples]
            print(f"Found {len(matching_image_ids)} images with occluded/truncated objects, returning first {max_samples}.")
        else:
            final_list = matching_image_ids
            print(f"Found {len(final_list)} images with occluded/truncated objects.")

        return final_list


    def find_images_with_high_density(
        self,
        min_count: int,
        category: Optional[str] = None,
        target_split: str = 'train',
        max_samples: Optional[int] = 20
        ) -> Optional[List[str]]:
        """
        Finds images containing more than 'min_count' objects.
        If 'category' is specified, counts only objects of that category.
        Otherwise, counts total objects (of the 10 target classes).
        """
        count_desc = f"{category}" if category else "total"
        print(f"\nFinding images with > {min_count} '{count_desc}' objects (split: {target_split})...")

        if target_split == 'train':
            objects_df = self.train_objects_df
        else:
            objects_df = self.val_objects_df

        if objects_df.empty:
            print("Error: Objects DataFrame is empty.")
            return []
        if 'image_id' not in objects_df.columns:
            print("Error: Missing 'image_id'. Cannot calculate density.")
            return []

        temp_df = objects_df
        if category:
            if 'category' not in objects_df.columns:
                 print(f"Error: 'category' column missing. Cannot filter density by category.")
                 return []
            temp_df = objects_df[objects_df['category'] == category]

        if temp_df.empty:
             print(f"No objects found for category '{category}' to calculate density.")
             return []

        image_counts = temp_df.groupby('image_id').size()

        high_density_image_ids = image_counts[image_counts > min_count].index.tolist()


        if not high_density_image_ids:
             print("No images found matching the high density criteria.")
             return []

        if max_samples is not None and len(high_density_image_ids) > max_samples:
            final_list = high_density_image_ids[:max_samples]
            print(f"Found {len(high_density_image_ids)} high density images, returning first {max_samples}.")
        else:
            final_list = high_density_image_ids
            print(f"Found {len(final_list)} high density images.")

        return final_list    


    def __call__(self, output_dir_path):
        outdir_path = os.path.join(output_dir_path, "gt_analysis")
        outdir_path = Path(outdir_path)
        os.makedirs(outdir_path, exist_ok=True)

        instance_counts = self.get_instance_counts(include_total=True)
        if instance_counts is not None:
            print("\n--- Instance Counts ---")
            print(instance_counts)
            file_path = os.path.join(outdir_path, "instance_counts_category_gt.parquet")
            instance_counts.to_parquet(file_path)


        ### 2. Size Distribution
        size_stats = self.get_object_size_distribution(stats=['mean', 'median', 'min', 'max', 'std', 'count'])
        if size_stats is not None:
            print("\n--- Object Size (Area) Stats ---")
            print(size_stats)
            file_path = os.path.join(outdir_path, "object_size_area_stats_gt.parquet")
            size_stats.to_parquet(file_path)


        ## 3. Location Data (ready for heatmap plotting)
        location_data = self.get_object_location_data()
        if location_data is not None:
            train_loc_df, val_loc_df = location_data
            print(f"\n--- Object Location Data Extracted ---")
            print(f"Train location points extracted: {len(train_loc_df)}")
            print(f"Validation location points extracted: {len(val_loc_df)}")


            train_loc_path = os.path.join(outdir_path, "bdd_train_object_locations.parquet")
            val_loc_path = os.path.join(outdir_path, "bdd_val_object_locations.parquet")



            try:
                print(f"Saving train location data to {train_loc_path}...")
                train_loc_df.to_parquet(train_loc_path, index=False)
                print(f"Saving validation location data to {val_loc_path}...")
                val_loc_df.to_parquet(val_loc_path, index=False)
                print("Location data saved successfully.")
            except Exception as e:
                print(f"Error saving location data: {e}")

        ## 4. Occlusion/Truncation Rates
        occ_trunc_rates = self.get_occlusion_truncation_rates()
        if occ_trunc_rates is not None:
            print("\n--- Occlusion/Truncation Rates (%) ---")
            print(occ_trunc_rates)
            # Save or visualize
            file_path = os.path.join(outdir_path, "occ_trunc_rates_gt.parquet")
            occ_trunc_rates.to_parquet(file_path)

        interesting_images_analysis = False
        print("")
    

        # To save all the interesting image ids
        if interesting_images_analysis:
            print("=-="*10)
            print(f" Find Interesting Samples Module Start ...")
            interesting_images_ids = set()

            # Condition 1: Rainy/Foggy Night with Pedestrians, Traffic Signs, Traffic Lights
            multi_cond_ids = self.find_multi_condition_images(
                weather_conditions=['rainy', 'foggy'],
                time_condition='night',
                required_categories=['person', 'traffic sign', 'traffic light'],
                target_split='val',
                max_samples=10
            )
            if multi_cond_ids:
                print(f"Condition 1 Samples: {multi_cond_ids}")

            # Condition 2: Images with very small pedestrians
            small_ped_ids = self.find_images_with_small_objects(
                max_area=500, # Example threshold in pixels^2
                category='person',
                target_split='val',
                max_samples=10
            )
            if small_ped_ids:
                print(f"Condition 2 Samples (Small Pedestrians): {small_ped_ids}")

            # Condition 3: Images with occluded or truncated cars
            occ_trunc_car_ids = self.find_images_with_occluded_or_truncated_objects(
                category='car',
                target_split='val',
                max_samples=10
            )
            if occ_trunc_car_ids:
                print(f"Condition 3 Samples (Occluded/Truncated Cars): {occ_trunc_car_ids}")

            # Condition 4: Images with high density of cars
            high_density_car_ids = self.find_images_with_high_density(
                min_count=15, # Example threshold: more than 15 cars
                category='car',
                target_split='val',
                max_samples=10
            )
            if high_density_car_ids:
                print(f"Condition 4 Samples (High Density Cars): {high_density_car_ids}")

            # Condition 4 (Alternative): Images with high total object density
            high_density_total_ids = self.find_images_with_high_density(
                min_count=25, # Example threshold: more than 25 objects total
                category=None,
                target_split='val',
                max_samples=10
            )
            if high_density_total_ids:
                print(f"Condition 4 Samples (High Total Density): {high_density_total_ids}")
            print("")
            print("=-="*10)

            interesting_images_ids.update(multi_cond_ids, small_ped_ids, occ_trunc_car_ids, high_density_car_ids, high_density_total_ids)

            interesting_images_ids = list(interesting_images_ids)

            print(f"Interesting Image ids (combined): {interesting_images_ids}")

        else:
            print(f"--- Interesting Images Analysis Skipped ---")

        print("\nGT analyser finished.")
        
      
class GTMetricsAnalyzer:
    def __init__(self, module_config: Dict[str, Any], 
                 project_config: Dict[str, Any],
                 main_output_dir: str):
        self.module_config = module_config
        self.main_output_dir = Path(main_output_dir)
        intermediate_subdir_name = self.module_config.get('intermediate_data_subdir', '02_gt_analysis_metrics')
        self.output_path_for_metrics = self.main_output_dir / intermediate_subdir_name
        self.output_path_for_metrics = Path(self.output_path_for_metrics)
        os.makedirs(self.output_path_for_metrics, exist_ok=True)

        self.project_path_config = project_config
        print(f"GTMetricsAnalyzer initialized. Outputs will be saved to: {self.output_path_for_metrics}")

    def __call__(self) -> Dict[str, str]:
        """
        Analyzes parsed GT data to calculate various statistics.

        Returns:
            Dict[str, str]: Dictionary of paths to the saved metrics/summary files.
        """
        print("GTMetricsAnalyzer: Starting ground truth metrics calculation...")

        train_objects_df_path = Path(self.project_path_config.get("train_objects_df_path"))
        train_images_df_path = Path(self.project_path_config.get("train_images_df_path"))
        val_objects_df_path = Path(self.project_path_config.get("val_objects_df_path"))
        val_images_df_path = Path(self.project_path_config.get("val_images_df_path"))

        
        gt_analyzer =BDDGroundTruthAnalyzer(train_objects_df_path, val_objects_df_path,
                               train_images_df_path, val_images_df_path,
                               self.output_path_for_metrics)
        
        gt_analyzer(self.output_path_for_metrics)

        print("GTMetricsAnalyzer: Metrics calculation complete.")

        
        return
