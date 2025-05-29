import math
import cv2, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any 

BDD_CLASSES = [
    "car", "traffic sign", "traffic light", "person", "truck", "bus",
    "bike", "rider", "motor", "train"
]

class BDDVisualizer:
    """
    Handles visualization of BDD dataset analysis results.
    Loads data from saved files (e.g., parquet) or uses pre-loaded DataFrames.
    """
    def __init__(self, plot_style: str = 'seaborn-v0_8-talk'):
        """
        Initializes the visualizer.

        Args:
            plot_style: Matplotlib/Seaborn style to use for plots.
        """
        try:
            plt.style.use(plot_style)
            print(f"Using plot style: {plot_style}")
        except OSError:
            print(f"Warning: Plot style '{plot_style}' not found. Using default.")
        plt.rcParams['figure.figsize'] = (12, 7)
        plt.rcParams['figure.dpi'] = 100


    def _load_data_if_needed(
        self,
        data: Optional[Union[pd.DataFrame, Path]],
        expected_columns: Optional[list] = None
        ) -> Optional[pd.DataFrame]:
        """Helper to load data from path if not already a DataFrame."""
        if data is None:
            print("Error: No data source (DataFrame or Path) provided.")
            return None

        if isinstance(data, pd.DataFrame):
            df = data
            print("Using provided DataFrame.")
        elif isinstance(data, Path):
            print(f"Attempting to load data from file: {data}")
            if not data.exists():
                print(f"Error: File not found at {data}")
                return None
            try:
                df = pd.read_parquet(data)
                print(f"Successfully loaded {len(df)} rows.")
            except Exception as e:
                print(f"Error loading file {data}: {e}")
                return None
        else:
            print("Error: Input data must be a pandas DataFrame or a pathlib.Path object.")
            return None

        if df.empty:
             print("Warning: Loaded DataFrame is empty.")

        if expected_columns:
            if not all(col in df.columns for col in expected_columns):
                print(f"Error: Loaded DataFrame missing one or more expected columns: {expected_columns}")
                return None

        return df


    def plot_instance_counts(
        self,
        counts_data: Union[pd.DataFrame, Path],
        use_log_scale: bool = False,
        save_path: Optional[Path] = None
        ):
        """
        Generates and optionally saves a grouped bar chart of instance counts
        per class for train vs validation sets.

        Loads data if a path is provided.

        Args:
            counts_data: DataFrame with classes as index and 'train_count',
                         'val_count' columns, OR a Path to the parquet/csv file
                         containing this data.
            use_log_scale: If True, use a logarithmic scale for the y-axis.
            save_path: If provided, saves the plot to this path.
        """
        print("\nAttempting to generate Instance Count Plot...")

        # --- Load data if path provided ---
        counts_df = self._load_data_if_needed(
            counts_data,
            expected_columns=['train_count', 'val_count']
        )

        if counts_df is None or counts_df.empty:
            print("Error: Could not load or use instance counts data. Cannot plot.")
            return
        # --- End Data Loading ---

        plot_df = counts_df.drop('total', errors='ignore')


        # --- Plotting --- (Same plotting code as before)
        fig, ax = plt.subplots()
        plot_df_melted = plot_df.reset_index().rename(columns={'index': 'category'})
        if 'category' not in plot_df_melted.columns and plot_df.index.name:
             plot_df_melted = plot_df.reset_index().rename(columns={plot_df.index.name: 'category'})
        elif 'category' not in plot_df_melted.columns:
             print("Error: Cannot identify category column after melting.")
             plt.close(fig)
             return

        plot_df_melted = plot_df_melted.melt(
            id_vars='category', var_name='Split', value_name='Count'
        )
        plot_df_melted['Split'] = plot_df_melted['Split'].replace({
             'train_count': 'Train', 'val_count': 'Validation'
        })

        sns.barplot(
            data=plot_df_melted, x='category', y='Count', hue='Split',
            ax=ax, palette='viridis', order=plot_df.index
        )
        ax.set_title('Instance Count per Class (Train vs Validation)')
        ax.set_xlabel('Object Class')
        ax.set_ylabel('Number of Instances' + (' (Log Scale)' if use_log_scale else ''))
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylim(bottom=1)
        plt.tight_layout()
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Dataset Split')

        if save_path:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving plot to {save_path}: {e}")
        else:
            plt.show()
        plt.close(fig)
 

    def plot_size_distribution(
        self,
        train_objects_data: Union[pd.DataFrame, Path],
        val_objects_data: Union[pd.DataFrame, Path],
        use_log_scale: bool = True, 
        show_outliers: bool = False,
        save_path: Optional[Path] = None
        ):
        """
        Generates grouped box plots comparing object bbox_area distribution
        per class for train vs validation sets.

        Loads data from object parquet files if paths are provided.

        Args:
            train_objects_data: DataFrame or Path to train objects parquet file
                                (must contain 'category' and 'bbox_area').
            val_objects_data: DataFrame or Path to validation objects parquet file
                              (must contain 'category' and 'bbox_area').
            use_log_scale: If True (recommended), use a logarithmic scale for the y-axis (area).
            show_outliers: If False, outliers will not be drawn on the box plots.
            save_path: If provided, saves the plot to this path.
        """
        print("\nGenerating Object Size (Area) Distribution Plot...")

        required_cols = ['category', 'bbox_area']

        # --- Load Data ---
        train_df = self._load_data_if_needed(train_objects_data, expected_columns=required_cols)
        val_df = self._load_data_if_needed(val_objects_data, expected_columns=required_cols)

        if train_df is None or val_df is None or train_df.empty or val_df.empty:
            print("Error: Could not load or use train/validation object data. Cannot plot size distribution.")
            return
        # --- End Data Loading ---

        try:
            print("Calculating median areas for sorting plot order...")
            median_sizes = train_df.groupby('category', observed=False)['bbox_area'].median()

            sorted_categories = median_sizes.fillna(0).sort_values().index.tolist()
            print(f"Category order for plot: {sorted_categories}")
        except KeyError:
            print("Warning: Could not calculate medians for sorting. Using default order.")
            sorted_categories = None 
        except Exception as e:
            print(f"Warning: Error during median calculation for sorting: {e}. Using default order.")
            sorted_categories = None

        # --- Prepare Data for Plotting ---
        train_df['split'] = 'Train'
        val_df['split'] = 'Validation'

        combined_df = pd.concat([train_df, val_df], ignore_index=True)

        original_count = len(combined_df)
        combined_df = combined_df[combined_df['bbox_area'] > 0].copy()
        if len(combined_df) < original_count:
             print(f"Warning: Removed {original_count - len(combined_df)} objects with non-positive area.")

        if combined_df.empty:
             print("Error: No objects with positive area found.")
             return

        # --- Plotting ---
        try:
            fig, ax = plt.subplots(figsize=(14, 8)) 

            sns.boxplot(
                data=combined_df,
                x='category',
                y='bbox_area',
                hue='split',
                ax=ax,
                order=sorted_categories,
                palette='viridis',
                showfliers=show_outliers
            )

            ax.set_title('Object Area Distribution per Class (Train vs Validation)')
            ax.set_xlabel('Object Class')
            ax.set_ylabel('Bounding Box Area (pixels²)' + (' (Log Scale)' if use_log_scale else ''))
            ax.tick_params(axis='x', rotation=45, right=True)

            if use_log_scale:
                ax.set_yscale('log')

            plt.tight_layout()
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.legend(title='Dataset Split')

            # --- Save or Show ---
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

            plt.close(fig)

        except Exception as e:
            print(f"Error during plotting size distribution: {e}")
            if 'fig' in locals(): 
                 plt.close(fig)


    def plot_location_heatmap(
        self,
        train_location_data: Union[pd.DataFrame, Path],
        val_location_data: Union[pd.DataFrame, Path],
        image_width: int = 1280,
        image_height: int = 720,
        bins: int = 50,
        cmap: str = 'inferno', 
        save_path_prefix: Optional[Path] = None
        ):
        """
        Generates and optionally saves 2D heatmaps showing the spatial
        distribution density of object center points (cx, cy).

        Generates separate heatmaps for train and validation sets.

        Args:
            train_location_data: DataFrame or Path to train location parquet file
                                 (must contain 'bbox_cx', 'bbox_cy').
            val_location_data: DataFrame or Path to validation location parquet file
                               (must contain 'bbox_cx', 'bbox_cy').
            image_width: The width of the image frame (default: 1280 for BDD).
            image_height: The height of the image frame (default: 720 for BDD).
            bins: The number of bins to use for histogram calculation along each axis.
            cmap: The matplotlib colormap to use for the heatmap.
            save_path_prefix: If provided, saves the plots using this prefix
                              (e.g., prefix_train.png, prefix_val.png).
        """
        print("\nGenerating Object Location Heatmaps...")

        required_cols = ['bbox_cx', 'bbox_cy']

        # --- Load Data ---
        train_df = self._load_data_if_needed(train_location_data, expected_columns=required_cols)
        val_df = self._load_data_if_needed(val_location_data, expected_columns=required_cols)

        if train_df is None or val_df is None or train_df.empty or val_df.empty:
            print("Error: Could not load or use train/validation location data. Cannot plot heatmaps.")
            return
        # --- End Data Loading ---

        # --- Define common plotting function ---
        def create_heatmap(ax, x_coords, y_coords, title):
            valid_mask = (
                x_coords.between(0, image_width, inclusive='both') &
                y_coords.between(0, image_height, inclusive='both') &
                x_coords.notna() & y_coords.notna()
            )
            x_coords_valid = x_coords[valid_mask]
            y_coords_valid = y_coords[valid_mask]

            if x_coords_valid.empty:
                 print(f"Warning: No valid coordinates found for {title}. Skipping plot.")
                 ax.set_title(f"{title}\n(No valid data)")
                 ax.set_xticks([])
                 ax.set_yticks([])
                 return None

            H, xedges, yedges = np.histogram2d(
                x_coords_valid, y_coords_valid,
                bins=bins,
                range=[[0, image_width], [0, image_height]]
            )

            H = H.T

            im = ax.imshow(
                 H,
                 interpolation='nearest',
                 origin='upper',
                 extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]], 
                 cmap=cmap
            )

            ax.set_title(title)
            ax.set_xlabel('Image Width (pixels)')
            ax.set_ylabel('Image Height (pixels)')
            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)

            return im

        # --- Create Plots ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        im_train = create_heatmap(axes[0], train_df['bbox_cx'], train_df['bbox_cy'], 'Train Set Object Location Density')

        im_val = create_heatmap(axes[1], val_df['bbox_cx'], val_df['bbox_cy'], 'Validation Set Object Location Density')

        last_im = im_val if im_val is not None else im_train
        if last_im:
             fig.colorbar(last_im, ax=axes, label='Log(1 + Object Count per Bin)', shrink=0.7)

        # --- Save or Show ---
        if save_path_prefix:
            try:
                save_path_fig = save_path_prefix.parent / f"{save_path_prefix.name}_comparison.png"
                save_path_fig.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path_fig, bbox_inches='tight')
                print(f"Comparison heatmap plot saved to {save_path_fig}")

            except Exception as e:
                print(f"Error saving heatmap plot: {e}")
        else:
            plt.show()

        plt.close(fig)


    def plot_location_heatmap_per_class(
        self,
        train_location_data: Union[pd.DataFrame, Path],
        val_location_data: Union[pd.DataFrame, Path],
        image_width: int = 1280,
        image_height: int = 720,
        bins: int = 50,
        cmap: str = 'inferno',
        save_path: Optional[Path] = None 
        ):
        """
        Generates and optionally saves a grid of 2D heatmaps showing the spatial
        distribution density of object center points (cx, cy) PER CLASS.

        Generates side-by-side heatmaps for train and validation sets for each class.

        Args:
            train_location_data: DataFrame or Path to train location parquet file
                                 (must contain 'category', 'bbox_cx', 'bbox_cy').
            val_location_data: DataFrame or Path to validation location parquet file
                               (must contain 'category', 'bbox_cx', 'bbox_cy').
            image_width: The width of the image frame (default: 1280 for BDD).
            image_height: The height of the image frame (default: 720 for BDD).
            bins: The number of bins to use for histogram calculation along each axis.
            cmap: The matplotlib colormap to use for the heatmap.
            save_path: If provided, saves the entire grid plot to this single path.
        """
        print("\nGenerating Per-Class Object Location Heatmaps...")

        required_cols = ['category', 'bbox_cx', 'bbox_cy']

        # --- Load Data ---
        train_df = self._load_data_if_needed(train_location_data, expected_columns=required_cols)
        val_df = self._load_data_if_needed(val_location_data, expected_columns=required_cols)

        if train_df is None or val_df is None or train_df.empty or val_df.empty:
            print("Error: Could not load or use train/validation location data. Cannot plot heatmaps.")
            return
        # --- End Data Loading ---

        # --- Define helper for individual heatmap ---
        def create_heatmap(ax, x_coords, y_coords, title, class_name):
            valid_mask = (
                x_coords.between(0, image_width, inclusive='both') &
                y_coords.between(0, image_height, inclusive='both') &
                x_coords.notna() & y_coords.notna()
            )
            x_coords_valid = x_coords[valid_mask]
            y_coords_valid = y_coords[valid_mask]

            if x_coords_valid.empty:
                print(f"Warning: No valid coordinates found for {title} - Class '{class_name}'. Skipping subplot.")
                ax.set_title(f"{title}\n(No data)")
                ax.set_xticks([])
                ax.set_yticks([])
                return None

            H, xedges, yedges = np.histogram2d(
                x_coords_valid, y_coords_valid,
                bins=bins, range=[[0, image_width], [0, image_height]]
            )
            H = H.T 

            im = ax.imshow(
                 np.log1p(H),
                 interpolation='nearest', origin='upper',
                 extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]],
                 cmap=cmap
            )
            ax.set_title(title)
            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)
            return im

        # --- Create Plot Grid ---
        num_classes = len(BDD_CLASSES)
        ncols = 4 
        nrows = num_classes
        fig_height = nrows * 3.5 
        fig_width = ncols * 4    

        ncols = 2
        fig_width = 10 
        fig, axes = plt.subplots(nrows=num_classes, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)
      

        print(f"Generating heatmap grid for {num_classes} classes...")

        for i, category in enumerate(BDD_CLASSES):
            ax_train = axes[i, 0]
            ax_val = axes[i, 1]

            train_cat_df = train_df[train_df['category'] == category]
            val_cat_df = val_df[val_df['category'] == category]

            im_train = create_heatmap(ax_train, train_cat_df['bbox_cx'], train_cat_df['bbox_cy'], f"Train: '{category}'", category)

            im_val = create_heatmap(ax_val, val_cat_df['bbox_cx'], val_cat_df['bbox_cy'], f"Validation: '{category}'", category)

            if i < num_classes - 1: 
                 ax_train.set_xlabel('')
                 ax_val.set_xlabel('')
                 ax_train.set_xticklabels([])
                 ax_val.set_xticklabels([])



        fig.suptitle("Per-Class Object Location Density (Train vs Validation)", fontsize=16, y=1.01)

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        if save_path:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Per-class heatmap grid saved to {save_path}")
            except Exception as e:
                print(f"Error saving per-class heatmap plot: {e}")
        else:
            plt.show()

        plt.close(fig)


    def plot_occlusion_rates(
        self,
        rates_data: Union[pd.DataFrame, Path], 
        save_path_prefix: Optional[Path] = None
        ):
        """
        Generates SORTED grouped bar charts comparing occlusion and truncation rates (%)
        per class for train vs validation sets, using the pre-calculated summary table.

        Saves two plots if save_path_prefix is provided:
        - {prefix}_occlusion_sorted.png
        - {prefix}_truncation_sorted.png

        Args:
            rates_data: DataFrame or Path to the summary rates parquet/csv file.
                        Expected columns like 'train_occluded_pct', 'val_truncated_pct'.
            save_path_prefix: If provided, saves the plots using this prefix.
        """
        print("\nGenerating Sorted Occlusion/Truncation Rate Plots...")

        expected_cols = [
            'train_occluded_pct', 'train_truncated_pct',
            'val_occluded_pct', 'val_truncated_pct'
        ]
        # --- Load data ---
        rates_df = self._load_data_if_needed(rates_data, expected_columns=expected_cols)
        if rates_df is None or rates_df.empty:
            print("Error: Could not load or use occlusion/truncation rates data. Cannot plot.")
            return
        # --- End Data Loading ---

        # --- Helper function (same as before) ---
        def create_rate_plot(data_subset, title, ylabel, save_path):
             if not isinstance(data_subset.index, pd.CategoricalIndex):
                  data_subset.index = pd.Categorical(data_subset.index, categories=data_subset.index, ordered=True)

             try:
                 ax = data_subset.plot(
                     kind='bar', figsize=(12, 7), grid=False, rot=45, colormap='plasma'
                 )
                 ax.set_title(title)
                 ax.set_xlabel('Object Class')
                 ax.set_ylabel(ylabel)
                 ax.legend(title='Dataset Split')
                 ax.set_xticklabels(data_subset.index, rotation=45, ha='right')
                 ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                 ax.xaxis.grid(False)
                 ax.set_ylim(0, 100)
                 plt.tight_layout()
                 if save_path:
                     save_path.parent.mkdir(parents=True, exist_ok=True)
                     plt.savefig(save_path, bbox_inches='tight')
                     print(f"Plot saved to {save_path}")
                 else:
                     plt.show()
                 plt.close(ax.figure)
             except Exception as e:
                 print(f"Error generating plot '{title}': {e}")
                 if 'ax' in locals() and hasattr(ax, 'figure'):
                     plt.close(ax.figure)
        # --- End Helper ---

        # --- Sort Data & Plot Occlusion Rates ---
        occ_cols = ['train_occluded_pct', 'val_occluded_pct']
        sort_col_occ = 'train_occluded_pct'
        if all(col in rates_df.columns for col in occ_cols):
            rates_df_sorted_occ = rates_df.sort_values(by=sort_col_occ, ascending=True)
            occ_df = rates_df_sorted_occ[occ_cols].copy()
            occ_df.columns = ['Train', 'Validation']
            occ_save_path = save_path_prefix.parent / f"{save_path_prefix.name}_occlusion_sorted.png" if save_path_prefix else None
            create_rate_plot(
                occ_df,
                'Object Occlusion Rate per Class (Train vs Validation, Sorted)',
                'Occlusion Rate (%)',
                occ_save_path
            )
        else:
             print(f"Warning: Missing one or more occlusion columns: {occ_cols}. Skipping occlusion plot.")


        # --- Sort Data & Plot Truncation Rates ---
        trunc_cols = ['train_truncated_pct', 'val_truncated_pct']
        sort_col_trunc = 'train_truncated_pct'
        if all(col in rates_df.columns for col in trunc_cols):
            rates_df_sorted_trunc = rates_df.sort_values(by=sort_col_trunc, ascending=True)
            trunc_df = rates_df_sorted_trunc[trunc_cols].copy()
            trunc_df.columns = ['Train', 'Validation'] # 
            trunc_save_path = save_path_prefix.parent / f"{save_path_prefix.name}_truncation_sorted.png" if save_path_prefix else None
            create_rate_plot(
                trunc_df,
                'Object Truncation Rate per Class (Train vs Validation, Sorted)',
                'Truncation Rate (%)',
                trunc_save_path
            )
        else:
            print(f"Warning: Missing one or more truncation columns: {trunc_cols}. Skipping truncation plot.")

        print("Sorted Occlusion/Truncation rate plotting complete.")

    
    def plot_recall_vs_size(
        self,
        per_class_data: Union[pd.DataFrame, Path],
        save_path: Optional[Path] = None
        ):
        """
        Generates a faceted bar chart showing Recall (%) vs. Size Bin
        for each object category.

        Args:
            per_class_data: DataFrame or Path to the per-class recall vs size
                            parquet/csv file. Expected columns: 'size_bin',
                            'category', 'Recall', 'GT_Count'.
            save_path: If provided, saves the plot to this path.
        """
        print("\nGenerating Recall vs. Size (Per Class) Plot...")

        required_cols = ['size_bin', 'category', 'Recall', 'GT_Count']

        # --- Load data ---
        recall_df = self._load_data_if_needed(per_class_data, expected_columns=required_cols)

        if recall_df is None or recall_df.empty:
            print("Error: Could not load or use per-class recall vs size data. Cannot plot.")
            return
        # --- End Data Loading ---

        # --- Ensure correct ordering for plotting ---
        
        size_bin_order = ['Small', 'Medium', 'Large']
        
        if pd.api.types.is_categorical_dtype(recall_df['size_bin']):
            recall_df['size_bin'] = recall_df['size_bin'].cat.reorder_categories(size_bin_order, ordered=True)
        else:
            recall_df['size_bin'] = pd.Categorical(recall_df['size_bin'], categories=size_bin_order, ordered=True)

        
        if pd.api.types.is_categorical_dtype(recall_df['category']):
            recall_df['category'] = recall_df['category'].cat.reorder_categories(BDD_CLASSES, ordered=True)
        else:
             recall_df['category'] = pd.Categorical(recall_df['category'], categories=BDD_CLASSES, ordered=True)

        recall_df = recall_df.sort_values(by=['category', 'size_bin'])


        # --- Plotting using Seaborn catplot ---
        try:
            
            g = sns.catplot(
                data=recall_df,
                x='size_bin',
                y='Recall',
                col='category',  
                kind='bar',
                col_wrap=5,       
                height=3,        
                aspect=1.2,      
                palette='viridis',
                sharex=True,     
                sharey=True      
            )

            # --- Customize Plot ---
            g.set(ylim=(0, 105)) 
            g.set_axis_labels("Object Size Bin", "Recall (%)")
            g.set_titles("Class: {col_name}") 
            g.fig.suptitle('Recall vs. Object Size per Class', y=1.03, fontsize=16) 
            g.despine(left=True)
            for ax in g.axes.flat:
                 ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                 ax.xaxis.grid(False)

            plt.tight_layout(rect=[0, 0, 1, 0.98]) 

            # --- Save or Show ---
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

            plt.close(g.fig) 

        except Exception as e:
            print(f"Error during plotting recall vs size: {e}")
            if 'g' in locals() and hasattr(g, 'fig'):
                plt.close(g.fig)


    def plot_recall_vs_visibility(
        self,
        recall_vis_data: Union[pd.DataFrame, Path], 
        sort_by_occlusion: str = 'NotOccluded', 
        sort_by_truncation: str = 'NotTruncated', 
        save_path_prefix: Optional[Path] = None 
        ):
        """
        Generates grouped bar charts comparing Recall (%) based on Occlusion
        and Truncation status per class.

        Loads data from the pre-calculated per-class recall vs visibility file.
        Saves two plots if save_path_prefix is provided.

        Args:
            recall_vis_data: DataFrame or Path to the per-class recall vs visibility
                             parquet/csv file. Expected columns like 'Recall_NotOccluded',
                             'Recall_Occluded', 'Recall_NotTruncated', 'Recall_Truncated'.
            sort_by_occlusion: Column basis for sorting occlusion plot ('NotOccluded' or 'Occluded').
            sort_by_truncation: Column basis for sorting truncation plot ('NotTruncated' or 'Truncated').
            save_path_prefix: If provided, saves the plots using this prefix
                              (e.g., prefix_occlusion.png, prefix_truncation.png).
        """
        print("\nGenerating Recall vs. Visibility Plots...")

        
        expected_cols = [
            'Recall_NotOccluded', 'Recall_Occluded',
            'Recall_NotTruncated', 'Recall_Truncated'
        ]

        # --- Load data ---
        recall_df = self._load_data_if_needed(recall_vis_data) 

        if recall_df is None or recall_df.empty:
            print("Error: Could not load or use recall vs visibility data. Cannot plot.")
            return
        if not all(f"{stat}_{suffix}" in recall_df.columns for stat in ['Recall', 'GT_Count'] for suffix in ['NotOccluded', 'Occluded', 'NotTruncated', 'Truncated']):
             print(f"Warning: Input DataFrame might be missing some expected Recall/Count columns.")
             
             if not all(col in recall_df.columns for col in expected_cols):
                  print(f"Error: DataFrame missing essential Recall columns: {expected_cols}")
                  return
        # --- End Data Loading ---

        # --- Helper function ---
        def create_recall_plot(data_subset, sort_col_suffix, title, ylabel, save_path):
             try:
                sort_col = f"Recall_{sort_col_suffix}"
                if sort_col not in data_subset.columns:
                     print(f"Warning: Sort column '{sort_col}' not found. Using index order.")
                     plot_data = data_subset
                else:
                     plot_data = data_subset.sort_values(by=sort_col, ascending=True)

                
                plot_data.columns = [col.replace(f'Recall_', '').replace('Not', 'Not ')
                                     for col in plot_data.columns]

                ax = plot_data.plot(
                    kind='bar', figsize=(12, 7), grid=False, rot=45, colormap='plasma'
                )
                ax.set_title(title)
                ax.set_xlabel('Object Class')
                ax.set_ylabel(ylabel)
                ax.legend(title='Status')
                ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
                ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                ax.xaxis.grid(False)
                ax.set_ylim(0, 105) 

                plt.tight_layout()

                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path, bbox_inches='tight')
                    print(f"Plot saved to {save_path}")
                else:
                    plt.show()
                plt.close(ax.figure)

             except Exception as e:
                print(f"Error generating plot '{title}': {e}")
                if 'ax' in locals() and hasattr(ax, 'figure'):
                    plt.close(ax.figure)
        # --- End Helper ---


        # --- Plot Recall vs Occlusion ---
        occ_cols = ['Recall_NotOccluded', 'Recall_Occluded']
        if all(col in recall_df.columns for col in occ_cols):
            occ_df = recall_df[occ_cols].copy()
            occ_save_path = save_path_prefix.parent / f"{save_path_prefix.name}_vs_occlusion.png" if save_path_prefix else None
            create_recall_plot(
                occ_df,
                sort_by_occlusion,
                f'Recall vs. Occlusion Status per Class (Sorted by Train {sort_by_occlusion})',
                'Recall (%)',
                occ_save_path
            )
        else:
             print(f"Warning: Missing one or more occlusion recall columns: {occ_cols}. Skipping plot.")


        # --- Plot Recall vs Truncation ---
        trunc_cols = ['Recall_NotTruncated', 'Recall_Truncated']
        if all(col in recall_df.columns for col in trunc_cols):
            trunc_df = recall_df[trunc_cols].copy()
            trunc_save_path = save_path_prefix.parent / f"{save_path_prefix.name}_vs_truncation.png" if save_path_prefix else None
            create_recall_plot(
                trunc_df,
                sort_by_truncation,
                f'Recall vs. Truncation Status per Class (Sorted by Train {sort_by_truncation})',
                'Recall (%)',
                trunc_save_path
            )
        else:
            print(f"Warning: Missing one or more truncation recall columns: {trunc_cols}. Skipping plot.")

        print("Recall vs. Visibility plotting complete.")       


    def plot_recall_vs_attribute(
        self,
        recall_data: Union[pd.DataFrame, Path],
        attribute_name: str, 
        group_by_class: bool, 
        save_path: Optional[Path] = None
        ):
        """
        Generates bar chart(s) showing Recall (%) vs. a specific image attribute.
        Creates a single bar chart for overall recall or a faceted bar chart for
        per-class recall.

        Args:
            recall_data: DataFrame or Path to the recall vs attribute parquet/csv file.
                         Expected columns: attribute_name, Recall, GT_Count,
                         and 'category' if group_by_class is True.
            attribute_name: The name of the attribute column (e.g., 'timeofday').
            group_by_class: Whether the input data is grouped by class.
            save_path: If provided, saves the plot to this path.
        """
        print(f"\nGenerating Recall vs. Attribute ('{attribute_name}', Group by class: {group_by_class}) Plot...")

        required_cols = [attribute_name, 'Recall', 'GT_Count']
        if group_by_class:
            required_cols.append('category')

        # --- Load data ---
        recall_df = self._load_data_if_needed(recall_data, expected_columns=required_cols)

        if recall_df is None or recall_df.empty:
            print(f"Error: Could not load or use recall vs '{attribute_name}' data. Cannot plot.")
            return
        # --- End Data Loading ---

        # --- Prepare Data (Sorting attribute values if needed) ---
        timeofday_order = ['daytime', 'night', 'dawn/dusk', 'undefined']
        weather_order = ['clear', 'overcast', 'partly cloudy', 'rainy', 'foggy', 'snowy', 'undefined']

        plot_order = None
        if attribute_name == 'timeofday':
            plot_order = [t for t in timeofday_order if t in recall_df[attribute_name].unique()]
        elif attribute_name == 'weather':
            plot_order = [w for w in weather_order if w in recall_df[attribute_name].unique()]

        
        recall_df['Recall'] = pd.to_numeric(recall_df['Recall'])
        recall_df[attribute_name] = recall_df[attribute_name].astype('category')
        if group_by_class:
             recall_df['category'] = recall_df['category'].astype('category')


        # --- Plotting ---
        fig = None 
        try:
            if group_by_class:
                g = sns.catplot(
                    data=recall_df,
                    x=attribute_name,
                    y='Recall',
                    col='category',
                    kind='bar',
                    col_wrap=5, 
                    height=3,
                    aspect=1.2,
                    palette='mako', 
                    order=plot_order, 
                    col_order=BDD_CLASSES,
                    sharex=True, sharey=True
                )
                g.set(ylim=(0, 105))
                g.set_axis_labels(attribute_name.replace('_', ' ').title(), "Recall (%)")
                g.set_titles("Class: {col_name}")
                g.fig.suptitle(f'Recall vs. {attribute_name.title()} per Class', y=1.03, fontsize=16)
                g.despine(left=True)
                for ax in g.axes.flat:
                     ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                     ax.xaxis.grid(False)
                     ax.tick_params(axis='x', rotation=45, right=True)

                plt.tight_layout(rect=[0, 0, 1, 0.98])
                fig = g.fig 

            else:
                fig, ax = plt.subplots(figsize=(10, 6)) 
                sns.barplot(
                    data=recall_df,
                    x=attribute_name,
                    y='Recall',
                    ax=ax,
                    palette='mako',
                    order=plot_order
                )
                ax.set_title(f'Overall Recall vs. {attribute_name.title()}')
                ax.set_xlabel(attribute_name.replace('_', ' ').title())
                ax.set_ylabel('Recall (%)')
                ax.set_ylim(0, 105)
                ax.tick_params(axis='x', rotation=45, right=True)
                ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                ax.xaxis.grid(False)
                plt.tight_layout()

            # --- Save or Show ---
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

            plt.close(fig)

        except Exception as e:
            print(f"Error during plotting recall vs {attribute_name}: {e}")
            if fig is not None:
                plt.close(fig)

    
    def display_image_with_boxes(
        self,
        image_id: str,
        image_dir: Path,
        gt_df: Optional[pd.DataFrame] = None,      
        pred_df: Optional[pd.DataFrame] = None,     
        show_labels: bool = True,
        show_confidence: bool = True,
        show_fp_iou: bool = False,                 
        color_map: Optional[Dict[str, Tuple]] = None, 
        line_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        save_path: Optional[Path] = None
        ):
        """
        Loads an image and draws ground truth and/or prediction boxes on it.

        Args:
            image_id: The ID (usually filename without extension) of the image.
            image_dir: Path to the directory containing the validation images.
            gt_df: DataFrame containing ground truth boxes for THIS image_id.
                   Expected columns: bbox_x1, y1, x2, y2, category, is_matched (optional).
            pred_df: DataFrame containing prediction boxes for THIS image_id.
                     Expected columns: bbox_x1, y1, x2, y2, category, confidence, status ('TP'/'FP').
            show_labels: Whether to draw class labels.
            show_confidence: Whether to draw confidence scores for predictions.
            show_fp_iou: If True, show IoU score for FP predictions (if 'iou' column exists).
            color_map: Dictionary mapping status/type ('GT', 'TP', 'FP', 'FN') to BGR tuples,
                       e.g., {'GT': (0, 255, 0), 'FP': (0, 0, 255)}.
            line_thickness: Thickness of bounding box lines.
            font_scale: Font scale for labels.
            font_thickness: Thickness for label font.
            save_path: If provided, saves the annotated image to this path. Otherwise, displays it.
        """
        print(f"\nVisualizing boxes for image: {image_id}")

        # --- Define Colors (BGR format for OpenCV) ---
        default_colors = {
            "GT": (0, 255, 0),     
            "FN": (0, 255, 255),  
            "TP": (255, 150, 0),   
            "FP": (0, 0, 255)     
        }
        colors = default_colors.copy()
        if color_map:
            colors.update(color_map)

        # --- Load Image ---
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            print(f"Error: Image file not found at {image_path}")
            return

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                 print(f"Error: Could not read image file {image_path}")
                 return
            img_vis = img.copy()
            image_height, image_width = img_vis.shape[:2]
        except Exception as e:
             print(f"Error loading image {image_path}: {e}")
             return

        # --- Draw Ground Truth Boxes ---
        if gt_df is not None and not gt_df.empty:
            print(f"  Drawing {len(gt_df)} ground truth box(es)...")
            has_match_status = 'is_matched' in gt_df.columns
            for _, row in gt_df.iterrows():
                try:
                    x1, y1, x2, y2 = int(row['bbox_x1']), int(row['bbox_y1']), int(row['bbox_x2']), int(row['bbox_y2'])
                    label = str(row['category'])
                    is_matched = row.get('is_matched', True) 

                    color = colors['GT']
                    label_prefix = "GT: "
                    if has_match_status and not is_matched:
                        color = colors['FN'] 
                        label_prefix = "FN: " 

                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, line_thickness)

                    if show_labels:
                        label_text = f"{label_prefix}{label}"
                        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        cv2.rectangle(img_vis, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
                        cv2.putText(img_vis, label_text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, (0,0,0), font_thickness, lineType=cv2.LINE_AA)

                except Exception as e:
                     print(f"Warning: Error drawing GT box for object {row.get('object_id', 'N/A')}: {e}")


        # --- Draw Prediction Boxes ---
        if pred_df is not None and not pred_df.empty:
             print(f"  Drawing {len(pred_df)} prediction box(es)...")
             for _, row in pred_df.iterrows():
                  try:
                      x1, y1, x2, y2 = int(row['bbox_x1']), int(row['bbox_y1']), int(row['bbox_x2']), int(row['bbox_y2'])
                      label = str(row['category'])
                      conf = row.get('confidence', 0.0)
                      status = row.get('status', 'FP') 

                      color = colors.get(status, (128, 128, 128)) 

                      cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, line_thickness + 1)

                      if show_labels:
                           label_text = f"{status}: {label}"
                           if show_confidence:
                                label_text += f" {conf:.2f}"
                           if show_fp_iou and status == 'FP' and 'iou' in row and pd.notna(row['iou']):
                                label_text += f" (IoU:{row['iou']:.2f})" 

                           (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                           text_y = y2 + h + 4 if y2 + h + 10 < image_height else y1 - 3 
                           bg_y1 = text_y - h - 4
                           bg_y2 = text_y

                           cv2.rectangle(img_vis, (x1, bg_y1), (x1 + w, bg_y2), color, -1)
                           cv2.putText(img_vis, label_text, (x1, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, (0,0,0), font_thickness, lineType=cv2.LINE_AA)

                  except Exception as e:
                      print(f"Warning: Error drawing prediction box: {e}")


        # --- Save or Display ---
        if save_path:
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), img_vis)
                print(f"Annotated image saved to {save_path}")
            except Exception as e:
                print(f"Error saving annotated image to {save_path}: {e}")
        else:
            print("Displaying image (close window to continue)...")
            try:
                img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(15, 10))
                plt.imshow(img_rgb)
                plt.title(f"Qualitative Results: {image_id}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                 print(f"Error displaying image using matplotlib: {e}")


    
    def plot_failure_counts_per_class(
        self,
        failure_data: Union[pd.DataFrame, Path],
        failure_type: str,
        save_path: Optional[Path] = None
        ):
        """
        Generates a bar chart showing the count of False Negatives (FN) or
        False Positives (FP) per object category, sorted descending.

        Args:
            failure_data: DataFrame or Path to the FN or FP data file.
                          Expected column: 'category'.
            failure_type: String indicating the type ('FN' or 'FP') for titles/labels.
            save_path: If provided, saves the plot to this path.
        """
        print(f"\nGenerating {failure_type} Counts per Class Plot...")

        required_cols = ['category']

        # --- Load data ---
        fail_df = self._load_data_if_needed(failure_data, expected_columns=required_cols)

        if fail_df is None or fail_df.empty:
            print(f"Error: Could not load or use {failure_type} data. Cannot plot counts.")
            return
        # --- End Data Loading ---

        # --- Calculate and Sort Counts ---
        counts = fail_df['category'].value_counts()
        counts = counts.reindex(BDD_CLASSES, fill_value=0)
        counts = counts.sort_values(ascending=True)

        # --- Plotting ---
        try:
            fig, ax = plt.subplots(figsize=(12, 7))

            sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='rocket')

            if failure_type == 'FP':
                ax.set_title(f'{failure_type} Count per predicted Class (Sorted)')
            else:
                ax.set_title(f'{failure_type} Count per Class (Sorted)')

            ax.set_xlabel('Object Class')
            ax.set_ylabel(f'Number of {failure_type}s')
            ax.tick_params(axis='x', rotation=45, right=True)

            for container in ax.containers:
                ax.bar_label(container, fmt='%d')

            plt.tight_layout()
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)

            # --- Save or Show ---
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

            plt.close(fig)

        except Exception as e:
            print(f"Error during plotting {failure_type} counts: {e}")
            if 'fig' in locals() and hasattr(fig, 'figure'):
                plt.close(fig)


    def plot_fn_size_distribution(
        self,
        fn_data: Union[pd.DataFrame, Path],
        bins: int = 50,
        use_log_scale: bool = True,
        save_path: Optional[Path] = None
        ):
        """
        Generates a histogram showing the distribution of bounding box areas
        for False Negative (FN) ground truth objects.

        Args:
            fn_data: DataFrame or Path to the False Negative data file.
                     Expected column: 'bbox_area'.
            bins: Number of bins for the histogram.
            use_log_scale: If True (recommended), use a logarithmic scale for the x-axis (area).
            save_path: If provided, saves the plot to this path.
        """
        print("\nGenerating False Negative (FN) Size Distribution Plot...")

        required_cols = ['bbox_area']

        # --- Load data ---
        fn_df = self._load_data_if_needed(fn_data, expected_columns=required_cols)

        if fn_df is None or fn_df.empty:
            print("Error: Could not load or use FN data. Cannot plot size distribution.")
            return
        # --- End Data Loading ---

        plot_data = fn_df[fn_df['bbox_area'] > 0]['bbox_area']
        if plot_data.empty:
             print("Warning: No FN objects with positive area found. Cannot plot.")
             return

        # --- Plotting ---
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.histplot(
                data=plot_data,
                log_scale=use_log_scale,
                bins=bins,
                ax=ax,
                kde=False
            )

            ax.set_title('Size Distribution of False Negative Objects')
            xlabel = 'Object Area (pixels²)'
            if use_log_scale:
                 xlabel += ' (Log Scale)'
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Frequency (Count)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            if use_log_scale:
                ax.axvline(x=1024, color='r', linestyle='--', alpha=0.6, label='Small/Medium Threshold (1024)')
                ax.axvline(x=9216, color='g', linestyle='--', alpha=0.6, label='Medium/Large Threshold (9216)')
                ax.legend()

            plt.tight_layout()

            # --- Save or Show ---
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

            plt.close(fig)

        except Exception as e:
            print(f"Error during plotting FN size distribution: {e}")
            if 'fig' in locals() and hasattr(fig, 'figure'):
                plt.close(fig)


    def plot_fp_confidence_distribution(
        self,
        fp_data: Union[pd.DataFrame, Path],
        bins: int = 20, 
        save_path: Optional[Path] = None
        ):
        """
        Generates a histogram showing the distribution of confidence scores
        for False Positive (FP) predictions.

        Args:
            fp_data: DataFrame or Path to the False Positive data file.
                     Expected column: 'confidence'.
            bins: Number of bins for the histogram.
            save_path: If provided, saves the plot to this path.
        """
        print("\nGenerating False Positive (FP) Confidence Distribution Plot...")

        required_cols = ['confidence']

        # --- Load data ---
        fp_df = self._load_data_if_needed(fp_data, expected_columns=required_cols)

        if fp_df is None or fp_df.empty:
            print("Error: Could not load or use FP data. Cannot plot confidence distribution.")
            return
        # --- End Data Loading ---

        plot_data = fp_df['confidence'].dropna()
        if plot_data.empty:
             print("Warning: No valid confidence scores found for FPs. Cannot plot.")
             return

        # --- Plotting ---
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.histplot(
                data=plot_data,
                bins=bins,
                ax=ax,
                kde=False
            )

            ax.set_title('Confidence Score Distribution of False Positives')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency (Count)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)


            plt.tight_layout()

            # --- Save or Show ---
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()

            plt.close(fig)

        except Exception as e:
            print(f"Error during plotting FP confidence distribution: {e}")
            if 'fig' in locals() and hasattr(fig, 'figure'):
                plt.close(fig)


    def plot_fn_visibility_comparison(
        self,
        fn_data: Union[pd.DataFrame, Path],
        overall_gt_rates_data: Union[pd.DataFrame, Path],
        save_path_prefix: Optional[Path] = None
        ):
        """
        Generates grouped bar charts comparing Occlusion and Truncation rates (%)
        for False Negatives (FNs) vs. Overall Ground Truth (GT) per class.

        Args:
            fn_data: DataFrame or Path to the False Negative data file.
                     Expected columns: 'category', 'occluded', 'truncated'.
            overall_gt_rates_data: DataFrame or Path to the overall GT occlusion/truncation
                                   rates file. Expected columns: 'val_occluded_pct',
                                   'val_truncated_pct', with category index or column.
            save_path_prefix: If provided, saves the plots using this prefix.
        """
        print("\nGenerating FN vs Overall GT Visibility Rate Comparison Plots...")

        # --- Load FN Data ---
        required_fn_cols = ['category', 'occluded', 'truncated']
        fn_df = self._load_data_if_needed(fn_data, expected_columns=required_fn_cols)
        if fn_df is None or fn_df.empty:
            print("Error: Could not load or use FN data. Cannot plot visibility comparison.")
            return

        # --- Load Overall GT Rates Data ---
        required_gt_cols = ['val_occluded_pct', 'val_truncated_pct']
        gt_rates_df = self._load_data_if_needed(overall_gt_rates_data)
        if gt_rates_df is None or gt_rates_df.empty:
            print("Error: Could not load or use overall GT rates data. Cannot plot comparison.")
            return
        if 'category' in gt_rates_df.columns:
             gt_rates_df = gt_rates_df.set_index('category')
        if not all(col in gt_rates_df.columns for col in required_gt_cols):
             print(f"Error: Overall GT rates DF missing required columns: {required_gt_cols}")
             return
        # --- End Data Loading ---

        # --- Calculate FN Rates ---
        try:
            fn_df['occluded'] = fn_df['occluded'].astype(pd.BooleanDtype())
            fn_df['truncated'] = fn_df['truncated'].astype(pd.BooleanDtype())
            fn_rates = fn_df.groupby('category')[['occluded', 'truncated']].mean().multiply(100)
            fn_rates.rename(columns={'occluded': 'FN_occluded_pct', 'truncated': 'FN_truncated_pct'}, inplace=True)
        except Exception as e:
            print(f"Error calculating FN rates: {e}")
            return

        # --- Prepare Data for Plotting ---
        occ_compare_df = pd.concat([
            fn_rates[['FN_occluded_pct']],
            gt_rates_df[['val_occluded_pct']]
        ], axis=1).fillna(0)
        occ_compare_df.rename(columns={'FN_occluded_pct': 'FN Rate (%)', 'val_occluded_pct': 'Overall GT Rate (%)'}, inplace=True)
        occ_compare_df.sort_values(by='FN Rate (%)', ascending=False, inplace=True)

        trunc_compare_df = pd.concat([
            fn_rates[['FN_truncated_pct']],
            gt_rates_df[['val_truncated_pct']]
        ], axis=1).fillna(0)
        trunc_compare_df.rename(columns={'FN_truncated_pct': 'FN Rate (%)', 'val_truncated_pct': 'Overall GT Rate (%)'}, inplace=True)
        trunc_compare_df.sort_values(by='FN Rate (%)', ascending=False, inplace=True) 

        # --- Helper function ---
        def create_comparison_plot(plot_data, title, save_path):
            try:
                ax = plot_data.plot(
                    kind='bar', figsize=(12, 7), grid=False, rot=45, colormap='coolwarm'
                )
                ax.set_title(title)
                ax.set_xlabel('Object Class')
                ax.set_ylabel('Rate (%)')
                ax.legend(title='Data Group')
                ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
                ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                ax.xaxis.grid(False)
                ax.set_ylim(0, 105)
                plt.tight_layout()
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path, bbox_inches='tight')
                    print(f"Plot saved to {save_path}")
                else:
                    plt.show()
                plt.close(ax.figure)
            except Exception as e:
                print(f"Error generating plot '{title}': {e}")
                if 'ax' in locals() and hasattr(ax, 'figure'):
                    plt.close(ax.figure)
        # --- End Helper ---

        # --- Generate Plots ---
        occ_save_path = save_path_prefix.parent / f"{save_path_prefix.name}_fn_vs_gt_occlusion.png" if save_path_prefix else None
        trunc_save_path = save_path_prefix.parent / f"{save_path_prefix.name}_fn_vs_gt_truncation.png" if save_path_prefix else None

        create_comparison_plot(
            occ_compare_df,
            'FN Occlusion Rate vs Overall GT Occlusion Rate per Class',
            occ_save_path
        )
        create_comparison_plot(
            trunc_compare_df,
            'FN Truncation Rate vs Overall GT Truncation Rate per Class',
            trunc_save_path
        )

        print("FN vs GT Visibility comparison plotting complete.")


class GTVisualizer:
    def __init__(self, module_config: Dict[str, Any], 
                  main_output_dir: str):
        self.module_config = module_config
        self.main_output_dir = Path(main_output_dir)

        plot_subdir_name = self.module_config.get('output_plots_subdir', '03_gt_visualization_plots')
        self.output_path_for_plots = self.main_output_dir / plot_subdir_name
        os.makedirs(self.output_path_for_plots, exist_ok=True)
        print(f"GTVisualizer initialized. Plots will be saved to: {self.output_path_for_plots}")

        self.visualizer = BDDVisualizer()

        intermediate_subdir_name = self.main_output_dir / Path(self.module_config.get('intermediate_data_subdir', '02_gt_analysis_metrics'))
        
        self.ANALYSIS_RESULTS_DIR = intermediate_subdir_name / "gt_analysis"

        self.RAW_DATA_DIR = intermediate_subdir_name / "gt_parsed_data"

        self.OUTPUT_PLOT_DIR = self.output_path_for_plots

    # --- End Configuration ---


    # --- Plotting Functions ---

    def generate_instance_count_plot(self, visualizer: BDDVisualizer):
        """Generates and saves the instance count plot."""
        print("-" * 10 + " Generating Instance Count Plot " + "-" * 10)
        try:
            instance_counts_path = self.ANALYSIS_RESULTS_DIR / "instance_counts_category_gt.parquet"
            plot_save_path = self.OUTPUT_PLOT_DIR / "instance_counts_comparison_log.png"

            if not instance_counts_path.exists():
                print(f"Warning: Input file not found: {instance_counts_path}. Skipping plot.")
                return

            print("\nGenerating Instance Count plot from file...")
            visualizer.plot_instance_counts(
                counts_data=instance_counts_path,
                use_log_scale=True,
                save_path=plot_save_path
            )

        except Exception as e:
            print(f"Error generating instance count plot: {e}")


    def generate_size_distribution_plot(self, visualizer: BDDVisualizer):
        """Generates and saves the object size distribution box plot."""
        print("-" * 10 + " Generating Size Distribution Plot " + "-" * 10)
        try:
            train_obj_path = self.RAW_DATA_DIR / "bdd_train_objects.parquet"
            val_obj_path = self.RAW_DATA_DIR / "bdd_val_objects.parquet"
            plot_save_path = self.OUTPUT_PLOT_DIR / "object_size_distribution_boxplot_log.png"

            if not train_obj_path.exists() or not val_obj_path.exists():
                print(f"Warning: Raw object data file(s) not found in {self.RAW_DATA_DIR}. Skipping plot.")
                return

            visualizer.plot_size_distribution(
                train_objects_data=train_obj_path,
                val_objects_data=val_obj_path,
                use_log_scale=True,
                show_outliers=False,
                save_path=plot_save_path
            )
        except Exception as e:
            print(f"Error generating size distribution plot: {e}")


    def generate_location_heatmap_plot(self, visualizer: BDDVisualizer):
        """Generates and saves the object location heatmap."""
        print("-" * 10 + " Generating Location Heatmap Plot " + "-" * 10)
        try:
            train_loc_path = self.ANALYSIS_RESULTS_DIR / "bdd_train_object_locations.parquet"
            val_loc_path = self.ANALYSIS_RESULTS_DIR / "bdd_val_object_locations.parquet"
            plot_save_prefix = self.OUTPUT_PLOT_DIR / "location_heatmap"

            if not train_loc_path.exists() or not val_loc_path.exists():
                print(f"Warning: Location data file(s) not found in {self.ANALYSIS_RESULTS_DIR}. Skipping plot.")
                return

            visualizer.plot_location_heatmap(
                train_location_data=train_loc_path,
                val_location_data=val_loc_path,
                image_width=1280,
                image_height=720,
                bins=64, 
                cmap='inferno',
                save_path_prefix=plot_save_prefix
            )

        except Exception as e:
            print(f"Error generating location heatmap plot: {e}")


    def generate_location_heatmap_plot_per_class(self, visualizer: BDDVisualizer): 
        """Generates and saves the per-class object location heatmap grid."""
        print("-" * 10 + " Generating Per-Class Location Heatmap Plot " + "-" * 10)
        try:
            train_loc_path = self.ANALYSIS_RESULTS_DIR / "bdd_train_object_locations.parquet"
            val_loc_path = self.ANALYSIS_RESULTS_DIR / "bdd_val_object_locations.parquet"
            plot_save_path = self.OUTPUT_PLOT_DIR / "location_heatmap_per_class.png"

            if not train_loc_path.exists() or not val_loc_path.exists():
                print(f"Warning: Location data file(s) not found in {self.ANALYSIS_RESULTS_DIR}. Skipping plot.")
                return

            visualizer.plot_location_heatmap_per_class(
                train_location_data=train_loc_path,
                val_location_data=val_loc_path,
                image_width=1280,
                image_height=720,
                bins=64,
                cmap='inferno',
                save_path=plot_save_path
            )

        except Exception as e:
            print(f"Error generating per-class location heatmap plot: {e}")


    def generate_occlusion_rates_plot(self, visualizer: BDDVisualizer):
        """Generates and saves the occlusion/truncation rates plot."""
        print("-" * 10 + " Generating Occlusion/Truncation Rates Plot " + "-" * 10)
        try:
            rates_path = self.ANALYSIS_RESULTS_DIR / "occ_trunc_rates_gt.parquet"
            plot_save_path = self.OUTPUT_PLOT_DIR / "occ_trunc_rates_gt"

            if not rates_path.exists():
                print(f"Warning: Occlusion/Truncation rates file not found: {rates_path}. Skipping plot.")
                return

            
            visualizer.plot_occlusion_rates(
                    rates_data=rates_path,
                    save_path_prefix=plot_save_path
                )

        except Exception as e:
            print(f"Error generating occlusion/truncation rates plot: {e}")
        
    def __call__(self):

        self.generate_instance_count_plot(self.visualizer)

        self.generate_size_distribution_plot(self.visualizer)

        self.generate_location_heatmap_plot(self.visualizer)

        self.generate_location_heatmap_plot_per_class(self.visualizer)

        self.generate_occlusion_rates_plot(self.visualizer)

        print("GTVisualizer: Visualization calculation complete.")

        return 