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

class BDDEvalVisualizer:
    """
    Handles visualization of BDD dataset analysis results.
    Loads data from saved files (e.g., parquet) or uses pre-loaded DataFrames.
    """
    def __init__(self, plot_style: str = 'seaborn-v0_8-talk'):
        """
        Initializes the visualizer

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
        # Define the desired order for size bins
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
            # Use catplot for easy faceting
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
                

            plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout slightly

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
                sort_by_occlusion, # e.g., 'NotOccluded' or 'Occluded'
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
                sort_by_truncation, # e.g., 'NotTruncated' or 'Truncated'
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
        weather_order = ['clear', 'overcast', 'partly cloudy', 'rainy', 'foggy', 'snowy', 'undefined'] # Add others if needed

        plot_order = None
        if attribute_name == 'timeofday':
            plot_order = [t for t in timeofday_order if t in recall_df[attribute_name].unique()]
        elif attribute_name == 'weather':
            plot_order = [w for w in weather_order if w in recall_df[attribute_name].unique()]

        # Ensure dtypes are correct
        recall_df['Recall'] = pd.to_numeric(recall_df['Recall'])
        recall_df[attribute_name] = recall_df[attribute_name].astype('category')
        if group_by_class:
             recall_df['category'] = recall_df['category'].astype('category')


        # --- Plotting ---
        fig = None # Initialize fig
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
            "GT": (0, 255, 0),     # Green
            "FN": (0, 255, 255),   # Yellow 
            "TP": (255, 150, 0),   # Blue-ish
            "FP": (0, 0, 255)      # Red
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

                    # Draw rectangle
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, line_thickness)

                    # Draw label
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
                plt.axis('off') # Hide axes
                plt.tight_layout()
                plt.show()
            except Exception as e:
                 print(f"Error displaying image using matplotlib: {e}")
          

    
    def plot_failure_counts_per_class(
        self,
        failure_data: Union[pd.DataFrame, Path],
        failure_type: str, # 'FN' or 'FP'
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

        fp_df = self._load_data_if_needed(fp_data, expected_columns=required_cols)

        if fp_df is None or fp_df.empty:
            print("Error: Could not load or use FP data. Cannot plot confidence distribution.")
            return

        plot_data = fp_df['confidence'].dropna()
        if plot_data.empty:
             print("Warning: No valid confidence scores found for FPs. Cannot plot.")
             return

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

        try:
            # Ensure boolean type
            fn_df['occluded'] = fn_df['occluded'].astype(pd.BooleanDtype())
            fn_df['truncated'] = fn_df['truncated'].astype(pd.BooleanDtype())
            # Calculate rates per class
            fn_rates = fn_df.groupby('category')[['occluded', 'truncated']].mean().multiply(100)
            fn_rates.rename(columns={'occluded': 'FN_occluded_pct', 'truncated': 'FN_truncated_pct'}, inplace=True)
        except Exception as e:
            print(f"Error calculating FN rates: {e}")
            return

        # Occlusion Comparison
        occ_compare_df = pd.concat([
            fn_rates[['FN_occluded_pct']],
            gt_rates_df[['val_occluded_pct']]
        ], axis=1).fillna(0)
        occ_compare_df.rename(columns={'FN_occluded_pct': 'FN Rate (%)', 'val_occluded_pct': 'Overall GT Rate (%)'}, inplace=True)
        occ_compare_df.sort_values(by='FN Rate (%)', ascending=False, inplace=True)

        # Truncation Comparison
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


class EvalVisualizerRunner:

    def __init__(self, config, main_output_dir):
        self.visualizer = BDDEvalVisualizer()

        self.module_config = config.get("eval_analysis_config")
        self.main_output_dir = Path(main_output_dir)

        # Subdirectory for this module's plot outputs
        plot_subdir_name = self.module_config.get('output_plots_subdir')

        self.output_path_for_plots = self.main_output_dir / plot_subdir_name

        os.makedirs(self.output_path_for_plots, exist_ok=True)

        print(f"Eval Visualizer initialized. Plots will be saved to: {self.output_path_for_plots}")

        self.visualizer = BDDEvalVisualizer()

        self.intermediate_subdir_name = self.main_output_dir / config.get("eval_analysis_config").get("intermediate_data_subdir")
    
        # Path where you want to SAVE the output plots
        self.OUTPUT_PLOT_DIR = self.output_path_for_plots



    def generate_recall_vs_size_plot(self):
        """Generates and saves the Recall vs. Size per class plot."""
        print("-" * 10 + " Generating Recall vs. Size Plot " + "-" * 10)
        try:
            # Define input data path (per-class results)
            recall_size_per_class_path = self.intermediate_subdir_name / "recall_vs_size_per_class.parquet"
            # Define output plot path
            plot_save_path = self.OUTPUT_PLOT_DIR / "recall_vs_size_per_class.png"

            if not recall_size_per_class_path.exists():
                print(f"Warning: Input file not found: {recall_size_per_class_path}. Skipping plot.")
                return

            # Call the visualizer method
            self.visualizer.plot_recall_vs_size(
                per_class_data=recall_size_per_class_path, 
                save_path=plot_save_path
            )
        except Exception as e:
            print(f"Error generating Recall vs Size plot: {e}")


    def generate_recall_vs_visibility_plot(self):
        """Generates and saves the Recall vs. Visibility per class plots."""
        print("-" * 10 + " Generating Recall vs. Visibility Plots " + "-" * 10)
        try:
            # Define input data path (per-class results from evaluator)
            recall_vis_per_class_path = self.intermediate_subdir_name / "recall_vs_visibility_per_class.parquet"
            # Define output plot path prefix
            plot_save_prefix = self.OUTPUT_PLOT_DIR / "recall_vs_visibility" # Saves _vs_occlusion.png and _vs_truncation.png

            if not recall_vis_per_class_path.exists():
                print(f"Warning: Input file not found: {recall_vis_per_class_path}. Skipping plot.")
                return

            # Call the visualizer method
            self.visualizer.plot_recall_vs_visibility(
                recall_vis_data=recall_vis_per_class_path, 
                sort_by_occlusion='NotOccluded', 
                sort_by_truncation='Truncated', 
                save_path_prefix=plot_save_prefix
            )
        except Exception as e:
            print(f"Error generating Recall vs Visibility plot: {e}")
            # import traceback; traceback.print_exc()
        

    def generate_recall_vs_timeofday_plot(self):
        """Generates and saves the Recall vs. TimeOfDay plots."""
        print("-" * 10 + " Generating Recall vs. TimeOfDay Plots " + "-" * 10)
        try:
            overall_path = self.intermediate_subdir_name / "recall_vs_timeofday_overall.parquet"
            per_class_path = self.intermediate_subdir_name / "recall_vs_timeofday_per_class.parquet"
            plot_save_overall = self.OUTPUT_PLOT_DIR / "recall_vs_timeofday_overall.png"
            plot_save_per_class = self.OUTPUT_PLOT_DIR / "recall_vs_timeofday_per_class.png"

            if overall_path.exists():
                self.visualizer.plot_recall_vs_attribute(
                    recall_data=overall_path,
                    attribute_name='timeofday',
                    group_by_class=False,
                    save_path=plot_save_overall
                )
            else:
                print(f"Warning: Input file not found: {overall_path}. Skipping overall plot.")

            if per_class_path.exists():
                self.visualizer.plot_recall_vs_attribute(
                    recall_data=per_class_path,
                    attribute_name='timeofday',
                    group_by_class=True,
                    save_path=plot_save_per_class
                )
            else:
                print(f"Warning: Input file not found: {per_class_path}. Skipping per-class plot.")

        except Exception as e:
            print(f"Error generating Recall vs TimeOfDay plots: {e}")


    def generate_recall_vs_weather_plot(self):
        """Generates and saves the Recall vs. Weather plots."""
        print("-" * 10 + " Generating Recall vs. Weather Plots " + "-" * 10)
        try:
            overall_path = self.intermediate_subdir_name / "recall_vs_weather_overall.parquet"
            per_class_path = self.intermediate_subdir_name / "recall_vs_weather_per_class.parquet"
            plot_save_overall = self.OUTPUT_PLOT_DIR / "recall_vs_weather_overall.png"
            plot_save_per_class = self.OUTPUT_PLOT_DIR / "recall_vs_weather_per_class.png"

            if overall_path.exists():
                self.visualizer.plot_recall_vs_attribute(
                    recall_data=overall_path,
                    attribute_name='weather',
                    group_by_class=False,
                    save_path=plot_save_overall
                )
            else:
                print(f"Warning: Input file not found: {overall_path}. Skipping overall plot.")

            if per_class_path.exists():
                self.visualizer.plot_recall_vs_attribute(
                    recall_data=per_class_path,
                    attribute_name='weather',
                    group_by_class=True,
                    save_path=plot_save_per_class
                )
            else:
                print(f"Warning: Input file not found: {per_class_path}. Skipping per-class plot.")

        except Exception as e:
            print(f"Error generating Recall vs Weather plots: {e}")


    def generate_fn_count_plot(self):
        """Generates and saves the FN Count per class plot."""
        print("-" * 10 + " Generating FN Count per Class Plot " + "-" * 10)
        try:
            fn_data_path = self.intermediate_subdir_name / "false_negatives.parquet"
            plot_save_path = self.OUTPUT_PLOT_DIR / "fn_counts_per_class.png"

            if not fn_data_path.exists():
                print(f"Warning: Input file not found: {fn_data_path}. Skipping plot.")
                return

            self.visualizer.plot_failure_counts_per_class(
                failure_data=fn_data_path,
                failure_type='FN',
                save_path=plot_save_path
            )
        except Exception as e:
            print(f"Error generating FN Count plot: {e}")


    def generate_fp_count_plot(self):
        """Generates and saves the FP Count per PREDICTED class plot."""
        print("-" * 10 + " Generating FP Count per Class Plot " + "-" * 10)
        try:
            fp_data_path = self.intermediate_subdir_name / "false_positives.parquet"
            plot_save_path = self.OUTPUT_PLOT_DIR / "fp_counts_per_predicted_class.png"

            if not fp_data_path.exists():
                print(f"Warning: Input file not found: {fp_data_path}. Skipping plot.")
                return

            self.visualizer.plot_failure_counts_per_class(
                failure_data=fp_data_path,
                failure_type='FP',
                save_path=plot_save_path
            )
        except Exception as e:
            print(f"Error generating FP Count plot: {e}")


    def generate_fn_size_plot(self):
        """Generates and saves the FN Size Distribution histogram."""
        print("-" * 10 + " Generating FN Size Distribution Plot " + "-" * 10)
        try:
   
            fn_data_path = self.intermediate_subdir_name / "fn_enriched_df.parquet" 
            plot_save_path = self.OUTPUT_PLOT_DIR / "fn_size_distribution_log.png"

            if not fn_data_path.exists():
                print(f"Warning: Input file not found: {fn_data_path}. Skipping plot.")
                return

            self.visualizer.plot_fn_size_distribution(
                fn_data=fn_data_path, 
                use_log_scale=True,   
                bins=50,              
                save_path=plot_save_path
            )
        except Exception as e:
            print(f"Error generating FN Size plot: {e}")


    def generate_fp_confidence_plot(self):
        """Generates and saves the FP Confidence Distribution histogram."""
        print("-" * 10 + " Generating FP Confidence Distribution Plot " + "-" * 10)
        try:
            
            fp_data_path = self.intermediate_subdir_name / "fp_enriched_df.parquet" 
            plot_save_path = self.OUTPUT_PLOT_DIR / "fp_confidence_distribution.png"

            if not fp_data_path.exists():
                print(f"Warning: Input file not found: {fp_data_path}. Skipping plot.")
                return

            self.visualizer.plot_fp_confidence_distribution(
                fp_data=fp_data_path, 
                bins=25,              
                save_path=plot_save_path
            )
        except Exception as e:
            print(f"Error generating FP Confidence plot: {e}")


    def generate_fn_visibility_comparison_plot(self):
        """Generates and saves the FN vs GT Visibility Rate comparison plots."""
        print("-" * 10 + " Generating FN vs GT Visibility Rate Plots " + "-" * 10)
        try:
            fn_data_path = self.intermediate_subdir_name / "fn_enriched_df.parquet"
            gt_rates_path = Path("data_analysis/out_data_analysis/gt_analysis/occ_trunc_rates_gt.parquet") 
            plot_save_prefix = self.OUTPUT_PLOT_DIR / "fn_vs_gt_visibility"

            if not fn_data_path.exists():
                print(f"Warning: FN data file not found: {fn_data_path}. Skipping plot.")
                return
            if not gt_rates_path.exists():
                print(f"Warning: Overall GT rates file not found: {gt_rates_path}. Skipping plot.")
                return

            self.visualizer.plot_fn_visibility_comparison(
                fn_data=fn_data_path,
                overall_gt_rates_data=gt_rates_path,
                save_path_prefix=plot_save_prefix
            )
        except Exception as e:
            print(f"Error generating FN Visibility comparison plot: {e}")


    def __call__(self):
        self.generate_recall_vs_size_plot()

        # Call the visibility plot function:
        self.generate_recall_vs_visibility_plot()

        # Call the attribute plot functions:
        self.generate_recall_vs_timeofday_plot()
        
        self.generate_recall_vs_weather_plot()

        # Call failure count plots:
        self.generate_fn_count_plot()
        
        self.generate_fp_count_plot()

        # Call the FN size distribution plot function:
        self.generate_fn_size_plot()

        # Call the FP confidence distribution plot function:
        self.generate_fp_confidence_plot()

        # Call the FN visibility comparison plot function:
        self.generate_fn_visibility_comparison_plot()
        
        print("=" * 30)
        print("Eval Visualization Generation Finished")