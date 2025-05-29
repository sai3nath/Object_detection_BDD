import os
import random
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

def create_subset(original_image_dir, original_label_dir, subset_image_dir, subset_label_dir, num_samples):
    """
    Creates a random subset of images and their corresponding YOLO labels.

    Args:
        original_image_dir (str): Path to the directory containing the full set of training images.
        original_label_dir (str): Path to the directory containing the full set of YOLO .txt labels.
        subset_image_dir (str): Path to the directory where subset images will be saved.
        subset_label_dir (str): Path to the directory where subset labels will be saved.
        num_samples (int): The number of image/label pairs to randomly sample.
    """
    print("--- Starting Subset Creation ---")
    orig_img_path = Path(original_image_dir)
    orig_lbl_path = Path(original_label_dir)
    subset_img_path = Path(subset_image_dir)
    subset_lbl_path = Path(subset_label_dir)

    # Create subset directories if they don't exist
    print(f"Creating subset image directory: {subset_img_path}")
    subset_img_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating subset label directory: {subset_lbl_path}")
    subset_lbl_path.mkdir(parents=True, exist_ok=True)

    # Get list of original images (assuming .jpg, change if needed)
    print(f"Scanning original image directory: {orig_img_path}")
    try:
        # Using list comprehension for potentially large directories
        image_files = [f for f in orig_img_path.glob('*.jpg') if f.is_file()]
        print(f"Found {len(image_files)} original images.")
        if not image_files:
            print("Error: No image files found in the original image directory.")
            return
    except Exception as e:
        print(f"Error scanning original image directory: {e}")
        return

    # Check if requested sample size is valid
    num_available = len(image_files)
    if num_samples > num_available:
        print(f"Warning: Requested {num_samples} samples, but only {num_available} images available.")
        print(f"Sampling all {num_available} available images.")
        num_samples = num_available
    elif num_samples <= 0:
        print("Error: Number of samples must be positive.")
        return

    # Randomly sample image paths
    print(f"Randomly selecting {num_samples} images...")
    selected_image_paths = random.sample(image_files, num_samples)

    # Copy selected images and their corresponding labels
    print(f"Copying selected image/label pairs...")
    copied_count = 0
    skipped_count = 0
    for img_path in tqdm(selected_image_paths, unit="pair"):
        # Construct corresponding label path
        label_filename = img_path.stem + '.txt'
        expected_label_path = orig_lbl_path / label_filename

        # Define destination paths
        dest_img_path = subset_img_path / img_path.name
        dest_lbl_path = subset_lbl_path / label_filename

        # Check if the label file exists before copying
        if expected_label_path.is_file():
            try:
                shutil.copy2(img_path, dest_img_path) # copy2 preserves metadata
                shutil.copy2(expected_label_path, dest_lbl_path)
                copied_count += 1
            except Exception as e:
                print(f"\nError copying {img_path.name} or its label: {e}")
                skipped_count += 1
        else:
            print(f"\nWarning: Label file not found for {img_path.name} at {expected_label_path}. Skipping pair.")
            skipped_count += 1

    print("\n--- Subset Creation Summary ---")
    print(f"Requested samples: {num_samples}")
    print(f"Successfully copied image/label pairs: {copied_count}")
    print(f"Skipped pairs (due to missing labels or copy errors): {skipped_count}")
    print(f"Subset images saved to: {subset_img_path}")
    print(f"Subset labels saved to: {subset_lbl_path}")
    print("------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a random subset of images and YOLO labels.")
    parser.add_argument('--original_image_dir', type=str, required=True,
                        help="Path to the full original training image directory.")
    parser.add_argument('--original_label_dir', type=str, required=True,
                        help="Path to the full original training label (.txt) directory.")
    parser.add_argument('--subset_image_dir', type=str, required=True,
                        help="Path to the directory where subset images will be saved.")
    parser.add_argument('--subset_label_dir', type=str, required=True,
                        help="Path to the directory where subset labels will be saved.")
    parser.add_argument('--num_samples', type=int, default=2000,
                        help="Number of images to randomly sample (default: 2000).")

    args = parser.parse_args()

    create_subset(
        args.original_image_dir,
        args.original_label_dir,
        args.subset_image_dir,
        args.subset_label_dir,
        args.num_samples
    )


# python3 create_subset.py \
#     --original_image_dir /Documents/personal/obd/data_bdd/bdd100k/images/100k/val \
#     --original_label_dir /Documents/personal/obd/data_bdd/bdd100k/labels/100k/val \
#     --subset_image_dir /Documents/personal/obd/data_bdd/bdd100k/subset_5000/images/val_subset \
#     --subset_label_dir /Documents/personal/obd/data_bdd/bdd100k/subset_5000/labels/val_subset \
#     --num_samples 1000


# python3 create_subset.py \
#     --original_image_dir /Documents/personal/obd/data_bdd/bdd100k/images/100k/train \
#     --original_label_dir /Documents/personal/obd/data_bdd/bdd100k/labels/100k/train \
#     --subset_image_dir /Documents/personal/obd/data_bdd/bdd100k/subset_5000/images/train_subset \
#     --subset_label_dir /Documents/personal/obd/data_bdd/bdd100k/subset_5000/labels/train_subset \
#     --num_samples 5000