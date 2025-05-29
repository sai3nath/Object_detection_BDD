import json
import os
from pathlib import Path
from PIL import Image
import argparse
from collections import Counter

# Define the mapping from BDD category names found in YOUR JSON data
# to YOLO class indices (0-9). This order must match your data.yaml 'names'.
CLASS_MAP = {
    "traffic light": 0,
    "traffic sign": 1,
    "car": 2,
    "person": 3,
    "bus": 4,
    "truck": 5,
    "rider": 6,
    "bike": 7,      
    "motor": 8,      
    "train": 9
}
print(f"Using Final Class Map: {CLASS_MAP}")

def convert_bdd_box_to_yolo(box, img_width, img_height):
    """
    Converts BDD box format [x1, y1, x2, y2] to YOLO format
    [x_center_norm, y_center_norm, width_norm, height_norm].

    Args:
        box (dict): BDD box2d dictionary {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        tuple: (x_center_norm, y_center_norm, width_norm, height_norm) or None if invalid box/coords.
    """
    x1 = box['x1']
    y1 = box['y1']
    x2 = box['x2']
    y2 = box['y2']

    # Basic validation for box coordinates relative to image dimensions
    if x2 <= x1 or y2 <= y1 or img_width <= 0 or img_height <= 0:
        return None
    x1, x2 = max(0.0, x1), min(img_width, x2)
    y1, y2 = max(0.0, y1), min(img_height, y2)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None


    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0


    x_center_norm = x_center * dw
    y_center_norm = y_center * dh
    width_norm = width * dw
    height_norm = height * dh

    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))


    return x_center_norm, y_center_norm, width_norm, height_norm

def convert_bdd_json_to_yolo(json_label_path, image_dir, output_label_dir):
    """
    Converts a BDD JSON label file to YOLO format .txt files.

    Args:
        json_label_path (str): Path to the BDD JSON label file (e.g., labels_images_val.json).
        image_dir (str): Path to the directory containing the corresponding images (e.g., Images/val).
        output_label_dir (str): Path to the directory where YOLO .txt files will be saved.
    """
    print(f"Loading JSON data from: {json_label_path}")
    try:
        with open(json_label_path, 'r') as f:
            bdd_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_label_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_label_path}")
        return

    print(f"Creating output directory: {output_label_dir}")
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(bdd_data)} image entries...")
    processed_count = 0
    label_count = 0
    skipped_images = 0
    category_counts = Counter() 

    for i, item in enumerate(bdd_data):
        image_name = item.get('name')
        if not image_name:
            skipped_images += 1 
            continue

        image_path = Path(image_dir) / image_name
        output_txt_path = Path(output_label_dir) / f"{Path(image_name).stem}.txt"

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except FileNotFoundError:
            skipped_images += 1
            continue
        except Exception as e:
            print(f"Warning: Could not open or read image {image_path}. Error: {e}. Skipping entry.")
            skipped_images += 1
            continue

        labels = item.get('labels')
        lines_to_write = [] 

        if labels:
            for label in labels:
                category = label.get('category')
                box2d = label.get('box2d')

                if category in CLASS_MAP and box2d:
                    class_index = CLASS_MAP[category]

                    yolo_coords = convert_bdd_box_to_yolo(box2d, img_width, img_height)

                    if yolo_coords:
                        x_center_norm, y_center_norm, width_norm, height_norm = yolo_coords
                        lines_to_write.append(f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
                        label_count += 1
                        category_counts[category] += 1 

        try:
            with open(output_txt_path, 'w') as f_out:
                for line in lines_to_write:
                    f_out.write(line + "\n")
            processed_count += 1
        except IOError as e:
             print(f"Warning: Could not write label file {output_txt_path}. Error: {e}. Skipping image entry.")
             skipped_images +=1


        if (processed_count + skipped_images) % 1000 == 0:
            print(f"  Checked {processed_count + skipped_images}/{len(bdd_data)} entries (Processed: {processed_count}, Skipped: {skipped_images})...")

    print("-" * 30)
    print("Conversion Summary:")
    print(f"  Total image entries in JSON: {len(bdd_data)}")
    print(f"  Images processed (labels potentially written): {processed_count}")
    print(f"  Images skipped (no name, file not found, read/write error): {skipped_images}")
    print(f"  Total object labels converted: {label_count}")
    print(f"  Labels per category (converted):")
    for category, count in category_counts.most_common():
         print(f"    - {category}: {count}")
    print(f"\n  YOLO format labels saved to: {output_label_dir}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BDD JSON labels to YOLO TXT format.")
    parser.add_argument('--json_path', type=str, required=True,
                        help="Path to the BDD JSON label file (e.g., bdd_100k/labels/labels_images_val.json)")
    parser.add_argument('--image_dir', type=str, required=True,
                        help="Path to the directory containing corresponding images (e.g., bdd_100k/Images/val)")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path to the directory where YOLO .txt label files will be saved (e.g., bdd_100k/labels_yolo/val)")

    args = parser.parse_args()

    # --- Run Conversion ---
    convert_bdd_json_to_yolo(args.json_path, args.image_dir, args.output_dir)
