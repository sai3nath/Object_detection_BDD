import argparse
from pathlib import Path
from PIL import Image
from collections import Counter

def get_image_size_distribution(image_dir):
    """
    Calculates the distribution of image sizes (width, height) in a directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        tuple: (collections.Counter, int, int)
               - Counter mapping (width, height) tuples to their counts.
               - Total number of valid images processed.
               - Total number of files skipped due to errors.
    """
    image_path = Path(image_dir)
    size_counter = Counter()
    image_count = 0
    error_count = 0

    print(f"Scanning directory: {image_path}")

    files_to_check = list(image_path.rglob('*'))
    total_files = len(files_to_check)
    print(f"Found {total_files} potential files to check...")

    for i, file_path in enumerate(files_to_check):
        if file_path.is_dir():
            continue

        try:
            with Image.open(file_path) as img:
                size = img.size 
                size_counter[size] += 1
                image_count += 1

        except FileNotFoundError:
            print(f"Warning: File not found during processing: {file_path}")
            error_count += 1
        except Image.UnidentifiedImageError:
            error_count += 1
        except Exception as e:
            print(f"Warning: Error processing file {file_path}: {e}. Skipping.")
            error_count += 1

        if (i + 1) % 1000 == 0:
            print(f"  Checked {i + 1}/{total_files} files...")

    print(f"Finished scanning. Processed {image_count} valid images.")
    return size_counter, image_count, error_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get distribution of image sizes in a directory.")
    parser.add_argument('--image_dir', type=str, required=True,
                        help="Path to the directory containing image files.")

    args = parser.parse_args()

    if not Path(args.image_dir).is_dir():
        print(f"Error: Directory not found at {args.image_dir}")
    else:
        size_distribution, total_images, total_errors = get_image_size_distribution(args.image_dir)

        print("\n" + "="*30)
        print("Image Size Distribution Summary")
        print("="*30)
        print(f"Directory Scanned: {args.image_dir}")
        print(f"Total Valid Images Found: {total_images}")
        if total_errors > 0:
             print(f"Total Skipped Files (Non-image/Error): {total_errors}")
        print(f"Number of Unique Image Sizes: {len(size_distribution)}")
        print("-" * 30)

        if size_distribution:
            print("Sizes Found (Width x Height): Count")
            for size, count in size_distribution.most_common():
                print(f"  {size[0]} x {size[1]}: {count}")
        else:
            print("No valid images found or processed.")
        print("="*30)