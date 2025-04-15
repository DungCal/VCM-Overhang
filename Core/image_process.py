import cv2
import numpy as np
import os
import time
import multiprocessing as mp
from pathlib import Path

# Constants (Consider moving these to a config file if they need to be easily changed)
MATCH_METHOD = cv2.TM_SQDIFF_NORMED
MIN_THRESHOLD_FACTOR = 0.1
IMAGE_SPLIT_RATIO = 0.5
DEBUG = False
CROP_WIDTH = 210  # Desired output crop width
CROP_HEIGHT = 60   # Desired output crop height
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg')  # Tuple for faster lookup


def load_templates(template_folder_path):
    """Loads templates from a folder.  Returns a list of tuples (template, width, height)."""
    templates = []
    try:
        template_folder = Path(template_folder_path)
        template_files = [f for f in template_folder.glob(
            "*") if f.suffix.lower() in ALLOWED_EXTENSIONS]

        for template_file in template_files:
            try:
                template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if template is None:
                    print(
                        f"Warning: Could not read template at: {template_file}")
                    continue  # Skip to the next template
                templates.append(
                    (template, template.shape[1], template.shape[0]))
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")
    except FileNotFoundError:
        print(f"Warning: Template folder not found at: {template_folder_path}")
    except Exception as e:
        print(f"Error loading templates from folder: {e}")
    return templates


def crop_and_save(image, center_x, center_y, crop_width, crop_height, output_path, filename_base, suffix, file_extension):
    """Crops a region from an image and saves it, centering the crop.  Handles boundary conditions."""
    try:
        # Calculate top-left corner of the crop
        x_start = int(center_x - crop_width / 2)
        y_start = int(center_y - crop_height / 2)

        # Clip the coordinates to ensure they're within the image boundaries.  This is much more concise.
        x_start = np.clip(x_start, 0, image.shape[1] - crop_width)
        y_start = np.clip(y_start, 0, image.shape[0] - crop_height)

        # Extract the crop.  Now safe because of clipping.
        crop = image[y_start:y_start + crop_height,
                     x_start:x_start + crop_width]

        # Save the crop
        output_filename = f"{filename_base}_crop_{suffix}{file_extension}"
        output_crop_path = os.path.join(output_path, output_filename)
        cv2.imwrite(output_crop_path, crop)

    except Exception as e:
        print(f"Error cropping and saving: {e}")


def detect_best_match(img_gray, templates, match_method, min_threshold_factor):
    """Detects the best match of a template in an image."""
    best_loc = None
    best_val = float('inf') if match_method in [
        cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else float('-inf')
    best_template = None
    best_template_width = None
    best_template_height = None
    is_sqdiff = match_method in [cv2.TM_SQDIFF,
                                 cv2.TM_SQDIFF_NORMED]

    for template, template_width, template_height in templates:
        try:
            res = cv2.matchTemplate(img_gray, template, match_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if is_sqdiff:
                best_loc_temp = min_loc
                best_val_temp = min_val
                threshold = min_val + \
                    (min_threshold_factor * (max_val - min_val))
                reject = best_val_temp > threshold
            else:
                best_loc_temp = max_loc
                best_val_temp = max_val
                threshold = max_val - \
                    (min_threshold_factor * (max_val - min_val))
                reject = best_val_temp < threshold

            if not reject:
                if is_sqdiff and best_val_temp < best_val:
                    best_val = best_val_temp
                    best_loc = best_loc_temp
                    best_template = template
                    best_template_width = template_width
                    best_template_height = template_height
                elif not is_sqdiff and best_val_temp > best_val:
                    best_val = best_val_temp
                    best_loc = best_loc_temp
                    best_template = template
                    best_template_width = template_width
                    best_template_height = template_height

        except Exception as e:
            print(f"Error in template matching: {e}")

    if DEBUG and best_loc is not None:
        print(f"Best value: {best_val},")
    return best_loc, best_template, best_template_width, best_template_height


def process_image(image_path, templates, output_path):
    """Processes a single image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(
                f"Warning: Could not read image {os.path.basename(image_path)}. Skipping.")
            return False

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        filename_base, file_extension = os.path.splitext(
            os.path.basename(image_path))

        image_width = img_gray.shape[1]
        split_point = int(image_width * IMAGE_SPLIT_RATIO)

        # Process left ROI
        left_roi = img_gray[:, :split_point]
        left_match, used_template, template_width, template_height = detect_best_match(
            left_roi, templates, MATCH_METHOD, MIN_THRESHOLD_FACTOR)

        if left_match:
            left_x, left_y = left_match
            center_x = left_x + template_width / 2
            center_y = left_y + template_height / 2
            save_name = "left"

            if DEBUG:
                cv2.rectangle(img, (left_x, left_y), (left_x + CROP_WIDTH,
                              left_y + CROP_HEIGHT), (0, 255, 0), 2)

            crop_and_save(img, center_x, center_y, CROP_WIDTH, CROP_HEIGHT,
                          output_path, filename_base, save_name, file_extension)

        # Process right ROI
        right_roi = img_gray[:, split_point:]
        right_match, used_template, template_width, template_height = detect_best_match(
            right_roi, templates, MATCH_METHOD, MIN_THRESHOLD_FACTOR)

        if right_match:
            right_x, right_y = right_match
            right_x += split_point
            center_x = right_x + template_width / 2
            center_y = right_y + template_height / 2
            save_name = "right"

            if DEBUG:
                cv2.rectangle(img, (right_x, right_y), (right_x +
                              CROP_WIDTH, right_y + CROP_HEIGHT), (0, 255, 0), 2)

            crop_and_save(img, center_x, center_y, CROP_WIDTH, CROP_HEIGHT,
                          output_path, filename_base, save_name, file_extension)

        if DEBUG:
            debug_output_path = os.path.join(
                output_path, f"{filename_base}_debug{file_extension}")
            cv2.imwrite(debug_output_path, img)

        return True

    except Exception as e:
        print(f"Error processing image {os.path.basename(image_path)}: {e}")
        return False


def process_image_wrapper(args):
    """Wrapper function for parallel processing."""
    return process_image(*args)


def process_images(input_folder_path, output_folder_path, template_folder_path):
    """Processes all images in a folder."""
    try:
        templates = load_templates(template_folder_path)
        if not templates:
            print("No templates loaded. Exiting.")
            return

        output_path_obj = Path(output_folder_path)
        output_path_obj.mkdir(parents=True, exist_ok=True)

        input_folder = Path(input_folder_path)
        image_files = [f for f in input_folder.glob(
            "*") if f.suffix.lower() in ALLOWED_EXTENSIONS]
        total_images = len(image_files)
        processed_count = 0
        skipped_count = 0

        image_args = [(str(image_file), templates, output_folder_path)
                      for image_file in image_files]

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(process_image_wrapper, image_args)

        processed_count = sum(results)
        skipped_count = total_images - processed_count

        print(
            f"\nProcessed {processed_count} images. Skipped {skipped_count} images.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    start_time = time.time()

    input_path = r"E:\02.pdx\pdx25-overhang\data\OK"
    output_path = r'E:\02.pdx\pdx25-overhang\data\cropped'
    template_folder_path = r'E:\02.pdx\pdx25-overhang\data\template'
    process_images(input_path, output_path, template_folder_path)

    print("Run time: ", time.time()-start_time)
