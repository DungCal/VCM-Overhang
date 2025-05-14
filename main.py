import os
import cv2
import numpy as np
import csv
import glob
from Core.image_process import load_templates, process_image as crop_and_template_match
from Core.overhang_detection import process_image as detect_overhang
import multiprocessing as mp
import shutil

# Define input and output paths
INPUT_TYPE = r"Validate"
INPUT_FOLDER = r"E:/02.pdx/pdx25-overhang/data/" + INPUT_TYPE
TEMPLATE_PATH = r'E:/02.pdx/pdx25-overhang/data/template'
CROPPED_OUTPUT_FOLDER = r'E:/02.pdx/pdx25-overhang/out/cropped_output'
DETECT_OUTPUT_FOLDER = r'E:/02.pdx/pdx25-overhang/out/detect_output'
CSV_OUTPUT_PATH = r'E:/02.pdx/pdx25-overhang/out/results.csv'
DRAW_FOLDER = r"E:/02.pdx/pdx25-overhang/out/draw_images"

# Define parameters for template matching
MATCH_METHOD = cv2.TM_SQDIFF_NORMED
MIN_THRESHOLD_FACTOR = 0.1
IMAGE_SPLIT_RATIO = 0.5

# Define parameters for line detection
MARGIN = 50
CANNY_THRESHOLD1_FULL = 100
CANNY_THRESHOLD2_FULL = 100
HOUGH_THRESHOLD_FULL = 20
HOUGH_MIN_LINE_LENGTH_FULL = 10
HOUGH_MAX_LINE_GAP_FULL = 10

CANNY_THRESHOLD1_CROPPED = 30
CANNY_THRESHOLD2_CROPPED = 30
HOUGH_THRESHOLD_CROPPED = 10
HOUGH_MIN_LINE_LENGTH_CROPPED = 3
HOUGH_MAX_LINE_GAP_CROPPED = 20

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180

DETECT_THREADHOLD = 1


def process_single_image(image_path, templates, CROPPED_OUTPUT_FOLDER, DETECT_OUTPUT_FOLDER, DRAW_FOLDER, MARGIN, CANNY_THRESHOLD1_FULL,
                         CANNY_THRESHOLD2_FULL, HOUGH_THRESHOLD_FULL,   HOUGH_MIN_LINE_LENGTH_FULL, HOUGH_MAX_LINE_GAP_FULL, CANNY_THRESHOLD1_CROPPED, CANNY_THRESHOLD2_CROPPED, HOUGH_THRESHOLD_CROPPED, HOUGH_MIN_LINE_LENGTH_CROPPED, HOUGH_MAX_LINE_GAP_CROPPED, HOUGH_RHO, HOUGH_THETA, INPUT_TYPE, DETECT_THREADHOLD):
    """
        Processes overhang detection for a single image.
    """
    image_name = os.path.basename(image_path)
    filename_base, file_extension = os.path.splitext(image_name)

    try:
        # Crop the image based on template matching
        crop_and_template_match(image_path, templates, CROPPED_OUTPUT_FOLDER)

        left_crop_path = os.path.join(
            CROPPED_OUTPUT_FOLDER, f"{filename_base}_crop_left{file_extension}")
        right_crop_path = os.path.join(
            CROPPED_OUTPUT_FOLDER, f"{filename_base}_crop_right{file_extension}")

        left_crop_exists = os.path.exists(left_crop_path)
        right_crop_exists = os.path.exists(right_crop_path)

    except Exception as e:
        print(f"Error during cropping {image_name}: {e}")
        return [image_name, INPUT_TYPE, "ERROR_CROPPING"]

    final_result = "OK"
    # Detect overhang on the left crop
    if left_crop_exists:
        try:
            result_left = detect_overhang(
                left_crop_path,
                DETECT_OUTPUT_FOLDER,
                DRAW_FOLDER,
                margin=MARGIN,
                canny_threshold1_full=CANNY_THRESHOLD1_FULL,
                canny_threshold2_full=CANNY_THRESHOLD2_FULL,
                hough_rho=HOUGH_RHO,
                hough_theta=HOUGH_THETA,
                hough_threshold_full=HOUGH_THRESHOLD_FULL,
                canny_threshold1_cropped=CANNY_THRESHOLD1_CROPPED,
                hough_threshold_cropped=HOUGH_THRESHOLD_CROPPED,
                hough_minLineLength_full=HOUGH_MIN_LINE_LENGTH_FULL,
                canny_threshold2_cropped=CANNY_THRESHOLD2_CROPPED,
                hough_minLineLength_cropped=HOUGH_MIN_LINE_LENGTH_CROPPED,
                hough_maxLineGap_full=HOUGH_MAX_LINE_GAP_FULL,
                hough_maxLineGap_cropped=HOUGH_MAX_LINE_GAP_CROPPED,
                detect_threadhold=DETECT_THREADHOLD
            )
            if result_left == "NG":
                final_result = "NG"

        except Exception as e:
            print(
                f"Error detecting overhang on left crop {image_name}: {e}")
            final_result = "ERROR_DETECT_LEFT"
    else:
        final_result = "NG"

    # Detect overhang on the right crop
    if right_crop_exists:
        try:
            result_right = detect_overhang(
                right_crop_path,
                DETECT_OUTPUT_FOLDER,
                DRAW_FOLDER,
                margin=MARGIN,
                canny_threshold1_full=CANNY_THRESHOLD1_FULL,
                canny_threshold2_full=CANNY_THRESHOLD2_FULL,
                hough_rho=HOUGH_RHO,
                hough_theta=HOUGH_THETA,
                hough_threshold_full=HOUGH_THRESHOLD_FULL,
                canny_threshold1_cropped=CANNY_THRESHOLD1_CROPPED,
                hough_threshold_cropped=HOUGH_THRESHOLD_CROPPED,
                hough_minLineLength_full=HOUGH_MIN_LINE_LENGTH_FULL,
                canny_threshold2_cropped=CANNY_THRESHOLD2_CROPPED,
                hough_minLineLength_cropped=HOUGH_MIN_LINE_LENGTH_CROPPED,
                hough_maxLineGap_full=HOUGH_MAX_LINE_GAP_FULL,
                hough_maxLineGap_cropped=HOUGH_MAX_LINE_GAP_CROPPED,
                detect_threadhold=DETECT_THREADHOLD
            )

            if result_right == "NG":
                final_result = "NG"
        except Exception as e:
            print(
                f"Error detecting overhang on right crop {image_name}: {e}")
            final_result = "ERROR_DETECT_RIGHT"
    else:
        final_result = "NG"

    # Save the image
    output_folder = os.path.join(DETECT_OUTPUT_FOLDER, final_result)
    output_filename = f"{filename_base}{file_extension}"
    output_path_final = os.path.join(output_folder, output_filename)

    try:

        os.makedirs(output_folder, exist_ok=True)
        image = cv2.imread(image_path)
        if image is not None:
            cv2.imwrite(output_path_final, image)
        else:
            print(f"Error: Could not read image {image_path} for saving.")
            return [image_name, INPUT_TYPE, "ERROR_SAVING_IMAGE_LOAD"]
    except Exception as e:
        print(
            f"Error saving image {image_name} to {output_folder} folder: {e}")
        return [image_name, INPUT_TYPE, "ERROR_SAVING"]

    print(f"{image_name} - Final Result: {final_result}")

    return [image_name, INPUT_TYPE, final_result]


def process_image_wrapper(args):
    """Wrapper function for parallel processing."""
    return process_single_image(*args)


def main():
    os.makedirs(CROPPED_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DETECT_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DRAW_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(DETECT_OUTPUT_FOLDER, "OK"), exist_ok=True)
    os.makedirs(os.path.join(DETECT_OUTPUT_FOLDER, "NG"), exist_ok=True)

    # Load templates
    try:
        templates = load_templates(TEMPLATE_PATH)
    except FileNotFoundError as e:
        print(f"Error: Template file not found: {e}")
        return
    except Exception as e:
        print(f"Error loading template: {e}")
        return

    image_files = [f for f in glob.glob(os.path.join(
        INPUT_FOLDER, "*.png")) + glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))]

    if not image_files:
        print("No images found in the input folder.")
        return

    image_args = [(
        image_path, templates,
        CROPPED_OUTPUT_FOLDER, DETECT_OUTPUT_FOLDER, DRAW_FOLDER,
        MARGIN,
        CANNY_THRESHOLD1_FULL, CANNY_THRESHOLD2_FULL,
        HOUGH_THRESHOLD_FULL, HOUGH_MIN_LINE_LENGTH_FULL, HOUGH_MAX_LINE_GAP_FULL,
        CANNY_THRESHOLD1_CROPPED, CANNY_THRESHOLD2_CROPPED,
        HOUGH_THRESHOLD_CROPPED, HOUGH_MIN_LINE_LENGTH_CROPPED, HOUGH_MAX_LINE_GAP_CROPPED,
        HOUGH_RHO, HOUGH_THETA, INPUT_TYPE, DETECT_THREADHOLD
    ) for image_path in image_files]

    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_image_wrapper, image_args)

    # Write results to a CSV file
    file_exists = os.path.exists(CSV_OUTPUT_PATH)
    mode = 'a' if file_exists else 'w'

    with open(CSV_OUTPUT_PATH, mode, newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(["Image Name", "Origin", "Result"])
        csvwriter.writerows(results)

    print(f"Results saved to {CSV_OUTPUT_PATH}")

    if os.path.exists(CROPPED_OUTPUT_FOLDER):
        shutil.rmtree(CROPPED_OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
