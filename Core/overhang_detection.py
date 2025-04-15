import cv2
import numpy as np
import os
import glob

# Constants
PI_180 = np.pi / 180
SAVE_IMAGE_CROP = False
SAVE_IMAGE_LINES = True

# --- Line Detection Functions ---


def detect_lines(image, margin, canny_threshold1, canny_threshold2, hough_rho,
                 hough_theta, hough_threshold, hough_min_line_length,
                 hough_max_line_gap, contrast_enhance=False, crop=False):
    try:
        working_image = image.copy()

        if contrast_enhance:
            working_image = increase_contrast(
                working_image, clip_limit=1.0, tile_grid_size=(3, 3))

        if crop:
            height, width, _ = working_image.shape
            working_image = working_image[:, margin:width - margin]

        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(
            blurred,
            threshold1=canny_threshold1,
            threshold2=canny_threshold2,
            apertureSize=3,
            L2gradient=True if not crop else False
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=hough_rho,
            theta=hough_theta,
            threshold=hough_threshold,
            minLineLength=hough_min_line_length,
            maxLineGap=hough_max_line_gap
        )

        lines_list = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                lines_list.append([x1, y1, x2, y2])

        return edges, lines_list

    except Exception as e:
        print(f"Error in detect_lines: {e}")
        return None, None


def detect_HTCC_line(image, margin, canny_threshold1, canny_threshold2, hough_rho,
                     hough_theta, hough_threshold, hough_min_line_length,
                     hough_max_line_gap, angle_threshold):
    edges, lines = detect_lines(image, margin, canny_threshold1, canny_threshold2, hough_rho,
                                hough_theta, hough_threshold, hough_min_line_length,
                                hough_max_line_gap, contrast_enhance=True, crop=False)

    max_len = 0
    max_line = None
    max_y = -1

    if lines is not None:
        height, width = image.shape[:2]
        for x1, y1, x2, y2 in lines:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if not (-angle_threshold <= angle <= angle_threshold and (x1 <= margin or x2 >= width - margin)):
                continue

            if length > max_len:
                max_len = length
                max_line = [x1, y1, x2, y2]
                max_y = max(y1, y2)

    return edges, max_y, max_line


def detect_VCM_line(image, margin, canny_threshold1, canny_threshold2, hough_rho,
                    hough_theta, hough_threshold, hough_min_line_length,
                    hough_max_line_gap, angle_threshold):
    edges, lines = detect_lines(image, margin, canny_threshold1, canny_threshold2, hough_rho,
                                hough_theta, hough_threshold, hough_min_line_length,
                                hough_max_line_gap, contrast_enhance=False, crop=True)
    max_y = -1
    max_line = None

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if not (-angle_threshold <= angle <= angle_threshold):
                continue

            current_max_y = max(y1, y2)
            if current_max_y > max_y:
                max_y = current_max_y
                max_line = [x1, y1, x2, y2]

    return edges, max_y, max_line


def increase_contrast(image, clip_limit=0.5, tile_grid_size=(3, 3)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_image


def draw_line(image, y, color=(0, 0, 255), thickness=1):
    height, width = image.shape[:2]
    cv2.line(image, (0, int(y)), (width - 1, int(y)), color, thickness)
    return image


def process_image(image_file, output_folder_base, draw_folder, margin,
                  canny_threshold1_full, canny_threshold1_cropped,
                  canny_threshold2_full, canny_threshold2_cropped,
                  hough_rho, hough_theta,
                  hough_threshold_full, hough_threshold_cropped,
                  hough_minLineLength_full, hough_minLineLength_cropped,
                  hough_maxLineGap_full, hough_maxLineGap_cropped):
    ok_folder = os.path.join(output_folder_base, "OK")
    ng_folder = os.path.join(output_folder_base, "NG")

    try:
        raw_image = cv2.imread(image_file)
        if raw_image is None:
            print(f"Error: Could not read image {image_file}")
            return

        image_copy = raw_image.copy()

        # Detect HTCC line
        edges_original, max_y_original, max_line_original = detect_HTCC_line(
            image_copy, margin, canny_threshold1_full, canny_threshold2_full, hough_rho, hough_theta,
            hough_threshold_full, hough_minLineLength_full, hough_maxLineGap_full, angle_threshold=3
        )

        # Detect VCM line
        edges_cropped, max_y_cropped, max_line_cropped = detect_VCM_line(
            image_copy, margin, canny_threshold1_cropped, canny_threshold2_cropped, hough_rho, hough_theta,
            hough_threshold_cropped, hough_minLineLength_cropped, hough_maxLineGap_cropped, angle_threshold=5
        )

        filename = os.path.basename(image_file)
        name, ext = os.path.splitext(filename)

        status = "NG"
        if max_line_original is not None and max_line_cropped is not None and max_y_original <= max_y_cropped:
            output_path = os.path.join(ok_folder, filename)

            if SAVE_IMAGE_CROP:
                print(f"Saving {filename} to OK folder (cropped line lower).")
                cv2.imwrite(output_path, raw_image)
            status = "OK"
        else:
            output_path = os.path.join(ng_folder, filename)
            if SAVE_IMAGE_CROP:
                print(f"Saving {filename} to NG folder.")
                cv2.imwrite(output_path, raw_image)

        drawn_image = raw_image.copy()
        if max_y_original != -1:
            drawn_image = draw_line(drawn_image, max_y_original, color=(
                0, 0, 255), thickness=1)  # Original line in red
        if max_y_cropped != -1:
            drawn_image = draw_line(drawn_image, max_y_cropped, color=(
                0, 255, 0), thickness=1)  # Cropped line in green

        # Save the draw image
        draw_output_path = os.path.join(
            draw_folder, f"{name}_{status}{ext}")
        edges_original_output_path = os.path.join(
            draw_folder, f"{name}_edges_original_{status}{ext}")
        edges_cropped_output_path = os.path.join(
            draw_folder, f"{name}_edges_cropped_{status}{ext}")

        if SAVE_IMAGE_LINES:
            print(
                f"Saving drawn image {filename} to DRAW folder as {status}_{name}{ext}")
            cv2.imwrite(draw_output_path, drawn_image)

            if edges_original is not None:
                cv2.imwrite(edges_original_output_path, edges_original)
            if edges_cropped is not None:
                cv2.imwrite(edges_cropped_output_path, edges_cropped)

    except Exception as e:
        print(f"Error processing {image_file}: {e}")

    return status


def process_images(input_folder, output_folder_base, draw_folder, margin=60,
                   canny_threshold1_full=50, canny_threshold1_cropped=30,
                   canny_threshold2_full=100, canny_threshold2_cropped=50,
                   hough_rho=1, hough_theta=PI_180,
                   hough_threshold_full=20, hough_threshold_cropped=10,
                   hough_minLineLength_full=15, hough_minLineLength_cropped=1,
                   hough_maxLineGap_full=3, hough_maxLineGap_cropped=20):
    """Processes images in a folder by calling process_image for each."""

    ok_folder = os.path.join(output_folder_base, "OK")
    ng_folder = os.path.join(output_folder_base, "NG")

    os.makedirs(output_folder_base, exist_ok=True)
    os.makedirs(ok_folder, exist_ok=True)
    os.makedirs(ng_folder, exist_ok=True)
    os.makedirs(draw_folder, exist_ok=True)

    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(
        os.path.join(input_folder, "*.png"))

    for image_file in image_files:
        process_image(
            image_file,
            output_folder_base,
            draw_folder,
            margin,
            canny_threshold1_full,
            canny_threshold1_cropped,
            canny_threshold2_full,
            canny_threshold2_cropped,
            hough_rho,
            hough_theta,
            hough_threshold_full,
            hough_threshold_cropped,
            hough_minLineLength_full,
            hough_minLineLength_cropped,
            hough_maxLineGap_full,
            hough_maxLineGap_cropped
        )


def main():
    # Configuration Constants
    INPUT_FOLDER = r"E:/02.pdx/pdx25-overhang/data/cropped\NG"
    OUTPUT_BASE = r"E:/02.pdx/pdx25-overhang/data/"
    OUTPUT_FOLDER = OUTPUT_BASE + r"detect"
    DRAW_FOLDER = OUTPUT_BASE + r"draw_images"

    # Constants
    MARGIN = 50
    CANNY_THRESHOLD1_FULL = 50
    CANNY_THRESHOLD1_CROPPED = 30
    CANNY_THRESHOLD2_FULL = 100
    CANNY_THRESHOLD2_CROPPED = 50
    HOUGH_RHO = 1
    HOUGH_THETA = np.pi / 180
    HOUGH_THRESHOLD_FULL = 20
    HOUGH_THRESHOLD_CROPPED = 10
    HOUGH_MIN_LINE_LENGTH_FULL = 20
    HOUGH_MIN_LINE_LENGTH_CROPPED = 3
    HOUGH_MAX_LINE_GAP_FULL = 3
    HOUGH_MAX_LINE_GAP_CROPPED = 20

    process_images(
        input_folder=INPUT_FOLDER,
        output_folder_base=OUTPUT_FOLDER,
        draw_folder=DRAW_FOLDER,
        margin=MARGIN,
        canny_threshold1_full=CANNY_THRESHOLD1_FULL,
        canny_threshold1_cropped=CANNY_THRESHOLD1_CROPPED,
        canny_threshold2_full=CANNY_THRESHOLD2_FULL,
        canny_threshold2_cropped=CANNY_THRESHOLD2_CROPPED,
        hough_rho=HOUGH_RHO,
        hough_theta=HOUGH_THETA,
        hough_threshold_full=HOUGH_THRESHOLD_FULL,
        hough_threshold_cropped=HOUGH_THRESHOLD_CROPPED,
        hough_minLineLength_full=HOUGH_MIN_LINE_LENGTH_FULL,
        hough_minLineLength_cropped=HOUGH_MIN_LINE_LENGTH_CROPPED,
        hough_maxLineGap_full=HOUGH_MAX_LINE_GAP_FULL,
        hough_maxLineGap_cropped=HOUGH_MAX_LINE_GAP_CROPPED
    )


if __name__ == "__main__":
    main()
