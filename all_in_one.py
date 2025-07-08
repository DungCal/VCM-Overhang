import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp

# File and Path Configuration
BASE_PATH = Path(r"E:/02.pdx/pdx25-overhang")
INPUT_SUBFOLDER = r"20250704/FAA00021PC2"
INPUT_FOLDER = BASE_PATH / "data" / INPUT_SUBFOLDER
TEMPLATE_FOLDER = BASE_PATH / "data/template"
OUTPUT_BASE = BASE_PATH / "output" / INPUT_SUBFOLDER

# General Settings
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg')
NUM_PROCESSES = max(1, mp.cpu_count() - 1)

# Output Settings
SAVE_IMAGE_LINES = False
SAVE_EDGES_IMAGE = False

# Cropping and Template Matching Parameters
MATCH_METHOD = cv2.TM_SQDIFF_NORMED
SEARCH_WIDTH_RATIO = 0.5
EARLY_EXIT_THRESHOLD = 0.05
CROP_WIDTH = 280
TOP_OFFSET = 100
BOTTOM_OFFSET = -10

# Line Detection Parameters
DETECTION_PARAMS = {
    'angle_threshold': 5,
    'detect_threshold': 2,
    'htcc_hough': {
        'canny_threshold1': 50,
        'canny_threshold2': 100,
        'hough_rho': 1,
        'hough_theta': np.pi / 180,
        'hough_threshold': 20,
        'hough_min_line_length': 10,
        'hough_max_line_gap': 3,
    },
    'vcm_hough': {
        'canny_threshold1': 30,
        'canny_threshold2': 50,
        'hough_rho': 1,
        'hough_theta': np.pi / 180,
        'hough_threshold': 10,
        'hough_min_line_length': 3,
        'hough_max_line_gap': 20,
    }
}


def load_templates(template_folder_path: Path):
    """Loads all image templates from a specified folder."""
    templates = []
    if not template_folder_path.is_dir():
        return templates

    for template_file in template_folder_path.glob("*"):
        if template_file.suffix.lower() in ALLOWED_EXTENSIONS:
            try:
                template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates.append(
                        (template, template.shape[1], template.shape[0]))
            except Exception:
                pass
    return templates


def create_output_folders():
    """Creates the necessary output folders."""
    output_folders = {
        'ok': OUTPUT_BASE / "OK",
        'ng': OUTPUT_BASE / "NG",
        'crop': OUTPUT_BASE / "crop",
        'edges': OUTPUT_BASE / "crop"
    }
    for folder in output_folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return output_folders


# Image Processing
worker_templates = []
worker_clahe = None


def init_worker(templates_list):
    global worker_templates, worker_clahe
    worker_templates = templates_list
    worker_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def increase_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def detect_lines(image, **params):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred,
                          threshold1=params['canny_threshold1'],
                          threshold2=params['canny_threshold2'],
                          L2gradient=True)
        lines = cv2.HoughLinesP(edges,
                                rho=params['hough_rho'],
                                theta=params['hough_theta'],
                                threshold=params['hough_threshold'],
                                minLineLength=params['hough_min_line_length'],
                                maxLineGap=params['hough_max_line_gap'])
        return edges, lines[:, 0].tolist() if lines is not None else []
    except Exception:
        return None, []


def detect_best_match(search_area_gray, templates):
    best_loc, best_val, best_t_w, best_t_h = None, float('inf'), 0, 0
    for template, t_w, t_h in templates:
        if t_h > search_area_gray.shape[0] or t_w > search_area_gray.shape[1]:
            continue
        try:
            res = cv2.matchTemplate(search_area_gray, template, MATCH_METHOD)
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            if min_val < best_val:
                best_val, best_loc, best_t_w, best_t_h = min_val, min_loc, t_w, t_h
            if EARLY_EXIT_THRESHOLD > 0 and best_val < EARLY_EXIT_THRESHOLD:
                break
        except Exception:
            pass
    return best_loc, best_t_w, best_t_h


def detect_HTCC_line(image, angle_threshold, params):
    contrasted_image = increase_contrast(
        image, clip_limit=1.0, tile_grid_size=(3, 3))
    edges, lines = detect_lines(contrasted_image, **params)
    if not lines:
        return edges, -1

    max_line, min_y = None, image.shape[0]
    for x1, y1, x2, y2 in lines:
        if x1 == x2:
            continue
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if -angle_threshold <= angle <= angle_threshold and y1 < min_y:
            min_y, max_line = y1, [x1, y1, x2, y2]
    return edges, min_y if max_line else -1


def detect_VCM_line(image, angle_threshold, params):
    edges, lines = detect_lines(image, **params)
    if not lines:
        return edges, -1

    max_y, max_line = -1, None
    for x1, y1, x2, y2 in lines:
        if x1 == x2:
            continue
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if -angle_threshold <= angle <= angle_threshold:
            current_max_y = max(y1, y2)
            if current_max_y > max_y:
                max_y, max_line = current_max_y, [x1, y1, x2, y2]
    return edges, max_y if max_line else -1


def draw_line_on_image(image, y, color=(0, 0, 255), thickness=1):
    cv2.line(image, (0, int(y)), (image.shape[1], int(y)), color, thickness)


def process_image_pipeline(image_path, output_folders):
    try:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return "FAILED_READ"

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if worker_clahe:
            img_gray = worker_clahe.apply(img_gray)

        h, w = img_gray.shape
        search_start_y = int(h * 4 / 5)
        search_width = int(w * SEARCH_WIDTH_RATIO)
        search_offset_x = (w - search_width) // 2
        search_area_gray = img_gray[search_start_y:,
                                    search_offset_x:search_offset_x + search_width]

        match_loc, t_w, t_h = detect_best_match(
            search_area_gray, worker_templates)
        if not (match_loc and t_w > 0):
            return "FAILED_CROP"

        x = match_loc[0] + search_offset_x
        y = match_loc[1] + search_start_y
        center_x = x + t_w / 2

        y_start = np.clip(y + TOP_OFFSET, 0, h)
        y_end = np.clip(y + t_h + BOTTOM_OFFSET, 0, h)
        x_start = np.clip(int(center_x - CROP_WIDTH / 2), 0, w)
        x_end = np.clip(x_start + CROP_WIDTH, 0, w)

        cropped_image = img_bgr[y_start:y_end, x_start:x_end]
        filename, ext = image_path.stem, image_path.suffix

        params = DETECTION_PARAMS
        edges_htcc, max_y_htcc = detect_HTCC_line(
            cropped_image, params['angle_threshold'], params['htcc_hough'])
        edges_vcm, max_y_vcm = detect_VCM_line(
            cropped_image, params['angle_threshold'], params['vcm_hough'])

        if SAVE_EDGES_IMAGE:
            if edges_htcc is not None:
                cv2.imwrite(
                    str(output_folders['edges'] / f"{filename}_htcc_edges{ext}"), edges_htcc)
            if edges_vcm is not None:
                cv2.imwrite(
                    str(output_folders['edges'] / f"{filename}_vcm_edges{ext}"), edges_vcm)

        status = "NG"
        if max_y_htcc != -1 and max_y_vcm != -1 and max_y_htcc <= max_y_vcm - params['detect_threshold']:
            status = "OK"

        cv2.imwrite(
            str(output_folders[status.lower()] / f"{filename}{ext}"), cropped_image)

        if SAVE_IMAGE_LINES:
            img_with_lines = cropped_image.copy()
            if max_y_htcc != -1:
                draw_line_on_image(
                    img_with_lines, max_y_htcc, color=(0, 0, 255))
            if max_y_vcm != -1:
                draw_line_on_image(
                    img_with_lines, max_y_vcm, color=(0, 255, 0))
            cv2.imwrite(
                str(output_folders['crop'] / f"{filename}_{status}{ext}"), img_with_lines)

        return status
    except Exception:
        return "FAILED_FATAL"


def main():
    """Main function to run the image processing pipeline."""
    templates = load_templates(TEMPLATE_FOLDER)
    if not templates:
        return

    output_folders = create_output_folders()

    image_files = [f for f in INPUT_FOLDER.glob(
        "*") if f.suffix.lower() in ALLOWED_EXTENSIONS]
    if not image_files:
        return

    tasks = [(path, output_folders) for path in image_files]

    with mp.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(templates,)) as pool:
        pool.starmap(process_image_pipeline, tasks)


if __name__ == "__main__":
    mp.freeze_support()
    main()
