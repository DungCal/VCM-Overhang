import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp

# --- File and Path Configuration ---
BASE_PATH = Path(r"E:/02.pdx/pdx25-overhang")
INPUT_SUBFOLDER = r"NG/Symptom 2"
INPUT_FOLDER = BASE_PATH / "data" / INPUT_SUBFOLDER
TEMPLATE_FOLDER = BASE_PATH / "data/template"
OUTPUT_BASE = BASE_PATH / "output" / INPUT_SUBFOLDER

# --- General Settings ---
DEBUG = False
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg')
NUM_PROCESSES = max(1, mp.cpu_count() - 1)


# --- Output Settings ---
SAVE_IMAGE_LINES = False
SAVE_EDGES_IMAGE = False

# --- Cropping and Template Matching Parameters ---
MATCH_METHOD = cv2.TM_SQDIFF_NORMED
SEARCH_WIDTH_RATIO = 0.5
EARLY_EXIT_THRESHOLD = 0.05
CROP_WIDTH = 280
TOP_OFFSET = 100
BOTTOM_OFFSET = -10

# --- Line Detection Parameters ---
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
