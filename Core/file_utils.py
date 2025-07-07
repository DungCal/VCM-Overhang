import cv2
from pathlib import Path
import Core.config as config


def load_templates(template_folder_path: Path):
    """Loads all image templates from a specified folder."""
    templates = []
    if not template_folder_path.is_dir():
        print(f"Warning: Template folder not found: {template_folder_path}")
        return templates

    for template_file in template_folder_path.glob("*"):
        if template_file.suffix.lower() in config.ALLOWED_EXTENSIONS:
            try:
                template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates.append(
                        (template, template.shape[1], template.shape[0]))
                else:
                    print(f"Warning: Could not read template: {template_file}")
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")
    return templates


def create_output_folders():
    """Creates the necessary output folders."""
    output_folders = {
        'ok': config.OUTPUT_BASE / "OK",
        'ng': config.OUTPUT_BASE / "NG",
        'crop': config.OUTPUT_BASE / "crop",
        'edges': config.OUTPUT_BASE / "crop"
    }
    for folder in output_folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return output_folders
