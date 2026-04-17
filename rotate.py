import cv2
from pathlib import Path

def rotate_images_folder(
    input_folder,
    output_folder,
    degree  # supports 90, -90, 180, 270, -270
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    degree = degree % 360

    if degree == 90:
        rotate_flag = cv2.ROTATE_90_CLOCKWISE
    elif degree == 270:
        rotate_flag = cv2.ROTATE_90_COUNTERCLOCKWISE
    elif degree == 180:
        rotate_flag = cv2.ROTATE_180
    elif degree == 0:
        rotate_flag = None
    else:
        raise ValueError("Only 90, -90, 180, 270 degree rotations are supported")

    for img_path in input_folder.iterdir():
        if img_path.suffix.lower() not in exts:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if rotate_flag is not None:
            img = cv2.rotate(img, rotate_flag)

        out_path = output_folder / img_path.name
        cv2.imwrite(str(out_path), img)


# -------- Example usage --------
if __name__ == "__main__":
    rotate_images_folder(
        input_folder=r"D:\AI\CM\pdx25-overhang\data\template",
        output_folder=r"D:\AI\CM\pdx25-overhang\data\new_template",
        degree=-90
    )
