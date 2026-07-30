import os
import shutil

FAINT_DIR = "train/faint_stamps"
BLANK_DIR = "train/blank_stamps"

IMAGE_DIR = "datasets/stamps/images/train"
LABEL_DIR = "datasets/stamps/labels/train"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


def copy_faint_images():
    count = 0

    for filename in os.listdir(FAINT_DIR):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        source = os.path.join(FAINT_DIR, filename)
        destination = os.path.join(IMAGE_DIR, filename)

        shutil.copy2(source, destination)

        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, label_name)

        with open(label_path, "w") as file:
            file.write("0 0.5 0.5 1.0 1.0\n")

        count += 1

    return count


def copy_blank_images():
    count = 0

    for filename in os.listdir(BLANK_DIR):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        source = os.path.join(BLANK_DIR, filename)
        destination = os.path.join(IMAGE_DIR, filename)

        shutil.copy2(source, destination)

        count += 1

    return count


if not os.path.exists(FAINT_DIR):
    raise FileNotFoundError(FAINT_DIR)

if not os.path.exists(BLANK_DIR):
    raise FileNotFoundError(BLANK_DIR)


faint_count = copy_faint_images()
blank_count = copy_blank_images()

print(f"Faint images: {faint_count}")
print(f"Blank images: {blank_count}")
print("Dataset ready in 'datasets/stamps'")
