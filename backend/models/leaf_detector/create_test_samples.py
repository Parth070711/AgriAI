import os
import shutil
import random

# Path to your dataset
SOURCE_DIR = r"D:\AgriAI\backend\dataset\PlantVillage"

# Folder where sample images will be copied
DEST_DIR = r"D:\AgriAI\backend\sample_test_images"

os.makedirs(DEST_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for class_name in sorted(os.listdir(SOURCE_DIR)):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not images:
        continue

    # Pick 2 random images (or all if fewer than 2)
    selected = random.sample(images, min(2, len(images)))

    for i, image in enumerate(selected, start=1):
        src = os.path.join(class_path, image)

        ext = os.path.splitext(image)[1]

        new_name = f"{class_name}_{i}{ext}"

        dst = os.path.join(DEST_DIR, new_name)

        shutil.copy2(src, dst)

print("Done!")
print(f"Sample images saved to:\n{DEST_DIR}")