from pathlib import Path
from PIL import Image
from collections import Counter

DATA_DIR = Path("data/raw/chest_xray/chest_xray")

image_files = list(DATA_DIR.rglob("*.jpeg"))

print(f"Total images: {len(image_files)}")

# Count images by split and class
for split in ["train", "test", "val"]:
    split_dir = DATA_DIR / split

    if not split_dir.exists():
        continue

    print(f"\n{split.upper()}")

    for class_name in ["NORMAL", "PNEUMONIA"]:
        class_dir = split_dir / class_name
        count = len(list(class_dir.glob("*.jpeg")))
        print(f"{class_name}: {count}")

# Inspect image properties
sizes = []
modes = Counter()

for image_path in image_files[:500]:
    with Image.open(image_path) as img:
        sizes.append(img.size)
        modes[img.mode] += 1

print("\nImage modes:")
print(modes)

print("\nMost common image sizes:")
for size, count in Counter(sizes).most_common(10):
    print(size, count)