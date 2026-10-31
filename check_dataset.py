
from pathlib import Path

# --------------------------------------------------
# PlantCare_AI - Dataset Verification
# --------------------------------------------------

# Project folder / data / PlantVillage
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "PlantVillage"

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def main():
    print("=" * 60)
    print("       PLANTCARE AI - DATASET VERIFICATION")
    print("=" * 60)

    # Check whether the dataset folder exists
    if not DATASET_PATH.exists():
        print("\nERROR: Dataset folder not found!")
        print("Expected location:")
        print(DATASET_PATH)
        return

    print(f"\nDataset location:\n{DATASET_PATH}")

    # Get only immediate subfolders, excluding nested
    # PlantVillage to avoid counting duplicate images.
    class_folders = sorted(
        folder
        for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
        and folder.name.lower() != "plantvillage"
    )

    if not class_folders:
        print("\nERROR: No disease-class folders found.")
        return

    total_images = 0

    print(f"\nNumber of classes: {len(class_folders)}")
    print("-" * 60)

    for folder in class_folders:
        image_count = sum(
            1
            for file in folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        total_images += image_count

        print(f"{folder.name}: {image_count} images")

    print("-" * 60)
    print(f"Total classes: {len(class_folders)}")
    print(f"Total images: {total_images}")
    print("=" * 60)

    if len(class_folders) == 15 and total_images == 20638:
        print("\nSUCCESS: Dataset count matches expected output.")
    else:
        print("\nDataset check completed.")
        print("Counts may differ depending on the dataset version.")


if __name__ == "__main__":
    main()
