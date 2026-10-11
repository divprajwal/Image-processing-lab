"""Run every GUI image-processing operation against every repository input."""

from pathlib import Path
import re

import numpy as np
from PIL import Image

from GUI.operations import get_operations


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Input_images"
OUTPUT_DIR = ROOT / "Output"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def safe_name(value: str) -> str:
    """Convert a display label into a stable filesystem name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def main() -> None:
    images = sorted(
        path for path in INPUT_DIR.iterdir() if path.suffix.lower() in VALID_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No input images found in {INPUT_DIR}")

    operations = get_operations()
    if not operations:
        raise RuntimeError("No image-processing operations were discovered")

    OUTPUT_DIR.mkdir(exist_ok=True)
    saved_count = 0

    for image_path in images:
        print(f"\nProcessing {image_path.name}")
        image = np.array(Image.open(image_path).convert("RGB"))
        image_dir = OUTPUT_DIR / safe_name(image_path.stem)

        for operation_name, process in operations.items():
            print(f"  - {operation_name}", end="", flush=True)
            results = process(image)
            operation_dir = image_dir / safe_name(operation_name)
            operation_dir.mkdir(parents=True, exist_ok=True)

            for result_name, result_image in results.items():
                output_path = operation_dir / f"{safe_name(result_name)}.png"
                result_image.save(output_path)
                saved_count += 1

            print(f": {len(results)} output(s)")

    print(
        f"\nCompleted {len(operations)} operations for {len(images)} image(s); "
        f"saved {saved_count} output files to {OUTPUT_DIR}."
    )


if __name__ == "__main__":
    main()
