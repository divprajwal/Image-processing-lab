"""Histogram Equalization (Manual DIP Implementation).

This script performs contrast enhancement via histogram equalization from first
principles:
1. Converts input image to greyscale.
2. Computes the frequency histogram (pixel counts).
3. Computes the Cumulative Distribution Function (CDF).
4. Creates a transformation lookup table s_k = round(255 * CDF(k) / N).
5. Maps input intensities to equalized intensities.
6. Generates individual outputs (input image, input histogram, equalized image, equalized histogram)
   and a comprehensive 2x2 comparison grid saved to Output/.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA numpy array to uint8 greyscale using luminance weights."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def compute_histogram_equalization(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform histogram equalization from first principles.

    Returns:
        equalized_img: np.ndarray (H, W) uint8
        hist_orig: np.ndarray (256,) frequency counts of input
        hist_eq: np.ndarray (256,) frequency counts of equalized image
        lut: np.ndarray (256,) mapping lookup table
    """
    # 1. Frequency histogram
    hist_orig = np.bincount(grey.ravel(), minlength=256)
    num_pixels = grey.size

    # 2. Cumulative Distribution Function (CDF)
    cdf = np.cumsum(hist_orig)

    # 3. Transformation Lookup Table: s_k = round((L - 1) * CDF(k) / TotalPixels)
    lut = np.round((cdf / num_pixels) * 255).astype(np.uint8)

    # 4. Map original pixel intensities to equalized values
    equalized_img = lut[grey]

    # 5. Output histogram
    hist_eq = np.bincount(equalized_img.ravel(), minlength=256)

    return equalized_img, hist_orig, hist_eq, lut


def plot_histogram(hist: np.ndarray, title: str, bar_color: str = "#3498db") -> Image.Image:
    """Render a clean histogram bar chart as a PIL Image."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#252526")

    bins = np.arange(256)
    ax.bar(bins, hist, width=1.0, color=bar_color, edgecolor=bar_color, alpha=0.9)
    ax.set_xlim([0, 255])
    ax.set_title(title, color="#ffffff", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Intensity Level (r)", color="#cccccc", fontsize=10)
    ax.set_ylabel("Pixel Count (n_k)", color="#cccccc", fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, linestyle="--", alpha=0.3, color="#666666")

    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    result = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return result


def plot_comparison_grid(
    grey_img: np.ndarray,
    hist_orig: np.ndarray,
    equalized_img: np.ndarray,
    hist_eq: np.ndarray
) -> Image.Image:
    """Render a 2x2 comparison grid containing images and their corresponding histograms."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 9), dpi=120)
    fig.patch.set_facecolor("#181818")

    # Top-Left: Input Image
    axs[0, 0].imshow(grey_img, cmap="gray", vmin=0, vmax=255)
    axs[0, 0].set_title("1. Input Greyscale Image", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    axs[0, 0].axis("off")

    # Top-Right: Input Histogram
    axs[0, 1].set_facecolor("#242424")
    axs[0, 1].bar(np.arange(256), hist_orig, width=1.0, color="#3498db", edgecolor="#3498db", alpha=0.9)
    axs[0, 1].set_title("2. Input Histogram", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    axs[0, 1].set_xlim([0, 255])
    axs[0, 1].tick_params(colors="#aaaaaa", labelsize=8)
    axs[0, 1].set_xlabel("Intensity Level (0-255)", color="#cccccc", fontsize=9)
    axs[0, 1].set_ylabel("Pixel Frequency", color="#cccccc", fontsize=9)
    for spine in axs[0, 1].spines.values():
        spine.set_color("#444444")
    axs[0, 1].grid(True, linestyle="--", alpha=0.25, color="#555555")

    # Bottom-Left: Equalized Image
    axs[1, 0].imshow(equalized_img, cmap="gray", vmin=0, vmax=255)
    axs[1, 0].set_title("3. Equalized Output Image", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    axs[1, 0].axis("off")

    # Bottom-Right: Equalized Histogram
    axs[1, 1].set_facecolor("#242424")
    axs[1, 1].bar(np.arange(256), hist_eq, width=1.0, color="#2ecc71", edgecolor="#2ecc71", alpha=0.9)
    axs[1, 1].set_title("4. Equalized Histogram", color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    axs[1, 1].set_xlim([0, 255])
    axs[1, 1].tick_params(colors="#aaaaaa", labelsize=8)
    axs[1, 1].set_xlabel("Intensity Level (0-255)", color="#cccccc", fontsize=9)
    axs[1, 1].set_ylabel("Pixel Frequency", color="#cccccc", fontsize=9)
    for spine in axs[1, 1].spines.values():
        spine.set_color("#444444")
    axs[1, 1].grid(True, linestyle="--", alpha=0.25, color="#555555")

    fig.tight_layout(pad=2.5)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    result = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return result


def main():
    # 1. Determine input and output paths
    possible_input_dirs = ["Input_images", "Input"]
    input_dir = next((d for d in possible_input_dirs if os.path.isdir(d) and len(os.listdir(d)) > 0), "Input_images")
    output_dir = "Output_images" if os.path.isdir("Output_images") else "Output"
    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    if not image_files:
        raise FileNotFoundError(f"No image found in '{input_dir}' directory.")

    img_path = os.path.join(input_dir, image_files[0])
    raw_img = np.array(Image.open(img_path))
    print(f"Loaded image: {img_path} ({raw_img.shape[1]}x{raw_img.shape[0]})")

    # 2. Greyscale conversion
    grey = rgb_to_grey(raw_img)
    grey_3ch = np.stack([grey, grey, grey], axis=-1)

    # 3. Histogram Equalization
    equalized, hist_in, hist_out, lut = compute_histogram_equalization(grey)
    equalized_3ch = np.stack([equalized, equalized, equalized], axis=-1)

    # 4. Generate histogram plots & comparison grid
    plot_hist_in = plot_histogram(hist_in, "Input Histogram", bar_color="#3498db")
    plot_hist_out = plot_histogram(hist_out, "Equalized Histogram", bar_color="#2ecc71")
    grid_comparison = plot_comparison_grid(grey, hist_in, equalized, hist_out)

    # 5. Save all outputs
    p_in_img = os.path.join(output_dir, "input_greyscale.png")
    p_in_hist = os.path.join(output_dir, "input_histogram.png")
    p_eq_img = os.path.join(output_dir, "equalized_image.png")
    p_eq_hist = os.path.join(output_dir, "equalized_histogram.png")
    p_grid = os.path.join(output_dir, "histogram_equalization_2x2_grid.png")

    Image.fromarray(grey_3ch).save(p_in_img)
    plot_hist_in.save(p_in_hist)
    Image.fromarray(equalized_3ch).save(p_eq_img)
    plot_hist_out.save(p_eq_hist)
    grid_comparison.save(p_grid)

    print(f"Saved Input Greyscale Image to : {p_in_img}")
    print(f"Saved Input Histogram to       : {p_in_hist}")
    print(f"Saved Equalized Image to       : {p_eq_img}")
    print(f"Saved Equalized Histogram to   : {p_eq_hist}")
    print(f"Saved 2x2 Comparison Grid to   : {p_grid}")
    print("\nHistogram Equalization completed successfully!")


if __name__ == "__main__":
    main()
