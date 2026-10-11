"""Histogram Equalization Types: Global, Local, Adaptive, and CLAHE.

This script implements four fundamental histogram equalization methods from first principles:
1. Global Histogram Equalization (GHE): Global cumulative distribution function.
2. Local Histogram Equalization (LHE): Sliding window rank-based local equalization.
3. Adaptive Histogram Equalization (AHE): Tile-based CDF with bilinear interpolation.
4. Contrast Limited Adaptive Histogram Equalization (CLAHE): Tile-based clipped CDF with redistribution & bilinear interpolation.

Each method outputs both the transformed image and its intensity histogram, alongside
a comprehensive 5x2 comparison overview grid saved to Output/.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA image array to uint8 greyscale using luminance weights."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


# -----------------------------------------------------------------------------
# 1. Global Histogram Equalization (GHE)
# -----------------------------------------------------------------------------
def global_histogram_equalization(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform Global Histogram Equalization (GHE) across the entire image."""
    hist_orig = np.bincount(grey.ravel(), minlength=256)
    num_pixels = grey.size
    cdf = np.cumsum(hist_orig)
    lut = np.round((cdf / num_pixels) * 255).astype(np.uint8)
    equalized = lut[grey]
    hist_eq = np.bincount(equalized.ravel(), minlength=256)
    return equalized, hist_orig, hist_eq


# -----------------------------------------------------------------------------
# 2. Local Histogram Equalization (LHE)
# -----------------------------------------------------------------------------
def local_histogram_equalization(grey: np.ndarray, window_size: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Perform Local Histogram Equalization (LHE) using a sliding window.

    For each pixel (r, c), computes the local rank of the center pixel intensity
    within a window_size x window_size neighbourhood.
    """
    H, W = grey.shape
    pad = window_size // 2
    padded = np.pad(grey, pad, mode="reflect")
    out = np.zeros_like(grey, dtype=np.uint8)
    total_win_pixels = window_size * window_size

    for i in range(H):
        for j in range(W):
            win = padded[i : i + window_size, j : j + window_size]
            center_val = grey[i, j]
            rank = np.count_nonzero(win <= center_val)
            out[i, j] = int(round((rank / total_win_pixels) * 255))

    hist_lhe = np.bincount(out.ravel(), minlength=256)
    return out, hist_lhe


# -----------------------------------------------------------------------------
# 3 & 4. Adaptive Histogram Equalization (AHE) & CLAHE
# -----------------------------------------------------------------------------
def tile_based_adaptive_equalization(
    grey: np.ndarray,
    grid_size: tuple[int, int] = (8, 8),
    clip_limit: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Adaptive HE (AHE) or CLAHE using contextual tiles and bilinear interpolation.

    Args:
        grey: 2D uint8 greyscale array
        grid_size: (n_rows, n_cols) tile grid layout (default: 8x8)
        clip_limit: Optional contrast clipping factor. If None, performs standard AHE.
                   If set (e.g. 2.5), clips histogram bins and redistributes excess (CLAHE).
    """
    H, W = grey.shape
    n_rows, n_cols = grid_size

    tile_h = int(np.ceil(H / n_rows))
    tile_w = int(np.ceil(W / n_cols))

    luts = np.zeros((n_rows, n_cols, 256), dtype=np.float32)

    # 1. Compute Transformation LUT for each tile
    for r in range(n_rows):
        r_start = r * tile_h
        r_end = min((r + 1) * tile_h, H)
        for c in range(n_cols):
            c_start = c * tile_w
            c_end = min((c + 1) * tile_w, W)

            tile = grey[r_start:r_end, c_start:c_end]
            tile_size = tile.size
            if tile_size == 0:
                continue

            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)

            # CLAHE: Contrast limiting and redistribution
            if clip_limit is not None and clip_limit > 0:
                actual_clip = clip_limit * (tile_size / 256.0) if clip_limit < 100 else clip_limit
                excess = np.sum(np.maximum(0, hist - actual_clip))
                hist = np.minimum(hist, actual_clip)
                hist += excess / 256.0

            # Cumulative Distribution Function (CDF)
            cdf = np.cumsum(hist)
            cdf_max = cdf[-1]
            if cdf_max > 0:
                luts[r, c] = np.round((cdf / cdf_max) * 255.0)
            else:
                luts[r, c] = np.arange(256)

    # 2. Vectorized Bilinear Interpolation across tile centers
    y_centers = (np.arange(n_rows) + 0.5) * (H / n_rows)
    x_centers = (np.arange(n_cols) + 0.5) * (W / n_cols)

    y_coords = np.arange(H)
    x_coords = np.arange(W)

    # Vertical interpolation indices and weights
    r1 = np.clip(np.searchsorted(y_centers, y_coords), 0, n_rows - 1)
    r0 = np.clip(r1 - 1, 0, n_rows - 1)
    y0 = y_centers[r0]
    y1 = y_centers[r1]
    denom_y = np.where(y1 != y0, y1 - y0, 1.0)
    ty = np.where(r0 == r1, 0.0, (y_coords - y0) / denom_y)[:, None]

    # Horizontal interpolation indices and weights
    c1 = np.clip(np.searchsorted(x_centers, x_coords), 0, n_cols - 1)
    c0 = np.clip(c1 - 1, 0, n_cols - 1)
    x0 = x_centers[c0]
    x1 = x_centers[c1]
    denom_x = np.where(x1 != x0, x1 - x0, 1.0)
    tx = np.where(c0 == c1, 0.0, (x_coords - x0) / denom_x)[None, :]

    R0, C0 = np.meshgrid(r0, c0, indexing="ij")
    R1, C1 = np.meshgrid(r1, c0, indexing="ij")
    R2, C2 = np.meshgrid(r0, c1, indexing="ij")
    R3, C3 = np.meshgrid(r1, c1, indexing="ij")

    val = grey
    lut00 = luts[R0, C0, val]
    lut10 = luts[R1, C1, val]
    lut01 = luts[R2, C2, val]
    lut11 = luts[R3, C3, val]

    interp = (1 - ty) * ((1 - tx) * lut00 + tx * lut01) + ty * ((1 - tx) * lut10 + tx * lut11)
    out = np.clip(np.round(interp), 0, 255).astype(np.uint8)
    hist_out = np.bincount(out.ravel(), minlength=256)
    return out, hist_out


# -----------------------------------------------------------------------------
# Matplotlib Visualization Helpers
# -----------------------------------------------------------------------------
def plot_histogram(hist: np.ndarray, title: str, bar_color: str = "#3498db") -> Image.Image:
    """Render an individual histogram bar plot as a PIL Image."""
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=110)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#252526")

    bins = np.arange(256)
    ax.bar(bins, hist, width=1.0, color=bar_color, edgecolor=bar_color, alpha=0.9)
    ax.set_xlim([0, 255])
    ax.set_title(title, color="#ffffff", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Intensity Level", color="#cccccc", fontsize=9)
    ax.set_ylabel("Pixel Count", color="#cccccc", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, linestyle="--", alpha=0.25, color="#555555")

    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return img


def plot_5x2_comparison_grid(
    items: list[tuple[str, np.ndarray, np.ndarray, str]]
) -> Image.Image:
    """Render a 5-row x 2-column grid comparing Input and all 4 Equalization techniques.

    items: list of (method_name, image_array, histogram_array, plot_color)
    """
    n_rows = len(items)
    fig, axs = plt.subplots(n_rows, 2, figsize=(11, 3.2 * n_rows), dpi=110)
    fig.patch.set_facecolor("#141414")

    for i, (name, img_arr, hist_arr, color) in enumerate(items):
        # Left column: Image
        axs[i, 0].imshow(img_arr, cmap="gray", vmin=0, vmax=255)
        axs[i, 0].set_title(f"{i + 1}. {name} - Image", color="#ffffff", fontsize=11, fontweight="bold", pad=6)
        axs[i, 0].axis("off")

        # Right column: Histogram
        axs[i, 1].set_facecolor("#202020")
        axs[i, 1].bar(np.arange(256), hist_arr, width=1.0, color=color, edgecolor=color, alpha=0.9)
        axs[i, 1].set_title(f"{i + 1}. {name} - Histogram", color="#ffffff", fontsize=11, fontweight="bold", pad=6)
        axs[i, 1].set_xlim([0, 255])
        axs[i, 1].tick_params(colors="#aaaaaa", labelsize=7.5)
        axs[i, 1].set_xlabel("Intensity", color="#cccccc", fontsize=8)
        axs[i, 1].set_ylabel("Count", color="#cccccc", fontsize=8)
        for spine in axs[i, 1].spines.values():
            spine.set_color("#444444")
        axs[i, 1].grid(True, linestyle="--", alpha=0.2, color="#555555")

    fig.tight_layout(pad=2.0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    res = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return res


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
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

    # 1. Greyscale Conversion
    grey = rgb_to_grey(raw_img)

    # 2. Process all 4 methods
    print("Computing Global Histogram Equalization (GHE)...")
    ghe_img, hist_orig, hist_ghe = global_histogram_equalization(grey)

    print("Computing Local Histogram Equalization (LHE, window=21)...")
    lhe_img, hist_lhe = local_histogram_equalization(grey, window_size=21)

    print("Computing Adaptive Histogram Equalization (AHE, grid=8x8)...")
    ahe_img, hist_ahe = tile_based_adaptive_equalization(grey, grid_size=(8, 8), clip_limit=None)

    print("Computing CLAHE (grid=8x8, clip_limit=2.5)...")
    clahe_img, hist_clahe = tile_based_adaptive_equalization(grey, grid_size=(8, 8), clip_limit=2.5)

    # 3. Save individual images and histogram plots
    outputs_to_save = [
        ("Input Greyscale", grey, hist_orig, "#3498db", "input_greyscale.png", "input_histogram.png"),
        ("Global HE (GHE)", ghe_img, hist_ghe, "#2ecc71", "global_he.png", "global_he_histogram.png"),
        ("Local HE (LHE)", lhe_img, hist_lhe, "#e67e22", "local_he.png", "local_he_histogram.png"),
        ("Adaptive HE (AHE)", ahe_img, hist_ahe, "#9b59b6", "adaptive_he.png", "adaptive_he_histogram.png"),
        ("CLAHE", clahe_img, hist_clahe, "#1abc9c", "clahe.png", "clahe_histogram.png"),
    ]

    print("\nSaving individual images and histograms to Output/...")
    for name, img_arr, hist_arr, color, img_file, hist_file in outputs_to_save:
        img_3ch = np.stack([img_arr, img_arr, img_arr], axis=-1)
        p_img = os.path.join(output_dir, img_file)
        p_hist = os.path.join(output_dir, hist_file)

        Image.fromarray(img_3ch).save(p_img)
        plot_histogram(hist_arr, f"{name} Histogram", bar_color=color).save(p_hist)
        print(f" - Saved: {img_file} & {hist_file}")

    # 4. Generate & save 5x2 Overview Comparison Grid
    print("\nGenerating 5x2 Comparison Grid...")
    grid_items = [(item[0], item[1], item[2], item[3]) for item in outputs_to_save]
    grid_img = plot_5x2_comparison_grid(grid_items)
    p_grid = os.path.join(output_dir, "he_types_comparison_grid.png")
    grid_img.save(p_grid)
    print(f"Saved 5x2 Comparison Grid to: {p_grid}")

    print("\nAll Histogram Equalization types processed successfully!")


if __name__ == "__main__":
    main()
