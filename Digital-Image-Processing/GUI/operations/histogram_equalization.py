# Histogram Equalization Operation for DIP Lab GUI

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

NAME = "Histogram Equalization"


def _rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA to uint8 greyscale."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def _compute_histogram_equalization(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform histogram equalization from first principles."""
    hist_orig = np.bincount(grey.ravel(), minlength=256)
    num_pixels = grey.size
    cdf = np.cumsum(hist_orig)
    lut = np.round((cdf / num_pixels) * 255).astype(np.uint8)
    equalized_img = lut[grey]
    hist_eq = np.bincount(equalized_img.ravel(), minlength=256)
    return equalized_img, hist_orig, hist_eq


def _plot_hist(hist: np.ndarray, title: str, color: str = "#3498db") -> Image.Image:
    """Render a single histogram plot as a PIL Image."""
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=100)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#252526")

    bins = np.arange(256)
    ax.bar(bins, hist, width=1.0, color=color, edgecolor=color, alpha=0.9)
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


def _plot_2x2_grid(
    grey: np.ndarray,
    hist_in: np.ndarray,
    equalized: np.ndarray,
    hist_out: np.ndarray
) -> Image.Image:
    """Render a 2x2 comparison grid figure as a PIL Image."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=110)
    fig.patch.set_facecolor("#181818")

    # 1. Input Image
    axs[0, 0].imshow(grey, cmap="gray", vmin=0, vmax=255)
    axs[0, 0].set_title("Input Greyscale", color="#ffffff", fontsize=11, fontweight="bold")
    axs[0, 0].axis("off")

    # 2. Input Histogram
    axs[0, 1].set_facecolor("#242424")
    axs[0, 1].bar(np.arange(256), hist_in, width=1.0, color="#3498db", edgecolor="#3498db", alpha=0.9)
    axs[0, 1].set_title("Input Histogram", color="#ffffff", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlim([0, 255])
    axs[0, 1].tick_params(colors="#aaaaaa", labelsize=8)
    axs[0, 1].set_xlabel("Intensity", color="#cccccc", fontsize=8)
    axs[0, 1].set_ylabel("Count", color="#cccccc", fontsize=8)
    for spine in axs[0, 1].spines.values():
        spine.set_color("#444444")
    axs[0, 1].grid(True, linestyle="--", alpha=0.25, color="#555555")

    # 3. Equalized Image
    axs[1, 0].imshow(equalized, cmap="gray", vmin=0, vmax=255)
    axs[1, 0].set_title("Equalized Output", color="#ffffff", fontsize=11, fontweight="bold")
    axs[1, 0].axis("off")

    # 4. Equalized Histogram
    axs[1, 1].set_facecolor("#242424")
    axs[1, 1].bar(np.arange(256), hist_out, width=1.0, color="#2ecc71", edgecolor="#2ecc71", alpha=0.9)
    axs[1, 1].set_title("Equalized Histogram", color="#ffffff", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlim([0, 255])
    axs[1, 1].tick_params(colors="#aaaaaa", labelsize=8)
    axs[1, 1].set_xlabel("Intensity", color="#cccccc", fontsize=8)
    axs[1, 1].set_ylabel("Count", color="#cccccc", fontsize=8)
    for spine in axs[1, 1].spines.values():
        spine.set_color("#444444")
    axs[1, 1].grid(True, linestyle="--", alpha=0.25, color="#555555")

    fig.tight_layout(pad=2.0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return img


def process(img: np.ndarray) -> dict[str, Image.Image]:
    """Perform histogram equalization and return input/output images and histograms."""
    # Greyscale
    grey = _rgb_to_grey(img)
    grey_3ch = np.stack([grey, grey, grey], axis=-1)

    # Histogram equalization
    equalized, hist_in, hist_out = _compute_histogram_equalization(grey)
    equalized_3ch = np.stack([equalized, equalized, equalized], axis=-1)

    # Render plots
    plot_in = _plot_hist(hist_in, "Input Histogram", color="#3498db")
    plot_out = _plot_hist(hist_out, "Equalized Histogram", color="#2ecc71")
    grid_img = _plot_2x2_grid(grey, hist_in, equalized, hist_out)

    return {
        "Combined 2x2 Grid": grid_img,
        "Input Image": Image.fromarray(grey_3ch),
        "Input Histogram": plot_in,
        "Equalized Image": Image.fromarray(equalized_3ch),
        "Equalized Histogram": plot_out,
    }
