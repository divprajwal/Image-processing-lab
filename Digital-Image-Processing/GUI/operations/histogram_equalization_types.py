# Histogram Equalization Types (GHE, LHE, AHE, CLAHE) for DIP Lab GUI

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

NAME = "Histogram Equalization Types (GHE, LHE, AHE, CLAHE)"


def _rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA image array to uint8 greyscale."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def _global_he(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global Histogram Equalization."""
    hist_orig = np.bincount(grey.ravel(), minlength=256)
    cdf = np.cumsum(hist_orig)
    lut = np.round((cdf / grey.size) * 255).astype(np.uint8)
    out = lut[grey]
    hist_out = np.bincount(out.ravel(), minlength=256)
    return out, hist_orig, hist_out


def _local_he(grey: np.ndarray, window_size: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Local Histogram Equalization via sliding window rank calculation."""
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

    hist_out = np.bincount(out.ravel(), minlength=256)
    return out, hist_out


def _tile_he(
    grey: np.ndarray,
    grid_size: tuple[int, int] = (8, 8),
    clip_limit: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Tile-based Adaptive HE (AHE) / CLAHE with bilinear interpolation."""
    H, W = grey.shape
    n_rows, n_cols = grid_size

    tile_h = int(np.ceil(H / n_rows))
    tile_w = int(np.ceil(W / n_cols))

    luts = np.zeros((n_rows, n_cols, 256), dtype=np.float32)

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

            if clip_limit is not None and clip_limit > 0:
                actual_clip = clip_limit * (tile_size / 256.0) if clip_limit < 100 else clip_limit
                excess = np.sum(np.maximum(0, hist - actual_clip))
                hist = np.minimum(hist, actual_clip)
                hist += excess / 256.0

            cdf = np.cumsum(hist)
            cdf_max = cdf[-1]
            if cdf_max > 0:
                luts[r, c] = np.round((cdf / cdf_max) * 255.0)
            else:
                luts[r, c] = np.arange(256)

    y_centers = (np.arange(n_rows) + 0.5) * (H / n_rows)
    x_centers = (np.arange(n_cols) + 0.5) * (W / n_cols)

    y_coords = np.arange(H)
    x_coords = np.arange(W)

    r1 = np.clip(np.searchsorted(y_centers, y_coords), 0, n_rows - 1)
    r0 = np.clip(r1 - 1, 0, n_rows - 1)
    y0 = y_centers[r0]
    y1 = y_centers[r1]
    denom_y = np.where(y1 != y0, y1 - y0, 1.0)
    ty = np.where(r0 == r1, 0.0, (y_coords - y0) / denom_y)[:, None]

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


def _plot_hist(hist: np.ndarray, title: str, color: str = "#3498db") -> Image.Image:
    """Render a single histogram bar plot."""
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


def _plot_5x2_grid(items: list[tuple[str, np.ndarray, np.ndarray, str]]) -> Image.Image:
    """Render a 5-row x 2-column grid comparing all techniques with histograms."""
    n_rows = len(items)
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 3.0 * n_rows), dpi=100)
    fig.patch.set_facecolor("#141414")

    for i, (name, img_arr, hist_arr, color) in enumerate(items):
        axs[i, 0].imshow(img_arr, cmap="gray", vmin=0, vmax=255)
        axs[i, 0].set_title(f"{name} - Image", color="#ffffff", fontsize=10, fontweight="bold", pad=4)
        axs[i, 0].axis("off")

        axs[i, 1].set_facecolor("#202020")
        axs[i, 1].bar(np.arange(256), hist_arr, width=1.0, color=color, edgecolor=color, alpha=0.9)
        axs[i, 1].set_title(f"{name} - Histogram", color="#ffffff", fontsize=10, fontweight="bold", pad=4)
        axs[i, 1].set_xlim([0, 255])
        axs[i, 1].tick_params(colors="#aaaaaa", labelsize=7)
        axs[i, 1].set_xlabel("Intensity", color="#cccccc", fontsize=7.5)
        axs[i, 1].set_ylabel("Count", color="#cccccc", fontsize=7.5)
        for spine in axs[i, 1].spines.values():
            spine.set_color("#444444")
        axs[i, 1].grid(True, linestyle="--", alpha=0.2, color="#555555")

    fig.tight_layout(pad=1.5)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return img


def process(img: np.ndarray) -> dict[str, Image.Image]:
    """Execute all 4 histogram equalization methods and return paired image/histogram cards."""
    grey = _rgb_to_grey(img)

    ghe_img, hist_orig, hist_ghe = _global_he(grey)
    lhe_img, hist_lhe = _local_he(grey, window_size=21)
    ahe_img, hist_ahe = _tile_he(grey, grid_size=(8, 8), clip_limit=None)
    clahe_img, hist_clahe = _tile_he(grey, grid_size=(8, 8), clip_limit=2.5)

    items = [
        ("Input", grey, hist_orig, "#3498db"),
        ("Global HE (GHE)", ghe_img, hist_ghe, "#2ecc71"),
        ("Local HE (LHE)", lhe_img, hist_lhe, "#e67e22"),
        ("Adaptive HE (AHE)", ahe_img, hist_ahe, "#9b59b6"),
        ("CLAHE", clahe_img, hist_clahe, "#1abc9c"),
    ]

    overview_grid = _plot_5x2_grid(items)

    to_3ch = lambda arr: Image.fromarray(np.stack([arr, arr, arr], axis=-1))

    return {
        "Overview Grid (All 4 Types)": overview_grid,
        "1. Input Image": to_3ch(grey),
        "1. Input Histogram": _plot_hist(hist_orig, "Input Histogram", color="#3498db"),
        "2. Global HE (GHE)": to_3ch(ghe_img),
        "2. Global HE Histogram": _plot_hist(hist_ghe, "Global HE Histogram", color="#2ecc71"),
        "3. Local HE (LHE)": to_3ch(lhe_img),
        "3. Local HE Histogram": _plot_hist(hist_lhe, "Local HE Histogram", color="#e67e22"),
        "4. Adaptive HE (AHE)": to_3ch(ahe_img),
        "4. Adaptive HE Histogram": _plot_hist(hist_ahe, "Adaptive HE Histogram", color="#9b59b6"),
        "5. CLAHE": to_3ch(clahe_img),
        "5. CLAHE Histogram": _plot_hist(hist_clahe, "CLAHE Histogram", color="#1abc9c"),
    }
