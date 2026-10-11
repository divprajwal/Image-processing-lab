# Edge Detection Operation for DIP Lab GUI
# Implemented from first principles using pure NumPy (no external vision libraries)

import numpy as np
from PIL import Image

NAME = "Edge Detection (1st & 2nd Order)"


# =====================================================================
# 1. First-Principles Greyscale & Spatial Convolution
# =====================================================================

def _rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA numpy array to uint8 greyscale using luminance weights."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] >= 3:
        return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
    return img.astype(np.uint8)


def _to_3ch(img_2d: np.ndarray) -> np.ndarray:
    """Convert a 2D uint8 array to a 3-channel (H, W, 3) uint8 image."""
    if img_2d.ndim == 3:
        return img_2d
    return np.stack([img_2d, img_2d, img_2d], axis=-1)


def _conv2d_scratch(image: np.ndarray, kernel: np.ndarray, pad_mode: str = "edge") -> np.ndarray:
    """Perform 2D spatial convolution from first principles using pure NumPy.
    
    Flips kernel 180 degrees and accumulates neighborhood contributions.
    """
    img_f = image.astype(np.float64)
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    k_rot = np.flipud(np.fliplr(kernel))
    padded = np.pad(img_f, ((pad_h, pad_h), (pad_w, pad_w)), mode=pad_mode)

    h, w = img_f.shape
    output = np.zeros((h, w), dtype=np.float64)

    for i in range(k_h):
        for j in range(k_w):
            weight = k_rot[i, j]
            if weight != 0.0:
                output += weight * padded[i : i + h, j : j + w]

    return output


def _contrast_stretch_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize float array linearly to [0, 255] uint8."""
    min_v = np.min(arr)
    max_v = np.max(arr)
    if max_v - min_v > 1e-8:
        norm = (arr - min_v) / (max_v - min_v) * 255.0
    else:
        norm = np.zeros_like(arr)
    return np.clip(np.round(norm), 0, 255).astype(np.uint8)


# =====================================================================
# 2. First-Order Difference Edge Operators
# =====================================================================

def _forward_difference(image: np.ndarray) -> np.ndarray:
    """Compute Forward Difference gradient magnitude."""
    img_f = image.astype(np.float64)
    h, w = img_f.shape
    padded = np.pad(img_f, ((0, 1), (0, 1)), mode="edge")
    
    gx = padded[:h, 1 : w + 1] - padded[:h, :w]
    gy = padded[1 : h + 1, :w] - padded[:h, :w]
    
    mag = np.sqrt(gx**2 + gy**2)
    return _contrast_stretch_uint8(mag)


def _backward_difference(image: np.ndarray) -> np.ndarray:
    """Compute Backward Difference gradient magnitude."""
    img_f = image.astype(np.float64)
    h, w = img_f.shape
    padded = np.pad(img_f, ((1, 0), (1, 0)), mode="edge")
    
    gx = padded[1 : h + 1, 1 : w + 1] - padded[1 : h + 1, :w]
    gy = padded[1 : h + 1, 1 : w + 1] - padded[:h, 1 : w + 1]
    
    mag = np.sqrt(gx**2 + gy**2)
    return _contrast_stretch_uint8(mag)


def _central_difference(image: np.ndarray) -> np.ndarray:
    """Compute Central Difference gradient magnitude."""
    img_f = image.astype(np.float64)
    h, w = img_f.shape
    padded = np.pad(img_f, ((1, 1), (1, 1)), mode="edge")
    
    gx = (padded[1 : h + 1, 2 : w + 2] - padded[1 : h + 1, :w]) / 2.0
    gy = (padded[2 : h + 2, 1 : w + 1] - padded[:h, 1 : w + 1]) / 2.0
    
    mag = np.sqrt(gx**2 + gy**2)
    return _contrast_stretch_uint8(mag)


# =====================================================================
# 3. Second-Order Operators: LoG, DoG, and Zero-Crossing Detection
# =====================================================================

def _generate_gaussian_kernel_2d(sigma: float, size: int | None = None) -> np.ndarray:
    """Generate 2D Gaussian filter kernel from first principles."""
    if size is None:
        radius = int(np.ceil(3.0 * sigma))
        size = 2 * radius + 1
    else:
        radius = size // 2

    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def _generate_log_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    """Generate discrete Laplacian of Gaussian (LoG) kernel."""
    if size is None:
        radius = int(np.ceil(3.0 * sigma))
        size = 2 * radius + 1
    else:
        radius = size // 2

    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    r2 = (x**2 + y**2).astype(np.float64)
    sigma2 = sigma**2
    sigma4 = sigma**4

    log_kernel = -(1.0 / (np.pi * sigma4)) * (1.0 - r2 / (2.0 * sigma2)) * np.exp(-r2 / (2.0 * sigma2))
    log_kernel = log_kernel - np.mean(log_kernel)
    return log_kernel


def _detect_zero_crossings(response: np.ndarray, threshold_ratio: float = 0.04) -> np.ndarray:
    """Detect zero crossings across 4 directional opposing neighbor pairs."""
    max_amp = np.max(np.abs(response))
    if max_amp < 1e-8:
        return np.zeros_like(response, dtype=np.uint8)

    tau = threshold_ratio * max_amp
    p = np.pad(response, 1, mode="edge")
    c = p[1:-1, 1:-1]

    # Horizontal
    left = p[1:-1, :-2]
    right = p[1:-1, 2:]
    h_cross = ((left * right < 0) | (c * left < 0) | (c * right < 0)) & (np.abs(left - right) > tau)

    # Vertical
    up = p[:-2, 1:-1]
    down = p[2:, 1:-1]
    v_cross = ((up * down < 0) | (c * up < 0) | (c * down < 0)) & (np.abs(up - down) > tau)

    # Main Diagonal
    ul = p[:-2, :-2]
    br = p[2:, 2:]
    d1_cross = ((ul * br < 0) | (c * ul < 0) | (c * br < 0)) & (np.abs(ul - br) > tau)

    # Anti-Diagonal
    ur = p[:-2, 2:]
    bl = p[2:, :-2]
    d2_cross = ((ur * bl < 0) | (c * ur < 0) | (c * bl < 0)) & (np.abs(ur - bl) > tau)

    edges = h_cross | v_cross | d1_cross | d2_cross
    return (edges.astype(np.uint8) * 255)


def _laplacian_of_gaussian(image: np.ndarray, sigma: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    """Execute LoG filtering and zero-crossing edge extraction."""
    kernel = _generate_log_kernel(sigma)
    response = _conv2d_scratch(image, kernel, pad_mode="edge")
    response_uint8 = _contrast_stretch_uint8(response)
    zero_crossings = _detect_zero_crossings(response, threshold_ratio=0.04)
    return response_uint8, zero_crossings


def _difference_of_gaussians(image: np.ndarray, sigma1: float = 1.0, sigma2: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    """Execute DoG filtering and zero-crossing edge extraction."""
    k1 = _generate_gaussian_kernel_2d(sigma1)
    k2 = _generate_gaussian_kernel_2d(sigma2)

    g1 = _conv2d_scratch(image, k1, pad_mode="edge")
    g2 = _conv2d_scratch(image, k2, pad_mode="edge")

    dog = g1 - g2
    response_uint8 = _contrast_stretch_uint8(dog)
    zero_crossings = _detect_zero_crossings(dog, threshold_ratio=0.04)
    return response_uint8, zero_crossings


# =====================================================================
# 4. Mosaic Grid Generation
# =====================================================================

def _build_mosaic_grid(
    images: list[np.ndarray],
    rows: int,
    cols: int,
    border_width: int = 2,
    border_color: int = 100,
) -> np.ndarray:
    """Tile a list of 2D uint8 images into a rows x cols mosaic."""
    h, w = images[0].shape[:2]
    std_images = []
    for im in images:
        if im.shape[:2] != (h, w):
            pil_im = Image.fromarray(im)
            pil_im = pil_im.resize((w, h), Image.Resampling.BILINEAR)
            im = np.array(pil_im)
        std_images.append(_to_3ch(im))

    total_slots = rows * cols
    while len(std_images) < total_slots:
        std_images.append(np.zeros((h, w, 3), dtype=np.uint8))

    row_blocks = []
    idx = 0
    for _ in range(rows):
        col_cells = []
        for c in range(cols):
            col_cells.append(std_images[idx])
            if c < cols - 1:
                col_cells.append(np.full((h, border_width, 3), border_color, dtype=np.uint8))
            idx += 1
        row_assembled = np.hstack(col_cells)
        row_blocks.append(row_assembled)

    grid_rows = []
    for r in range(rows):
        grid_rows.append(row_blocks[r])
        if r < rows - 1:
            sep_h = np.full((border_width, row_blocks[r].shape[1], 3), border_color, dtype=np.uint8)
            grid_rows.append(sep_h)

    return np.vstack(grid_rows)


# =====================================================================
# 5. Main GUI Operation Interface
# =====================================================================

def process(img: np.ndarray) -> dict[str, Image.Image]:
    """Execute 1st and 2nd Order Edge Detection and return visual artifacts."""
    # Convert input to greyscale
    grey = _rgb_to_grey(img)

    # 1. First-Order Difference Operators
    fwd_mag = _forward_difference(grey)
    bwd_mag = _backward_difference(grey)
    cen_mag = _central_difference(grey)

    # 2. Second-Order Operators (LoG & DoG)
    log_resp, log_zc = _laplacian_of_gaussian(grey, sigma=1.4)
    dog_resp, dog_zc = _difference_of_gaussians(grey, sigma1=1.0, sigma2=1.6)

    # 3. Composite Visual Grids
    grid_1st = _build_mosaic_grid([grey, fwd_mag, bwd_mag, cen_mag], rows=1, cols=4)
    grid_2nd = _build_mosaic_grid([log_resp, log_zc, dog_resp, dog_zc], rows=2, cols=2)
    grid_master = _build_mosaic_grid(
        [grey, fwd_mag, bwd_mag, cen_mag, log_resp, log_zc, dog_resp, dog_zc],
        rows=2,
        cols=4,
    )

    return {
        "1. Forward Difference (1st Order)": Image.fromarray(_to_3ch(fwd_mag)),
        "2. Backward Difference (1st Order)": Image.fromarray(_to_3ch(bwd_mag)),
        "3. Central Difference (1st Order)": Image.fromarray(_to_3ch(cen_mag)),
        "4. LoG Response (2nd Order)": Image.fromarray(_to_3ch(log_resp)),
        "5. LoG Zero-Crossing Edges": Image.fromarray(_to_3ch(log_zc)),
        "6. DoG Response (2nd Order)": Image.fromarray(_to_3ch(dog_resp)),
        "7. DoG Zero-Crossing Edges": Image.fromarray(_to_3ch(dog_zc)),
        "8. First-Order Comparison (1x4 Grid)": Image.fromarray(grid_1st),
        "9. Second-Order Comparison (2x2 Grid)": Image.fromarray(grid_2nd),
        "10. Master Overview (2x4 Grid)": Image.fromarray(grid_master),
    }
