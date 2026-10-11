"""Discrete Wavelet Transform (Manual DIP Implementation).

This script performs 2D Discrete Wavelet Transform (DWT) from first principles
using pure NumPy (no external wavelet libraries like pywt or scipy):
1. Converts input image to greyscale.
2. Performs 1-Level 2D Haar DWT decomposition into 4 subbands:
   - LL: Approximation (Low-pass row, Low-pass col)
   - HL: Vertical Details (High-pass row, Low-pass col)
   - LH: Horizontal Details (Low-pass row, High-pass col)
   - HH: Diagonal Details (High-pass row, High-pass col)
3. Performs 2-Level Multiresolution DWT decomposition on the LL subband.
4. Performs 2D Inverse DWT (IDWT) for exact image reconstruction and
   calculates Mean Squared Error (MSE) to verify mathematical accuracy.
5. Saves individual subbands, 1-level/2-level mosaic grids, and a comprehensive
   comparison figure to Output/.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


# =====================================================================
# 1. First-Principles Haar Wavelet Transform (Forward & Inverse)
# =====================================================================

def rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA numpy array to uint8 greyscale using luminance weights."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def _pad_to_even(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad image with edge replication if height or width is odd."""
    h, w = img.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
        return padded, (h, w)
    return img, (h, w)


def dwt1d_haar_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1D Haar DWT along rows (horizontal direction).
    
    Analysis filters (orthonormal):
        Low-pass (Approximation): L[c] = (x[2c] + x[2c + 1]) / sqrt(2)
        High-pass (Detail):        H[c] = (x[2c] - x[2c + 1]) / sqrt(2)
    """
    sqrt2 = np.sqrt(2.0)
    low = (mat[:, 0::2] + mat[:, 1::2]) / sqrt2
    high = (mat[:, 0::2] - mat[:, 1::2]) / sqrt2
    return low, high


def idwt1d_haar_rows(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """1D Inverse Haar DWT along rows.
    
    Synthesis filters (orthonormal):
        x[2c]     = (L[c] + H[c]) / sqrt(2)
        x[2c + 1] = (L[c] - H[c]) / sqrt(2)
    """
    sqrt2 = np.sqrt(2.0)
    h, w_half = low.shape
    reconstructed = np.zeros((h, w_half * 2), dtype=np.float64)
    reconstructed[:, 0::2] = (low + high) / sqrt2
    reconstructed[:, 1::2] = (low - high) / sqrt2
    return reconstructed


def dwt1d_haar_cols(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1D Haar DWT along columns (vertical direction)."""
    sqrt2 = np.sqrt(2.0)
    low = (mat[0::2, :] + mat[1::2, :]) / sqrt2
    high = (mat[0::2, :] - mat[1::2, :]) / sqrt2
    return low, high


def idwt1d_haar_cols(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """1D Inverse Haar DWT along columns."""
    sqrt2 = np.sqrt(2.0)
    h_half, w = low.shape
    reconstructed = np.zeros((h_half * 2, w), dtype=np.float64)
    reconstructed[0::2, :] = (low + high) / sqrt2
    reconstructed[1::2, :] = (low - high) / sqrt2
    return reconstructed


def dwt2d_haar(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Perform 2D Haar Discrete Wavelet Transform on a 2D image matrix.
    
    Process:
        1. Apply 1D DWT to rows -> separates into Low-pass (L) and High-pass (H) halves.
        2. Apply 1D DWT to columns of L -> produces LL (Approximation) and LH (Horizontal detail).
        3. Apply 1D DWT to columns of H -> produces HL (Vertical detail) and HH (Diagonal detail).
    
    Returns:
        LL: Low-Low subband (Approximation)
        LH: Low-High subband (Horizontal details / vertical frequency)
        HL: High-Low subband (Vertical details / horizontal frequency)
        HH: High-High subband (Diagonal details)
        orig_shape: (H, W) before any padding
    """
    img_f = image.astype(np.float64)
    img_padded, orig_shape = _pad_to_even(img_f)
    
    # Row-wise transform
    l_row, h_row = dwt1d_haar_rows(img_padded)
    
    # Column-wise transform on each half
    ll, lh = dwt1d_haar_cols(l_row)
    hl, hh = dwt1d_haar_cols(h_row)
    
    return ll, lh, hl, hh, orig_shape


def idwt2d_haar(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
    orig_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Perform 2D Inverse Haar Discrete Wavelet Transform.
    
    Process:
        1. Invert column transforms on (LL, LH) -> l_row and (HL, HH) -> h_row.
        2. Invert row transform on (l_row, h_row) -> reconstructed spatial image.
        3. Crop back to original shape if padding was applied.
    """
    # Column-wise inverse
    l_row = idwt1d_haar_cols(ll, lh)
    h_row = idwt1d_haar_cols(hl, hh)
    
    # Row-wise inverse
    reconstructed = idwt1d_haar_rows(l_row, h_row)
    
    if orig_shape is not None:
        reconstructed = reconstructed[:orig_shape[0], :orig_shape[1]]
        
    return reconstructed


# =====================================================================
# 2. Subband Normalization and Grid Compositing
# =====================================================================

def normalize_detail_subband(subband: np.ndarray) -> np.ndarray:
    """Normalize detail coefficients (LH, HL, HH) to uint8 [0, 255] for clear visual contrast.
    
    Uses absolute magnitude contrast stretching so edge boundaries are prominent.
    """
    abs_sub = np.abs(subband)
    max_val = np.max(abs_sub)
    if max_val > 1e-8:
        norm = (abs_sub / max_val) * 255.0
    else:
        norm = np.zeros_like(abs_sub)
    return np.clip(norm, 0, 255).astype(np.uint8)


def normalize_approx_subband(ll: np.ndarray) -> np.ndarray:
    """Normalize approximation subband (LL) to uint8 [0, 255].
    
    Since 2D orthonormal Haar scales constants by 2 (sqrt(2) * sqrt(2)),
    dividing by 2 maps the original [0, 255] range back faithfully.
    """
    norm = ll / 2.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def build_1level_grid(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
    border_color: int = 128,
    border_width: int = 2,
) -> np.ndarray:
    """Assemble classic 2x2 DWT subband mosaic:
    +----------+----------+
    |    LL    |    HL    |
    | (Approx) |(Vertical)|
    +----------+----------+
    |    LH    |    HH    |
    |(Horizont)|(Diagonal)|
    +----------+----------+
    """
    ll_u = normalize_approx_subband(ll)
    hl_u = normalize_detail_subband(hl)
    lh_u = normalize_detail_subband(lh)
    hh_u = normalize_detail_subband(hh)
    
    # Horizontal border
    sep_h = np.full((border_width, ll_u.shape[1]), border_color, dtype=np.uint8)
    left_col = np.vstack((ll_u, sep_h, lh_u))
    right_col = np.vstack((hl_u, sep_h, hh_u))
    
    # Vertical border
    sep_v = np.full((left_col.shape[0], border_width), border_color, dtype=np.uint8)
    grid = np.hstack((left_col, sep_v, right_col))
    return grid


def build_2level_grid(
    image: np.ndarray,
    border_color: int = 128,
    border_width: int = 2,
) -> tuple[np.ndarray, dict]:
    """Perform 2-Level DWT decomposition and compose hierarchical pyramid grid.
    
    Level 1 decomposes Image -> LL1, LH1, HL1, HH1.
    Level 2 decomposes LL1 -> LL2, LH2, HL2, HH2.
    """
    ll1, lh1, hl1, hh1, orig_shape1 = dwt2d_haar(image)
    ll2, lh2, hl2, hh2, orig_shape2 = dwt2d_haar(ll1)
    
    # Level 2 grid replaces LL1
    grid_lvl2 = build_1level_grid(ll2, lh2, hl2, hh2, border_color, border_width)
    
    hl1_u = normalize_detail_subband(hl1)
    lh1_u = normalize_detail_subband(lh1)
    hh1_u = normalize_detail_subband(hh1)
    
    # Match sizes if needed due to rounding/padding
    h2, w2 = grid_lvl2.shape
    if hl1_u.shape[0] != h2:
        grid_lvl2 = np.pad(grid_lvl2, ((0, max(0, hl1_u.shape[0] - h2)), (0, 0)), mode="edge")[:hl1_u.shape[0], :]
    if hl1_u.shape[1] != w2:
        grid_lvl2 = np.pad(grid_lvl2, ((0, 0), (0, max(0, hl1_u.shape[1] - w2))), mode="edge")[:, :hl1_u.shape[1]]
        
    sep_h = np.full((border_width, grid_lvl2.shape[1]), border_color, dtype=np.uint8)
    left_col = np.vstack((grid_lvl2, sep_h, lh1_u))
    right_col = np.vstack((hl1_u, sep_h, hh1_u))
    
    sep_v = np.full((left_col.shape[0], border_width), border_color, dtype=np.uint8)
    grid_2level = np.hstack((left_col, sep_v, right_col))
    
    subbands_data = {
        "level1": {"LL": ll1, "LH": lh1, "HL": hl1, "HH": hh1, "shape": orig_shape1},
        "level2": {"LL": ll2, "LH": lh2, "HL": hl2, "HH": hh2, "shape": orig_shape2},
    }
    return grid_2level, subbands_data


# =====================================================================
# 3. Overview Figure Generation (Matplotlib)
# =====================================================================

def generate_comparison_figure(
    original_grey: np.ndarray,
    grid_1level: np.ndarray,
    grid_2level: np.ndarray,
    reconstructed_grey: np.ndarray,
    mse: float,
    max_err: float,
) -> Image.Image:
    """Render a 4-panel overview comparison: Original, 1-Level DWT, 2-Level DWT, IDWT Reconstruction."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), facecolor="#1e1e2e")
    fig.suptitle(
        "Discrete Wavelet Transform (2D Haar DWT & IDWT)",
        fontsize=16,
        fontweight="bold",
        color="#cdd6f4",
        y=0.96,
    )

    panels = [
        (axes[0, 0], original_grey, "Original Image (Greyscale)", "Spatial domain"),
        (axes[0, 1], grid_1level, "1-Level DWT Subbands (2×2)", "Top-L: LL (Approx) | Top-R: HL (Vertical)\nBot-L: LH (Horiz)   | Bot-R: HH (Diagonal)"),
        (axes[1, 0], grid_2level, "2-Level Multiresolution DWT", "LL decomposed recursively into LL2, HL2, LH2, HH2"),
        (axes[1, 1], reconstructed_grey, "IDWT Reconstructed Image", f"Exact mathematical inverse\nMSE: {mse:.2e} | Max Diff: {max_err:.1f}"),
    ]

    for ax, data, title, subtitle in panels:
        ax.set_facecolor("#181825")
        ax.imshow(data, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12, fontweight="semibold", color="#89b4fa", pad=8)
        ax.set_xlabel(subtitle, fontsize=9, color="#a6adc8", labelpad=6)
        ax.tick_params(colors="#6c7086", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#45475a")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    result_img = Image.fromarray(rgba[:, :, :3])
    plt.close(fig)
    return result_img


# =====================================================================
# 4. Main Standalone Script Entrypoint
# =====================================================================

def main():
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / "Output"
    output_dir.mkdir(exist_ok=True)

    # Locate input image
    possible_dirs = [root_dir / "Input_images", root_dir / "Input"]
    input_dir = next((d for d in possible_dirs if d.is_dir() and any(d.iterdir())), None)
    if input_dir is None:
        raise FileNotFoundError("Could not find an Input_images or Input directory with files.")

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts]
    if not images:
        raise FileNotFoundError(f"No valid image files found in {input_dir}.")

    # Prefer scenary or tree if available, otherwise take the first
    target_img_path = images[0]
    for candidate in images:
        if "scenar" in candidate.name.lower() or "tree" in candidate.name.lower():
            target_img_path = candidate
            break

    print(f"Loading input image: {target_img_path.name}")
    pil_raw = Image.open(target_img_path).convert("RGB")
    rgb_arr = np.array(pil_raw)

    # 1. Greyscale conversion
    grey = rgb_to_grey(rgb_arr)
    h, w = grey.shape
    print(f"Image converted to greyscale ({w}x{h}).")

    # 2. 1-Level 2D Haar DWT
    ll, lh, hl, hh, orig_shape = dwt2d_haar(grey)
    print(f"1-Level DWT computed. Subband shapes: {ll.shape}")

    # 3. Inverse DWT (IDWT) Reconstruction & Verification
    recon = idwt2d_haar(ll, lh, hl, hh, orig_shape=orig_shape)
    recon_uint8 = np.clip(np.round(recon), 0, 255).astype(np.uint8)

    # Calculate error metrics
    mse_float = float(np.mean((grey.astype(np.float64) - recon) ** 2))
    max_abs_float = float(np.max(np.abs(grey.astype(np.float64) - recon)))
    mse_uint8 = float(np.mean((grey.astype(np.int32) - recon_uint8.astype(np.int32)) ** 2))
    max_abs_uint8 = float(np.max(np.abs(grey.astype(np.int32) - recon_uint8.astype(np.int32))))

    print("\n--- IDWT Mathematical Verification ---")
    print(f"Floating-point MSE      : {mse_float:.4e}")
    print(f"Floating-point Max Diff : {max_abs_float:.4e}")
    print(f"Uint8 Rounding MSE      : {mse_uint8:.4f}")
    print(f"Uint8 Max Absolute Diff : {max_abs_uint8:.1f}")
    assert mse_float < 1e-10, "IDWT reconstruction error is too high! Forward/inverse math mismatch."
    print("Verification PASSED: Perfect reconstruction confirmed!\n")

    # 4. Create Subbands, Grids, and Figures
    ll_vis = normalize_approx_subband(ll)
    lh_vis = normalize_detail_subband(lh)
    hl_vis = normalize_detail_subband(hl)
    hh_vis = normalize_detail_subband(hh)

    grid_1level = build_1level_grid(ll, lh, hl, hh)
    grid_2level, _ = build_2level_grid(grey)

    comparison_fig = generate_comparison_figure(
        original_grey=grey,
        grid_1level=grid_1level,
        grid_2level=grid_2level,
        reconstructed_grey=recon_uint8,
        mse=mse_float,
        max_err=max_abs_float,
    )

    # 5. Save all outputs
    outputs = {
        "dwt_original_grey.png": Image.fromarray(grey),
        "dwt_LL_approximation.png": Image.fromarray(ll_vis),
        "dwt_LH_horizontal.png": Image.fromarray(lh_vis),
        "dwt_HL_vertical.png": Image.fromarray(hl_vis),
        "dwt_HH_diagonal.png": Image.fromarray(hh_vis),
        "dwt_subbands_1level.png": Image.fromarray(grid_1level),
        "dwt_subbands_2level.png": Image.fromarray(grid_2level),
        "dwt_reconstructed.png": Image.fromarray(recon_uint8),
        "dwt_comparison_overview.png": comparison_fig,
    }

    for fname, img_obj in outputs.items():
        save_path = output_dir / fname
        img_obj.save(save_path)
        print(f"Saved: {save_path.relative_to(root_dir)}")

    print("\nDiscrete Wavelet Transform processing completed successfully!")


if __name__ == "__main__":
    main()
