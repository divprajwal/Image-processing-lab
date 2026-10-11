# Discrete Wavelet Transform (DWT) Operation for DIP Lab GUI
# Implemented from first principles using pure NumPy (no external wavelet libraries)

import numpy as np
from PIL import Image

NAME = "Discrete Wavelet Transform (DWT)"


# =====================================================================
# 1. Manual 2D Haar DWT & IDWT from First Principles
# =====================================================================

def _rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA to uint8 greyscale."""
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


def _dwt1d_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1D Haar DWT along rows."""
    sqrt2 = np.sqrt(2.0)
    low = (mat[:, 0::2] + mat[:, 1::2]) / sqrt2
    high = (mat[:, 0::2] - mat[:, 1::2]) / sqrt2
    return low, high


def _idwt1d_rows(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """1D Inverse Haar DWT along rows."""
    sqrt2 = np.sqrt(2.0)
    h, w_half = low.shape
    rec = np.zeros((h, w_half * 2), dtype=np.float64)
    rec[:, 0::2] = (low + high) / sqrt2
    rec[:, 1::2] = (low - high) / sqrt2
    return rec


def _dwt1d_cols(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1D Haar DWT along columns."""
    sqrt2 = np.sqrt(2.0)
    low = (mat[0::2, :] + mat[1::2, :]) / sqrt2
    high = (mat[0::2, :] - mat[1::2, :]) / sqrt2
    return low, high


def _idwt1d_cols(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """1D Inverse Haar DWT along columns."""
    sqrt2 = np.sqrt(2.0)
    h_half, w = low.shape
    rec = np.zeros((h_half * 2, w), dtype=np.float64)
    rec[0::2, :] = (low + high) / sqrt2
    rec[1::2, :] = (low - high) / sqrt2
    return rec


def _dwt2d(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """2D Haar DWT returning (LL, LH, HL, HH, orig_shape)."""
    img_f = image.astype(np.float64)
    img_padded, orig_shape = _pad_to_even(img_f)
    
    l_row, h_row = _dwt1d_rows(img_padded)
    ll, lh = _dwt1d_cols(l_row)
    hl, hh = _dwt1d_cols(h_row)
    return ll, lh, hl, hh, orig_shape


def _idwt2d(
    ll: np.ndarray,
    lh: np.ndarray,
    hl: np.ndarray,
    hh: np.ndarray,
    orig_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """2D Inverse Haar DWT reconstructing spatial image."""
    l_row = _idwt1d_cols(ll, lh)
    h_row = _idwt1d_cols(hl, hh)
    rec = _idwt1d_rows(l_row, h_row)
    if orig_shape is not None:
        rec = rec[:orig_shape[0], :orig_shape[1]]
    return rec


# =====================================================================
# 2. Normalization & Grid Assembly
# =====================================================================

def _norm_approx(ll: np.ndarray) -> np.ndarray:
    """Normalize LL approximation to uint8 [0, 255]."""
    return np.clip(ll / 2.0, 0, 255).astype(np.uint8)


def _norm_detail(sub: np.ndarray) -> np.ndarray:
    """Contrast stretch detail subband absolute values to uint8 [0, 255]."""
    abs_sub = np.abs(sub)
    m = np.max(abs_sub)
    if m > 1e-8:
        norm = (abs_sub / m) * 255.0
    else:
        norm = np.zeros_like(abs_sub)
    return np.clip(norm, 0, 255).astype(np.uint8)


def _build_1level_grid(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray) -> np.ndarray:
    """Assemble 2x2 DWT mosaic grid."""
    ll_u = _norm_approx(ll)
    hl_u = _norm_detail(hl)
    lh_u = _norm_detail(lh)
    hh_u = _norm_detail(hh)
    
    border_w = 2
    border_col = 128
    sep_h = np.full((border_w, ll_u.shape[1]), border_col, dtype=np.uint8)
    left_col = np.vstack((ll_u, sep_h, lh_u))
    right_col = np.vstack((hl_u, sep_h, hh_u))
    
    sep_v = np.full((left_col.shape[0], border_w), border_col, dtype=np.uint8)
    return np.hstack((left_col, sep_v, right_col))


def _build_2level_grid(image: np.ndarray) -> np.ndarray:
    """Assemble 2-level hierarchical multiresolution grid."""
    ll1, lh1, hl1, hh1, _ = _dwt2d(image)
    ll2, lh2, hl2, hh2, _ = _dwt2d(ll1)
    
    grid_lvl2 = _build_1level_grid(ll2, lh2, hl2, hh2)
    hl1_u = _norm_detail(hl1)
    lh1_u = _norm_detail(lh1)
    hh1_u = _norm_detail(hh1)
    
    h2, w2 = grid_lvl2.shape
    if hl1_u.shape[0] != h2:
        grid_lvl2 = np.pad(grid_lvl2, ((0, max(0, hl1_u.shape[0] - h2)), (0, 0)), mode="edge")[:hl1_u.shape[0], :]
    if hl1_u.shape[1] != w2:
        grid_lvl2 = np.pad(grid_lvl2, ((0, 0), (0, max(0, hl1_u.shape[1] - w2))), mode="edge")[:, :hl1_u.shape[1]]
        
    border_w = 2
    border_col = 128
    sep_h = np.full((border_w, grid_lvl2.shape[1]), border_col, dtype=np.uint8)
    left_col = np.vstack((grid_lvl2, sep_h, lh1_u))
    right_col = np.vstack((hl1_u, sep_h, hh1_u))
    
    sep_v = np.full((left_col.shape[0], border_w), border_col, dtype=np.uint8)
    return np.hstack((left_col, sep_v, right_col))


# =====================================================================
# 3. Main GUI Operation Interface
# =====================================================================

def process(img: np.ndarray) -> dict[str, Image.Image]:
    """Execute Discrete Wavelet Transform and return all visual artifacts."""
    # Convert input to greyscale
    grey = _rgb_to_grey(img)

    # Forward 1-Level 2D DWT
    ll, lh, hl, hh, orig_shape = _dwt2d(grey)

    # Inverse 2D DWT (Reconstruction)
    recon = _idwt2d(ll, lh, hl, hh, orig_shape=orig_shape)
    recon_uint8 = np.clip(np.round(recon), 0, 255).astype(np.uint8)

    # Normalize subbands
    ll_u = _norm_approx(ll)
    lh_u = _norm_detail(lh)
    hl_u = _norm_detail(hl)
    hh_u = _norm_detail(hh)

    # Construct grids
    grid_1level = _build_1level_grid(ll, lh, hl, hh)
    grid_2level = _build_2level_grid(grey)

    # Convert 1-channel arrays to 3-channel for GUI consistency
    def _to_3ch_img(arr: np.ndarray) -> Image.Image:
        return Image.fromarray(np.stack([arr, arr, arr], axis=-1))

    return {
        "1-Level DWT Grid (2x2)": _to_3ch_img(grid_1level),
        "2-Level DWT Grid": _to_3ch_img(grid_2level),
        "LL (Approximation)": _to_3ch_img(ll_u),
        "LH (Horizontal Detail)": _to_3ch_img(lh_u),
        "HL (Vertical Detail)": _to_3ch_img(hl_u),
        "HH (Diagonal Detail)": _to_3ch_img(hh_u),
        "IDWT Reconstructed": _to_3ch_img(recon_uint8),
    }
