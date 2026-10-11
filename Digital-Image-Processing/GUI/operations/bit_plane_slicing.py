# Bit Plane Slicing (Greyscale & 8 Individual Bit Planes)

import numpy as np
from PIL import Image

NAME = "Bit Plane Slicing"


def process(img: np.ndarray) -> dict[str, Image.Image]:
    """Extract greyscale, 8 individual bit planes (Bit 0 to 7), 3x3 comparison grid, and upper bits reconstruction."""
    # 1. Manual Greyscale Conversion
    grey = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
    grey_3ch = np.stack([grey, grey, grey], axis=-1)

    # 2. Extract 8 Bit Planes (Bit 0 LSB to Bit 7 MSB)
    bit_planes_3ch = []
    for bit in range(8):
        plane_uint8 = (((grey >> bit) & 1) * 255).astype(np.uint8)
        p3 = np.stack([plane_uint8, plane_uint8, plane_uint8], axis=-1)
        bit_planes_3ch.append(p3)

    # 3. Create 3x3 Combined Grid
    # [ Greyscale       ] [ Bit Plane 7 (MSB) ] [ Bit Plane 6       ]
    # [ Bit Plane 5     ] [ Bit Plane 4       ] [ Bit Plane 3       ]
    # [ Bit Plane 2     ] [ Bit Plane 1       ] [ Bit Plane 0 (LSB) ]
    row0 = np.hstack((grey_3ch, bit_planes_3ch[7], bit_planes_3ch[6]))
    row1 = np.hstack((bit_planes_3ch[5], bit_planes_3ch[4], bit_planes_3ch[3]))
    row2 = np.hstack((bit_planes_3ch[2], bit_planes_3ch[1], bit_planes_3ch[0]))
    grid_3x3 = np.vstack((row0, row1, row2))

    # 4. Reconstruction using upper 4 bits (Bits 7..4) to show compression / significance
    recon_upper4 = (grey & 0b11110000).astype(np.uint8)
    recon_upper4_3ch = np.stack([recon_upper4, recon_upper4, recon_upper4], axis=-1)

    return {
        "Combined Grid (3x3)": Image.fromarray(grid_3x3),
        "Greyscale": Image.fromarray(grey_3ch),
        "Bit Plane 7 (MSB)": Image.fromarray(bit_planes_3ch[7]),
        "Bit Plane 6": Image.fromarray(bit_planes_3ch[6]),
        "Bit Plane 5": Image.fromarray(bit_planes_3ch[5]),
        "Bit Plane 4": Image.fromarray(bit_planes_3ch[4]),
        "Bit Plane 3": Image.fromarray(bit_planes_3ch[3]),
        "Bit Plane 2": Image.fromarray(bit_planes_3ch[2]),
        "Bit Plane 1": Image.fromarray(bit_planes_3ch[1]),
        "Bit Plane 0 (LSB)": Image.fromarray(bit_planes_3ch[0]),
        "Reconstruction (Bits 7-4)": Image.fromarray(recon_upper4_3ch),
    }
