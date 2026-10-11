# RGB to Greyscale & Individual Channels (from rgb2grey.py)

import numpy as np
from PIL import Image

NAME = "RGB to Greyscale & Channels"


def process(img: np.ndarray) -> dict[str, Image.Image]:
    """Extract R, G, B channels, greyscale, and a 2x2 grid."""
    red_img = np.zeros_like(img)
    red_img[:, :, 0] = img[:, :, 0]

    green_img = np.zeros_like(img)
    green_img[:, :, 1] = img[:, :, 1]

    blue_img = np.zeros_like(img)
    blue_img[:, :, 2] = img[:, :, 2]

    grey = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
    grey_img = np.stack([grey, grey, grey], axis=-1)

    top_row = np.hstack((red_img, green_img))
    bottom_row = np.hstack((blue_img, grey_img))
    final_grid = np.vstack((top_row, bottom_row))

    return {
        "Red Channel": Image.fromarray(red_img),
        "Green Channel": Image.fromarray(green_img),
        "Blue Channel": Image.fromarray(blue_img),
        "Greyscale": Image.fromarray(grey_img),
        "Grid (2x2)": Image.fromarray(final_grid),
    }
