import os
import numpy as np
from PIL import Image

input_dir = "Input"
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
if not image_files:
    raise FileNotFoundError("No image found in Input directory.")

img_path = os.path.join(input_dir, image_files[0])
img = np.array(Image.open(img_path).convert("RGB"))

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

output_path = os.path.join(output_dir, "output_inlayed.png")
Image.fromarray(final_grid).save(output_path)
print(f"Inlayed image successfully saved to {output_path}")
