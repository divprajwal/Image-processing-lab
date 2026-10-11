import os
import numpy as np
from PIL import Image

# Determine input and output directories
possible_input_dirs = ["Input_images", "Input"]
input_dir = next((d for d in possible_input_dirs if os.path.isdir(d) and len(os.listdir(d)) > 0), "Input_images")
output_dir = "Output_images" if os.path.isdir("Output_images") else "Output"
os.makedirs(output_dir, exist_ok=True)

# Find image files
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
if not image_files:
    raise FileNotFoundError(f"No image found in '{input_dir}' directory.")

img_path = os.path.join(input_dir, image_files[0])
img = np.array(Image.open(img_path).convert("RGB"))
print(f"Loaded image: {img_path} ({img.shape[1]}x{img.shape[0]})")

# 1. Manual Greyscale Conversion using standard luminance weights
grey = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
grey_3ch = np.stack([grey, grey, grey], axis=-1)

# Save Greyscale Image
grey_output_path = os.path.join(output_dir, "greyscale.png")
Image.fromarray(grey_3ch).save(grey_output_path)
print(f"Saved greyscale image to: {grey_output_path}")

# 2. Extract 8 Bit Planes (Bit 0 = LSB, Bit 7 = MSB)
# Each bit plane is extracted by shifting bits and masking with 1, then scaled to 0-255 for visualization
bit_planes_3ch = []
for bit in range(8):
    plane = ((grey >> bit) & 1) * 255
    plane_uint8 = plane.astype(np.uint8)
    plane_3ch = np.stack([plane_uint8, plane_uint8, plane_uint8], axis=-1)
    bit_planes_3ch.append(plane_3ch)
    
    # Save individual bit plane
    bit_name = f"bit_plane_{bit}_LSB.png" if bit == 0 else (f"bit_plane_{bit}_MSB.png" if bit == 7 else f"bit_plane_{bit}.png")
    plane_path = os.path.join(output_dir, bit_name)
    Image.fromarray(plane_3ch).save(plane_path)
    print(f"Saved Bit Plane {bit} ({'MSB' if bit==7 else ('LSB' if bit==0 else 'Bit ' + str(bit))}) to: {plane_path}")

# 3. Create 3x3 Combined Grid
# Layout:
# [ Greyscale       ] [ Bit Plane 7 (MSB) ] [ Bit Plane 6       ]
# [ Bit Plane 5     ] [ Bit Plane 4       ] [ Bit Plane 3       ]
# [ Bit Plane 2     ] [ Bit Plane 1       ] [ Bit Plane 0 (LSB) ]
row0 = np.hstack((grey_3ch, bit_planes_3ch[7], bit_planes_3ch[6]))
row1 = np.hstack((bit_planes_3ch[5], bit_planes_3ch[4], bit_planes_3ch[3]))
row2 = np.hstack((bit_planes_3ch[2], bit_planes_3ch[1], bit_planes_3ch[0]))
final_grid = np.vstack((row0, row1, row2))

grid_output_path = os.path.join(output_dir, "bit_planes_grid_3x3.png")
Image.fromarray(final_grid).save(grid_output_path)
print(f"Saved 3x3 bit planes grid to: {grid_output_path}")
