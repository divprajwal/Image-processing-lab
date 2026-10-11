# Digital Image Processing Lab

Course work and experiments for the Digital Image Processing course.

## Files

| File | Description |
| :--- | :--- |
| `rgb2grey.py` | Extracts RGB channels and converts an image to greyscale, outputs a 2×2 grid. |
| `bit_plane_slicing.py` | Converts an image to greyscale, extracts all 8 bit planes (Bit 0 to Bit 7), and outputs a 3×3 comparison grid. |
| `histogram_equalization.py` | Contrast enhancement via histogram equalization from first principles, outputs input/equalized images, histograms, and a 2×2 comparison grid. |
| `histogram_equalization_types.py` | Implementation of Global (GHE), Local (LHE), Adaptive (AHE), and CLAHE techniques with paired histograms and a 5×2 overview comparison grid. |
| `discrete_wavelet_transform.py` | 2D Discrete Wavelet Transform (Haar DWT/IDWT) from first principles, 1-level/2-level multiresolution subbands, and exact reconstruction. |
| `shannon_fano_coding.py` | Shannon-Fano source coding from first principles with prefix-tree decoding, entropy, efficiency metrics, and codeword length heatmaps. |
| `huffman_coding.py` | Optimal Huffman coding from first principles with binary tree construction, prefix decoding, entropy, efficiency metrics, and bit-length heatmaps. |
| `edge_detection.py` | First-order (Forward, Backward, Central Difference) and Second-order (LoG, DoG) edge detection from scratch with zero-crossing detection. |
| `housie_tickets.py` | Generates and renders valid Housie/Tambola tickets as a PNG. |
| `GUI/app.py` | Interactive desktop toolkit for running DIP operations on images. |
| `GUI/operations/` | Drop-in modules for the GUI (RGB & Channels, Bit Plane Slicing, Histogram Equalization, HE Types, DWT, Shannon-Fano, Huffman, Edge Detection). |

## Running the GUI

```bash
pip install numpy pillow matplotlib ttkbootstrap
python GUI/app.py
```
