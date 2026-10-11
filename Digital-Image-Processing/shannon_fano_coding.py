"""Shannon-Fano Coding (Manual DIP Implementation).

This script performs Shannon-Fano source coding from first principles on an image:
1. Converts input image to greyscale.
2. Computes the probability distribution of pixel intensity symbols (0-255).
3. Recursively constructs the Shannon-Fano prefix codebook by dividing symbols
   into two sub-groups of approximately equal cumulative probabilities.
4. Computes core Information Theory & DIP compression metrics:
   - Shannon Entropy H
   - Average Codeword Length L_avg
   - Coding Efficiency eta
   - Redundancy R
   - Compression Ratio CR
5. Encodes the image into a binary bitstream and decodes it via prefix-tree traversal.
6. Verifies exact mathematical lossless reconstruction (MSE = 0, 100% pixel match).
7. Generates informative visual outputs (Decoded image, Bit-length spatial heatmap,
   Metrics summary card, and Codeword length distribution) saved to Output/.
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
# 1. First-Principles Shannon-Fano Coding & Decoding
# =====================================================================

def rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA numpy array to uint8 greyscale using luminance weights."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def compute_symbol_probabilities(grey: np.ndarray) -> tuple[dict[int, float], dict[int, int]]:
    """Compute normalized probabilities and frequency counts for all present symbols."""
    counts = np.bincount(grey.ravel(), minlength=256)
    total_pixels = grey.size
    freq_dict = {i: int(counts[i]) for i in range(256) if counts[i] > 0}
    prob_dict = {i: counts[i] / total_pixels for i in freq_dict}
    return prob_dict, freq_dict


def _shannon_fano_recursive(symbols_with_prob: list[tuple[int, float]]) -> dict[int, str]:
    """Recursively split sorted symbols into two subsets with minimal probability difference."""
    if len(symbols_with_prob) <= 1:
        return {symbols_with_prob[0][0]: ""}

    total_prob = sum(p for _, p in symbols_with_prob)
    running_sum = 0.0
    best_diff = float("inf")
    split_idx = 1

    for i in range(len(symbols_with_prob) - 1):
        running_sum += symbols_with_prob[i][1]
        rest = total_prob - running_sum
        diff = abs(running_sum - rest)
        if diff < best_diff:
            best_diff = diff
            split_idx = i + 1

    left = symbols_with_prob[:split_idx]
    right = symbols_with_prob[split_idx:]

    left_codes = _shannon_fano_recursive(left)
    right_codes = _shannon_fano_recursive(right)

    codes = {}
    for sym, code in left_codes.items():
        codes[sym] = "0" + code
    for sym, code in right_codes.items():
        codes[sym] = "1" + code

    return codes


def build_shannon_fano_codebook(prob_dict: dict[int, float]) -> dict[int, str]:
    """Generate complete Shannon-Fano codebook from symbol probabilities."""
    sorted_symbols = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_symbols) == 1:
        return {sorted_symbols[0][0]: "0"}
    return _shannon_fano_recursive(sorted_symbols)


def compute_compression_metrics(
    prob_dict: dict[int, float],
    freq_dict: dict[int, int],
    codebook: dict[int, str],
    total_pixels: int,
) -> dict[str, float]:
    """Compute Shannon Entropy, Average Length, Efficiency, Redundancy, and Compression Ratio."""
    # Entropy: H = -sum(p * log2(p))
    entropy = -sum(p * np.log2(p) for p in prob_dict.values() if p > 0)

    # Average codeword length: L_avg = sum(p * len(code))
    l_avg = sum(prob_dict[s] * len(codebook[s]) for s in prob_dict)

    # Coding efficiency: eta = H / L_avg * 100%
    efficiency = (entropy / l_avg * 100.0) if l_avg > 0 else 100.0

    # Redundancy: R = L_avg - H
    redundancy = max(0.0, l_avg - entropy)

    # Compression Ratio: CR = 8 / L_avg
    compression_ratio = 8.0 / l_avg if l_avg > 0 else 1.0

    # Total bits
    orig_bits = total_pixels * 8
    compressed_bits = sum(freq_dict[s] * len(codebook[s]) for s in freq_dict)
    space_savings = (1.0 - (compressed_bits / orig_bits)) * 100.0

    return {
        "entropy": float(entropy),
        "avg_length": float(l_avg),
        "efficiency": float(efficiency),
        "redundancy": float(redundancy),
        "compression_ratio": float(compression_ratio),
        "orig_bits": int(orig_bits),
        "compressed_bits": int(compressed_bits),
        "space_savings": float(space_savings),
    }


# Fast Prefix-Tree (Trie) for Decoding
class TrieNode:
    __slots__ = ("children", "symbol")

    def __init__(self):
        self.children = [None, None]
        self.symbol = None


def encode_image(grey: np.ndarray, codebook: dict[int, str]) -> str:
    """Encode 2D image into binary string using codebook."""
    flat = grey.ravel()
    return "".join(codebook[s] for s in flat)


def decode_image(bitstream: str, codebook: dict[int, str], shape: tuple[int, int]) -> np.ndarray:
    """Decode binary bitstream back into original spatial image using a Prefix Trie."""
    # Build trie
    root = TrieNode()
    for symbol, code in codebook.items():
        curr = root
        for bit in code:
            idx = 0 if bit == "0" else 1
            if curr.children[idx] is None:
                curr.children[idx] = TrieNode()
            curr = curr.children[idx]
        curr.symbol = symbol

    # Traverse trie
    total_pixels = shape[0] * shape[1]
    decoded = np.empty(total_pixels, dtype=np.uint8)
    ptr = 0
    curr = root

    for bit in bitstream:
        curr = curr.children[0 if bit == "0" else 1]
        if curr.symbol is not None:
            decoded[ptr] = curr.symbol
            ptr += 1
            curr = root
            if ptr == total_pixels:
                break

    return decoded.reshape(shape)


# =====================================================================
# 2. Visualizations & Figures
# =====================================================================

def generate_bit_length_heatmap(grey: np.ndarray, codebook: dict[int, str]) -> Image.Image:
    """Create a color-coded heatmap showing how many bits were required for each pixel."""
    len_lut = np.zeros(256, dtype=np.float32)
    for sym, code in codebook.items():
        len_lut[sym] = len(code)

    bit_map = len_lut[grey]
    min_len, max_len = np.min(bit_map), np.max(bit_map)

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=100, facecolor="#181825")
    ax.set_facecolor("#181825")
    cmap = plt.cm.plasma
    im = ax.imshow(bit_map, cmap=cmap, vmin=min_len, vmax=max_len)

    ax.set_title(
        f"Shannon-Fano Codeword Length Map\n(Min: {int(min_len)} bits, Max: {int(max_len)} bits)",
        color="#cdd6f4",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Bits per Pixel", color="#a6adc8", fontsize=9)
    cbar.ax.tick_params(colors="#a6adc8", labelsize=8)

    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return img


def generate_metrics_summary_card(
    metrics: dict[str, float],
    prob_dict: dict[int, float],
    freq_dict: dict[int, int],
    codebook: dict[int, str],
    total_pixels: int,
) -> Image.Image:
    """Render an infographic card with entropy, compression metrics, and top symbols."""
    fig = plt.figure(figsize=(9, 7), dpi=110, facecolor="#1e1e2e")

    # Header
    fig.text(0.5, 0.94, "Shannon-Fano Coding: Performance & Metrics", color="#cdd6f4",
             fontsize=14, fontweight="bold", ha="center")
    fig.text(0.5, 0.90, f"Total Pixels: {total_pixels:,} | Unique Symbols: {len(prob_dict)}",
             color="#a6adc8", fontsize=10, ha="center")

    # Metric boxes (4 cards)
    box_data = [
        ("Shannon Entropy (H)", f"{metrics['entropy']:.3f} bpp", "#89b4fa"),
        ("Avg Length (L_avg)", f"{metrics['avg_length']:.3f} bpp", "#a6e3a1"),
        ("Coding Efficiency (η)", f"{metrics['efficiency']:.2f}%", "#fab387"),
        ("Compression Ratio", f"{metrics['compression_ratio']:.2f} : 1", "#f38ba8"),
    ]

    for i, (title, val, col) in enumerate(box_data):
        x = 0.08 + i * 0.22
        rect = plt.Rectangle((x, 0.74), 0.19, 0.12, transform=fig.transFigure,
                             facecolor="#313244", edgecolor=col, linewidth=1.5,
                             clip_on=False, zorder=2)
        fig.add_artist(rect)
        fig.text(x + 0.095, 0.81, val, color=col, fontsize=13, fontweight="bold",
                 ha="center", va="center", transform=fig.transFigure, zorder=3)
        fig.text(x + 0.095, 0.76, title, color="#cdd6f4", fontsize=8,
                 ha="center", va="center", transform=fig.transFigure, zorder=3)

    # Sub-metrics line
    fig.text(
        0.5, 0.69,
        f"Redundancy: {metrics['redundancy']:.3f} bits/pixel  |  "
        f"Original: {metrics['orig_bits'] / 8000:.1f} KB  ->  "
        f"Compressed: {metrics['compressed_bits'] / 8000:.1f} KB  "
        f"({metrics['space_savings']:.1f}% space saved)",
        color="#bac2de", fontsize=9, ha="center", style="italic"
    )

    # Table of top 10 frequent symbols
    sorted_syms = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)[:10]
    table_data = [
        [
            f"{sym}",
            f"{freq_dict[sym]:,}",
            f"{prob:.4f}",
            codebook[sym],
            f"{len(codebook[sym])} bits",
        ]
        for sym, prob in sorted_syms
    ]

    col_labels = ["Symbol (Intensity)", "Pixel Count", "Probability (p)", "Shannon-Fano Code", "Code Length"]
    ax_table = fig.add_axes([0.08, 0.06, 0.84, 0.58])
    ax_table.axis("off")

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    # Style table
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#45475a")
        if r == 0:
            cell.set_facecolor("#181825")
            cell.set_text_props(color="#89b4fa", fontweight="bold")
        else:
            cell.set_facecolor("#313244" if r % 2 == 0 else "#24273a")
            cell.set_text_props(color="#cdd6f4")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return img


def generate_length_distribution_chart(codebook: dict[int, str], prob_dict: dict[int, float]) -> Image.Image:
    """Bar chart showing codeword length distribution weighted by symbol probabilities."""
    lengths = [len(codebook[s]) for s in prob_dict]
    max_l = max(lengths)
    min_l = min(lengths)

    bins = np.arange(min_l, max_l + 2)
    # Probability per length
    prob_per_len = np.zeros(max_l + 1, dtype=np.float32)
    symbol_count_per_len = np.zeros(max_l + 1, dtype=np.int32)
    for s, p in prob_dict.items():
        l = len(codebook[s])
        prob_per_len[l] += p
        symbol_count_per_len[l] += 1

    x = np.arange(min_l, max_l + 1)
    y_prob = prob_per_len[min_l:max_l + 1] * 100.0

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=100, facecolor="#1e1e2e")
    ax1.set_facecolor("#181825")

    bars = ax1.bar(x, y_prob, color="#89b4fa", edgecolor="#b4befe", alpha=0.85, width=0.6)
    ax1.set_xlabel("Codeword Length (Bits)", color="#cdd6f4", fontsize=10)
    ax1.set_ylabel("Pixel Frequency (%)", color="#89b4fa", fontsize=10)
    ax1.set_title("Codeword Length Distribution", color="#cdd6f4", fontsize=12, fontweight="bold", pad=8)
    ax1.tick_params(colors="#a6adc8", labelsize=9)
    ax1.set_xticks(x)
    for spine in ax1.spines.values():
        spine.set_color("#45475a")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#585b70")

    for bar in bars:
        h = bar.get_height()
        if h > 1.0:
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{h:.1f}%",
                     ha="center", va="bottom", color="#cdd6f4", fontsize=8)

    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.fromarray(np.asarray(buf)[:, :, :3])
    plt.close(fig)
    return img


# =====================================================================
# 3. Main Standalone Script Entrypoint
# =====================================================================

def main():
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / "Output"
    output_dir.mkdir(exist_ok=True)

    possible_dirs = [root_dir / "Input_images", root_dir / "Input"]
    input_dir = next((d for d in possible_dirs if d.is_dir() and any(d.iterdir())), None)
    if input_dir is None:
        raise FileNotFoundError("Could not find an Input_images or Input directory with files.")

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts]
    if not images:
        raise FileNotFoundError(f"No valid image files found in {input_dir}.")

    target_img_path = images[0]
    for candidate in images:
        if "scenar" in candidate.name.lower() or "tree" in candidate.name.lower():
            target_img_path = candidate
            break

    print(f"Loading input image: {target_img_path.name}")
    pil_raw = Image.open(target_img_path).convert("RGB")
    rgb_arr = np.array(pil_raw)

    # 1. Greyscale Conversion
    grey = rgb_to_grey(rgb_arr)
    h, w = grey.shape
    total_pixels = grey.size
    print(f"Image converted to greyscale ({w}x{h}, {total_pixels:,} pixels).")

    # 2. Probability Distribution & Shannon-Fano Coding
    prob_dict, freq_dict = compute_symbol_probabilities(grey)
    codebook = build_shannon_fano_codebook(prob_dict)
    metrics = compute_compression_metrics(prob_dict, freq_dict, codebook, total_pixels)

    print("\n--- Shannon-Fano Coding Performance ---")
    print(f"Unique Symbols          : {len(prob_dict)}")
    print(f"Shannon Entropy (H)     : {metrics['entropy']:.4f} bits/pixel")
    print(f"Average Codeword Length : {metrics['avg_length']:.4f} bits/pixel")
    print(f"Coding Efficiency (eta) : {metrics['efficiency']:.2f}%")
    print(f"Redundancy (R)          : {metrics['redundancy']:.4f} bits/pixel")
    print(f"Compression Ratio (CR)  : {metrics['compression_ratio']:.2f} : 1")
    print(f"Theoretical Space Saved : {metrics['space_savings']:.2f}%\n")

    # 3. Encoding & Decoding Verification
    print("Encoding image bitstream...")
    bitstream = encode_image(grey, codebook)
    print(f"Bitstream length: {len(bitstream):,} bits ({len(bitstream) / 8000:.1f} KB)")

    print("Decoding image bitstream with Prefix Trie...")
    decoded = decode_image(bitstream, codebook, grey.shape)

    mse = float(np.mean((grey.astype(int) - decoded.astype(int)) ** 2))
    diff_count = int(np.sum(grey != decoded))
    print("\n--- Decoding Verification ---")
    print(f"Reconstruction MSE      : {mse:.4f}")
    print(f"Pixel Mismatches        : {diff_count} / {total_pixels} ({(diff_count/total_pixels)*100:.2f}%)")
    assert mse == 0.0, "Decoded image does not match original image! Lossless verification failed."
    print("Verification PASSED: 100.0% Exact Lossless Reconstruction Confirmed!\n")

    # 4. Generate Visual Artifacts
    heatmap_img = generate_bit_length_heatmap(grey, codebook)
    summary_card = generate_metrics_summary_card(metrics, prob_dict, freq_dict, codebook, total_pixels)
    dist_chart = generate_length_distribution_chart(codebook, prob_dict)

    outputs = {
        "shannon_fano_original_grey.png": Image.fromarray(grey),
        "shannon_fano_decoded.png": Image.fromarray(decoded),
        "shannon_fano_bit_length_map.png": heatmap_img,
        "shannon_fano_metrics_summary.png": summary_card,
        "shannon_fano_length_distribution.png": dist_chart,
    }

    for fname, img_obj in outputs.items():
        save_path = output_dir / fname
        img_obj.save(save_path)
        print(f"Saved: {save_path.relative_to(root_dir)}")

    print("\nShannon-Fano Coding completed successfully!")


if __name__ == "__main__":
    main()
