# Shannon-Fano Coding Operation for DIP Lab GUI
# Implemented from first principles using pure NumPy (no external compression libraries)

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

NAME = "Shannon-Fano Coding"


# =====================================================================
# 1. First-Principles Shannon-Fano Coding & Decoding
# =====================================================================

def _rgb_to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA to uint8 greyscale."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def _compute_symbol_probabilities(grey: np.ndarray) -> tuple[dict[int, float], dict[int, int]]:
    """Compute normalized probabilities and frequency counts for present symbols."""
    counts = np.bincount(grey.ravel(), minlength=256)
    total = grey.size
    freq_dict = {i: int(counts[i]) for i in range(256) if counts[i] > 0}
    prob_dict = {i: counts[i] / total for i in freq_dict}
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


def _build_shannon_fano_codebook(prob_dict: dict[int, float]) -> dict[int, str]:
    """Generate complete Shannon-Fano codebook."""
    sorted_symbols = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_symbols) == 1:
        return {sorted_symbols[0][0]: "0"}
    return _shannon_fano_recursive(sorted_symbols)


def _compute_metrics(
    prob_dict: dict[int, float],
    freq_dict: dict[int, int],
    codebook: dict[int, str],
    total_pixels: int,
) -> dict[str, float]:
    """Compute Shannon Entropy, Average Length, Efficiency, Redundancy, and Compression Ratio."""
    entropy = -sum(p * np.log2(p) for p in prob_dict.values() if p > 0)
    l_avg = sum(prob_dict[s] * len(codebook[s]) for s in prob_dict)
    efficiency = (entropy / l_avg * 100.0) if l_avg > 0 else 100.0
    redundancy = max(0.0, l_avg - entropy)
    compression_ratio = 8.0 / l_avg if l_avg > 0 else 1.0
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


class _TrieNode:
    __slots__ = ("children", "symbol")

    def __init__(self):
        self.children = [None, None]
        self.symbol = None


def _decode_image(bitstream: str, codebook: dict[int, str], shape: tuple[int, int]) -> np.ndarray:
    """Decode bitstream back into original spatial image."""
    root = _TrieNode()
    for symbol, code in codebook.items():
        curr = root
        for bit in code:
            idx = 0 if bit == "0" else 1
            if curr.children[idx] is None:
                curr.children[idx] = _TrieNode()
            curr = curr.children[idx]
        curr.symbol = symbol

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
# 2. Visualizations
# =====================================================================

def _plot_heatmap(grey: np.ndarray, codebook: dict[int, str]) -> Image.Image:
    """Render spatial heatmap of codeword bit lengths."""
    len_lut = np.zeros(256, dtype=np.float32)
    for sym, code in codebook.items():
        len_lut[sym] = len(code)

    bit_map = len_lut[grey]
    min_len, max_len = np.min(bit_map), np.max(bit_map)

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=100, facecolor="#181825")
    ax.set_facecolor("#181825")
    im = ax.imshow(bit_map, cmap="plasma", vmin=min_len, vmax=max_len)
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


def _plot_metrics_card(
    metrics: dict[str, float],
    prob_dict: dict[int, float],
    freq_dict: dict[int, int],
    codebook: dict[int, str],
    total_pixels: int,
) -> Image.Image:
    """Render metrics summary infographic card."""
    fig = plt.figure(figsize=(9, 7), dpi=110, facecolor="#1e1e2e")

    # Header
    fig.text(0.5, 0.94, "Shannon-Fano Coding: Performance & Metrics", color="#cdd6f4",
             fontsize=14, fontweight="bold", ha="center")
    fig.text(0.5, 0.90, f"Total Pixels: {total_pixels:,} | Unique Symbols: {len(prob_dict)}",
             color="#a6adc8", fontsize=10, ha="center")

    # Metric boxes
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


def _plot_length_distribution(codebook: dict[int, str], prob_dict: dict[int, float]) -> Image.Image:
    """Bar chart showing codeword length distribution."""
    lengths = [len(codebook[s]) for s in prob_dict]
    max_l = max(lengths)
    min_l = min(lengths)

    prob_per_len = np.zeros(max_l + 1, dtype=np.float32)
    for s, p in prob_dict.items():
        prob_per_len[len(codebook[s])] += p

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
# 3. Main GUI Operation Interface
# =====================================================================

def process(img: np.ndarray) -> dict[str, Image.Image]:
    """Execute Shannon-Fano Coding and return visual cards for GUI."""
    grey = _rgb_to_grey(img)
    total_pixels = grey.size

    # Codebook & metrics
    prob_dict, freq_dict = _compute_symbol_probabilities(grey)
    codebook = _build_shannon_fano_codebook(prob_dict)
    metrics = _compute_metrics(prob_dict, freq_dict, codebook, total_pixels)

    # Encode & Decode
    bitstream = "".join(codebook[s] for s in grey.ravel())
    decoded = _decode_image(bitstream, codebook, grey.shape)

    # Visual cards
    heatmap_img = _plot_heatmap(grey, codebook)
    summary_card = _plot_metrics_card(metrics, prob_dict, freq_dict, codebook, total_pixels)
    dist_chart = _plot_length_distribution(codebook, prob_dict)

    decoded_3ch = np.stack([decoded, decoded, decoded], axis=-1)

    return {
        "Decoded Reconstruction": Image.fromarray(decoded_3ch),
        "Bit Length Heatmap": heatmap_img,
        "Compression Metrics Card": summary_card,
        "Codeword Length Distribution": dist_chart,
    }
