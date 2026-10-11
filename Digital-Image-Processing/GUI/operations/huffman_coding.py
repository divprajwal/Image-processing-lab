# Huffman Coding Operation for DIP Lab GUI
# Implemented from first principles using pure NumPy & heapq (no external compression libraries)

import heapq
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

NAME = "Huffman Coding"


# =====================================================================
# 1. First-Principles Huffman Tree & Coding
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


class _HuffmanNode:
    __slots__ = ("symbol", "freq", "left", "right", "uid")

    def __init__(self, symbol: int | None, freq: int, left=None, right=None, uid: int = 0):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
        self.uid = uid

    def __lt__(self, other):
        if self.freq == other.freq:
            return self.uid < other.uid
        return self.freq < other.freq


def _build_huffman_tree(freq_dict: dict[int, int]) -> _HuffmanNode:
    """Construct optimal Huffman binary tree from symbol frequencies."""
    heap = []
    uid_counter = 0
    for sym, freq in freq_dict.items():
        heapq.heappush(heap, _HuffmanNode(symbol=sym, freq=freq, uid=uid_counter))
        uid_counter += 1

    if len(heap) == 1:
        only = heapq.heappop(heap)
        return _HuffmanNode(symbol=None, freq=only.freq, left=only, uid=uid_counter)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = _HuffmanNode(
            symbol=None,
            freq=left.freq + right.freq,
            left=left,
            right=right,
            uid=uid_counter,
        )
        uid_counter += 1
        heapq.heappush(heap, parent)

    return heap[0]


def _generate_huffman_codebook(root: _HuffmanNode) -> dict[int, str]:
    """Generate prefix codebook by traversing the Huffman tree."""
    codes = {}

    def _traverse(node: _HuffmanNode | None, prefix: str):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = prefix if prefix else "0"
            return
        _traverse(node.left, prefix + "0")
        _traverse(node.right, prefix + "1")

    _traverse(root, "")
    return codes


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


def _decode_image(bitstream: str, root: _HuffmanNode, shape: tuple[int, int]) -> np.ndarray:
    """Decode bitstream back into original spatial image by traversing the Huffman tree."""
    total_pixels = shape[0] * shape[1]
    decoded = np.empty(total_pixels, dtype=np.uint8)
    ptr = 0
    curr = root

    for bit in bitstream:
        curr = curr.left if bit == "0" else curr.right
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
        f"Huffman Codeword Length Map\n(Min: {int(min_len)} bits, Max: {int(max_len)} bits)",
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
    fig.text(0.5, 0.94, "Huffman Coding: Performance & Metrics", color="#cdd6f4",
             fontsize=14, fontweight="bold", ha="center")
    fig.text(0.5, 0.90, f"Total Pixels: {total_pixels:,} | Unique Symbols: {len(prob_dict)}",
             color="#a6adc8", fontsize=10, ha="center")

    # Metric boxes
    box_data = [
        ("Shannon Entropy (H)", f"{metrics['entropy']:.3f} bpp", "#89b4fa"),
        ("Avg Length (L_avg)", f"{metrics['avg_length']:.3f} bpp", "#a6e3a1"),
        ("Coding Efficiency (eta)", f"{metrics['efficiency']:.2f}%", "#fab387"),
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

    col_labels = ["Symbol (Intensity)", "Pixel Count", "Probability (p)", "Huffman Code", "Code Length"]
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

    bars = ax1.bar(x, y_prob, color="#a6e3a1", edgecolor="#94e2d5", alpha=0.85, width=0.6)
    ax1.set_xlabel("Codeword Length (Bits)", color="#cdd6f4", fontsize=10)
    ax1.set_ylabel("Pixel Frequency (%)", color="#a6e3a1", fontsize=10)
    ax1.set_title("Huffman Codeword Length Distribution", color="#cdd6f4", fontsize=12, fontweight="bold", pad=8)
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
    """Execute Huffman Coding and return visual cards for GUI."""
    grey = _rgb_to_grey(img)
    total_pixels = grey.size

    # Codebook & metrics
    prob_dict, freq_dict = _compute_symbol_probabilities(grey)
    tree_root = _build_huffman_tree(freq_dict)
    codebook = _generate_huffman_codebook(tree_root)
    metrics = _compute_metrics(prob_dict, freq_dict, codebook, total_pixels)

    # Encode & Decode
    bitstream = "".join(codebook[s] for s in grey.ravel())
    decoded = _decode_image(bitstream, tree_root, grey.shape)

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
