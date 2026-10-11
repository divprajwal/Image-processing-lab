# DIP Lab - Image Processing Toolkit GUI

import os
import sys
import tkinter as _tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ttkbootstrap for modern theme; fallback to plain ttk
try:
    import ttkbootstrap as ttkb
    from ttkbootstrap.constants import *
    from ttkbootstrap.scrolled import ScrolledFrame
    _USE_BOOTSTRAP = True
except ImportError:
    import tkinter.ttk as ttk
    _USE_BOOTSTRAP = False

from operations import get_operations

# Constants
OUTPUT_DIR = _PROJECT_ROOT / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)
THUMB_MAX = 280
INPUT_THUMB_MAX = 320
WINDOW_MIN_W = 1100
WINDOW_MIN_H = 700


def _make_thumb(pil_img: Image.Image, max_size: int) -> Image.Image:
    """Return a thumbnail copy fitting within max_size x max_size."""
    img = pil_img.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


class DIPLabApp:
    def __init__(self):
        # Window setup
        if _USE_BOOTSTRAP:
            self.root = ttkb.Window(
                title="DIP Lab - Image Processing Toolkit",
                themename="darkly",
                minsize=(WINDOW_MIN_W, WINDOW_MIN_H),
            )
            self._Frame = ttkb.Frame
            self._Label = ttkb.Label
            self._Button = ttkb.Button
            self._Combobox = ttkb.Combobox
            self._Separator = ttkb.Separator
            self._LabelFrame = ttkb.Labelframe
        else:
            self.root = _tk.Tk()
            self.root.title("DIP Lab - Image Processing Toolkit")
            self.root.minsize(WINDOW_MIN_W, WINDOW_MIN_H)
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass
            self._Frame = ttk.Frame
            self._Label = ttk.Label
            self._Button = ttk.Button
            self._Combobox = ttk.Combobox
            self._Separator = ttk.Separator
            self._LabelFrame = ttk.Labelframe

        self.root.configure(padx=16, pady=16)

        # State
        self._input_pil: Image.Image | None = None
        self._input_path: str = ""
        self._results: dict[str, Image.Image] = {}
        self._photo_refs: list[ImageTk.PhotoImage] = []

        # Load operations
        self._ops = get_operations()
        if not self._ops:
            messagebox.showerror("Error", "No operations found in GUI/operations/")
            sys.exit(1)

        self._build_ui()
        self.root.mainloop()

    def _build_ui(self):
        """Build the complete GUI layout."""
        # Top bar - operation selector and controls
        top = self._Frame(self.root)
        top.pack(fill="x", pady=(0, 12))

        self._Label(top, text="Operation:").pack(side="left", padx=(0, 6))
        self._op_var = _tk.StringVar(value=list(self._ops.keys())[0])
        op_combo = self._Combobox(
            top,
            textvariable=self._op_var,
            values=list(self._ops.keys()),
            state="readonly",
            width=35,
        )
        op_combo.pack(side="left", padx=(0, 16))

        self._btn_select = self._Button(
            top, text="Select Image", command=self._select_image
        )
        self._btn_select.pack(side="left", padx=(0, 8))

        self._btn_run = self._Button(
            top, text="Run", command=self._run, state="disabled"
        )
        self._btn_run.pack(side="left", padx=(0, 8))

        if _USE_BOOTSTRAP:
            self._btn_run.configure(bootstyle="success")

        # Main body - input preview and output results
        body = self._Frame(self.root)
        body.pack(fill="both", expand=True)

        # Input panel
        left = self._LabelFrame(body, text="  Input Image  ", padding=10)
        left.pack(side="left", fill="both", padx=(0, 10))

        self._input_label = self._Label(left, text="No image selected")
        self._input_label.pack(expand=True)

        self._filename_label = self._Label(left, text="", wraplength=250)
        self._filename_label.pack(pady=(6, 0))

        # Output panel
        right = self._LabelFrame(body, text="  Output Results  ", padding=10)
        right.pack(side="left", fill="both", expand=True)

        if _USE_BOOTSTRAP:
            self._output_scroll = ScrolledFrame(right, autohide=True)
            self._output_scroll.pack(fill="both", expand=True)
            self._output_inner = self._output_scroll
        else:
            # Fallback scrollable canvas
            canvas = _tk.Canvas(right, highlightthickness=0)
            vbar = ttk.Scrollbar(right, orient="vertical", command=canvas.yview)
            self._output_inner = self._Frame(canvas)
            self._output_inner.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
            )
            canvas.create_window((0, 0), window=self._output_inner, anchor="nw")
            canvas.configure(yscrollcommand=vbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            vbar.pack(side="right", fill="y")

            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._placeholder = self._Label(
            self._output_inner, text="Run an operation to see results here"
        )
        self._placeholder.pack(expand=True, pady=40)

        # Bottom bar - save and status
        bottom = self._Frame(self.root)
        bottom.pack(fill="x", pady=(12, 0))

        self._btn_save = self._Button(
            bottom, text="Save All to Output/", command=self._save_all, state="disabled"
        )
        self._btn_save.pack(side="left")

        self._btn_open_folder = self._Button(
            bottom, text="Open Output Folder", command=self._open_output_folder
        )
        self._btn_open_folder.pack(side="left", padx=(8, 0))

        self._status = self._Label(bottom, text="Ready")
        self._status.pack(side="right")

    def _select_image(self):
        """Open file dialog and load the selected image."""
        input_dir = _PROJECT_ROOT / "Input_images" if (_PROJECT_ROOT / "Input_images").exists() else _PROJECT_ROOT / "Input"
        path = filedialog.askopenfilename(
            title="Select an image",
            initialdir=str(input_dir),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            pil = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot open image:\n{exc}")
            return

        self._input_pil = pil
        self._input_path = path

        thumb = _make_thumb(pil, INPUT_THUMB_MAX)
        photo = ImageTk.PhotoImage(thumb)
        self._input_label.configure(image=photo, text="")
        self._input_label.image = photo

        self._filename_label.configure(text=os.path.basename(path))
        self._btn_run.configure(state="normal")
        self._status.configure(text=f"Loaded: {os.path.basename(path)}")

    def _run(self):
        """Run the selected operation on the loaded image."""
        if self._input_pil is None:
            return

        op_name = self._op_var.get()
        process_fn = self._ops.get(op_name)
        if process_fn is None:
            messagebox.showerror("Error", f"Operation '{op_name}' not found.")
            return

        self._status.configure(text=f"Running '{op_name}'...")
        self.root.update_idletasks()

        try:
            img_array = np.array(self._input_pil)
            results = process_fn(img_array)
        except Exception as exc:
            messagebox.showerror("Processing Error", str(exc))
            self._status.configure(text="Error")
            return

        self._results = results
        self._display_results(results)
        self._btn_save.configure(state="normal")
        self._status.configure(text=f"Done - {len(results)} output(s)")

    def _display_results(self, results: dict[str, Image.Image]):
        """Show result thumbnails in the output panel."""
        for w in self._output_inner.winfo_children():
            w.destroy()
        self._photo_refs.clear()

        cols = 2
        for idx, (name, pil_img) in enumerate(results.items()):
            row, col = divmod(idx, cols)

            card = self._Frame(self._output_inner, padding=6)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="n")

            thumb = _make_thumb(pil_img, THUMB_MAX)
            photo = ImageTk.PhotoImage(thumb)
            self._photo_refs.append(photo)

            img_label = self._Label(card, image=photo, cursor="hand2")
            img_label.pack()

            img_label.bind(
                "<Button-1>",
                lambda e, _img=pil_img, _name=name: self._preview_full(_img, _name),
            )

            title = self._Label(card, text=name, wraplength=THUMB_MAX)
            title.pack(pady=(4, 0))

    def _preview_full(self, pil_img: Image.Image, name: str):
        """Save temporarily and open with system viewer."""
        safe = name.replace(" ", "_").replace("/", "-")
        tmp_path = OUTPUT_DIR / f"_preview_{safe}.png"
        pil_img.save(str(tmp_path))
        os.startfile(str(tmp_path))

    def _save_all(self):
        """Save all results to the Output folder."""
        if not self._results:
            return

        base = Path(self._input_path).stem
        saved = []
        for name, pil_img in self._results.items():
            safe = name.replace(" ", "_").replace("/", "-")
            out_path = OUTPUT_DIR / f"{base}_{safe}.png"
            pil_img.save(str(out_path))
            saved.append(out_path.name)

        self._status.configure(text=f"Saved {len(saved)} file(s) to Output/")
        messagebox.showinfo(
            "Saved",
            f"Saved {len(saved)} file(s) to Output/:\n\n" + "\n".join(saved),
        )

    def _open_output_folder(self):
        """Open the Output directory in file explorer."""
        os.startfile(str(OUTPUT_DIR))


if __name__ == "__main__":
    DIPLabApp()
