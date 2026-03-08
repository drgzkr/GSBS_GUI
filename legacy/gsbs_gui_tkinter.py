#!/usr/bin/env python3
"""
GSBS GUI - Greedy State Boundary Search visualization tool
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from statesegmentation import gsbs


class GSBSApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("GSBS GUI")
        self.root.minsize(1100, 820)

        self.roi_data: np.ndarray | None = None
        self.gsbs_object = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self._build_controls()
        self._build_plots()
        self._build_statusbar()

    def _build_controls(self) -> None:
        ctrl = ttk.LabelFrame(self.root, text="Controls", padding=6)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        ctrl.columnconfigure(1, weight=1)

        # Row 0 — file path + load buttons
        ttk.Label(ctrl, text="File (.npy):").grid(row=0, column=0, sticky="w")
        self.data_path_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.data_path_var).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(ctrl, text="Browse", command=self._browse_data).grid(
            row=0, column=2, padx=2)
        ttk.Button(ctrl, text="Load ROI Data", command=self.load_roi_data).grid(
            row=0, column=3, padx=2)
        ttk.Button(ctrl, text="Load GSBS Object", command=self.load_gsbs_object).grid(
            row=0, column=4, padx=2)

        # Row 1 — parameters + run/save
        pf = ttk.Frame(ctrl)
        pf.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(6, 0))

        ttk.Label(pf, text="kmax:").pack(side="left")
        self.kmax_var = tk.IntVar(value=10)
        ttk.Spinbox(pf, from_=1, to=9999, textvariable=self.kmax_var,
                    width=6).pack(side="left", padx=(2, 12))

        ttk.Label(pf, text="finetune:").pack(side="left")
        self.finetune_var = tk.IntVar(value=1)
        ttk.Spinbox(pf, from_=0, to=100, textvariable=self.finetune_var,
                    width=6).pack(side="left", padx=(2, 12))

        self.statewise_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(pf, text="Statewise detection",
                        variable=self.statewise_var).pack(side="left", padx=(0, 12))

        self.run_btn = ttk.Button(pf, text="Run GSBS", command=self.run_gsbs)
        self.run_btn.pack(side="left", padx=(0, 12))

        ttk.Separator(pf, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(pf, text="Save as:").pack(side="left")
        self.save_path_var = tk.StringVar(value="gsbs_result.npy")
        ttk.Entry(pf, textvariable=self.save_path_var, width=22).pack(
            side="left", padx=(2, 4))
        ttk.Button(pf, text="Browse", command=self._browse_save).pack(
            side="left", padx=2)
        ttk.Button(pf, text="Save GSBS Object",
                   command=self.save_gsbs_object).pack(side="left", padx=2)

        # Row 2 — progress bar (hidden until running)
        self.progress = ttk.Progressbar(ctrl, mode="indeterminate")
        self.progress.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(4, 0))
        self.progress.grid_remove()

    def _build_plots(self) -> None:
        pa = ttk.Frame(self.root)
        pa.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        pa.columnconfigure((0, 1), weight=1)
        pa.rowconfigure((0, 1), weight=1)

        self.fig_corrmat, self.ax_corrmat, self.canvas_corrmat = \
            self._make_plot_cell(pa, 0, 0, "Correlation Matrix")
        self.fig_tdist, self.ax_tdist, self.canvas_tdist = \
            self._make_plot_cell(pa, 0, 1, "T-dist Curve")
        self.fig_raw, self.ax_raw, self.canvas_raw = \
            self._make_plot_cell(pa, 1, 0, "Voxel Timeseries")
        self.fig_state, self.ax_state, self.canvas_state = \
            self._make_plot_cell(pa, 1, 1, "State Activity Timeseries")

        # Solution explorer slider
        sf = ttk.LabelFrame(self.root, text="Solution Explorer  (drag to change k)",
                            padding=4)
        sf.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        sf.columnconfigure(0, weight=1)

        self.k_var = tk.IntVar(value=0)
        self.k_slider = ttk.Scale(sf, from_=0, to=1, orient="horizontal",
                                  variable=self.k_var, command=self._on_slider)
        self.k_slider.grid(row=0, column=0, sticky="ew", padx=4)
        self.k_label = ttk.Label(sf, text="k = —", width=8)
        self.k_label.grid(row=0, column=1, padx=8)

    def _make_plot_cell(self, parent, row, col, title):
        frame = ttk.Frame(parent, relief="sunken", borderwidth=1)
        frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        fig, ax = plt.subplots(tight_layout=True)
        fig.suptitle(title, fontsize=9)

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(canvas, toolbar_frame)

        return fig, ax, canvas

    def _build_statusbar(self) -> None:
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief="sunken", anchor="w", padding=(4, 2)
                  ).grid(row=3, column=0, sticky="ew")

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _browse_data(self) -> None:
        path = filedialog.askopenfilename(
            title="Select data or GSBS object",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
        if path:
            self.data_path_var.set(path)

    def _browse_save(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save GSBS object",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
        if path:
            self.save_path_var.set(path)

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_roi_data(self) -> None:
        path = self.data_path_var.get().strip()
        if not path:
            messagebox.showwarning("No path", "Enter or browse to a .npy file first.")
            return
        try:
            data = np.load(path, allow_pickle=True)
            if data.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {data.shape}")
            self.roi_data = data
            self._status(f"Loaded ROI data  shape={data.shape}")
            self._plot_roi_data()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def load_gsbs_object(self) -> None:
        path = self.data_path_var.get().strip()
        if not path:
            messagebox.showwarning("No path", "Enter or browse to a .npy file first.")
            return
        try:
            obj = np.load(path, allow_pickle=True).item()
            if not hasattr(obj, "tdists") or not hasattr(obj, "all_bounds"):
                raise ValueError("File does not look like a GSBS object.")
            self.gsbs_object = obj
            kmax = len(obj.tdists) - 1
            self.kmax_var.set(kmax)
            best_k = int(np.argmax(obj.tdists))
            self._configure_slider(kmax, best_k)
            self._status(f"Loaded GSBS object  kmax={kmax}  best k={best_k}")
            self._refresh_all_plots(best_k)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def save_gsbs_object(self) -> None:
        if self.gsbs_object is None:
            messagebox.showwarning("Nothing to save", "No GSBS object in memory.")
            return
        path = self.save_path_var.get().strip()
        if not path:
            messagebox.showwarning("No path", "Enter a save path first.")
            return
        try:
            np.save(path, self.gsbs_object, allow_pickle=True)
            self._status(f"Saved GSBS object → {path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    # ------------------------------------------------------------------
    # GSBS computation (runs in background thread)
    # ------------------------------------------------------------------

    def run_gsbs(self) -> None:
        if self.roi_data is None:
            messagebox.showwarning("No data", "Load ROI data first.")
            return

        kmax = self.kmax_var.get()
        finetune = self.finetune_var.get()
        statewise = self.statewise_var.get()
        data = self.roi_data

        self.run_btn.config(state="disabled")
        self.progress.grid()
        self.progress.start(10)
        self._status("Running GSBS…")

        def _worker() -> None:
            try:
                obj = gsbs.GSBS(kmax=kmax, x=data,
                                statewise_detection=statewise,
                                finetune=finetune)
                obj.fit(showProgressBar=False)
                self.root.after(0, self._gsbs_done, obj)
            except Exception as exc:
                self.root.after(0, self._gsbs_error, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _gsbs_done(self, obj) -> None:
        self.gsbs_object = obj
        self.progress.stop()
        self.progress.grid_remove()
        self.run_btn.config(state="normal")

        kmax = len(obj.tdists) - 1
        best_k = int(np.argmax(obj.tdists))
        self._configure_slider(kmax, best_k)
        self._status(f"GSBS done  kmax={kmax}  best k={best_k}")
        self._refresh_all_plots(best_k)

    def _gsbs_error(self, msg: str) -> None:
        self.progress.stop()
        self.progress.grid_remove()
        self.run_btn.config(state="normal")
        messagebox.showerror("GSBS error", msg)
        self._status("GSBS failed.")

    # ------------------------------------------------------------------
    # Slider
    # ------------------------------------------------------------------

    def _configure_slider(self, kmax: int, value: int) -> None:
        self.k_slider.config(from_=1, to=kmax)
        self.k_var.set(value)
        self.k_label.config(text=f"k = {value}")

    def _on_slider(self, _=None) -> None:
        if self.gsbs_object is None:
            return
        k = int(round(self.k_var.get()))
        self.k_label.config(text=f"k = {k}")
        self._refresh_all_plots(k)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_roi_data(self) -> None:
        data = self.roi_data

        self.ax_raw.clear()
        self.ax_raw.imshow(data.T, aspect="auto", origin="lower")
        self.ax_raw.set_ylabel("Channels")
        self.ax_raw.set_xlabel("Timepoints")
        self.canvas_raw.draw()

        self.ax_corrmat.clear()
        self.ax_corrmat.imshow(np.corrcoef(data), cmap="viridis",
                               vmin=-1, vmax=1, aspect="equal")
        self.ax_corrmat.set_ylabel("Timepoints")
        self.canvas_corrmat.draw()

    def _refresh_all_plots(self, k: int) -> None:
        obj = self.gsbs_object
        bounds = np.where(obj.all_bounds[k] > 0)[0]
        n_time = obj.x.shape[0]

        # T-dist curve
        self.ax_tdist.clear()
        self.ax_tdist.plot(obj.tdists, color="steelblue")
        self.ax_tdist.axvline(x=k, color="red", linestyle="--", linewidth=1.2,
                              label=f"k = {k}")
        self.ax_tdist.set_xlabel("k (boundaries)")
        self.ax_tdist.set_ylabel("T-dist")
        self.ax_tdist.legend(fontsize=8)

        # Correlation matrix with state rectangles
        self.ax_corrmat.clear()
        corr = np.corrcoef(obj.x)
        self.ax_corrmat.imshow(corr, cmap="viridis", vmin=-1, vmax=1, aspect="equal")
        self.ax_corrmat.set_ylabel("Timepoints")
        edges = np.concatenate(([0], bounds, [n_time]))
        for i in range(len(edges) - 1):
            x0, x1 = edges[i], edges[i + 1]
            rect = patches.Rectangle(
                (x0, x0), x1 - x0, x1 - x0,
                linewidth=1.5, edgecolor="red", facecolor="none")
            self.ax_corrmat.add_patch(rect)

        # Raw timeseries
        self.ax_raw.clear()
        self.ax_raw.imshow(obj.x.T, aspect="auto", origin="lower")
        self.ax_raw.set_ylabel("Channels")
        for b in bounds:
            self.ax_raw.axvline(x=b, color="red", linewidth=0.8)

        # State-averaged timeseries
        self.ax_state.clear()
        self.ax_state.imshow(self._state_timeseries(k).T, aspect="auto", origin="lower")
        self.ax_state.set_ylabel("Channels")
        self.ax_state.set_xlabel("Timepoints")
        for b in bounds:
            self.ax_state.axvline(x=b, color="red", linewidth=0.8)

        for canvas in (self.canvas_corrmat, self.canvas_tdist,
                       self.canvas_raw, self.canvas_state):
            canvas.draw()

    def _state_timeseries(self, k: int) -> np.ndarray:
        obj = self.gsbs_object
        patterns = obj.get_state_patterns(k=k)        # shape: (k, n_channels)
        bounds = np.where(obj.all_bounds[k] > 0)[0]
        n_time = obj.x.shape[0]

        state_ts = np.empty_like(obj.x)
        edges = np.concatenate(([0], bounds, [n_time]))
        for i, pattern in enumerate(patterns):
            state_ts[edges[i]:edges[i + 1]] = pattern
        return state_ts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()


def main() -> None:
    root = tk.Tk()
    GSBSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
