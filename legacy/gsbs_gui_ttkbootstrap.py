#!/usr/bin/env python3
"""
GSBS GUI - Greedy State Boundary Search visualization tool (ttkbootstrap)
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from statesegmentation import gsbs

plt.style.use("dark_background")
_FIG_COLOR = "#222222"   # matches ttkbootstrap darkly window bg


class GSBSApp:
    def __init__(self, root: ttk.Window) -> None:
        self.root = root
        self.root.title("GSBS GUI")
        self.root.minsize(1100, 820)

        self.roi_data: np.ndarray | None = None
        self.gsbs_object = None
        self._corr_cache: np.ndarray | None = None   # cached to avoid recompute on slider
        self._gsbs_running: bool = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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
        ctrl = ttk.LabelFrame(self.root, text="Controls")
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4), ipadx=6, ipady=4)
        ctrl.columnconfigure(1, weight=1)

        # Row 0 — file path + load buttons
        ttk.Label(ctrl, text="File (.npy):").grid(row=0, column=0, sticky="w")
        self.data_path_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.data_path_var).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(ctrl, text="Browse", command=self._browse_data,
                   bootstyle="secondary").grid(row=0, column=2, padx=2)
        ttk.Button(ctrl, text="Load ROI Data", command=self.load_roi_data,
                   bootstyle="primary").grid(row=0, column=3, padx=2)
        ttk.Button(ctrl, text="Load GSBS Object", command=self.load_gsbs_object,
                   bootstyle="info").grid(row=0, column=4, padx=2)

        # Row 1 — parameters + run/save
        pf = ttk.Frame(ctrl)
        pf.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(8, 0))

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
                        variable=self.statewise_var,
                        bootstyle="round-toggle").pack(side="left", padx=(0, 12))

        self.run_btn = ttk.Button(pf, text="Run GSBS", command=self.run_gsbs,
                                  bootstyle="success")
        self.run_btn.pack(side="left", padx=(0, 12))

        ttk.Separator(pf, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(pf, text="Save as:").pack(side="left")
        self.save_path_var = tk.StringVar(value="gsbs_result.npy")
        ttk.Entry(pf, textvariable=self.save_path_var, width=22).pack(
            side="left", padx=(2, 4))
        ttk.Button(pf, text="Browse", command=self._browse_save,
                   bootstyle="secondary").pack(side="left", padx=2)
        ttk.Button(pf, text="Save GSBS Object", command=self.save_gsbs_object,
                   bootstyle="secondary").pack(side="left", padx=2)

        # Row 2 — progress bar + patience label (hidden until running)
        self.progress = ttk.Progressbar(ctrl, mode="indeterminate",
                                        bootstyle="success-striped")
        self.progress.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.progress.grid_remove()

        self.patience_label = ttk.Label(
            ctrl,
            text="⏳ This can take tens of minutes with large kmax or statewise detection, please be patient.",
            bootstyle="warning", justify="left")
        self.patience_label.bind(
            "<Configure>",
            lambda e: self.patience_label.config(wraplength=e.width))
        self.patience_label.grid(row=3, column=0, columnspan=5, sticky="w", pady=(2, 0))
        self.patience_label.grid_remove()

    def _build_plots(self) -> None:
        pa = ttk.Frame(self.root)
        pa.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        pa.columnconfigure(0, weight=1)
        pa.columnconfigure(1, weight=1)
        pa.rowconfigure(0, weight=1)

        # Left column: Correlation Matrix — fills full height
        self.fig_corrmat, self.ax_corrmat, self.canvas_corrmat = \
            self._make_plot_cell(pa, 0, 0, "Correlation Matrix")

        # Right column: T-dist → Solution Explorer → stacked Timeseries
        right = ttk.Frame(pa)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3)   # T-dist: larger share
        right.rowconfigure(1, weight=0)   # Solution Explorer: natural height
        right.rowconfigure(2, weight=2)   # Timeseries: slightly smaller share

        self.fig_tdist, self.ax_tdist, self.canvas_tdist = \
            self._make_plot_cell(right, 0, 0, "T-dist Curve")

        self._build_solution_explorer(right, row=1)

        # Timeseries: single figure, two subplots sharing x-axis
        self.fig_ts, self.ax_raw, self.ax_state, self.canvas_ts = \
            self._make_timeseries_cell(right, 2, 0)

    def _build_solution_explorer(self, parent, row: int) -> None:
        sf = ttk.LabelFrame(parent, text="Solution Explorer")
        sf.grid(row=row, column=0, sticky="ew", padx=4, pady=4, ipadx=4, ipady=6)
        sf.columnconfigure(0, weight=1)

        self.k_var = tk.IntVar(value=0)

        self.k_slider = ttk.Scale(sf, from_=0, to=1, orient="horizontal",
                                  variable=self.k_var, command=self._on_slider,
                                  bootstyle="info")
        self.k_slider.grid(row=0, column=0, sticky="ew", padx=(4, 4))

        self.k_spinbox = ttk.Spinbox(sf, from_=0, to=1, textvariable=self.k_var,
                                     width=5, command=self._on_k_spinbox,
                                     bootstyle="info")
        self.k_spinbox.bind("<Return>", self._on_k_spinbox)
        self.k_spinbox.bind("<FocusOut>", self._on_k_spinbox)
        self.k_spinbox.grid(row=0, column=1, padx=(0, 8))

        self.best_k_label = ttk.Label(sf, text="Best k: —", bootstyle="info")
        self.best_k_label.grid(row=0, column=2, padx=(0, 4))

    def _make_plot_cell(self, parent, row, col, title, figsize=None):
        frame = ttk.Frame(parent, padding=2)
        frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        fig, ax = plt.subplots(tight_layout=True, figsize=figsize)
        fig.patch.set_facecolor(_FIG_COLOR)
        ax.set_facecolor(_FIG_COLOR)
        fig.suptitle(title, fontsize=9)

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(canvas, toolbar_frame)

        return fig, ax, canvas

    def _make_timeseries_cell(self, parent, row, col):
        frame = ttk.Frame(parent, padding=2)
        frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        fig, (ax_raw, ax_state) = plt.subplots(
            2, 1, sharex=True, tight_layout=True, figsize=(6, 4))
        fig.patch.set_facecolor(_FIG_COLOR)
        for ax in (ax_raw, ax_state):
            ax.set_facecolor(_FIG_COLOR)

        ax_raw.set_ylabel("Channels")
        ax_raw.set_title("Voxel Timeseries", fontsize=9)
        ax_state.set_ylabel("Channels")
        ax_state.set_xlabel("Timepoints")
        ax_state.set_title("State Activity Timeseries", fontsize=9)

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(canvas, toolbar_frame)

        return fig, ax_raw, ax_state, canvas

    def _build_statusbar(self) -> None:
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var,
                  anchor="w", bootstyle="inverse-secondary"
                  ).grid(row=3, column=0, sticky="ew", ipady=3, ipadx=6)

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
            self._corr_cache = np.corrcoef(obj.x)
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

        self._gsbs_running = True
        self.run_btn.config(state="disabled")
        self.progress.grid()
        self.progress.start(10)
        self.patience_label.grid()
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
        self._gsbs_running = False
        self._corr_cache = np.corrcoef(obj.x)
        self.progress.stop()
        self.progress.grid_remove()
        self.patience_label.grid_remove()
        self.run_btn.config(state="normal")

        kmax = len(obj.tdists) - 1
        best_k = int(np.argmax(obj.tdists))
        self._configure_slider(kmax, best_k)
        self._status(f"GSBS done  kmax={kmax}  best k={best_k}")
        self._refresh_all_plots(best_k)

    def _gsbs_error(self, msg: str) -> None:
        self._gsbs_running = False
        self.progress.stop()
        self.progress.grid_remove()
        self.patience_label.grid_remove()
        self.run_btn.config(state="normal")
        messagebox.showerror("GSBS error", msg)
        self._status("GSBS failed.")

    # ------------------------------------------------------------------
    # Slider / k control
    # ------------------------------------------------------------------

    def _configure_slider(self, kmax: int, value: int) -> None:
        self.k_slider.config(from_=1, to=kmax)
        self.k_spinbox.config(from_=1, to=kmax)
        self.k_var.set(value)
        self.best_k_label.config(text=f"Best k: {value}")

    def _on_slider(self, _=None) -> None:
        if self.gsbs_object is None:
            return
        self._apply_k(int(round(self.k_var.get())))

    def _on_k_spinbox(self, _=None) -> None:
        if self.gsbs_object is None:
            return
        try:
            self._apply_k(int(self.k_var.get()))
        except (ValueError, tk.TclError):
            pass

    def _apply_k(self, k: int) -> None:
        kmax = len(self.gsbs_object.tdists) - 1
        k = max(1, min(k, kmax))
        self.k_var.set(k)
        self._refresh_all_plots(k)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_roi_data(self) -> None:
        data = self.roi_data

        self.ax_raw.clear()
        self.ax_raw.imshow(data.T, aspect="auto", origin="lower")
        self.ax_raw.set_ylabel("Channels")
        self.ax_raw.set_title("Voxel Timeseries", fontsize=9)
        self.canvas_ts.draw()

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
        self.ax_tdist.plot(obj.tdists, color="#4fc3f7")
        self.ax_tdist.axvline(x=k, color="#ef5350", linestyle="--", linewidth=1.2,
                              label=f"k = {k}")
        self.ax_tdist.set_xlabel("k (boundaries)")
        self.ax_tdist.set_ylabel("T-dist")
        self.ax_tdist.legend(fontsize=8)

        # Correlation matrix — use cached corrcoef, only redraws boundaries
        self.ax_corrmat.clear()
        self.ax_corrmat.imshow(self._corr_cache, cmap="viridis",
                               vmin=-1, vmax=1, aspect="equal")
        self.ax_corrmat.set_ylabel("Timepoints")
        edges = np.concatenate(([0], bounds, [n_time]))
        for i in range(len(edges) - 1):
            x0, x1 = edges[i], edges[i + 1]
            rect = patches.Rectangle(
                (x0, x0), x1 - x0, x1 - x0,
                linewidth=1.5, edgecolor="#ef5350", facecolor="none")
            self.ax_corrmat.add_patch(rect)

        # Raw timeseries
        self.ax_raw.clear()
        self.ax_raw.imshow(obj.x.T, aspect="auto", origin="lower")
        self.ax_raw.set_ylabel("Channels")
        self.ax_raw.set_title("Voxel Timeseries", fontsize=9)
        for b in bounds:
            self.ax_raw.axvline(x=b, color="#ef5350", linewidth=0.8)

        # State-averaged timeseries
        self.ax_state.clear()
        self.ax_state.imshow(self._state_timeseries(k).T, aspect="auto", origin="lower")
        self.ax_state.set_ylabel("Channels")
        self.ax_state.set_xlabel("Timepoints")
        self.ax_state.set_title("State Activity Timeseries", fontsize=9)
        for b in bounds:
            self.ax_state.axvline(x=b, color="#ef5350", linewidth=0.8)

        for canvas in (self.canvas_corrmat, self.canvas_tdist, self.canvas_ts):
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

    def _on_close(self) -> None:
        if self._gsbs_running:
            if not messagebox.askyesno(
                "GSBS running",
                "GSBS is still running. Closing will cancel the computation.\n\nClose anyway?",
                icon="warning",
            ):
                return
        self.root.destroy()

    def _status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()


def main() -> None:
    root = ttk.Window(themename="darkly")
    GSBSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
