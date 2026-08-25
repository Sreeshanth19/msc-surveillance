"""Desktop application (macOS/Tkinter) for mask + social-distance monitoring.

Runs entirely locally on a recorded video. Open a file, Play/Pause/Stop,
watch the five-state risk assignment in real time, optionally calibrate for
metric distance, and save the processed output. Reuses the project's
MonitoringPipeline. Live-camera monitoring is a separate application:
scripts/live_camera.py.

Launch (from the project root, with the virtual environment active):
    python -m scripts.desktop_app
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import cv2
from PIL import Image, ImageTk

from src.config import Config
from src.pipeline import MonitoringPipeline
from src.distance import DistanceEstimator
from src.calibration import compute_homography

VIDEO_TYPES = [("Video files", "*.mp4 *.mov *.avi *.mkv *.MP4 *.MOV *.AVI *.MKV"),
               ("All files", "*.*")]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Face Mask & Social Distance Monitor")
        self.cfg = Config().resolve(ROOT)
        self.disp_w = int(self.cfg.process_width)
        self.cfg.process_width = self.disp_w

        self.pipeline = None
        self.cap = None
        self.source = None
        self.worker = None
        self.playing = threading.Event()
        self.stop_flag = threading.Event()
        self.frame_q: queue.Queue = queue.Queue(maxsize=2)
        self.writer = None
        self.save_path = None
        self.last_raw = None
        self.canvas_img = None
        self._imgref = None
        self.calib_mode = False
        self.calib_points = []
        self._poll_id = None

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()

    def _build_ui(self):
        bar = tk.Frame(self.root, pady=6)
        bar.pack(side=tk.TOP, fill=tk.X)
        self.b_open = tk.Button(bar, text="Open Video", width=10, command=self.open_video)
        self.b_play = tk.Button(bar, text="Play", width=8, command=self.toggle_play, state=tk.DISABLED)
        self.b_stop = tk.Button(bar, text="Stop", width=8, command=self.stop, state=tk.DISABLED)
        self.b_cal = tk.Button(bar, text="Calibrate", width=9, command=self.start_calibration, state=tk.DISABLED)
        self.b_save = tk.Button(bar, text="Save Output…", width=12, command=self.choose_save)
        for b in (self.b_open, self.b_play, self.b_stop, self.b_cal, self.b_save):
            b.pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(self.root, width=self.disp_w,
                                height=int(self.disp_w * 9 / 16), bg="black", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.status = tk.Label(self.root, text="Open a video to begin.",
                               anchor="w", padx=8, pady=4)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, msg):
        self.status.config(text=msg)

    def _resize(self, frame):
        h, w = frame.shape[:2]
        return cv2.resize(frame, (self.disp_w, int(h * self.disp_w / w)))

    def _draw(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._imgref = img
        self.canvas.config(height=bgr.shape[0])
        if self.canvas_img is None:
            self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        else:
            self.canvas.itemconfig(self.canvas_img, image=img)
            self.canvas.tag_lower(self.canvas_img)

    def ensure_pipeline(self):
        if self.pipeline is None:
            self.set_status("Loading detection models… (first time may take ~20s)")
            self.root.update()
            self.pipeline = MonitoringPipeline(self.cfg, enable_mask=True)

    def _enable_controls(self):
        for b in (self.b_play, self.b_stop, self.b_cal):
            b.config(state=tk.NORMAL)

    # ---------------- sources ----------------
    def open_video(self):
        path = filedialog.askopenfilename(filetypes=VIDEO_TYPES)
        if not path:
            return
        self.stop()
        self.source = path
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("Error", "Could not read this video file.")
            return
        self.last_raw = self._resize(frame)
        self.canvas.delete("calibdot")
        self._draw(self.last_raw)
        self._enable_controls()
        self.set_status(f"Loaded {Path(path).name}. Press Play. (Calibrate for metres.)")

    # ---------------- playback ----------------
    def toggle_play(self):
        if self.source is None:
            return
        if self.playing.is_set():
            self.playing.clear()
            self.b_play.config(text="Play")
            self.set_status("Paused.")
            return
        try:
            self.ensure_pipeline()
        except Exception as e:
            messagebox.showerror("Model load failed", str(e))
            return
        if self.worker is None or not self.worker.is_alive():
            self.stop_flag.clear()
            self.cap = cv2.VideoCapture(self.source)
            self.writer = None
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()
        self.playing.set()
        self.b_play.config(text="Pause")
        self.set_status("Playing…")

    def _run(self):
        while not self.stop_flag.is_set():
            if not self.playing.is_set():
                time.sleep(0.04)
                continue
            ok, frame = self.cap.read()
            if not ok:
                break
            annotated, _ = self.pipeline.process_frame(frame)
            if self.save_path:
                if self.writer is None:
                    h, w = annotated.shape[:2]
                    self.writer = cv2.VideoWriter(self.save_path,
                                                  cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                self.writer.write(annotated)
            try:
                self.frame_q.put_nowait(annotated)
            except queue.Full:
                try:
                    self.frame_q.get_nowait()
                    self.frame_q.put_nowait(annotated)
                except (queue.Empty, queue.Full):
                    pass
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.playing.clear()
        saved = f" Saved to {Path(self.save_path).name}." if self.save_path else ""
        self.root.after(0, lambda: self.b_play.config(text="Play"))
        self.root.after(0, lambda: self.set_status("Finished." + saved))

    def _poll(self):
        try:
            frame = self.frame_q.get_nowait()
            self._draw(frame)
        except queue.Empty:
            pass
        self._poll_id = self.root.after(15, self._poll)

    def stop(self):
        self.stop_flag.set()
        self.playing.clear()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=1.5)
        self.worker = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.b_play.config(text="Play")
        if self.source is not None:
            cap = cv2.VideoCapture(self.source)
            ok, fr = cap.read()
            cap.release()
            if ok:
                self.last_raw = self._resize(fr)
                self.canvas.delete("calibdot")
                self._draw(self.last_raw)
        self.set_status("Stopped.")

    def choose_save(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                            filetypes=[("MP4 video", "*.mp4")])
        if path:
            self.save_path = path
            self.set_status(f"Output will be saved to {Path(path).name}. Press Stop then Play.")

    # ---------------- calibration ----------------
    def start_calibration(self):
        if self.last_raw is None:
            return
        self.playing.clear()
        self.b_play.config(text="Play")
        self.calib_mode = True
        self.calib_points = []
        self.canvas.delete("calibdot")
        self._draw(self.last_raw)
        self.set_status("Calibration: click 4 ground points — TL, TR, BR, BL.")

    def on_canvas_click(self, ev):
        if not self.calib_mode:
            return
        self.calib_points.append((ev.x, ev.y))
        n = len(self.calib_points)
        self.canvas.create_oval(ev.x - 4, ev.y - 4, ev.x + 4, ev.y + 4,
                                fill="#00ff00", outline="black", tags="calibdot")
        self.canvas.create_text(ev.x + 9, ev.y, text=str(n), fill="#00ff00",
                                anchor="w", tags="calibdot")
        if n == 4:
            self.finish_calibration()

    def finish_calibration(self):
        self.calib_mode = False
        w = simpledialog.askfloat("Calibration", "Real width  (point 1 → point 2) in metres:",
                                  minvalue=0.1, parent=self.root)
        h = simpledialog.askfloat("Calibration", "Real height (point 1 → point 4) in metres:",
                                  minvalue=0.1, parent=self.root)
        if not w or not h:
            self.set_status("Calibration cancelled.")
            return
        H = compute_homography(self.calib_points, [(0, 0), (w, 0), (w, h), (0, h)])
        try:
            self.ensure_pipeline()
        except Exception as e:
            messagebox.showerror("Model load failed", str(e))
            return
        self.pipeline.distance = DistanceEstimator(H, self.cfg.min_safe_distance_m,
                                                   self.cfg.fallback_pixel_distance)
        self.set_status("Calibrated — distances now in metres. Press Play.")

    def on_close(self):
        self.stop_flag.set()
        self.playing.clear()
        # Cancel the repeating canvas poll before the interpreter goes away.
        # Without this a queued callback fires against destroyed widgets and
        # prints a TclError traceback, which reads as a crash on a clean exit.
        if self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except Exception:
                pass
            self._poll_id = None
        self.root.after(150, self.root.destroy)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
