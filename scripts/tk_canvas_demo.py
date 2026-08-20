"""Minimal test: does Tk canvas + PIL PhotoImage rendering work on this machine?

This has nothing to do with the mask/distance app. It just draws a solid
red rectangle on a canvas. If you see red, canvas image rendering works
and the bug is something else in desktop_app.py. If the window stays
black, this confirms a system-Tk rendering problem unrelated to your code.

Run with:
    python3 tk_canvas_test.py

Close the window (or press q... actually just click the red X) when done.
"""
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

root = tk.Tk()
root.title("Tk Canvas Test — you should see a RED rectangle below")

canvas = tk.Canvas(root, width=400, height=300, bg="black")
canvas.pack()

arr = np.zeros((300, 400, 3), dtype="uint8")
arr[:, :, 0] = 255  # solid red in RGB

img = ImageTk.PhotoImage(Image.fromarray(arr))
_imgref = img  # keep a reference alive, same pattern as desktop_app.py
canvas.create_image(0, 0, anchor=tk.NW, image=img)

label = tk.Label(root, text="If this window is black instead of red, that's the bug.")
label.pack()

root.mainloop()
