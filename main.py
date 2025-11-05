import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib
import os

DEVICE = torch.device('cpu')
MODEL_FOLDER = 'models_folder'


# --- Function to update the plot ---
def update_plot(ax, canvas, func_var, xrange_var):
    """Updates the plot based on the selected function and x-range."""
    func = func_var.get()

    # Parse user range
    try:
        x_min, x_max = map(float, xrange_var.get().split(","))
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter range as two numbers separated by a comma.")
        return

    # Generate data
    x = np.linspace(x_min, x_max, 300)

    # Choose function
    if func == "sin(x)":
        y = np.sin(x)
    elif func == "cos(x)":
        y = np.cos(x)
    elif func == "exp(x)":
        y = np.exp(x)
    elif func == "x^2":
        y = x**2
    else:
        y = np.zeros_like(x)

    # Update plot
    ax.clear()
    ax.plot(x, y, label=func)
    ax.legend()
    ax.set_title(f"f(x) = {func}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    canvas.draw()


# --- Main GUI ---
def main():
    root = tk.Tk()
    root.title("Function Plotter")
    root.geometry("800x500")

    # Layout setup
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)
    root.rowconfigure(0, weight=1)

    # --- Left control panel ---
    control_frame = ttk.Frame(root, padding=10)
    control_frame.grid(row=0, column=0, sticky="nswe")

    ttk.Label(control_frame, text="Select function:").pack(anchor="w", pady=5)
    func_var = tk.StringVar()
    func_combo = ttk.Combobox(control_frame, textvariable=func_var, state="readonly")
    func_combo['values'] = ("sin(x)", "cos(x)", "exp(x)", "x^2")
    func_combo.current(0)
    func_combo.pack(fill="x", pady=5)

    ttk.Label(control_frame, text="Enter x range (e.g. 0,10):").pack(anchor="w", pady=5)
    xrange_var = tk.StringVar(value="0,10")
    ttk.Entry(control_frame, textvariable=xrange_var).pack(fill="x", pady=5)

    # --- Right plot area ---
    plot_frame = ttk.Frame(root)
    plot_frame.grid(row=0, column=1, sticky="nswe")

    # Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 4))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Button to trigger update_plot
    ttk.Button(
        control_frame,
        text="Update Plot",
        command=lambda: update_plot(ax, canvas, func_var, xrange_var)
    ).pack(fill="x", pady=10)

    # Initial plot
    update_plot(ax, canvas, func_var, xrange_var)

    root.mainloop()


if __name__ == "__main__":
    main()