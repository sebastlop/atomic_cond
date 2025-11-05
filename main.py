import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib
import os, sys
from model import predict_conditional, FlexibleNN

DEVICE = torch.device('cpu')
MODEL_FOLDER = 'models_folder'

encoder = joblib.load(os.path.join(MODEL_FOLDER, 'one_hot_encoder_models.joblib'))
model = FlexibleNN(6, [], "elu", 0.0)
model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER,'./modelo_state_dict.pt'),map_location=DEVICE))
model.eval()

# --- Function to update the plot ---
def update_plot(ax, canvas, func_var, Zp_var, Zt_var, xrange_var):
    """Updates the plot based on the selected function and x-range."""
    func = func_var.get()
    Zp = int(Zp_var.get())
    Zt = int(Zt_var.get())

    # Parse user range
    try:
        x_min, x_max = map(float, xrange_var.get().split(","))
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter range as two numbers separated by a comma.")
        return

    # Generate data
    x = np.linspace(x_min, x_max, 300)

    # Choose function
    if func in ["CDW-EIS", "CTMC", "Semiempiric_1985Rudd"]:
        y = predict_conditional(model, encoder, func, Zp, Zt, np.log10(x))
    else:
        y = np.zeros_like(x)
        messagebox.showerror("Wrong Model")

    # Update plot
    ax.clear()
    ax.loglog(x, y, label=func)
    ax.legend()
    ax.set_title(f"Theory: {func}")
    ax.set_xlabel("Projectile Energy (keV/u)")
    ax.set_ylabel(r"Cross Section (cm$^2$)")
    canvas.draw()


# --- Main GUI ---
def main():
    root = tk.Tk()
    root.title("Function Plotter")
    root.geometry("900x500")

    # Layout setup
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)
    root.rowconfigure(0, weight=1)

    # --- Left control panel ---
    control_frame = ttk.Frame(root, padding=10)
    control_frame.grid(row=0, column=0, sticky="nswe")

    ttk.Label(control_frame, text="Select Theory:").pack(anchor="w", pady=5)
    func_var = tk.StringVar()
    func_combo = ttk.Combobox(control_frame, textvariable=func_var, state="readonly")
    func_combo['values'] = ("CDW-EIS", "CTMC", "Semiempiric_1985Rudd")
    func_combo.current(0)
    func_combo.pack(fill="x", pady=5)

    ttk.Label(control_frame, text="Enter x range (e.g. 0,10):").pack(anchor="w", pady=5)
    xrange_var = tk.StringVar(value="100,200")
    ttk.Entry(control_frame, textvariable=xrange_var).pack(fill="x", pady=5)

    Zp_var = tk.StringVar()
    Zp_var.set('1')
    label_Zp = ttk.Label(text="Projectile Charge:")
    label_Zp.place(x=10, y=300, width=200)
    spin_Zp = ttk.Spinbox(from_=1, to=9, increment=1, textvariable=Zp_var, state="readonly")
    spin_Zp.place(x=170, y=300, width=50)

    Zt_var = tk.StringVar()
    Zt_var.set('1')
    label_Zt = ttk.Label(text="Target atomic number:")
    label_Zt.place(x=10, y=350, width=200)
    spin_Zt = ttk.Spinbox(from_=1, to=1, increment=1, textvariable=Zt_var, state="readonly")
    spin_Zt.place(x=170, y=350, width=50)


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
        command=lambda: update_plot(ax, canvas, func_var, Zp_var, Zt_var, xrange_var)
    ).pack(fill="x", pady=10)

    # Initial plot
    update_plot(ax, canvas, func_var, Zp_var, Zt_var, xrange_var)

    def on_closing_window():
        root.destroy()
        sys.exit() # Terminate the Python script

    root.protocol("WM_DELETE_WINDOW", on_closing_window)

    root.mainloop()



if __name__ == "__main__":
    main()