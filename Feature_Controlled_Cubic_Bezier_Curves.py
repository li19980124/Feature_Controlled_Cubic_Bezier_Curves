import math
import os
import numpy as np
import matplotlib.pyplot as plt
from _tkinter import TclError
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox


# Global style
STYLE_CONFIG = {
    'regular':
        {'color': '#FF8A00', 'marker': 's', 'fill': False, 'size': 8, 'edge_width': 2},
    'inflection':
        {'color': '#0000FF', 'marker': '*', 'fill': False, 'size': 10, 'edge_width': 2},
    'loop':
        {'color': '#00FF00', 'marker': 'o', 'fill': False, 'size': 8, 'edge_width': 2},
    'cusp':
        {'color': '#FF0000', 'marker': '^', 'fill': False, 'size': 8, 'edge_width': 2},
    'open_end':
        {'color': 'black', 'marker': 's', 'fill': False, 'size': 10, 'edge_width': 2},
    'selected':
        {'color': 'yellow', 'size_offset': 2, 'edge_color': 'black', 'edge_width': 2},
    'curve':
        {'color': 'black', 'line_width': 2},
    'polygon':
        {'color': 'c', 'line_width': 1, 'alpha': 0.6, 'linestyle': '--'},
    'curvature_comb':
        {'color': '#FF1493', 'line_width': 1, 'scale': 5000.0, 'density': 5}
}

# Load model
class Curve:
    def __init__(self, points, index, pa_a, is_closed, color):
        self.points = points
        self.index = index
        self.pa_a = pa_a
        self.is_closed = is_closed
        self.color = color

class Model:
    def __init__(self, name, curves):
        self.name = name
        self.curves = curves

#
def load_model_from_txt(file_path):
    curves = []
    color_idx = 0
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split('|')
            if len(parts) != 4:

                print(f"Warning: The format of line {line_num + 1} in {file_path} is incorrect, skipping")
                continue
            try:
                points = np.array(eval(parts[0]))
                index = np.array(eval(parts[1])).tolist()
                pa_a = np.array(eval(parts[2])).tolist()
                is_closed = eval(parts[3])
                # color = curve_colors[color_idx % len(curve_colors)]
                color = '#000000'
                color_idx += 1
                curves.append(Curve(points, index, pa_a, is_closed, color))
            except Exception as e:
                print(f"Warning: Failed to parse line {line_num + 1} of {file_path}: {str(e)}, skipping")
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    return Model(model_name, curves)

# Load the txt file
def auto_scan_models():
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(model_dir, filename)
            models.append(load_model_from_txt(file_path))
    return models

# Calculate the area of a triangle
def areastri(a, b, c):
    return abs(np.linalg.det(np.vstack([b - a, c - a]))) / 2.0

# Calculate the endpoints c0 and c2 of the closed curve
def compute_endpoints_closed(N, lamad, c1):
    c2 = np.zeros_like(c1)
    for i in range(N - 1):
        c2[i] = (1 - lamad[i]) * c1[i] + lamad[i] * c1[i + 1]
    c2[-1] = (1 - lamad[-1]) * c1[-1] + lamad[-1] * c1[0]
    c0 = np.vstack([c2[-1], c2[:-1]])
    return c0, c2

# Calculate the λ parameter of the closed curve
def compute_lamad_closed(c0, c1, c2, param_a, N):
    tmp = np.zeros(N)
    denom = np.zeros(N)
    for i in range(N - 1):
        area = areastri(c0[i], c1[i], c1[i + 1])
        tmp[i] = param_a[i + 1] * np.sqrt(abs(1 - param_a[i]) * area)
        area2 = areastri(c1[i], c1[i + 1], c2[i + 1])
        denom[i] = tmp[i] + param_a[i] * np.sqrt(abs(1 - param_a[i + 1]) * area2)
    area = areastri(c0[-1], c1[-1], c1[0])
    tmp[-1] = param_a[0] * np.sqrt(abs(1 - param_a[-1]) * area)
    area2 = areastri(c1[-1], c1[0], c2[0])
    denom[-1] = tmp[-1] + param_a[-1] * np.sqrt(abs(1 - param_a[0]) * area2)
    denom[denom < 1e-10] = 1e-10
    return tmp / denom

# Calculate the λ parameter of the closed curve
def compute_middlepoints_closed(N, lamad, tt, p, param_a):
    triag = np.zeros((N, N))
    A1 = np.zeros(N)
    A2 = np.zeros(N)
    A3 = np.zeros(N)
    for i in range(1, N):
        t = tt[i]
        pa = param_a[i]
        A1[i] = (1 - lamad[i - 1]) * ((1 - t) ** 3 + 3 * t * (1 - pa) * (1 - t) ** 2)
        term2 = lamad[i - 1] * ((1 - t) ** 3 + 3 * (1 - t) ** 2 * t * (1 - pa))
        term3 = (1 - lamad[i]) * (t ** 3 + 3 * (1 - t) * t ** 2 * (1 - pa))
        term4 = pa * 3 * (1 - t) * t
        A2[i] = term2 + term3 + term4
        A3[i] = lamad[i] * (t ** 3 + 3 * (1 - pa) * (1 - t) * t ** 2)
    t = tt[0]
    pa = param_a[0]
    A1[0] = (1 - lamad[-1]) * ((1 - t) ** 3 + 3 * (1 - t) ** 2 * t * (1 - pa))
    term2 = lamad[-1] * ((1 - t) ** 3 + 3 * (1 - t) ** 2 * t * (1 - pa))
    term3 = (1 - lamad[0]) * (t ** 3 + 3 * (1 - t) * t ** 2 * (1 - pa))
    term4 = pa * 3 * (1 - t) * t
    A2[0] = term2 + term3 + term4
    A3[0] = lamad[0] * (t ** 3 + 3 * (1 - pa) * (1 - t) * t ** 2)
    for i in range(N):
        triag[i, i] = A2[i]
        if i > 0:
            triag[i, i - 1] = A1[i]
        if i < N - 1:
            triag[i, i + 1] = A3[i]
    triag[0, -1] = A1[0]
    triag[-1, 0] = A3[-1]
    return np.linalg.solve(triag, p)

# Calculate the endpoint c2 of the open curve
def compute_endpoints_open(N_lamad, lamad, c1):
    c2 = np.zeros_like(c1[:-1])
    for i in range(N_lamad):
        c2[i] = (1 - lamad[i]) * c1[i] + lamad[i] * c1[i + 1]
    return c2

# Calculate the endpoint c2 of the open curve
def compute_lamad_open(c0, c1, c2, param_a, N):
    N_lamad = N - 3
    tmp = np.zeros(N_lamad)
    denom = np.zeros(N_lamad)
    for i in range(N_lamad):
        area = areastri(c0[i], c1[i], c1[i + 1])
        tmp[i] = param_a[i + 1] * np.sqrt(abs(1 - param_a[i]) * area)
        area2 = areastri(c1[i], c1[i + 1], c2[i + 1])
        denom[i] = tmp[i] + param_a[i] * np.sqrt(abs(1 - param_a[i + 1]) * area2)
    denom[denom < 1e-10] = 1e-10
    return tmp / denom

# Calculate the intermediate control point c1 of the open curve
def compute_middlepoints_open(N, lamad, tt, p, param_a):
    N_mid = N - 2
    triag = np.zeros((N_mid, N_mid))
    fixed = np.zeros((N_mid, 2))
    if N == 3:
        t = tt[0]
        pa = param_a[0]
        c0 = p[0]
        c2 = p[2]
        numerator = p[1] - c0 * (1 - t) ** 3 - 3 * (1 - t) * t * (1 - pa) * ((1 - t) * c0 + t * c2) - c2 * t ** 3
        denominator = 3 * pa * (1 - t) * t
        return numerator / denominator.reshape(1, -1)
    t_first = tt[0]
    pa_first = param_a[0]
    fixed[0] = p[0] * ((1 - t_first) ** 3 + 3 * (1 - t_first) ** 2 * t_first * (1 - pa_first))
    t_last = tt[-1]
    pa_last = param_a[-1]
    fixed[-1] = p[-1] * (t_last ** 3 + 3 * (1 - t_last) * t_last ** 2 * (1 - pa_last))
    t = tt[0]
    pa = param_a[0]
    term1 = (1 - lamad[0]) * (t ** 3 + 3 * (1 - t) * t ** 2 * (1 - pa))
    term2 = pa * 3 * (1 - t) * t
    triag[0, 0] = term1 + term2
    triag[0, 1] = lamad[0] * (t ** 3 + 3 * (1 - pa) * (1 - t) * t ** 2)
    t = tt[-1]
    pa = param_a[-1]
    triag[-1, -2] = (1 - lamad[-1]) * ((1 - t) ** 3 + 3 * (1 - t) ** 2 * t * (1 - pa))
    term1 = lamad[-1] * ((1 - t) ** 3 + 3 * (1 - t) ** 2 * t * (1 - pa))
    term2 = pa * 3 * (1 - t) * t
    triag[-1, -1] = term1 + term2
    for i in range(1, N_mid - 1):
        t = tt[i]
        pa = param_a[i]
        triag[i, i - 1] = (1 - lamad[i - 1]) * ((1 - t) ** 3 + 3 * t * (1 - pa) * (1 - t) ** 2)
        term2 = lamad[i - 1] * ((1 - t) ** 3 + 3 * (1 - t) ** 2 * t * (1 - pa))
        term3 = (1 - lamad[i]) * (t ** 3 + 3 * (1 - t) * t ** 2 * (1 - pa))
        term4 = pa * 3 * (1 - t) * t
        triag[i, i] = term2 + term3 + term4
        triag[i, i + 1] = lamad[i] * (t ** 3 + 3 * (1 - pa) * (1 - t) * t ** 2)
    return np.linalg.solve(triag, p[1:-1] - fixed)

# Generate the point set of the closed curve and Bezier control points
def generate_closed_curve(p, index, pa_a, inflection_signs):
    N = len(p)
    param_a = np.ones(N) * 2.0
    tt = np.ones(N) / 2.0
    if N == 2:
        t = np.linspace(0, 1, 201)
        x = p[0][0] * (1 - t) + p[1][0] * t
        y = p[0][1] * (1 - t) + p[1][1] * t
        curve = np.column_stack((x, y))
        c0 = np.array([p[0]])
        c1 = np.array([p[1]])
        c2 = np.array([p[0]])
        return curve, c0, c1, c2, param_a
    for i in range(N):
        if index[i] == 1: # inflection point
            param_a[i] = pa_a[0]
            sign = inflection_signs[i] if i < len(inflection_signs) else -1
            tt[i] += sign * np.sqrt((2 - param_a[i]) / (3 * param_a[i] - 2)) / 2
        elif index[i] == 2: # loop
            param_a[i] = pa_a[1]
            tt[i] -= np.sqrt(3 * (param_a[i] - 2) / (3 * param_a[i] - 2)) / 2
        elif index[i] == 3: # regular point
            param_a[i] = pa_a[2]
    lamad = np.ones(N) / 2.0
    c1 = p.copy()
    max_iter = 60
    tolerance = 1e-10
    iter_count = 0
    flag = True
    while flag:
        c0, c2 = compute_endpoints_closed(N, lamad, c1)
        lamad = compute_lamad_closed(c0, c1, c2, param_a, N)
        cc1 = compute_middlepoints_closed(N, lamad, tt, p, param_a)
        max_deviation = np.max(np.linalg.norm(cc1 - c1, axis=1))
        if max_deviation <= tolerance or iter_count >= max_iter:
            flag = False
        else:
            c1 = cc1
            iter_count += 1
    t_global = np.linspace(0, N, 201 * N)
    curve_points = []
    for i in range(N):
        c0_i = c0[i]
        c1_i = c1[i]
        c2_i = c2[i]
        pa = param_a[i]
        b1 = (1 - pa) * c0_i + pa * c1_i
        b2 = pa * c1_i + (1 - pa) * c2_i
        mask = (t_global >= i) & (t_global < i + 1)
        t_local = t_global[mask] - i
        xp = (1 - t_local) ** 3 * c0_i[0] + 3 * (1 - t_local) ** 2 * t_local * b1[0] + 3 * (1 - t_local) * t_local ** 2 * b2[0] + t_local ** 3 * c2_i[0]
        yp = (1 - t_local) ** 3 * c0_i[1] + 3 * (1 - t_local) ** 2 * t_local * b1[1] + 3 * (1 - t_local) * t_local ** 2 * b2[1] + t_local ** 3 * c2_i[1]
        curve_points.append(np.column_stack((xp, yp)))
    return np.vstack(curve_points), c0, c1, c2, param_a

# Generate the point set of the open curve and the Bezier control points
def generate_open_curve(p, index, pa_a, inflection_signs):
    N = len(p)
    N_mid = N - 2
    if len(index) != N_mid:
        index = [3] * N_mid
    if len(inflection_signs) != N_mid:
        inflection_signs = [-1] * N_mid
    param_a = np.ones(N_mid) * 2.0
    tt = np.ones(N_mid) / 2.0
    if N < 2:
        return np.array([]), np.array([]), np.array([]), np.array([]), param_a
    if N == 2:
        t = np.linspace(0, 1, 201)
        x = p[0][0] * (1 - t) + p[1][0] * t
        y = p[0][1] * (1 - t) + p[1][1] * t
        curve = np.column_stack((x, y))
        c0 = np.array([p[0]])
        c1 = np.array([(p[0] + p[1]) / 2])
        c2 = np.array([p[1]])
        return curve, c0, c1, c2, param_a
    for i in range(N_mid):
        if index[i] == 1: # inflection point
            param_a[i] = pa_a[0]
            sign = inflection_signs[i]
            tt[i] += sign * np.sqrt((2 - param_a[i]) / (3 * param_a[i] - 2)) / 2
        elif index[i] == 2: # loop
            param_a[i] = pa_a[1]
            tt[i] -= np.sqrt(3 * (param_a[i] - 2) / (3 * param_a[i] - 2)) / 2
        elif index[i] == 3: # regular point
            param_a[i] = pa_a[2]
    if N == 3:
        c1 = compute_middlepoints_open(N, [], tt, p, param_a)
        c0 = np.array([p[0]])
        c2 = np.array([p[2]])
        t = np.linspace(0, 1, 201)
        curve_points = []
        c0_i = c0[0]
        c1_i = c1[0]
        c2_i = c2[0]
        pa = param_a[0]
        b1 = (1 - pa) * c0_i + pa * c1_i
        b2 = pa * c1_i + (1 - pa) * c2_i
        xp = (1 - t) ** 3 * c0_i[0] + 3 * (1 - t) ** 2 * t * b1[0] + 3 * (1 - t) * t ** 2 * b2[0] + t ** 3 * c2_i[0]
        yp = (1 - t) ** 3 * c0_i[1] + 3 * (1 - t) ** 2 * t * b1[1] + 3 * (1 - t) * t ** 2 * b2[1] + t ** 3 * c2_i[1]
        curve_points.append(np.column_stack((xp, yp)))
        return np.vstack(curve_points), c0, c1, c2 ,param_a
    else:
        lamad = np.ones(N - 3) / 2.0
        c1 = p[1:-1].copy()
        max_iter = 60
        tolerance = 1e-10
        iter_count = 0
        flag = True
        while flag:
            cc2 = compute_endpoints_open(N - 3, lamad, c1)
            c0 = np.vstack([p[0], cc2])
            c2 = np.vstack([cc2, p[-1]])
            lamad = compute_lamad_open(c0, c1, c2, param_a, N)
            cc1 = compute_middlepoints_open(N, lamad, tt, p, param_a)
            max_deviation = np.max(np.linalg.norm(cc1 - c1, axis=1))
            if max_deviation <= tolerance or iter_count >= max_iter:
                flag = False
            else:
                c1 = cc1
                iter_count += 1
                cc2 = compute_endpoints_open(N - 3, lamad, c1)
                c0 = np.vstack([p[0], cc2])
                c2 = np.vstack([cc2, p[-1]])

        # Generate curve points
        # t = np.linspace(0, 1, 201)
        t_global = np.linspace(0, N_mid, 201 * N_mid)
        curve_points = []
        for i in range(N_mid):
            c0_i = c0[i]
            c1_i = c1[i]
            c2_i = c2[i]
            pa = param_a[i]
            b1 = (1 - pa) * c0_i + pa * c1_i
            b2 = pa * c1_i + (1 - pa) * c2_i
            mask = (t_global >= i) & (t_global < i + 1)
            t_local = t_global[mask] - i
            xp = (1 - t_local) ** 3 * c0_i[0] + 3 * (1 - t_local) ** 2 * t_local * b1[0] + 3 * (1 - t_local) * t_local ** 2 * b2[0] + t_local ** 3 * c2_i[0]
            yp = (1 - t_local) ** 3 * c0_i[1] + 3 * (1 - t_local) ** 2 * t_local * b1[1] + 3 * (1 - t_local) * t_local ** 2 * b2[1] + t_local ** 3 * c2_i[1]
            curve_points.append(np.column_stack((xp, yp)))
        return np.vstack(curve_points) if curve_points else np.array([]), c0, c1, c2, param_a

# Calculate the derivative
def bezier_derivatives(curve, t):
    n = len(curve) - 1
    # 0th
    b0 = np.zeros(2)
    for i in range(n+1):
        b0 += curve[i] * binomial(n, i) * (1-t)**(n-i) * t**i
    # 1th
    b1 = np.zeros(2)
    for i in range(n):
        b1 += n * (curve[i+1] - curve[i]) * binomial(n-1, i) * (1-t)**(n-1-i) * t**i
    # 2th
    b2 = np.zeros(2)
    for i in range(n-1):
        b2 += n*(n-1) * (curve[i+2] - 2*curve[i+1] + curve[i]) * binomial(n-2, i) * (1-t)**(n-2-i) * t**i
    return b0, b1, b2

# Binomial coefficient C(n,k)
def binomial(n, k):
    return math.comb(n, k)

# Calculate the curvature based on the analytical derivative
def compute_curvature_analytical(curve_segments, param_a):
    curvature = []

    # Check the validity
    if len(curve_segments) == 0 or len(param_a) == 0 or len(param_a) != len(curve_segments):
        return np.array([])

    for i, seg in enumerate(curve_segments):
        pa = param_a[i]

        # Construct the four control vertices of the cubic Bezier
        c0, c1, c2 = seg[0], seg[1], seg[2]
        b1 = (1 - pa) * c0 + pa * c1
        b2 = pa * c1 + (1 - pa) * c2
        curve = np.array([c0, b1, b2, c2])
        t_global = np.linspace(i, i+1, 201)
        for t in t_global:
            t_local = t - i
            _, b1, b2 = bezier_derivatives(curve, t_local)
            # Curvature
            numerator = b1[0] * b2[1] - b1[1] * b2[0]
            denominator = (b1[0]**2 + b1[1]** 2) ** 1.5
            denominator = max(denominator, 1e-10)
            curvature.append(numerator / denominator)
    return np.array(curvature)


# GUI
class FCaCurveDesigner:
    def __init__(self, root):
        self.root = root
        self.root.title("Curve design")
        try:
            self.root.state('zoomed')
        except TclError:
            try:
                self.root.attributes('-zoomed', True)
            except TclError:
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height}")

        # Curve data
        self.param_a = []
        self.points = []
        self.index = []
        self.inflection_signs = []
        self.pa_a = [1.1, 3.0, 2 / 3] # Default
        self.is_closed = True
        self.loaded_models = auto_scan_models()
        self.current_model = None
        self.in_model_mode = False

        # Display options
        self.show_control_polygon = False
        self.show_curvature_comb = False
        self.show_only_curve = False

        # Interaction state
        self.selected_point = None
        self.dragging_point = None
        self.hover_point = None

        # Store curve and curvature data
        self.curve_points = np.array([])
        self.curvature = np.array([])
        self.bezier_c0 = np.array([])
        self.bezier_c1 = np.array([])
        self.bezier_c2 = np.array([])

        # Create
        self._create_widgets()

        # Initialize
        self._init_plot()
        self._bind_events()


        if self.loaded_models:
            self.model_combo['values'] = [model.name for model in self.loaded_models]
        else:
            self.model_combo['values'] = []
            self.status_var.set("Status: Model file not scanned")

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left control panel
        control_frame = ttk.LabelFrame(main_frame, text="Parameter setting", width=220)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        # Model selection area
        ttk.Label(control_frame, text="Models:", font=("SimHei", 10, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, state="readonly")
        self.model_combo.pack(fill=tk.X, padx=10, pady=2)
        #
        ttk.Button(control_frame, text="load model", command=self._load_selected_model).pack(fill=tk.X, padx=10,
                                                                                               pady=2)

        ttk.Button(control_frame, text="Exit model", command=self._exit_model_mode).pack(fill=tk.X, padx=10, pady=2)


        # Curve type selection
        ttk.Label(control_frame, text="Curve Type:", font=("SimHei", 10, "bold")).pack(anchor=tk.W, padx=5, pady=0)
        self.curve_type = tk.BooleanVar(value=True)
        ttk.Radiobutton(control_frame, text="closed curve", variable=self.curve_type, value=True,
                        command=self._on_curve_type_changed).pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(control_frame, text="open curve", variable=self.curve_type, value=False,
                        command=self._on_curve_type_changed).pack(anchor=tk.W, padx=10, pady=2)

        # Display options
        ttk.Label(control_frame, text="Display options:", font=("SimHei", 10, "bold")).pack(anchor=tk.W, padx=5, pady=(10, 5))
        self.show_polygon_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="show the control polygon", variable=self.show_polygon_var,
                        command=self._on_display_option_changed).pack(anchor=tk.W, padx=10)
        self.show_curvature_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="show the curvature", variable=self.show_curvature_var,
                        command=self._on_display_option_changed).pack(anchor=tk.W, padx=10)
        self.show_only_curve_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="only show the curve", variable=self.show_only_curve_var,
                        command=self._on_display_option_changed).pack(anchor=tk.W, padx=10, pady=0)

        # Feature point type
        ttk.Label(control_frame, text="Feature point:", font=("SimHei", 10, "bold")).pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.point_type_frame = ttk.Frame(control_frame)
        self.point_type_frame.pack(anchor=tk.W, padx=5, pady=0)
        self.point_type = tk.IntVar(value=3)
        types = [(0, "Cusp(a=2)"), (1, "Inflection point"), (2, "Loop"), (3, "Regular point")]
        for i, (val, text) in enumerate(types):
            ttk.Radiobutton(self.point_type_frame, text=text, variable=self.point_type,
                            value=val, command=self._on_point_type_changed).grid(row=i // 2, column=i % 2, sticky=tk.W,
                                                                                 padx=5, pady=2)

        # Parameter setting
        ttk.Label(control_frame, text="Parameter a:", font=("SimHei", 10, "bold")).pack(anchor=tk.W, padx=5, pady=(10, 5))
        ttk.Label(control_frame, text="Inflection point value (1 < a < 2):").pack(anchor=tk.W, padx=10)
        self.inflection_a = tk.DoubleVar(value=self.pa_a[0])
        a1_scale = ttk.Scale(control_frame, from_=1.01, to=1.99, variable=self.inflection_a,
                             command=lambda v: self._on_global_param_changed(0, float(v)), orient=tk.HORIZONTAL)
        a1_scale.pack(fill=tk.X, padx=10)
        self.a1_label = ttk.Label(control_frame, text=f"Current value: {self.pa_a[0]:.2f}")
        self.a1_label.pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(control_frame, text="Inflection point symbol:", font=("SimHei", 10, "bold")).pack(anchor=tk.W, padx=5, pady=(10, 5))
        self.sign_var = tk.IntVar(value=-1)
        sign_frame = ttk.Frame(control_frame)
        sign_frame.pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(sign_frame, text="Sub(-)", variable=self.sign_var, value=-1,
                        command=self._on_sign_changed).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(sign_frame, text="Add(+)", variable=self.sign_var, value=1,
                        command=self._on_sign_changed).pack(side=tk.LEFT, padx=10)
        ttk.Label(control_frame, text="Loop value (a > 2):").pack(anchor=tk.W, padx=10)
        self.loop_a = tk.DoubleVar(value=self.pa_a[1])
        a2_scale = ttk.Scale(control_frame, from_=2.01, to=10.0, variable=self.loop_a,
                             command=lambda v: self._on_global_param_changed(1, float(v)), orient=tk.HORIZONTAL)
        a2_scale.pack(fill=tk.X, padx=10)
        self.a2_label = ttk.Label(control_frame, text=f"Current value: {self.pa_a[1]:.2f}")
        self.a2_label.pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(control_frame, text="Regular point (0 < a < 1):").pack(anchor=tk.W, padx=10)
        self.regular_a = tk.DoubleVar(value=self.pa_a[2])
        a3_scale = ttk.Scale(control_frame, from_=0.1, to=0.99, variable=self.regular_a,
                             command=lambda v: self._on_global_param_changed(2, float(v)), orient=tk.HORIZONTAL)
        a3_scale.pack(fill=tk.X, padx=10)
        self.a3_label = ttk.Label(control_frame, text=f"Current value: {self.pa_a[2]:.2f}")
        self.a3_label.pack(anchor=tk.W, padx=10, pady=2)


        # Curvature comb setting
        ttk.Label(control_frame, text="Curvature setting:", font=("SimHei", 10, "bold")).pack(anchor=tk.W, padx=5,
                                                                                       pady=(10, 5))

        # Curvature comb scaling factor
        ttk.Label(control_frame, text="scale:").pack(anchor=tk.W, padx=10)
        self.curvature_scale = tk.DoubleVar(value=STYLE_CONFIG['curvature_comb']['scale'])
        curv_scale = ttk.Scale(control_frame, from_=5000, to=10000.0, variable=self.curvature_scale,
                               command=lambda v: self._on_curvature_param_changed('scale', float(v)),
                               orient=tk.HORIZONTAL)
        curv_scale.pack(fill=tk.X, padx=10)
        self.curv_scale_label = ttk.Label(control_frame, text=f"Current value: {STYLE_CONFIG['curvature_comb']['scale']:.1f}")

        # Curvature comb density control
        ttk.Label(control_frame, text="density:").pack(anchor=tk.W, padx=10, pady=(5, 0))
        self.curvature_density = tk.IntVar(value=STYLE_CONFIG['curvature_comb']['density'])
        curv_density = ttk.Scale(control_frame, from_=1, to=10, variable=self.curvature_density,
                                 command=lambda v: self._on_curvature_param_changed('density', int(float(v))),
                                 orient=tk.HORIZONTAL)
        curv_density.pack(fill=tk.X, padx=10)
        self.curv_density_label = ttk.Label(control_frame, text=f"Current value: {STYLE_CONFIG['curvature_comb']['density']}")



        # button
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=0, pady=10)
        ttk.Button(control_frame, text="Clear all", command=self._clear_all_points).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="Reset the view", command=self._reset_view).pack(fill=tk.X, padx=5, pady=5)

        # Status tag
        self.status_var = tk.StringVar(value="Status: Please select model or click")

        # The drawing area on the right
        plot_frame = ttk.LabelFrame(main_frame, text="Drawing Area")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fig = plt.figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

    def _load_selected_model(self):

        # Check if there is any model loading
        if not self.loaded_models:
            messagebox.showwarning("Prompt "," No model files")
            return
        #
        selected_name = self.model_var.get()
        if not selected_name:
            messagebox.showwarning("Prompt", "Please select a model")
            return
        # Load the selected model and draw the plot
        self.current_model = next(model for model in self.loaded_models if model.name == selected_name)
        self.in_model_mode = True  # Enter the model mode
        self._redraw()  # Draw the curve of the loaded model
        # Update status prompt
        self.status_var.set(f"Status: Loading model {selected_name}")

    def _on_model_selected(self, event):
        if not self.loaded_models:
            return
        selected_name = self.model_var.get()
        self.status_var.set(f"Status: Selected model {selected_name}")

    def _exit_model_mode(self):
        self.in_model_mode = False
        self.current_model = None
        self._clear_all_points()
        self.status_var.set("Status: Please draw the curve")

    def _init_plot(self):
        self.ax.set_xlim(0, 1200)
        self.ax.set_ylim(0, 1200)
        self.ax.set_aspect('auto')
        self.ax.set_title("Curve design (click to add points, drag to adjust, right-click to select points)")
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.canvas.draw()

    def _bind_events(self):
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)

    def _on_curve_type_changed(self):
        old_closed = self.is_closed
        self.is_closed = self.curve_type.get()
        if not self.is_closed and old_closed and len(self.points) > 2:
            self.index = self.index[1:-1] if len(self.index) >= len(self.points) else [3] * (len(self.points) - 2)
            self.inflection_signs = self.inflection_signs[1:-1] if len(self.inflection_signs) >= len(self.points) else [-1] * (len(self.points) - 2)
        elif self.is_closed and not old_closed:
            num_points = len(self.points)
            current_indices = len(self.index)
            current_signs = len(self.inflection_signs)
            if current_indices < num_points:
                self.index += [3] * (num_points - current_indices)
            if current_signs < num_points:
                self.inflection_signs += [-1] * (num_points - current_signs)
        self.status_var.set(f"Status: {'Closed' if self.is_closed else 'Open'} Curve")
        self._redraw()

    def _on_display_option_changed(self):
        self.show_control_polygon = self.show_polygon_var.get()
        self.show_curvature_comb = self.show_curvature_var.get()
        self.show_only_curve = self.show_only_curve_var.get()
        if self.show_only_curve:
            self.show_control_polygon = False
            self.show_curvature_comb = False
            self.show_polygon_var.set(False)
            self.show_curvature_var.set(False)
        self._redraw()

    def _on_global_param_changed(self, idx, value):
        self.pa_a[idx] = value
        if idx == 0:
            self.a1_label.config(text=f"Current value: {value:.2f}")
        elif idx == 1:
            self.a2_label.config(text=f"Current value: {value:.2f}")
        elif idx == 2:
            self.a3_label.config(text=f"Current value: {value:.2f}")
        if self.points:
            self._redraw()


    def _on_curvature_param_changed(self, param_type, value):
        """Curvature comb parameter change event (ensure parameters take effect in real-time)"""
        if param_type == 'scale':
            STYLE_CONFIG['curvature_comb']['scale'] = value
            self.curv_scale_label.config(text=f"Current value: {value:.1f}")
        elif param_type == 'density':
            STYLE_CONFIG['curvature_comb']['density'] = value
            self.curv_density_label.config(text=f"Current value: {value}")

        # Redraw the curvature comb immediately after the parameters change
        if self.show_curvature_comb and len(self.curvature) > 0 and len(self.curve_points) > 0:
            self._redraw()

    def _on_sign_changed(self):
        if self.selected_point is not None and self._can_modify_point(self.selected_point):
            actual_idx = self._get_actual_index(self.selected_point)
            if 0 <= actual_idx < len(self.inflection_signs) and 0 <= actual_idx < len(self.index) and self.index[actual_idx] == 1:
                self.inflection_signs[actual_idx] = self.sign_var.get()
                sign_char = "+" if self.sign_var.get() == 1 else "-"
                self.status_var.set(f"Status: Change the {self.selected_point} symbol to {sign_char}")
                self._redraw()
            else:
                self.status_var.set("Status: Only inflection point")
        else:
            self.status_var.set("Status: Please select an inflection point")

    def _on_point_type_changed(self):
        if not self._can_modify_point(self.selected_point):
            return
        if self.selected_point is not None:
            actual_idx = self._get_actual_index(self.selected_point)
            while len(self.inflection_signs) <= actual_idx:
                self.inflection_signs.append(-1)
            if 0 <= actual_idx < len(self.index):
                old_type = self.index[actual_idx]
                new_type = self.point_type.get()
                self.index[actual_idx] = new_type
                if new_type == 1 and actual_idx >= len(self.inflection_signs):
                    self.inflection_signs.append(self.sign_var.get())
                self.status_var.set(f"Status: {self.selected_point} change to {self._get_type_name(new_type)}")
                self._redraw()

    def _add_default_point(self):
        if self.in_model_mode:
            self.status_var.set("Status: Please clear the canvas")
            return
        center_x, center_y = 500, 600
        angle = len(self.points) * np.pi / 6
        radius = 150 + len(self.points) * 10
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        self.points.append((x, y))
        if self.is_closed:
            self.index.append(3)
            self.inflection_signs.append(-1)
        else:
            if len(self.points) > 2:
                self.index.append(3)
                self.inflection_signs.append(-1)
        self.status_var.set(f"Status: Add new points({len(self.points) - 1})")
        self._redraw()

    def _delete_selected_point(self):
        if self.in_model_mode:
            self.status_var.set("Status: Please clear the canvas")
            return
        if self.selected_point is None or not 0 <= self.selected_point < len(self.points):
            messagebox.showwarning("Warning "," Please right-click and select a point")
            return
        self.points.pop(self.selected_point)
        if self._can_modify_point(self.selected_point):
            actual_idx = self._get_actual_index(self.selected_point)
            if 0 <= actual_idx < len(self.index):
                self.index.pop(actual_idx)
            if 0 <= actual_idx < len(self.inflection_signs):
                self.inflection_signs.pop(actual_idx)
        self.status_var.set(f" status: Deleted point {self.selected_point}")
        self.selected_point = None
        self._redraw()

    def _clear_all_points(self):

        self.in_model_mode = False
        self.current_model = None


        self.points = []
        self.index = []
        self.inflection_signs = []
        self.selected_point = None
        self.curve_points = np.array([])
        self.curvature = np.array([])
        self.bezier_c0 = np.array([])
        self.bezier_c1 = np.array([])
        self.bezier_c2 = np.array([])
        self.status_var.set("Status: Cleared")
        self._redraw()

    def _reset_view(self):
        self.ax.set_xlim(0, 1200)
        self.ax.set_ylim(0, 1200)
        self.ax.set_aspect('auto')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self._redraw()
        self.status_var.set("Status: The view has been reset")

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        closest_idx = self._find_closest_point(x, y)
        if event.button == 1:
            if closest_idx is not None:
                self.dragging_point = closest_idx
                self.status_var.set(f"Status: Dragging point {closest_idx}")
            else:
                self.points.append((x, y))
                if self.is_closed:
                    self.index.append(3)
                    self.inflection_signs.append(-1)
                else:
                    if len(self.points) > 2:
                        self.index.append(3)
                        self.inflection_signs.append(-1)
                self.status_var.set(f"Status: Add new points({len(self.points) - 1})")
                self._redraw()
        elif event.button == 3:
            if closest_idx is not None:
                if not self.is_closed and (closest_idx == 0 or closest_idx == len(self.points) - 1):
                    self.status_var.set("Status: Uneditable")
                    self.selected_point = None
                else:
                    self.selected_point = closest_idx
                    actual_idx = self._get_actual_index(closest_idx)
                    if 0 <= actual_idx < len(self.index):
                        self.point_type.set(self.index[actual_idx])
                        if self.index[actual_idx] == 1 and 0 <= actual_idx < len(self.inflection_signs):
                            self.sign_var.set(self.inflection_signs[actual_idx])
                            sign_char = "+" if self.inflection_signs[actual_idx] == 1 else "-"
                            self.status_var.set(f"Status: Inflection point {closest_idx},symbol:{sign_char}")
                        else:
                            self.status_var.set(f"Status: Selected {closest_idx},{self._get_type_name(self.index[actual_idx])}")
                    else:
                        self.point_type.set(3)
                        self.status_var.set(f"Status: Selected {closest_idx},Regular point")
            else:
                self.selected_point = None
                self.status_var.set("Status: Unselected")
            self._redraw()

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax:
            self.hover_point = None
            return
        x, y = event.xdata, event.ydata
        if self.dragging_point is not None:
            self.points[self.dragging_point] = (x, y)
            self._redraw()
            return
        closest_idx = self._find_closest_point(x, y)
        if closest_idx != self.hover_point:
            self.hover_point = closest_idx
            self._redraw()

    def _on_mouse_release(self, event):
        if self.dragging_point is not None:
            self.status_var.set(f"Status: Dragged point {self.dragging_point}")
            self.dragging_point = None

    def _find_closest_point(self, x, y, threshold=15):
        if not self.points:
            return None
        min_dist = float('inf')
        closest_idx = None
        for i, (px, py) in enumerate(self.points):
            dist = np.hypot(px - x, py - y)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_idx = i
        return closest_idx

    def _get_type_name(self, typ):
        names = ["Cusp", "Inflection point", "Loop", "Regular point"]
        return names[typ]

    def _can_modify_point(self, point_idx):
        if point_idx is None:
            return False
        if not self.is_closed and (point_idx == 0 or point_idx == len(self.points) - 1):
            self.status_var.set("Status: Uneditable")
            return False
        return True

    def _get_actual_index(self, point_idx):
        if self.is_closed:
            return point_idx
        else:
            return point_idx - 1

    def _draw_point(self, x, y, typ, is_selected=False, sign=None):
        if typ == 0:
            config = STYLE_CONFIG['cusp']
        elif typ == 1:
            config = STYLE_CONFIG['inflection']
        elif typ == 2:
            config = STYLE_CONFIG['loop']
        else:
            config = STYLE_CONFIG['regular']
        if is_selected:
            sel_config = STYLE_CONFIG['selected']
            self.ax.plot(
                x, y,
                marker=config['marker'],
                color=sel_config['color'],
                markersize=config['size'] + sel_config['size_offset'],
                markeredgecolor=sel_config['edge_color'],
                markeredgewidth=sel_config['edge_width'],
                fillstyle='full'
            )
        self.ax.plot(
            x, y,
            marker=config['marker'],
            color=config['color'],
            markersize=config['size'],
            markeredgewidth=config['edge_width'],
            fillstyle='none' if not config['fill'] else 'full'
        )
        if typ == 1 and sign is not None:
            sign_text = "+" if sign == 1 else "-"
            self.ax.text(
                x + 10, y + 10, sign_text,
                fontsize=12,
                color=config['color'],
                fontweight='bold'
            )

    def _draw_curvature_comb(self, curve_points, curvature):
        """Draw a closed curvature comb"""
        if len(curve_points) == 0 or len(curvature) == 0:
            return

        dx = np.gradient(curve_points[:, 0])
        dy = np.gradient(curve_points[:, 1])
        norm = np.sqrt(dx ** 2 + dy ** 2)
        norm[norm < 1e-10] = 1e-10



    def _draw_curvature_comb(self, curve_points, curvature, color):
        if len(curve_points) == 0 or len(curvature) == 0:
            return
        dx = np.gradient(curve_points[:, 0])
        dy = np.gradient(curve_points[:, 1])
        norm = np.sqrt(dx ** 2 + dy ** 2)
        norm[norm < 1e-10] = 1e-10
        tx = dx / norm
        ty = dy / norm
        nx = ty
        ny = -tx
        scale = STYLE_CONFIG['curvature_comb']['scale']
        step = STYLE_CONFIG['curvature_comb']['density']
        line_width = STYLE_CONFIG['curvature_comb']['line_width']
        sample_indices = range(0, len(curve_points), step)
        if len(sample_indices) < 2:
            return
        comb_endpoints = []
        for i in sample_indices:
            x_curve, y_curve = curve_points[i]
            curv_length = curvature[i] * scale
            x_end = x_curve + nx[i] * curv_length
            y_end = y_curve + ny[i] * curv_length
            comb_endpoints.append([x_end, y_end])
            self.ax.plot(
                [x_curve, x_end], [y_curve, y_end],
                color=color,
                linewidth=line_width,
                alpha=0.8
            )
        comb_endpoints = np.array(comb_endpoints)
        self.ax.plot(
            comb_endpoints[:, 0], comb_endpoints[:, 1],
            color=color,
            linewidth=line_width + 0.5,
            linestyle='-',
            alpha=0.9
        )
        if self.is_closed and len(comb_endpoints) > 2:
            self.ax.plot(
                [comb_endpoints[-1, 0], comb_endpoints[0, 0]],
                [comb_endpoints[-1, 1], comb_endpoints[0, 1]],
                color=color,
                linewidth=line_width + 0.5,
                alpha=0.9
            )


    def _redraw(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()

        self.ax.set_xlim(0, 1200)
        self.ax.set_ylim(0, 1200)
        self.ax.set_aspect('auto')

        # Loading model drawing
        if self.in_model_mode and self.current_model:
            for curve in self.current_model.curves:
                p = curve.points
                index = curve.index
                pa_a = curve.pa_a
                is_closed = curve.is_closed
                color = curve.color
                # inflection_signs = [-1] * len(index)
                inflection_signs = [1] * len(index)

                # Generate curve
                if is_closed:
                    curve_points, c0, c1, c2, param_a = generate_closed_curve(p, index, pa_a, inflection_signs)
                else:
                    curve_points, c0, c1, c2, param_a = generate_open_curve(p, index, pa_a, inflection_signs)

                if len(curve_points) > 0:
                    # Draw curve
                    self.ax.plot(
                        curve_points[:, 0], curve_points[:, 1],
                        color=color,
                        linewidth=STYLE_CONFIG['curve']['line_width']
                    )


                    # Draw control polygon
                    if self.show_control_polygon and not self.show_only_curve and len(p) > 1:
                        for i in range(len(c0)):
                            c0_i = c0[i]
                            c1_i = c1[i]
                            c2_i = c2[i]
                            pa = param_a[i]
                            p2_x = (1 - pa) * c0_i[0] + pa * c1_i[0]
                            p2_y = (1 - pa) * c0_i[1] + pa * c1_i[1]
                            p3_x = (1 - pa) * c2_i[0] + pa * c1_i[0]
                            p3_y = (1 - pa) * c2_i[1] + pa * c1_i[1]
                            seg_points_x = [c0_i[0], p2_x, p3_x, c2_i[0]]
                            seg_points_y = [c0_i[1], p2_y, p3_y, c2_i[1]]
                            self.ax.plot(
                                seg_points_x, seg_points_y,
                                color='c',
                                marker='o',
                                linestyle='--',
                                linewidth=STYLE_CONFIG['polygon']['line_width'],
                                markersize=4,
                                alpha=STYLE_CONFIG['polygon']['alpha']
                            )
                        if is_closed and len(c0) > 0:
                            last_end_x, last_end_y = c2[-1][0], c2[-1][1]
                            first_start_x, first_start_y = c0[0][0], c0[0][1]
                            self.ax.plot(
                                [last_end_x, first_start_x],
                                [last_end_y, first_start_y],
                                color=color,
                                linestyle='-',
                                linewidth=STYLE_CONFIG['polygon']['line_width'],
                                alpha=STYLE_CONFIG['polygon']['alpha']
                            )

                    # Draw control points
                    if not self.show_only_curve and len(p) > 0:
                        num_points = len(p)
                        for i, (x, y) in enumerate(p):
                            if not is_closed:
                                if i == 0 or i == num_points - 1:
                                    config = STYLE_CONFIG['open_end']
                                    self.ax.plot(
                                        x, y, marker=config['marker'], color=config['color'],
                                        markersize=config['size'], markeredgewidth=config['edge_width'],
                                        fillstyle='none'
                                    )

                                else:
                                    index_idx = i - 1
                                    if 0 <= index_idx < len(index):
                                        typ = index[index_idx]
                                    else:
                                        typ = 3  # default:Regular point
                                    self._draw_point(x, y, typ, sign=inflection_signs[index_idx] if typ == 1 else None)
                                # if closed curve
                            else:
                                typ = index[i] if i < len(index) else 3
                                self._draw_point(x, y, typ, sign=inflection_signs[i] if typ == 1 else None)
        # Mouse click to draw
        else:
            curve_drawn = False
            if len(self.points) >= 2:
                p = np.array(self.points)
                if self.is_closed:
                    idx = np.array(self.index[:len(p)]) if len(self.index) >= len(p) else np.array([3] * len(p))
                    signs = np.array(self.inflection_signs[:len(p)]) if len(self.inflection_signs) >= len(p) else np.array([-1] * len(p))
                    self.curve_points, self.bezier_c0, self.bezier_c1, self.bezier_c2, self.param_a = generate_closed_curve(p, idx, self.pa_a, signs)
                else:
                    self.curve_points, self.bezier_c0, self.bezier_c1, self.bezier_c2, self.param_a = generate_open_curve(p, self.index, self.pa_a, self.inflection_signs)

                if len(self.curve_points) > 0:
                    curve_segments = list(zip(self.bezier_c0, self.bezier_c1, self.bezier_c2))

                    if len(curve_segments) > 0 and len(self.param_a) > 0 and len(curve_segments) == len(self.param_a):
                        self.curvature = compute_curvature_analytical(curve_segments, self.param_a)
                    else:
                        self.curvature = np.array([])
                    self.ax.plot(
                        self.curve_points[:, 0],
                        self.curve_points[:, 1],
                        color=STYLE_CONFIG['curve']['color'],
                        linewidth=STYLE_CONFIG['curve']['line_width'],
                        label="Curve design (Click to add points, drag to adjust, right-click to select points"
                    )
                    curve_drawn = True

            if self.show_curvature_comb and len(self.curve_points) > 0 and len(self.curvature) > 0:
                self._draw_curvature_comb(self.curve_points, self.curvature, STYLE_CONFIG['curvature_comb']['color'])

            if self.show_control_polygon and len(self.points) > 1 and not self.show_only_curve:
                if (len(self.bezier_c0) > 0 and len(self.bezier_c1) > 0 and len(self.bezier_c2) > 0 and self.param_a is not None):
                    for i in range(len(self.bezier_c0)):
                        c0_i = self.bezier_c0[i]
                        c1_i = self.bezier_c1[i]
                        c2_i = self.bezier_c2[i]
                        pa = self.param_a[i]
                        p2_x = (1 - pa) * c0_i[0] + pa * c1_i[0]
                        p2_y = (1 - pa) * c0_i[1] + pa * c1_i[1]
                        p3_x = (1 - pa) * c2_i[0] + pa * c1_i[0]
                        p3_y = (1 - pa) * c2_i[1] + pa * c1_i[1]
                        seg_points_x = [c0_i[0], p2_x, p3_x, c2_i[0]]
                        seg_points_y = [c0_i[1], p2_y, p3_y, c2_i[1]]
                        self.ax.plot(
                            seg_points_x, seg_points_y,
                            color='c',
                            marker='o',
                            linestyle='--',
                            linewidth=STYLE_CONFIG['polygon']['line_width'],
                            markersize=4,
                            alpha=STYLE_CONFIG['polygon']['alpha'],
                            label="Control polygon" if i == 0 else ""
                        )
                    if self.is_closed and len(self.bezier_c0) > 0:
                        last_end_x, last_end_y = self.bezier_c2[-1][0], self.bezier_c2[-1][1]
                        first_start_x, first_start_y = self.bezier_c0[0][0], self.bezier_c0[0][1]
                        self.ax.plot(
                            [last_end_x, first_start_x],
                            [last_end_y, first_start_y],
                            color='c',
                            linestyle='-',
                            linewidth=STYLE_CONFIG['polygon']['line_width'],
                            alpha=STYLE_CONFIG['polygon']['alpha']
                        )

            if not self.show_only_curve and self.points:
                legend_markers = {}
                for i, (x, y) in enumerate(self.points):
                    is_selected = (i == self.selected_point)
                    if not self.is_closed and (i == 0 or i == len(self.points) - 1):
                        config = STYLE_CONFIG['open_end']
                        self.ax.plot(
                            x, y,
                            marker=config['marker'],
                            color=config['color'],
                            markersize=config['size'],
                            markeredgewidth=config['edge_width'],
                            fillstyle='none'
                        )
                        if 'Open curve endpoint' not in legend_markers:
                            legend_markers['Open curve endpoint'] = (config['marker'], config['color'])
                    else:
                        if self.is_closed:
                            typ = self.index[i] if i < len(self.index) else 3
                            sign = self.inflection_signs[i] if i < len(self.inflection_signs) else -1
                        else:
                            actual_idx = i - 1
                            typ = self.index[actual_idx] if actual_idx < len(self.index) and actual_idx >= 0 else 3
                            sign = self.inflection_signs[actual_idx] if actual_idx < len(self.inflection_signs) and actual_idx >= 0 else -1
                        self._draw_point(x, y, typ, is_selected, sign if typ == 1 else None)
                        type_name = self._get_type_name(typ)
                        if type_name not in legend_markers:
                            legend_markers[type_name] = (STYLE_CONFIG[self._get_type_key(typ)]['marker'], STYLE_CONFIG[self._get_type_key(typ)]['color'])
                handles = []
                labels = []
                for label, (marker, color) in legend_markers.items():
                    handles.append(plt.Line2D([], [], marker=marker, color='w', markerfacecolor='w',
                                              markeredgecolor=color, markersize=8, label=label))
                    labels.append(label)

            handles_comb = []
            labels_comb = []
            if curve_drawn:
                handles_comb.append(plt.Line2D([], [], color=STYLE_CONFIG['curve']['color'],
                                               linewidth=STYLE_CONFIG['curve']['line_width'], label="Curve"))
                labels_comb.append("Curve")
            if self.show_control_polygon and len(self.points) > 1 and not self.show_only_curve:
                handles_comb.append(plt.Line2D([], [], color=STYLE_CONFIG['polygon']['color'],
                                               linewidth=STYLE_CONFIG['polygon']['line_width'],
                                               linestyle=STYLE_CONFIG['polygon']['linestyle'],
                                               label="Control polygon"))
                labels_comb.append("Control polygon")
            if self.show_curvature_comb and len(self.curve_points) > 0 and len(self.curvature) > 0:
                handles_comb.append(plt.Line2D([], [], color=STYLE_CONFIG['curvature_comb']['color'],
                                               linewidth=STYLE_CONFIG['curvature_comb']['line_width'],
                                               label="Curvature comb"))
                labels_comb.append("Curvature comb")
            if handles_comb:
                self.ax.legend(handles_comb, labels_comb, loc='upper right', framealpha=0.9)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('auto')
        self.ax.set_title("Curve design (click to add points, drag to adjust, right-click to select points)")
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.canvas.draw()

    def _get_type_key(self, typ):
        if typ == 0:
            return 'cusp'
        elif typ == 1:
            return 'inflection'
        elif typ == 2:
            return 'loop'
        else:
            return 'regular'

# -------------------------- Main --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FCaCurveDesigner(root)
    root.mainloop()