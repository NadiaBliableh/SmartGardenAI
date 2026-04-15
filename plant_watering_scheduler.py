"""
Smart Plant Watering Scheduler
================================
Pure Python only — no numpy, pandas, matplotlib, or any third-party library.
Only standard-library modules are used:
    tkinter  (built-in GUI)
    csv      (read Excel-exported CSV or the xlsx via manual parsing)
    math     (sqrt, exp)
    random   (shuffle, sample, random)
    openpyxl is NOT used — we read the xlsx with the built-in zipfile + xml.etree
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import math
import random
import zipfile
import xml.etree.ElementTree as ET
import os

# ─────────────────────────────────────────────
#  PURE-PYTHON HELPERS  (no numpy)
# ─────────────────────────────────────────────

def dot(w, x):
    """Dot product of two lists."""
    return sum(wi * xi for wi, xi in zip(w, x))

def vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def vec_scale(v, s):
    return [vi * s for vi in v]

def mean_list(lst):
    return sum(lst) / len(lst) if lst else 0.0

def std_list(lst):
    m = mean_list(lst)
    var = sum((x - m) ** 2 for x in lst) / len(lst) if lst else 0.0
    return math.sqrt(var)

def normalise_dataset(rows):
    """
    rows : list of lists  [[f0,f1,f2], ...]
    Returns normalised rows + (means, stds) for later use.
    """
    n_feat = len(rows[0])
    cols   = [[r[i] for r in rows] for i in range(n_feat)]
    means  = [mean_list(c) for c in cols]
    stds   = [std_list(c) + 1e-8 for c in cols]
    normed = [[(rows[r][i] - means[i]) / stds[i]
               for i in range(n_feat)]
              for r in range(len(rows))]
    return normed, means, stds

def normalise_one(feat, means, stds):
    return [(feat[i] - means[i]) / stds[i] for i in range(len(feat))]

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# ─────────────────────────────────────────────
#  XLSX READER  (pure stdlib: zipfile + xml)
# ─────────────────────────────────────────────

def read_xlsx(path):
    """
    Returns list of dicts.  Reads the first sheet only.
    Values are returned as strings; caller converts.
    """
    rows = []
    with zipfile.ZipFile(path) as zf:
        # shared strings
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            tree = ET.parse(zf.open("xl/sharedStrings.xml"))
            ns   = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in tree.getroot().findall(".//s:si", ns):
                t_nodes = si.findall(".//s:t", ns)
                shared.append("".join((t.text or "") for t in t_nodes))

        # first sheet
        sheet_name = "xl/worksheets/sheet1.xml"
        tree  = ET.parse(zf.open(sheet_name))
        ns    = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheet_rows = tree.getroot().findall(".//s:row", ns)

        header = None
        for row_el in sheet_rows:
            cells = {}
            for c in row_el.findall("s:c", ns):
                ref  = c.get("r")             # e.g. A1
                col  = ''.join(ch for ch in ref if ch.isalpha())
                t    = c.get("t", "n")        # type
                v_el = c.find("s:v", ns)
                if v_el is None or v_el.text is None:
                    cells[col] = ""
                    continue
                if t == "s":                  # shared string
                    cells[col] = shared[int(v_el.text)]
                else:
                    cells[col] = v_el.text

            if header is None:
                header = list(cells.values())
            else:
                row_dict = {header[i]: list(cells.values())[i]
                            if i < len(list(cells.values())) else ""
                            for i in range(len(header))}
                rows.append(row_dict)
    return rows


# ─────────────────────────────────────────────
#  PERCEPTRON  (pure Python)
# ─────────────────────────────────────────────

class Perceptron:
    def __init__(self, lr=0.1, epochs=50):
        self.lr      = lr
        self.epochs  = epochs
        self.weights = None
        self.bias    = 0.0
        self.loss_history = []   # errors per epoch
        self.acc_history  = []   # accuracy per epoch

    def _step(self, val):
        return 1 if val >= 0 else 0

    def predict_one(self, x):
        return self._step(dot(self.weights, x) + self.bias)

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def fit(self, X, y):
        n = len(X[0])
        self.weights      = [0.0] * n
        self.bias         = 0.0
        self.loss_history = []
        self.acc_history  = []

        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred  = self.predict_one(xi)
                delta = self.lr * (yi - pred)
                self.weights = vec_add(self.weights, vec_scale(xi, delta))
                self.bias   += delta
                errors += int(pred != yi)
            self.loss_history.append(errors)
            preds = self.predict(X)
            correct = sum(p == t for p, t in zip(preds, y))
            self.acc_history.append(correct / len(y))


# ─────────────────────────────────────────────
#  SIMULATED ANNEALING  (pure Python)
# ─────────────────────────────────────────────

def sa_cost(sequence, plants, predictions):
    needs_water = set(i for i, p in enumerate(predictions) if p == 1)
    seq_set     = set(sequence)

    missed  = len(needs_water - seq_set)
    extra   = len(seq_set - needs_water)
    dist    = 0.0
    for k in range(len(sequence) - 1):
        dist += euclidean(plants[sequence[k]]['pos'],
                          plants[sequence[k+1]]['pos'])
    return missed + dist + extra

def simulated_annealing(sequence, plants, predictions,
                        T=100.0, cooling=0.95, iterations=500):
    seq          = sequence[:]
    current_cost = sa_cost(seq, plants, predictions)
    best_seq     = seq[:]
    best_cost    = current_cost
    history      = [current_cost]

    for _ in range(iterations):
        if len(seq) < 2:
            break
        i, j = random.sample(range(len(seq)), 2)
        new_seq      = seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_cost = sa_cost(new_seq, plants, predictions)
        delta    = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            seq          = new_seq
            current_cost = new_cost
            if current_cost < best_cost:
                best_seq  = seq[:]
                best_cost = current_cost

        T *= cooling
        history.append(current_cost)

    return best_seq, best_cost, history


# ─────────────────────────────────────────────
#  PURE-PYTHON CANVAS CHART HELPERS
# ─────────────────────────────────────────────

def draw_line_chart(canvas, data, title,
                    x0=50, y0=10, x1=None, y1=None,
                    line_color="#66bb6a", label_y=""):
    """Draw a simple line chart on a tk.Canvas (no matplotlib)."""
    w = int(canvas["width"])
    h = int(canvas["height"])
    x1 = x1 or w - 20
    y1 = y1 or h - 30

    canvas.delete("all")
    # background
    canvas.create_rectangle(0, 0, w, h, fill="#253225", outline="")
    # axes
    canvas.create_line(x0, y0, x0, y1, fill="#4caf50", width=2)
    canvas.create_line(x0, y1, x1, y1, fill="#4caf50", width=2)
    # title
    canvas.create_text(w//2, 6, text=title, fill="#e8f5e9",
                       font=("Segoe UI", 9, "bold"), anchor="n")
    if not data:
        return

    mn  = min(data)
    mx  = max(data) if max(data) != mn else mn + 1
    n   = len(data)
    cw  = (x1 - x0)
    ch  = (y1 - y0)

    def px(i):   return x0 + int(i / max(n-1,1) * cw)
    def py(val): return y1 - int((val - mn) / (mx - mn) * ch)

    # grid lines
    for step in range(0, 5):
        gy = y0 + int(step / 4 * ch)
        canvas.create_line(x0, gy, x1, gy, fill="#2a3d2a", dash=(3,3))
        val = mx - step / 4 * (mx - mn)
        canvas.create_text(x0-4, gy, text=f"{val:.1f}",
                           fill="#aed581", font=("Segoe UI",7), anchor="e")

    # line
    pts = [(px(i), py(v)) for i, v in enumerate(data)]
    for k in range(len(pts)-1):
        canvas.create_line(pts[k][0], pts[k][1],
                           pts[k+1][0], pts[k+1][1],
                           fill=line_color, width=2)
    # dots at start/end
    for pt in [pts[0], pts[-1]]:
        canvas.create_oval(pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3,
                           fill=line_color, outline="white")

    # x labels
    for i in [0, n//2, n-1]:
        canvas.create_text(px(i), y1+10, text=str(i),
                           fill="#aed581", font=("Segoe UI",7))

    canvas.create_text(x0-30, (y0+y1)//2, text=label_y,
                       fill="#aed581", font=("Segoe UI",7), angle=90)


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class PlantWateringApp(tk.Tk):

    BG   = "#1e2a1e"
    CARD = "#253225"
    ACC  = "#4caf50"
    TXT  = "#e8f5e9"
    BTN  = "#388e3c"

    def __init__(self):
        super().__init__()
        self.title("🌿 Smart Plant Watering Scheduler  |  Pure Python")
        self.geometry("1250x780")
        self.configure(bg=self.BG)
        self.resizable(True, True)

        # state
        self.plants      = []
        self.perceptron  = Perceptron()
        self.trained     = False
        self.X_mean      = []
        self.X_std       = []
        self.predictions = []
        self.optimal_seq = []
        self.placing_mode = False

        self._build_ui()
        self._auto_train()

    # ══════════════════════════════════════════
    #  BUILD UI
    # ══════════════════════════════════════════
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        BG, CARD, ACC, TXT, BTN = (self.BG, self.CARD, self.ACC, self.TXT, self.BTN)

        style.configure("TNotebook",         background=BG,   borderwidth=0)
        style.configure("TNotebook.Tab",     background=CARD, foreground=TXT,
                        padding=[12,6], font=("Segoe UI",10,"bold"))
        style.map("TNotebook.Tab",           background=[("selected", ACC)],
                                             foreground=[("selected","#000")])
        style.configure("TFrame",            background=BG)
        style.configure("TLabel",            background=BG, foreground=TXT,
                        font=("Segoe UI",10))
        style.configure("TButton",           background=BTN, foreground=TXT,
                        font=("Segoe UI",10,"bold"), borderwidth=0, padding=6)
        style.map("TButton",                 background=[("active","#2e7d32")])
        style.configure("Treeview",          background=CARD, foreground=TXT,
                        fieldbackground=CARD, rowheight=24)
        style.configure("Treeview.Heading",  background=BTN, foreground=TXT,
                        font=("Segoe UI",9,"bold"))
        style.map("Treeview",                background=[("selected","#4caf50")],
                                             foreground=[("selected","#000")])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_garden  = ttk.Frame(nb)
        self.tab_percept = ttk.Frame(nb)
        self.tab_sa      = ttk.Frame(nb)

        nb.add(self.tab_garden,  text="🌱  Garden")
        nb.add(self.tab_percept, text="🧠  Perceptron")
        nb.add(self.tab_sa,      text="🔄  SA Optimizer")

        self._build_garden_tab()
        self._build_perceptron_tab()
        self._build_sa_tab()

    # ──────────────────────────────────────────
    #  TAB 1 — GARDEN
    # ──────────────────────────────────────────
    def _build_garden_tab(self):
        BG, CARD, TXT, ACC = self.BG, self.CARD, self.TXT, self.ACC
        f = self.tab_garden

        left = tk.Frame(f, bg=CARD, width=265)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="➕  Add Plant", bg=CARD, fg=ACC,
                 font=("Segoe UI",12,"bold")).pack(pady=(12,4))

        def lbl(t):
            tk.Label(left, text=t, bg=CARD, fg=TXT,
                     font=("Segoe UI",9)).pack(anchor="w", padx=10, pady=(6,0))

        lbl("Plant Name")
        self.e_name = tk.Entry(left, bg="#2e3d2e", fg=TXT, insertbackground=TXT,
                               font=("Segoe UI",10), relief="flat")
        self.e_name.pack(fill="x", padx=10)
        self.e_name.insert(0, "Plant A")

        lbl("Soil Moisture  (0 – 100)")
        self.sl_moist = tk.Scale(left, from_=0, to=100, orient="horizontal",
                                 bg=CARD, fg=TXT, troughcolor=ACC,
                                 highlightthickness=0, activebackground="#81c784")
        self.sl_moist.set(30)
        self.sl_moist.pack(fill="x", padx=10)

        lbl("Last Watered  (hours ago, 0 – 48)")
        self.sl_last = tk.Scale(left, from_=0, to=48, orient="horizontal",
                                bg=CARD, fg=TXT, troughcolor=ACC,
                                highlightthickness=0, activebackground="#81c784")
        self.sl_last.set(12)
        self.sl_last.pack(fill="x", padx=10)

        lbl("Plant Type")
        self.v_type = tk.StringVar(value="0")
        frm_t = tk.Frame(left, bg=CARD)
        frm_t.pack(fill="x", padx=10, pady=4)
        for lbl_t, val in [("Cactus 🌵","0"),("Flower 🌸","1"),("Herb 🌿","2")]:
            tk.Radiobutton(frm_t, text=lbl_t, variable=self.v_type, value=val,
                           bg=CARD, fg=TXT, selectcolor=ACC,
                           font=("Segoe UI",9), activebackground=CARD).pack(anchor="w")

        tk.Label(left, text="→ Click map to place",
                 bg=CARD, fg="#aed581",
                 font=("Segoe UI",9,"italic")).pack(pady=4)

        self.btn_place = tk.Button(left, text="📍  Click to Place Plant",
                                   bg="#1565c0", fg="white",
                                   font=("Segoe UI",10,"bold"), relief="flat",
                                   command=self._toggle_place)
        self.btn_place.pack(fill="x", padx=10, pady=3)

        tk.Button(left, text="🗑  Clear All",
                  bg="#c62828", fg="white",
                  font=("Segoe UI",10,"bold"), relief="flat",
                  command=self._clear_plants).pack(fill="x", padx=10, pady=3)

        # plant list
        tk.Label(left, text="Plants in Garden", bg=CARD, fg=ACC,
                 font=("Segoe UI",10,"bold")).pack(pady=(8,2))
        cols = ("Name","Moist","Hrs","Type","💧")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=9)
        for c, w in zip(cols, [68,40,40,48,30]):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=4)

        # garden canvas
        right = tk.Frame(f, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(4,8), pady=8)

        tk.Label(right, text="Garden Map — click to place plants",
                 bg=BG, fg=ACC, font=("Segoe UI",11,"bold")).pack(pady=(4,2))

        self.canvas = tk.Canvas(right, bg="#1b2e1b", cursor="crosshair",
                                highlightthickness=1, highlightbackground=ACC)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        self.lbl_status = tk.Label(right, text="⏳  Loading…",
                                   bg=BG, fg="#aed581",
                                   font=("Segoe UI",9,"italic"))
        self.lbl_status.pack(pady=4)

    # ──────────────────────────────────────────
    #  TAB 2 — PERCEPTRON
    # ──────────────────────────────────────────
    def _build_perceptron_tab(self):
        BG, CARD, TXT, ACC = self.BG, self.CARD, self.TXT, self.ACC
        f = self.tab_percept

        top = tk.Frame(f, bg=BG)
        top.pack(fill="both", expand=True, padx=10, pady=6)

        # --- left controls ---
        ctrl = tk.Frame(top, bg=CARD, width=240)
        ctrl.pack(side="left", fill="y", padx=(0,8))
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="Perceptron Settings", bg=CARD, fg=ACC,
                 font=("Segoe UI",11,"bold")).pack(pady=(12,6))

        def row(lbl_t, default):
            frm = tk.Frame(ctrl, bg=CARD); frm.pack(fill="x", padx=10, pady=3)
            tk.Label(frm, text=lbl_t, bg=CARD, fg=TXT,
                     font=("Segoe UI",9), width=16, anchor="w").pack(side="left")
            e = tk.Entry(frm, width=8, bg="#2e3d2e", fg=TXT,
                         insertbackground=TXT, font=("Segoe UI",10), relief="flat")
            e.insert(0, default)
            e.pack(side="left", padx=4)
            return e

        self.e_lr = row("Learning Rate", "0.1")
        self.e_ep = row("Epochs",        "50")

        tk.Button(ctrl, text="🔁  Re-Train Perceptron",
                  bg=self.BTN, fg="white",
                  font=("Segoe UI",10,"bold"), relief="flat",
                  command=self._retrain).pack(fill="x", padx=10, pady=8)

        self.lbl_acc = tk.Label(ctrl, text="Accuracy: —",
                                bg=CARD, fg="#aed581",
                                font=("Segoe UI",11,"bold"))
        self.lbl_acc.pack(pady=4)

        self.lbl_wts = tk.Label(ctrl, text="Weights:\n  w1=—  w2=—  w3=—\nBias: —",
                                bg=CARD, fg=TXT,
                                font=("Segoe UI",9), justify="left")
        self.lbl_wts.pack(pady=4, padx=10, anchor="w")

        # separator
        tk.Frame(ctrl, bg="#4caf50", height=1).pack(fill="x", padx=10, pady=8)

        # test area
        tk.Label(ctrl, text="🔍  Test Perceptron", bg=CARD, fg=ACC,
                 font=("Segoe UI",10,"bold")).pack(pady=(0,4))

        self.test_entries = []
        for lbl_t, def_v in [("Moisture (0-100)","50"),
                              ("Hours ago (0-48)","20"),
                              ("Type (0/1/2)",    "1")]:
            frm = tk.Frame(ctrl, bg=CARD); frm.pack(fill="x", padx=10, pady=2)
            tk.Label(frm, text=lbl_t, bg=CARD, fg=TXT,
                     font=("Segoe UI",8), width=16, anchor="w").pack(side="left")
            e = tk.Entry(frm, width=6, bg="#2e3d2e", fg=TXT,
                         insertbackground=TXT, font=("Segoe UI",9), relief="flat")
            e.insert(0, def_v)
            e.pack(side="left", padx=2)
            self.test_entries.append(e)

        tk.Button(ctrl, text="Predict →",
                  bg="#1565c0", fg="white",
                  font=("Segoe UI",10,"bold"), relief="flat",
                  command=self._test_perceptron).pack(fill="x", padx=10, pady=6)

        self.lbl_pred = tk.Label(ctrl, text="Result: —",
                                 bg=CARD, fg="#fff176",
                                 font=("Segoe UI",11,"bold"))
        self.lbl_pred.pack(pady=2)

        # --- right charts (pure tk.Canvas) ---
        chart_frame = tk.Frame(top, bg=BG)
        chart_frame.pack(side="left", fill="both", expand=True)

        self.chart_loss = tk.Canvas(chart_frame, bg="#253225",
                                    width=320, height=260,
                                    highlightthickness=1,
                                    highlightbackground="#4caf50")
        self.chart_loss.pack(side="left", fill="both", expand=True, padx=(0,6))

        self.chart_acc = tk.Canvas(chart_frame, bg="#253225",
                                   width=320, height=260,
                                   highlightthickness=1,
                                   highlightbackground="#4caf50")
        self.chart_acc.pack(side="left", fill="both", expand=True)

    # ──────────────────────────────────────────
    #  TAB 3 — SA
    # ──────────────────────────────────────────
    def _build_sa_tab(self):
        BG, CARD, TXT, ACC = self.BG, self.CARD, self.TXT, self.ACC
        f = self.tab_sa

        top = tk.Frame(f, bg=BG)
        top.pack(fill="both", expand=True, padx=10, pady=6)

        # --- left controls ---
        ctrl = tk.Frame(top, bg=CARD, width=240)
        ctrl.pack(side="left", fill="y", padx=(0,8))
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="SA Settings", bg=CARD, fg=ACC,
                 font=("Segoe UI",11,"bold")).pack(pady=(12,6))

        def row(lbl_t, default):
            frm = tk.Frame(ctrl, bg=CARD); frm.pack(fill="x", padx=10, pady=3)
            tk.Label(frm, text=lbl_t, bg=CARD, fg=TXT,
                     font=("Segoe UI",9), width=16, anchor="w").pack(side="left")
            e = tk.Entry(frm, width=8, bg="#2e3d2e", fg=TXT,
                         insertbackground=TXT, font=("Segoe UI",10), relief="flat")
            e.insert(0, default)
            e.pack(side="left", padx=4)
            return e

        self.e_T    = row("Initial Temp",  "100")
        self.e_cool = row("Cooling Rate",  "0.95")
        self.e_iter = row("Iterations",    "500")

        tk.Button(ctrl, text="🚀  Run SA Optimizer",
                  bg=self.BTN, fg="white",
                  font=("Segoe UI",10,"bold"), relief="flat",
                  command=self._run_sa).pack(fill="x", padx=10, pady=8)

        self.lbl_sa_cost = tk.Label(ctrl, text="Best Cost: —",
                                    bg=CARD, fg="#fff176",
                                    font=("Segoe UI",12,"bold"))
        self.lbl_sa_cost.pack(pady=4)

        tk.Frame(ctrl, bg="#4caf50", height=1).pack(fill="x", padx=10, pady=6)

        tk.Label(ctrl, text="Optimal Watering Order:", bg=CARD, fg=ACC,
                 font=("Segoe UI",9,"bold")).pack(anchor="w", padx=10)

        self.txt_order = tk.Text(ctrl, height=12, width=22,
                                 bg="#2e3d2e", fg="#aed581",
                                 font=("Segoe UI",9), relief="flat",
                                 state="disabled")
        self.txt_order.pack(fill="x", padx=10, pady=4)

        # --- right charts ---
        chart_frame = tk.Frame(top, bg=BG)
        chart_frame.pack(side="left", fill="both", expand=True)

        self.chart_sa_cost = tk.Canvas(chart_frame, bg="#253225",
                                       width=310, height=280,
                                       highlightthickness=1,
                                       highlightbackground="#4caf50")
        self.chart_sa_cost.pack(side="left", fill="both", expand=True, padx=(0,6))

        self.chart_path = tk.Canvas(chart_frame, bg="#1b2e1b",
                                    width=310, height=280,
                                    highlightthickness=1,
                                    highlightbackground="#4caf50")
        self.chart_path.pack(side="left", fill="both", expand=True)

    # ══════════════════════════════════════════
    #  DATA LOADING & TRAINING
    # ══════════════════════════════════════════
    def _auto_train(self):
        # try default path next to script
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data.xlsx")
        if os.path.exists(default):
            self._train_from_file(default)
        else:
            self.lbl_status.config(
                text="⚠️  Data.xlsx not found — use Re-Train to load manually")

    def _train_from_file(self, path):
        try:
            raw = read_xlsx(path)
            X_raw, y = [], []
            for r in raw:
                try:
                    x1 = float(r.get("soil_moisture", 0))
                    x2 = float(r.get("last_watered",  0))
                    x3 = float(r.get("plant_type",    0))
                    yi = int(float(r.get("needs_water", 0)))
                    X_raw.append([x1, x2, x3])
                    y.append(yi)
                except (ValueError, TypeError):
                    continue

            if not X_raw:
                raise ValueError("No valid rows found in xlsx")

            X_norm, self.X_mean, self.X_std = normalise_dataset(X_raw)

            # 80/20 split (shuffle indices manually)
            indices = list(range(len(X_norm)))
            random.shuffle(indices)
            split   = int(0.8 * len(indices))
            tr, te  = indices[:split], indices[split:]

            X_tr = [X_norm[i] for i in tr]; y_tr = [y[i] for i in tr]
            X_te = [X_norm[i] for i in te]; y_te = [y[i] for i in te]

            self.perceptron.fit(X_tr, y_tr)
            preds_te = self.perceptron.predict(X_te)
            acc = sum(p == t for p, t in zip(preds_te, y_te)) / len(y_te)

            self.trained = True
            self._update_perceptron_ui(acc)
            self.lbl_status.config(
                text=f"✅  Trained on {len(X_tr)} samples — Val Accuracy: {acc*100:.1f}%")
        except Exception as ex:
            messagebox.showerror("Training Error", str(ex))

    def _retrain(self):
        try:
            self.perceptron.lr     = float(self.e_lr.get())
            self.perceptron.epochs = int(self.e_ep.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid LR or Epochs value.")
            return

        path = filedialog.askopenfilename(
            title="Select Data.xlsx",
            filetypes=[("Excel files","*.xlsx"), ("All files","*.*")])
        if path:
            self._train_from_file(path)

    def _update_perceptron_ui(self, acc):
        self.lbl_acc.config(text=f"Val Accuracy: {acc*100:.1f}%")
        w = self.perceptron.weights
        self.lbl_wts.config(
            text=f"Weights:\n  w1={w[0]:.4f}\n  w2={w[1]:.4f}\n  w3={w[2]:.4f}\nBias: {self.perceptron.bias:.4f}")

        draw_line_chart(self.chart_loss, self.perceptron.loss_history,
                        "Training Loss (errors per epoch)",
                        line_color="#ef5350", label_y="Errors")
        draw_line_chart(self.chart_acc,
                        [a*100 for a in self.perceptron.acc_history],
                        "Training Accuracy (%)",
                        line_color="#66bb6a", label_y="Acc %")

    # ══════════════════════════════════════════
    #  GARDEN INTERACTION
    # ══════════════════════════════════════════
    def _toggle_place(self):
        self.placing_mode = not self.placing_mode
        if self.placing_mode:
            self.btn_place.config(bg="#e65100", text="🖱  Click map to place…")
            self.canvas.config(cursor="crosshair")
        else:
            self.btn_place.config(bg="#1565c0", text="📍  Click to Place Plant")
            self.canvas.config(cursor="arrow")

    def _on_canvas_click(self, event):
        if not self.placing_mode:
            return
        x, y     = event.x, event.y
        name     = self.e_name.get().strip() or f"Plant {len(self.plants)+1}"
        moisture = self.sl_moist.get()
        last_w   = self.sl_last.get()
        ptype    = int(self.v_type.get())
        pred     = self._predict_one(moisture, last_w, ptype)

        self.plants.append({
            "pos": (x, y), "name": name,
            "moisture": moisture, "last_watered": last_w,
            "plant_type": ptype, "pred": pred
        })

        # auto-increment name
        import re
        m = re.match(r"^(.*?)(\d+)$", name)
        if m:
            self.e_name.delete(0, "end")
            self.e_name.insert(0, m.group(1) + str(int(m.group(2))+1))

        self.placing_mode = False
        self.btn_place.config(bg="#1565c0", text="📍  Click to Place Plant")
        self.canvas.config(cursor="arrow")

        self._redraw_garden()
        self._update_tree()

    def _predict_one(self, moisture, last_w, ptype):
        if not self.trained:
            return 0
        feat = normalise_one([float(moisture), float(last_w), float(ptype)],
                             self.X_mean, self.X_std)
        return self.perceptron.predict_one(feat)

    def _update_all_predictions(self):
        for p in self.plants:
            p['pred'] = self._predict_one(p['moisture'], p['last_watered'], p['plant_type'])
        self.predictions = [p['pred'] for p in self.plants]

    def _redraw_garden(self, highlight_seq=None):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()  or 600
        h = self.canvas.winfo_height() or 500

        # grid
        for gx in range(0, w, 50):
            self.canvas.create_line(gx, 0, gx, h, fill="#2a3d2a")
        for gy in range(0, h, 50):
            self.canvas.create_line(0, gy, w, gy, fill="#2a3d2a")

        # path
        if highlight_seq and len(highlight_seq) > 1:
            for k in range(len(highlight_seq)-1):
                p1 = self.plants[highlight_seq[k]]['pos']
                p2 = self.plants[highlight_seq[k+1]]['pos']
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                        fill="#ffd54f", width=2, dash=(6,4))

        icons = {0:"🌵", 1:"🌸", 2:"🌿"}
        for i, p in enumerate(self.plants):
            px, py = p['pos']
            col = "#ef5350" if p['pred'] == 1 else "#66bb6a"
            self.canvas.create_oval(px-18, py-18, px+18, py+18,
                                    fill=col, outline="white", width=2)
            self.canvas.create_text(px, py-2, text=icons.get(p['plant_type'],'?'),
                                    font=("Segoe UI",13))
            if highlight_seq and i in highlight_seq:
                rank = highlight_seq.index(i) + 1
                self.canvas.create_text(px+16, py-16, text=str(rank),
                                        fill="#ffd54f",
                                        font=("Segoe UI",8,"bold"))
            self.canvas.create_text(px, py+28, text=p['name'],
                                    fill="white", font=("Segoe UI",8,"bold"))

        # legend
        self.canvas.create_rectangle(6, 6, 195, 58, fill="#253225", outline="#4caf50")
        self.canvas.create_oval(14,14,26,26, fill="#ef5350", outline="")
        self.canvas.create_text(115, 20, text="💧 Needs Watering",
                                fill="#ef5350", font=("Segoe UI",8,"bold"))
        self.canvas.create_oval(14,34,26,46, fill="#66bb6a", outline="")
        self.canvas.create_text(115, 40, text="✅ Does NOT Need Water",
                                fill="#66bb6a", font=("Segoe UI",8,"bold"))

    def _update_tree(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        type_map = {0:"Cactus", 1:"Flower", 2:"Herb"}
        for p in self.plants:
            self.tree.insert("", "end", values=(
                p['name'], p['moisture'], p['last_watered'],
                type_map.get(p['plant_type'],'?'),
                "💧" if p['pred']==1 else "✅"))

    def _clear_plants(self):
        self.plants.clear()
        self.predictions.clear()
        self.optimal_seq = []
        self._redraw_garden()
        self._update_tree()

    # ══════════════════════════════════════════
    #  PERCEPTRON TEST
    # ══════════════════════════════════════════
    def _test_perceptron(self):
        if not self.trained:
            messagebox.showwarning("Not Trained", "Load data and train first!")
            return
        try:
            vals = [float(e.get()) for e in self.test_entries]
        except ValueError:
            messagebox.showerror("Input Error", "Enter valid numbers.")
            return
        pred = self._predict_one(*vals)
        result = "💧 Needs Watering" if pred == 1 else "✅ Does NOT Need Water"
        self.lbl_pred.config(text=f"Result: {result}")

    # ══════════════════════════════════════════
    #  SIMULATED ANNEALING
    # ══════════════════════════════════════════
    def _run_sa(self):
        if len(self.plants) < 2:
            messagebox.showwarning("Too few plants",
                                   "Add at least 2 plants to the garden first!")
            return
        if not self.trained:
            messagebox.showwarning("Not Trained", "Train the Perceptron first!")
            return
        try:
            T    = float(self.e_T.get())
            cool = float(self.e_cool.get())
            itr  = int(self.e_iter.get())
        except ValueError:
            messagebox.showerror("Input Error", "Invalid SA parameter.")
            return

        self._update_all_predictions()

        seq = list(range(len(self.plants)))
        random.shuffle(seq)

        best_seq, best_cost, history = simulated_annealing(
            seq, self.plants, self.predictions,
            T=T, cooling=cool, iterations=itr)

        self.optimal_seq = best_seq

        self.lbl_sa_cost.config(text=f"Best Cost: {best_cost:.2f}")

        # order text
        self.txt_order.config(state="normal")
        self.txt_order.delete("1.0", "end")
        icons = {0:"🌵", 1:"🌸", 2:"🌿"}
        for rank, idx in enumerate(best_seq, 1):
            p    = self.plants[idx]
            icon = icons.get(p['plant_type'], '?')
            need = "💧" if p['pred'] == 1 else "✅"
            self.txt_order.insert("end", f"{rank}. {icon} {p['name']} {need}\n")
        self.txt_order.config(state="disabled")

        # redraw garden with path
        self._redraw_garden(highlight_seq=best_seq)

        # SA convergence chart
        draw_line_chart(self.chart_sa_cost, history,
                        "SA Cost Convergence",
                        line_color="#ffa726", label_y="Cost")

        # watering path mini-map
        self._draw_path_minimap(best_seq)

    def _draw_path_minimap(self, seq):
        canvas = self.chart_path
        canvas.delete("all")
        cw = int(canvas["width"])
        ch = int(canvas["height"])
        canvas.create_rectangle(0, 0, cw, ch, fill="#1b2e1b", outline="")
        canvas.create_text(cw//2, 10, text="Watering Path (minimap)",
                           fill="#e8f5e9", font=("Segoe UI",9,"bold"))

        if not self.plants:
            return

        # scale positions to fit canvas
        all_x = [p['pos'][0] for p in self.plants]
        all_y = [p['pos'][1] for p in self.plants]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        rng_x = max(max_x - min_x, 1)
        rng_y = max(max_y - min_y, 1)
        PAD = 30

        def sx(x): return PAD + int((x-min_x)/rng_x*(cw-2*PAD))
        def sy(y): return PAD + int((y-min_y)/rng_y*(ch-2*PAD))

        # path lines
        if len(seq) > 1:
            for k in range(len(seq)-1):
                p1 = self.plants[seq[k]]['pos']
                p2 = self.plants[seq[k+1]]['pos']
                canvas.create_line(sx(p1[0]), sy(p1[1]),
                                   sx(p2[0]), sy(p2[1]),
                                   fill="#ffd54f", width=2, dash=(5,3))

        icons = {0:"🌵", 1:"🌸", 2:"🌿"}
        for rank, idx in enumerate(seq):
            p  = self.plants[idx]
            px, py = sx(p['pos'][0]), sy(p['pos'][1])
            col = "#ef5350" if p['pred'] == 1 else "#66bb6a"
            canvas.create_oval(px-10, py-10, px+10, py+10,
                               fill=col, outline="white", width=1)
            canvas.create_text(px, py, text=str(rank+1),
                               fill="white", font=("Segoe UI",7,"bold"))

        # non-sequenced plants (if any)
        seq_set = set(seq)
        for i, p in enumerate(self.plants):
            if i not in seq_set:
                px, py = sx(p['pos'][0]), sy(p['pos'][1])
                canvas.create_oval(px-8, py-8, px+8, py+8,
                                   fill="#546e7a", outline="white")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = PlantWateringApp()
    app.mainloop()