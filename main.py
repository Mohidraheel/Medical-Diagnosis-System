"""
Medical Diagnosis System
FAST-NUCES Karachi — AI Section 6B
Group: Syed Suhaib Raza, Mohid Raheel Khan, Rizwan Vadsariya

Uses a Naive Bayes (Bayesian Network) classifier trained on the
Disease Symptom Dataset to predict the most likely disease
given user-selected symptoms.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ─────────────────────────────────────────────
# Resolve dataset path (same folder as script)
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "Disease_symptom_dataset.csv")

# ─────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────
BG_DARK      = "#0d1117"
BG_CARD      = "#161b22"
BG_HOVER     = "#1c2128"
ACCENT_BLUE  = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED   = "#f85149"
ACCENT_YELL  = "#e3b341"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"
BORDER       = "#30363d"
TAG_BG       = "#1f6feb"

# ─────────────────────────────────────────────
# Load & train the Bayesian model
# ─────────────────────────────────────────────
def load_and_train(path):
    df = pd.read_csv(path)
    df["prognosis"] = df["prognosis"].str.strip()
    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = BernoulliNB()
    model.fit(X, y_enc)
    symptoms = list(X.columns)
    return model, le, symptoms

# ─────────────────────────────────────────────
# Symptom display name helper
# ─────────────────────────────────────────────
def fmt(s):
    return s.replace("_", " ").title()


# ═════════════════════════════════════════════
# Main Application Window
# ═════════════════════════════════════════════
class MedicalDiagnosisApp:

    def __init__(self, root, model, le, symptoms):
        self.root      = root
        self.model     = model
        self.le        = le
        self.symptoms  = symptoms
        self.selected  = {}          # symptom → BooleanVar
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_symptoms)

        root.title("Medical Diagnosis System — FAST-NUCES Karachi")
        root.configure(bg=BG_DARK)
        root.geometry("1200x780")
        root.minsize(900, 640)

        self._build_ui()

    # ── UI Construction ──────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────
        header = tk.Frame(self.root, bg=BG_CARD, pady=12)
        header.pack(fill="x")

        tk.Label(header, text="⚕  Medical Diagnosis System",
                 font=("Segoe UI", 18, "bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(side="left", padx=24)

        tk.Label(header,
                 text="Bayesian Network  |  FAST-NUCES Karachi  |  AI Section 6B",
                 font=("Segoe UI", 9), bg=BG_CARD, fg=TEXT_MUTED).pack(
                     side="right", padx=24)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

        # ── Body (left panel + right panel) ─
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        self._build_left(body)
        self._build_right(body)

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG_DARK, width=420)
        left.pack(side="left", fill="both", padx=(0, 10))
        left.pack_propagate(False)

        # ── Search ──────────────────────────
        sf = tk.Frame(left, bg=BG_CARD, pady=8, padx=8,
                      highlightbackground=BORDER, highlightthickness=1)
        sf.pack(fill="x", pady=(0, 8))

        tk.Label(sf, text="🔍  Search symptoms",
                 font=("Segoe UI", 10, "bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(anchor="w")

        entry = tk.Entry(sf, textvariable=self.search_var,
                         bg=BG_HOVER, fg=TEXT_PRIMARY,
                         insertbackground=TEXT_PRIMARY,
                         relief="flat", font=("Segoe UI", 11),
                         bd=4)
        entry.pack(fill="x", pady=4)

        # ── Symptom list ────────────────────
        lf = tk.LabelFrame(left, text="  Select Symptoms  ",
                           font=("Segoe UI", 10, "bold"),
                           bg=BG_CARD, fg=ACCENT_BLUE,
                           bd=1, relief="solid",
                           labelanchor="n")
        lf.pack(fill="both", expand=True)

        canvas = tk.Canvas(lf, bg=BG_CARD, highlightthickness=0)
        sb = ttk.Scrollbar(lf, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)

        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.symp_frame = tk.Frame(canvas, bg=BG_CARD)
        self.symp_win   = canvas.create_window((0, 0), window=self.symp_frame,
                                                anchor="nw")
        self.symp_frame.bind("<Configure>",
                             lambda e: canvas.configure(
                                 scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(self.symp_win, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-int(e.delta/120), "units"))
        self._canvas = canvas

        self._populate_symptoms(self.symptoms)

        # ── Buttons ─────────────────────────
        bf = tk.Frame(left, bg=BG_DARK)
        bf.pack(fill="x", pady=8)

        self._btn_diagnose = self._make_btn(
            bf, "  Diagnose  ", ACCENT_BLUE, self._diagnose)
        self._btn_diagnose.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self._make_btn(bf, "  Clear  ", ACCENT_RED, self._clear).pack(
            side="left", fill="x", expand=True, padx=(4, 0))

        # ── Selected badge ───────────────────
        self.badge_var = tk.StringVar(value="0 symptoms selected")
        tk.Label(left, textvariable=self.badge_var,
                 font=("Segoe UI", 9), bg=BG_DARK, fg=TEXT_MUTED).pack()

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)

        # ── Diagnosis card ───────────────────
        dc = tk.Frame(right, bg=BG_CARD, pady=14, padx=18,
                      highlightbackground=BORDER, highlightthickness=1)
        dc.pack(fill="x", pady=(0, 10))

        tk.Label(dc, text="Diagnosis Result",
                 font=("Segoe UI", 11, "bold"),
                 bg=BG_CARD, fg=TEXT_MUTED).pack(anchor="w")

        self.disease_var = tk.StringVar(value="— Awaiting Input —")
        tk.Label(dc, textvariable=self.disease_var,
                 font=("Segoe UI", 22, "bold"),
                 bg=BG_CARD, fg=ACCENT_GREEN,
                 wraplength=640, justify="left").pack(anchor="w", pady=4)

        self.conf_var = tk.StringVar(value="")
        tk.Label(dc, textvariable=self.conf_var,
                 font=("Segoe UI", 12),
                 bg=BG_CARD, fg=ACCENT_YELL).pack(anchor="w")

        self.symp_used_var = tk.StringVar(value="")
        tk.Label(dc, textvariable=self.symp_used_var,
                 font=("Segoe UI", 9),
                 bg=BG_CARD, fg=TEXT_MUTED, wraplength=640, justify="left"
                 ).pack(anchor="w", pady=(6, 0))

        # ── Probability chart ────────────────
        cf = tk.LabelFrame(right, text="  Top-10 Disease Probabilities  ",
                           font=("Segoe UI", 10, "bold"),
                           bg=BG_CARD, fg=ACCENT_BLUE,
                           bd=1, relief="solid", labelanchor="n")
        cf.pack(fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4.2))
        self.fig.patch.set_facecolor(BG_CARD)
        self.ax.set_facecolor(BG_CARD)
        self._draw_empty_chart()

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=cf)
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True,
                                                 padx=8, pady=8)

    # ── Helpers ──────────────────────────────
    def _make_btn(self, parent, text, color, cmd):
        btn = tk.Button(parent, text=text, command=cmd,
                        bg=color, fg=BG_DARK,
                        font=("Segoe UI", 11, "bold"),
                        relief="flat", cursor="hand2",
                        activebackground=BG_HOVER,
                        activeforeground=TEXT_PRIMARY,
                        padx=10, pady=6)
        return btn

    def _populate_symptoms(self, symp_list):
        for w in self.symp_frame.winfo_children():
            w.destroy()

        for i, s in enumerate(symp_list):
            if s not in self.selected:
                self.selected[s] = tk.BooleanVar()

            row = tk.Frame(self.symp_frame, bg=BG_CARD)
            row.pack(fill="x", pady=1)

            cb = tk.Checkbutton(
                row, text=fmt(s),
                variable=self.selected[s],
                bg=BG_CARD, fg=TEXT_PRIMARY,
                selectcolor=TAG_BG,
                activebackground=BG_HOVER,
                activeforeground=TEXT_PRIMARY,
                font=("Segoe UI", 10),
                anchor="w", relief="flat",
                command=self._update_badge
            )
            cb.pack(fill="x", padx=6)

    def _filter_symptoms(self, *_):
        q = self.search_var.get().lower().replace(" ", "_")
        filtered = [s for s in self.symptoms if q in s.lower()]
        self._populate_symptoms(filtered)

    def _update_badge(self):
        n = sum(1 for v in self.selected.values() if v.get())
        self.badge_var.set(f"{n} symptom{'s' if n != 1 else ''} selected")

    def _get_input_vector(self):
        vec = np.zeros(len(self.symptoms))
        for i, s in enumerate(self.symptoms):
            if self.selected.get(s) and self.selected[s].get():
                vec[i] = 1
        return vec

    # ── Diagnosis ────────────────────────────
    def _diagnose(self):
        vec = self._get_input_vector()
        n_selected = int(vec.sum())

        if n_selected == 0:
            messagebox.showwarning("No Symptoms",
                                   "Please select at least one symptom.")
            return

        probs = self.model.predict_proba([vec])[0]
        top_n = 10
        top_idx  = np.argsort(probs)[::-1][:top_n]
        top_prob = probs[top_idx]
        top_name = self.le.classes_[top_idx]

        best_disease = top_name[0]
        best_conf    = top_prob[0] * 100

        # ── Update result card ───────────────
        self.disease_var.set(best_disease)
        self.conf_var.set(f"Confidence: {best_conf:.1f}%")

        used = [fmt(s) for s in self.symptoms
                if self.selected.get(s) and self.selected[s].get()]
        self.symp_used_var.set(
            f"Symptoms used ({n_selected}): " + ", ".join(used))

        # ── Update chart ─────────────────────
        self._update_chart(top_name, top_prob * 100)

    def _clear(self):
        for v in self.selected.values():
            v.set(False)
        self.badge_var.set("0 symptoms selected")
        self.disease_var.set("— Awaiting Input —")
        self.conf_var.set("")
        self.symp_used_var.set("")
        self._draw_empty_chart()
        self.canvas_widget.draw()

    # ── Chart helpers ─────────────────────────
    def _draw_empty_chart(self):
        self.ax.clear()
        self.ax.set_facecolor(BG_CARD)
        self.ax.text(0.5, 0.5, "Select symptoms and click Diagnose",
                     ha="center", va="center",
                     color=TEXT_MUTED, fontsize=11,
                     transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.fig.tight_layout()

    def _update_chart(self, names, probs):
        self.ax.clear()
        self.ax.set_facecolor(BG_CARD)

        colors = [ACCENT_GREEN if i == 0 else ACCENT_BLUE
                  for i in range(len(names))]

        y_pos = range(len(names) - 1, -1, -1)
        bars  = self.ax.barh(list(y_pos), probs, color=colors,
                             edgecolor="none", height=0.65)

        self.ax.set_yticks(list(y_pos))
        self.ax.set_yticklabels(names, color=TEXT_PRIMARY, fontsize=9)
        self.ax.set_xlabel("Probability (%)", color=TEXT_MUTED, fontsize=9)
        self.ax.tick_params(axis="x", colors=TEXT_MUTED, labelsize=8)
        self.ax.set_xlim(0, max(probs) * 1.18)

        for spine in self.ax.spines.values():
            spine.set_color(BORDER)

        for bar, prob in zip(bars, probs):
            if prob >= 0.1:
                self.ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                             f"{prob:.1f}%",
                             va="center", ha="left",
                             color=TEXT_PRIMARY, fontsize=8)

        self.fig.tight_layout()
        self.canvas_widget.draw()


# ═════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════
def main():
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at: {DATASET_PATH}")
        print("Place Disease_symptom_dataset.csv in the same folder as main.py")
        sys.exit(1)

    print("Loading dataset and training Bayesian model…")
    model, le, symptoms = load_and_train(DATASET_PATH)
    print(f"  ✓ Trained on {len(symptoms)} symptoms, {len(le.classes_)} diseases")

    root = tk.Tk()

    # Dark title bar on Windows
    try:
        root.wm_attributes("-alpha", 1.0)
    except Exception:
        pass

    app = MedicalDiagnosisApp(root, model, le, symptoms)
    root.mainloop()


if __name__ == "__main__":
    main()
