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
# Enhanced colour palette with gradients
# ─────────────────────────────────────────────
BG_DARK      = "#0a0e12"
BG_CARD      = "#161b22"
BG_HOVER     = "#1f2937"
BG_ELEVATED  = "#1c2128"
ACCENT_BLUE  = "#58a6ff"
ACCENT_BLUE_LIGHT = "#79c0ff"
ACCENT_GREEN = "#3fb950"
ACCENT_GREEN_LIGHT = "#56d364"
ACCENT_RED   = "#f85149"
ACCENT_YELL  = "#e3b341"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"
BORDER       = "#30363d"
BORDER_LIGHT = "#484f58"
TAG_BG       = "#1f6feb"
SHADOW       = "#000000"

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
# Custom styled widgets
# ═════════════════════════════════════════════
class HoverButton(tk.Button):
    """Button with hover effects"""
    def __init__(self, parent, **kwargs):
        self.default_bg = kwargs.get('bg', ACCENT_BLUE)
        self.hover_bg = kwargs.pop('hover_bg', BG_HOVER)
        super().__init__(parent, **kwargs)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_enter(self, e):
        self['bg'] = self.hover_bg
    
    def _on_leave(self, e):
        self['bg'] = self.default_bg


class StyledFrame(tk.Frame):
    """Frame with shadow effect simulation"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)


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
        root.geometry("1280x820")
        root.minsize(1000, 700)

        self._build_ui()

    # ── UI Construction ──────────────────────
    def _build_ui(self):
        # ── Header with gradient effect ──────
        header = tk.Frame(self.root, bg=BG_ELEVATED, pady=20)
        header.pack(fill="x")

        # Title with icon
        title_frame = tk.Frame(header, bg=BG_ELEVATED)
        title_frame.pack()
        
        tk.Label(title_frame, text="⚕", 
                 font=("Segoe UI", 32),
                 bg=BG_ELEVATED, fg=ACCENT_BLUE).pack(side="left", padx=(0, 15))
        
        title_text = tk.Frame(title_frame, bg=BG_ELEVATED)
        title_text.pack(side="left")
        
        tk.Label(title_text, text="Medical Diagnosis System",
                 font=("Segoe UI", 24, "bold"),
                 bg=BG_ELEVATED, fg=TEXT_PRIMARY).pack(anchor="w")
        
        tk.Label(title_text,
                 text="Bayesian Network Classification  •  FAST-NUCES Karachi  •  AI Section 6B",
                 font=("Segoe UI", 10), bg=BG_ELEVATED, fg=TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        # Decorative line
        separator = tk.Frame(self.root, height=3, bg=ACCENT_BLUE)
        separator.pack(fill="x")

        # ── Body (left panel + right panel) ─
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        self._build_left(body)
        self._build_right(body)

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG_DARK, width=450)
        left.pack(side="left", fill="both", padx=(0, 15))
        left.pack_propagate(False)

        # ── Search with modern styling ───────
        search_container = StyledFrame(left, bg=BG_DARK)
        search_container.pack(fill="x", pady=(0, 15))
        
        sf = tk.Frame(search_container, bg=BG_CARD, pady=14, padx=16,
                      highlightbackground=BORDER_LIGHT, highlightthickness=2)
        sf.pack(fill="x")

        search_header = tk.Frame(sf, bg=BG_CARD)
        search_header.pack(fill="x", pady=(0, 8))
        
        tk.Label(search_header, text="", 
                 font=("Segoe UI", 16),
                 bg=BG_CARD, fg=ACCENT_BLUE).pack(side="left", padx=(0, 8))
        
        tk.Label(search_header, text="Search Symptoms",
                 font=("Segoe UI", 12, "bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(side="left")

        # Search entry with better styling
        entry_frame = tk.Frame(sf, bg=BG_HOVER, highlightbackground=BORDER_LIGHT,
                               highlightthickness=1)
        entry_frame.pack(fill="x")
        
        entry = tk.Entry(entry_frame, textvariable=self.search_var,
                         bg=BG_HOVER, fg=TEXT_PRIMARY,
                         insertbackground=ACCENT_BLUE,
                         relief="flat", font=("Segoe UI", 12),
                         bd=8)
        entry.pack(fill="x")

        # ── Symptom list with modern card ────
        list_container = StyledFrame(left, bg=BG_DARK)
        list_container.pack(fill="both", expand=True, pady=(0, 15))
        
        lf = tk.Frame(list_container, bg=BG_CARD,
                      highlightbackground=BORDER_LIGHT, highlightthickness=2)
        lf.pack(fill="both", expand=True)
        
        # List header
        list_header = tk.Frame(lf, bg=BG_ELEVATED, pady=12, padx=16)
        list_header.pack(fill="x")
        
        tk.Label(list_header, text="Select Symptoms",
                 font=("Segoe UI", 12, "bold"),
                 bg=BG_ELEVATED, fg=ACCENT_BLUE).pack(side="left")

        # Scrollable symptom area
        scroll_container = tk.Frame(lf, bg=BG_CARD)
        scroll_container.pack(fill="both", expand=True, padx=2, pady=2)
        
        canvas = tk.Canvas(scroll_container, bg=BG_CARD, highlightthickness=0)
        sb = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
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

        # ── Buttons with enhanced styling ────
        bf = tk.Frame(left, bg=BG_DARK)
        bf.pack(fill="x", pady=(0, 12))

        self._btn_diagnose = HoverButton(
            bf, text="Diagnose", 
            bg=ACCENT_BLUE, 
            hover_bg=ACCENT_BLUE_LIGHT,
            fg=BG_DARK,
            font=("Segoe UI", 12, "bold"),
            relief="flat", cursor="hand2",
            activebackground=ACCENT_BLUE_LIGHT,
            activeforeground=BG_DARK,
            padx=16, pady=12,
            command=self._diagnose)
        self._btn_diagnose.pack(side="left", fill="x", expand=True, padx=(0, 8))

        HoverButton(bf, text="Clear", 
                    bg=ACCENT_RED,
                    hover_bg="#ff6b6b",
                    fg=BG_DARK,
                    font=("Segoe UI", 12, "bold"),
                    relief="flat", cursor="hand2",
                    activebackground="#ff6b6b",
                    activeforeground=BG_DARK,
                    padx=16, pady=12,
                    command=self._clear).pack(side="left", fill="x", expand=True)

        # ── Enhanced badge ───────────────────
        badge_frame = tk.Frame(left, bg=BG_CARD, 
                               highlightbackground=BORDER, highlightthickness=1)
        badge_frame.pack(fill="x")
        
        self.badge_var = tk.StringVar(value="0 symptoms selected")
        tk.Label(badge_frame, textvariable=self.badge_var,
                 font=("Segoe UI", 10, "bold"), 
                 bg=BG_CARD, fg=ACCENT_YELL,
                 pady=10).pack()

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)

        # ── Enhanced diagnosis card ───────────
        dc_container = StyledFrame(right, bg=BG_DARK)
        dc_container.pack(fill="x", pady=(0, 15))
        
        dc = tk.Frame(dc_container, bg=BG_CARD, pady=20, padx=24,
                      highlightbackground=BORDER_LIGHT, highlightthickness=2)
        dc.pack(fill="x")

        # Card header
        header = tk.Frame(dc, bg=BG_CARD)
        header.pack(fill="x", pady=(0, 12))
        
        tk.Label(header, text="",
                 font=("Segoe UI", 20),
                 bg=BG_CARD, fg=ACCENT_GREEN).pack(side="left", padx=(0, 10))
        
        tk.Label(header, text="Diagnosis Result",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_CARD, fg=TEXT_MUTED).pack(side="left")

        # Disease name with larger, more prominent display
        self.disease_var = tk.StringVar(value="— Awaiting Input —")
        disease_label = tk.Label(dc, textvariable=self.disease_var,
                 font=("Segoe UI", 26, "bold"),
                 bg=BG_CARD, fg=ACCENT_GREEN,
                 wraplength=700, justify="left")
        disease_label.pack(anchor="w", pady=(0, 8))

        # Confidence with progress bar style
        conf_frame = tk.Frame(dc, bg=BG_CARD)
        conf_frame.pack(fill="x", pady=(0, 12))
        
        self.conf_var = tk.StringVar(value="")
        tk.Label(conf_frame, textvariable=self.conf_var,
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_CARD, fg=ACCENT_YELL).pack(side="left")

        # Symptoms used section
        symp_header = tk.Frame(dc, bg=BG_ELEVATED, pady=8, padx=12)
        symp_header.pack(fill="x", pady=(0, 4))
        
        tk.Label(symp_header, text="Selected Symptoms:",
                 font=("Segoe UI", 10, "bold"),
                 bg=BG_ELEVATED, fg=TEXT_MUTED).pack(anchor="w")
        
        self.symp_used_var = tk.StringVar(value="")
        tk.Label(dc, textvariable=self.symp_used_var,
                 font=("Segoe UI", 10),
                 bg=BG_CARD, fg=TEXT_PRIMARY, 
                 wraplength=700, justify="left"
                 ).pack(anchor="w", padx=12)

        # ── Enhanced probability chart ────────
        cf_container = StyledFrame(right, bg=BG_DARK)
        cf_container.pack(fill="both", expand=True)
        
        cf = tk.Frame(cf_container, bg=BG_CARD,
                      highlightbackground=BORDER_LIGHT, highlightthickness=2)
        cf.pack(fill="both", expand=True)
        
        # Chart header
        chart_header = tk.Frame(cf, bg=BG_ELEVATED, pady=12, padx=16)
        chart_header.pack(fill="x")
        
        tk.Label(chart_header, text="Top-10 Disease Probabilities",
                 font=("Segoe UI", 12, "bold"),
                 bg=BG_ELEVATED, fg=ACCENT_BLUE).pack(side="left")

        # Chart area
        chart_bg = tk.Frame(cf, bg=BG_CARD)
        chart_bg.pack(fill="both", expand=True, padx=2, pady=2)

        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.fig.patch.set_facecolor(BG_CARD)
        self.ax.set_facecolor(BG_CARD)
        self._draw_empty_chart()

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=chart_bg)
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True,
                                                 padx=12, pady=12)

    # ── Helpers ──────────────────────────────
    def _populate_symptoms(self, symp_list):
        for w in self.symp_frame.winfo_children():
            w.destroy()

        for i, s in enumerate(symp_list):
            if s not in self.selected:
                self.selected[s] = tk.BooleanVar()

            # Alternating row colors for better readability
            row_bg = BG_CARD if i % 2 == 0 else BG_ELEVATED
            row = tk.Frame(self.symp_frame, bg=row_bg, pady=2)
            row.pack(fill="x")

            cb = tk.Checkbutton(
                row, text=fmt(s),
                variable=self.selected[s],
                bg=row_bg, fg=TEXT_PRIMARY,
                selectcolor=TAG_BG,
                activebackground=BG_HOVER,
                activeforeground=TEXT_PRIMARY,
                font=("Segoe UI", 11),
                anchor="w", relief="flat",
                padx=16, pady=4,
                cursor="hand2",
                command=self._update_badge
            )
            cb.pack(fill="x")
            
            # Hover effect
            cb.bind("<Enter>", lambda e, r=row: r.config(bg=BG_HOVER))
            cb.bind("<Leave>", lambda e, r=row, bg=row_bg: r.config(bg=bg))

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
        self.symp_used_var.set(", ".join(used))

        # ── Update chart ─────────────────────
        self._update_chart(top_name, top_prob * 100)

    def _clear(self):
        for v in self.selected.values():
            v.set(False)
        self.badge_var.set("0 symptoms selected")
        self.disease_var.set("Awaiting Input")
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
                     color=TEXT_MUTED, fontsize=13,
                     transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.fig.tight_layout()

    def _update_chart(self, names, probs):
        self.ax.clear()
        self.ax.set_facecolor(BG_CARD)

        # Gradient colors
        colors = []
        for i, prob in enumerate(probs):
            if i == 0:
                colors.append(ACCENT_GREEN_LIGHT)
            else:
                # Fade from blue to darker blue
                intensity = 1 - (i / len(probs)) * 0.5
                colors.append(ACCENT_BLUE if intensity > 0.7 else "#3d7ab8")

        y_pos = range(len(names) - 1, -1, -1)
        bars  = self.ax.barh(list(y_pos), probs, color=colors,
                             edgecolor="none", height=0.7)

        self.ax.set_yticks(list(y_pos))
        self.ax.set_yticklabels(names, color=TEXT_PRIMARY, fontsize=10, weight='bold')
        self.ax.set_xlabel("Probability (%)", color=TEXT_MUTED, fontsize=10, weight='bold')
        self.ax.tick_params(axis="x", colors=TEXT_MUTED, labelsize=9)
        self.ax.set_xlim(0, max(probs) * 1.15)

        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
            spine.set_linewidth(1.5)

        # Enhanced bar labels
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            if prob >= 0.1:
                self.ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                             f"{prob:.1f}%",
                             va="center", ha="left",
                             color=TEXT_PRIMARY, fontsize=9, weight='bold')

        self.ax.grid(axis='x', alpha=0.2, color=BORDER, linestyle='--', linewidth=0.5)
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