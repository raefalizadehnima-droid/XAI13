#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DYNAMIC FIGURE GENERATOR  (make_figures.py)
===========================================
Reproduces and generalises three figure types for ANY table of metrics:

  1) hist_<column>   histogram + KDE for each chosen metric column  (e.g. "Accuracy (%)")
  2) corr_matrix     annotated correlation heatmap (Pearson / Spearman / Kendall), title shows n
  3) hist_CTI        histogram + KDE of the Composite Trust Index
  4) panel           (optional) all of the above in one multi-panel figure

CTI source (choose one):
  --cti-column NAME           use a CTI column already in the file
  --weights w1 w2 ...         compute CTI = sum(w_i * normalised metric_i), one weight per metric
                              (weights are re-scaled to sum to 1; omitted -> equal weights)
Normalisation: --norm minmax (default) or range100; --invert marks lower-is-better metrics.

INSTALL   python -m pip install numpy pandas scipy matplotlib seaborn openpyxl

EXAMPLES
  # all figures, columns auto-detected (non-numeric columns such as IDs are skipped)
  python make_figures.py --data benchmark.xlsx --sheet Data_120 --invert "Batch_Failure_Rate (%)"

  # exactly the three figures of the paper, custom weights, PNG + PDF at 300 dpi
  python make_figures.py --data benchmark.xlsx --sheet Data_120 \
      --hist-columns "AI_Model_Accuracy (%)" --labels '{"AI_Model_Accuracy (%)":"Accuracy (%)"}' \
      --weights 0.3 0.2 0.1 0.25 0.15 --invert "Batch_Failure_Rate (%)" --formats png pdf --dpi 300

  # CTI already in the file, Spearman correlation, 15 bins, no KDE
  python make_figures.py --data d.csv --cti-column CTI --corr-method spearman --bins 15 --no-kde
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ID_LIKE = re.compile(r"(^id$|record|index|status|source|name)", re.I)

# ----------------------------------------------------------------------------- data
def read_table(path, sheet=None):
    p = Path(path)
    if p.suffix.lower() == ".csv": return pd.read_csv(p)
    return pd.read_excel(p, sheet_name=sheet or 0)

def pick_metrics(df, requested, exclude):
    if requested:
        miss = [m for m in requested if m not in df.columns]
        if miss: sys.exit(f"Columns not found: {miss}\nAvailable: {list(df.columns)}")
        return list(requested)
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not ID_LIKE.search(str(c)) and c not in exclude]
    if not cols: sys.exit("No numeric metric columns detected; pass --metrics.")
    return cols

def compute_cti(X, weights, invert, norm):
    Z = X.astype(float).copy()
    for c in Z.columns:
        lo, hi = (Z[c].min(), Z[c].max()) if norm == "minmax" else (0.0, 100.0)
        Z[c] = (Z[c] - lo) / (hi - lo)
        if c in invert: Z[c] = 1 - Z[c]
    w = np.full(Z.shape[1], 1 / Z.shape[1]) if not weights else np.asarray(weights, float)
    if len(w) != Z.shape[1]: sys.exit(f"--weights needs {Z.shape[1]} values (one per metric), got {len(w)}.")
    if (w < 0).any() or w.sum() <= 0: sys.exit("Weights must be non-negative with a positive sum.")
    return pd.Series(Z.values @ (w / w.sum()), index=X.index, name="CTI Score")

# ----------------------------------------------------------------------------- plotting
def style():
    sns.set_theme(style="white"); plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": True, "axes.spines.right": True})

def hist_kde(ax, x, xlabel, color, bins, kde, edge="black", alpha=0.5, line_color=None):
    x = pd.Series(x).dropna()
    sns.histplot(x, bins=bins, kde=kde, color=color, edgecolor=edge, alpha=alpha, ax=ax, line_kws={"linewidth": 1.5})
    if kde and line_color and ax.lines: ax.lines[-1].set_color(line_color)
    ax.set_xlabel(xlabel); ax.set_ylabel("Frequency")
    return len(x)

def corr_heatmap(ax, X, method, cmap, labels, fmt, annot_size):
    C = X.corr(method=method)
    C = C.rename(index=lambda c: labels.get(c, c), columns=lambda c: labels.get(c, c))
    sns.heatmap(C, annot=True, fmt=fmt, cmap=cmap, vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={"size": annot_size}, cbar_kws={"label": f"{method.capitalize()} r"})
    ax.set_title(f"Correlation Matrix (n = {len(X.dropna())})")
    plt.setp(ax.get_xticklabels(), rotation=90); plt.setp(ax.get_yticklabels(), rotation=0)
    return C

def save(fig, outdir, stem, formats, dpi):
    paths = []
    for f in formats:
        p = Path(outdir) / f"{stem}.{f}"; fig.savefig(p, dpi=dpi, bbox_inches="tight"); paths.append(p)
    plt.close(fig); return paths

def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--sheet")
    ap.add_argument("--metrics", nargs="+", help="metric columns (default: all numeric, ID/status-like columns skipped)")
    ap.add_argument("--invert", nargs="*", default=[], help="lower-is-better metrics (used for CTI only)")
    ap.add_argument("--norm", choices=["minmax", "range100"], default="minmax")
    ap.add_argument("--cti-column"); ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--plots", nargs="+", choices=["hist", "corr", "cti", "panel"], default=["hist", "corr", "cti", "panel"])
    ap.add_argument("--hist-columns", nargs="+", help="columns for histograms (default: all metrics)")
    ap.add_argument("--labels", default="{}", help='JSON {"column":"axis label"}; or path to a .json file')
    ap.add_argument("--bins", default="auto", help="'auto', 'fd', 'sturges' or an integer")
    ap.add_argument("--kde", dest="kde", action="store_true", default=True); ap.add_argument("--no-kde", dest="kde", action="store_false")
    ap.add_argument("--corr-method", choices=["pearson", "spearman", "kendall"], default="pearson")
    ap.add_argument("--color", default="#1f77b4", help="histogram colour (metrics)")
    ap.add_argument("--cti-color", default="darkblue"); ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--fmt", default=".3f", help="annotation format of the heatmap"); ap.add_argument("--annot-size", type=float, default=10)
    ap.add_argument("--figsize", nargs=2, type=float, default=[6.4, 4.8]); ap.add_argument("--corr-figsize", nargs=2, type=float, default=[6.5, 5.5])
    ap.add_argument("--dpi", type=int, default=200); ap.add_argument("--formats", nargs="+", default=["png"])
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--export-cti", help="also save the computed CTI per row to this CSV")
    a = ap.parse_args()

    df = read_table(a.data, a.sheet)
    labels = json.load(open(a.labels)) if a.labels.endswith(".json") else json.loads(a.labels)
    exclude = {a.cti_column} if a.cti_column else set()
    metrics = pick_metrics(df, a.metrics, exclude)
    X = df[metrics].apply(pd.to_numeric, errors="coerce")
    bad = X.isna().any(axis=1)
    if bad.any(): print(f"[warn] {int(bad.sum())} rows with missing/non-numeric values are ignored in correlations/CTI")
    X = X[~bad]
    bins = int(a.bins) if re.fullmatch(r"\d+", str(a.bins)) else a.bins
    Path(a.outdir).mkdir(parents=True, exist_ok=True); style(); made = []

    cti = None
    if "cti" in a.plots or "panel" in a.plots:
        if a.cti_column:
            if a.cti_column not in df.columns: sys.exit(f"CTI column not found: {a.cti_column}")
            cti = pd.to_numeric(df.loc[X.index, a.cti_column], errors="coerce").dropna()
        else:
            cti = compute_cti(X, a.weights, set(a.invert), a.norm)
            print("[cti] computed with weights:", "equal" if not a.weights else np.round(np.asarray(a.weights) / sum(a.weights), 4).tolist())
        print(f"[cti] mean={cti.mean():.4f} sd={cti.std(ddof=1):.4f} min={cti.min():.4f} max={cti.max():.4f} n={len(cti)}")
        if a.export_cti: cti.to_csv(a.export_cti, header=True)

    hcols = a.hist_columns or metrics
    if "hist" in a.plots:
        for c in hcols:
            if c not in X.columns: sys.exit(f"--hist-columns: {c} is not among the metrics")
            fig, ax = plt.subplots(figsize=a.figsize); hist_kde(ax, X[c], labels.get(c, c), a.color, bins, a.kde)
            made += save(fig, a.outdir, f"hist_{safe(c)}", a.formats, a.dpi)
    if "corr" in a.plots and X.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=a.corr_figsize); corr_heatmap(ax, X, a.corr_method, a.cmap, labels, a.fmt, a.annot_size)
        made += save(fig, a.outdir, "corr_matrix", a.formats, a.dpi)
    if "cti" in a.plots:
        fig, ax = plt.subplots(figsize=a.figsize)
        hist_kde(ax, cti, "CTI Score", a.cti_color, bins, a.kde, alpha=0.5, line_color=a.cti_color)
        made += save(fig, a.outdir, "hist_CTI", a.formats, a.dpi)
    if "panel" in a.plots:
        n_h = len(hcols); cols = min(3, n_h + 1)
        items = [("hist", c) for c in hcols] + [("cti", None)] + ([("corr", None)] if X.shape[1] >= 2 else [])
        rows = int(np.ceil(len(items) / cols)); fig, axs = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), squeeze=False)
        for ax in axs.ravel()[len(items):]: ax.axis("off")
        for ax, (kind, c) in zip(axs.ravel(), items):
            if kind == "hist": hist_kde(ax, X[c], labels.get(c, c), a.color, bins, a.kde)
            elif kind == "cti": hist_kde(ax, cti, "CTI Score", a.cti_color, bins, a.kde, line_color=a.cti_color)
            else: corr_heatmap(ax, X, a.corr_method, a.cmap, labels, a.fmt, max(6, a.annot_size - 3))
        fig.tight_layout(); made += save(fig, a.outdir, "panel", a.formats, a.dpi)
    print("[done] figures:"); [print("  ", p) for p in made]

if __name__ == "__main__":
    main()
