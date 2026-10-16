#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cti_analysis.py
---------------
Composite Trust Index (CTI) analysis for the 120-row study dataset.

Background (from package config / manuscript notes)
  • The paper does NOT publish the exact CTI formula or weights.
  • Section 3.3 mentions four categories (accuracy, regulatory adherence,
    resilience, cost-effectiveness), min-max normalised, with accuracy and
    regulatory adherence weighted more heavily.
  • Table 3 reports CTI summary: mean≈0.486, sd≈0.083, min≈0.21, max≈0.675.

This script therefore provides:
  1. Explicit, documented CTI computation (4-metric paper-style OR 5-metric)
  2. Comparison of the resulting CTI distribution with Table 3 targets
  3. Optional weight search so that the CTI summary matches Table 3 as closely
     as possible (post-hoc fit — does NOT prove the original weights)
  4. Export of per-row CTI, normalised components, weights used, and diagnostics

Usage examples
  # Paper-style 4-metric CTI (default illustrative weights)
  python cti_analysis.py

  # Equal weights on all five metrics (Batch Failure inverted)
  python cti_analysis.py --mode five --equal

  # Search weights that best match Table 3 summary
  python cti_analysis.py --fit-table3 --out ../cti_fit_table3.xlsx

  # Custom weights (must match the chosen metric set)
  python cti_analysis.py --mode four --weights 0.35 0.35 0.15 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution

# --------------------------------------------------------------------------- metrics
METRICS_FIVE = [
    "AI_Model_Accuracy (%)",
    "Batch_Failure_Rate (%)",
    "Production_Cost_Reduction (%)",
    "Regulatory_Compliance_Score",
    "System_Resilience_Score",
]

# Paper-style four categories (Batch Failure excluded from CTI formula in config notes)
METRICS_FOUR = [
    "AI_Model_Accuracy (%)",
    "Regulatory_Compliance_Score",
    "System_Resilience_Score",
    "Production_Cost_Reduction (%)",
]

# Illustrative weights from config.json (accuracy & regulatory heavier)
DEFAULT_WEIGHTS_FOUR = [0.35, 0.35, 0.15, 0.15]
DEFAULT_WEIGHTS_FIVE = [0.30, 0.15, 0.15, 0.25, 0.15]

# Lower-is-better (only relevant when the metric is included)
INVERT_DEFAULT = {"Batch_Failure_Rate (%)"}

# Table 3 targets from paper_claims.json
TABLE3 = {
    "mean": 0.486,
    "std": 0.083,
    "min": 0.210,
    "q25": 0.438,
    "median": 0.480,
    "q75": 0.547,
    "max": 0.675,
}


# --------------------------------------------------------------------------- helpers
def load_data(path: Path, sheet) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet)
    return df


def normalise_column(x: pd.Series, method: str, invert: bool) -> pd.Series:
    x = x.astype(float)
    if method == "minmax":
        lo, hi = x.min(), x.max()
        z = pd.Series(0.5, index=x.index) if hi <= lo else (x - lo) / (hi - lo)
    elif method == "range100":
        z = (x / 100.0).clip(0.0, 1.0)
    else:
        raise ValueError(f"Unknown norm method: {method}")
    return (1.0 - z) if invert else z


def compute_cti(
    df: pd.DataFrame,
    metrics: list[str],
    weights: list[float],
    invert: set[str],
    norm: str,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Return (CTI series, Z matrix, normalised weight vector)."""
    w = np.asarray(weights, dtype=float)
    if len(w) != len(metrics):
        raise SystemExit(f"Need {len(metrics)} weights, got {len(w)}")
    if (w < 0).any() or w.sum() <= 0:
        raise SystemExit("Weights must be non-negative and sum > 0")
    w = w / w.sum()

    Z = pd.DataFrame(index=df.index)
    for m in metrics:
        if m not in df.columns:
            raise SystemExit(f"Column not found: {m}")
        Z[m] = normalise_column(df[m], norm, invert=(m in invert))

    cti = pd.Series(Z.values @ w, index=df.index, name="CTI")
    return cti, Z, w


def summarise(cti: pd.Series) -> dict:
    return {
        "n": int(len(cti)),
        "mean": float(cti.mean()),
        "std": float(cti.std(ddof=1)),
        "min": float(cti.min()),
        "q25": float(cti.quantile(0.25)),
        "median": float(cti.median()),
        "q75": float(cti.quantile(0.75)),
        "max": float(cti.max()),
    }


def compare_to_table3(s: dict) -> pd.DataFrame:
    rows = []
    for k, target in TABLE3.items():
        got = s[k]
        rows.append({
            "Statistic": k,
            "Computed": round(got, 6),
            "Table3_Target": target,
            "Abs_Diff": round(abs(got - target), 6),
        })
    return pd.DataFrame(rows)


def fit_weights_to_table3(
    df: pd.DataFrame,
    metrics: list[str],
    invert: set[str],
    norm: str,
    seed: int = 42,
    maxiter: int = 80,
) -> tuple[np.ndarray, dict]:
    """
    Post-hoc search for non-negative weights that make the CTI summary
    as close as possible to Table 3 (mean, std, quantiles, min, max).
    This does NOT recover the true original weights; it only shows one
    plausible weight vector consistent with the published summary.
    """
    n_w = len(metrics)

    def loss(raw):
        w = np.abs(raw)
        if w.sum() <= 0:
            return 1e6
        w = w / w.sum()
        cti, _, _ = compute_cti(df, metrics, w.tolist(), invert, norm)
        s = summarise(cti)
        # weighted absolute error on key moments
        err = (
            4.0 * abs(s["mean"] - TABLE3["mean"])
            + 3.0 * abs(s["std"] - TABLE3["std"])
            + 2.0 * abs(s["median"] - TABLE3["median"])
            + 1.5 * abs(s["q25"] - TABLE3["q25"])
            + 1.5 * abs(s["q75"] - TABLE3["q75"])
            + 1.0 * abs(s["min"] - TABLE3["min"])
            + 1.0 * abs(s["max"] - TABLE3["max"])
        )
        return err

    bounds = [(0.0, 1.0)] * n_w
    result = differential_evolution(
        loss, bounds=bounds, seed=seed, maxiter=maxiter, polish=True, workers=1
    )
    w_hat = np.abs(result.x)
    w_hat = w_hat / w_hat.sum()
    cti, _, _ = compute_cti(df, metrics, w_hat.tolist(), invert, norm)
    return w_hat, summarise(cti)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "Data_120_Study_Dataset_with_Provenance.xlsx",
    )
    ap.add_argument("--sheet", default="Data_120")
    ap.add_argument(
        "--mode",
        choices=["four", "five"],
        default="four",
        help="four = paper-style (Accuracy, Compliance, Resilience, Cost); "
             "five = all metrics including Batch Failure",
    )
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--equal", action="store_true", help="Equal weights")
    ap.add_argument(
        "--invert",
        nargs="*",
        default=None,
        help="Metrics to invert (default: Batch Failure when present)",
    )
    ap.add_argument("--norm", choices=["minmax", "range100"], default="minmax")
    ap.add_argument(
        "--fit-table3",
        action="store_true",
        help="Search weights so CTI summary matches Table 3 (post-hoc fit)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.data.exists():
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        return 1

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = load_data(args.data, sheet)

    metrics = METRICS_FOUR if args.mode == "four" else METRICS_FIVE
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        return 1

    if args.invert is not None:
        invert = set(args.invert)
    else:
        invert = INVERT_DEFAULT & set(metrics)

    # ----- choose weights -----
    if args.fit_table3:
        print("Searching weights to match Table 3 (post-hoc fit) …")
        w_hat, fitted_summary = fit_weights_to_table3(
            df, metrics, invert, args.norm, seed=args.seed, maxiter=args.maxiter
        )
        weights = w_hat.tolist()
        print("Optimisation finished.")
    elif args.equal:
        weights = [1.0] * len(metrics)
    elif args.weights is not None:
        weights = args.weights
    else:
        weights = (
            DEFAULT_WEIGHTS_FOUR if args.mode == "four" else DEFAULT_WEIGHTS_FIVE
        )

    cti, Z, w_used = compute_cti(df, metrics, weights, invert, args.norm)
    summary = summarise(cti)
    cmp = compare_to_table3(summary)

    # ----- console report -----
    print("=" * 64)
    print("COMPOSITE TRUST INDEX (CTI) ANALYSIS")
    print("=" * 64)
    print(f"Data file     : {args.data.name}")
    print(f"Mode          : {args.mode}-metric")
    print(f"Normalisation : {args.norm}")
    print(f"Inverted      : {sorted(invert) if invert else '(none)'}")
    print(f"Fit to Table3 : {'YES (post-hoc)' if args.fit_table3 else 'NO'}")
    print()
    print("Weights used (re-scaled to sum = 1):")
    for m, wi in zip(metrics, w_used):
        print(f"  {m:40s}  {wi:.4f}")
    print()
    print("CTI distribution:")
    for k, v in summary.items():
        print(f"  {k:8s}  {v:.6f}" if k != "n" else f"  {k:8s}  {v}")
    print()
    print("Comparison with manuscript Table 3:")
    print(cmp.to_string(index=False))
    mae = cmp["Abs_Diff"].mean()
    print(f"\nMean absolute difference vs Table 3: {mae:.6f}")
    if mae < 0.02:
        print("→ Close match to published CTI summary.")
    elif mae < 0.05:
        print("→ Moderate match; weights are illustrative / approximate.")
    else:
        print("→ Distribution differs from Table 3; consider --fit-table3 or different weights.")

    # one-sample style note on mean
    print()
    print(f"CTI mean vs Table 3 mean (0.486): difference = {summary['mean'] - 0.486:+.4f}")

    # ----- export -----
    if args.out is None:
        args.out = (
            Path(__file__).resolve().parent.parent / "cti_analysis_results.xlsx"
        )

    out_df = df.copy()
    # keep useful ID columns if present
    out_df["CTI"] = cti
    for m in metrics:
        out_df[f"z_{m}"] = Z[m]

    weights_df = pd.DataFrame({
        "Metric": metrics,
        "Weight": w_used,
        "Inverted": [m in invert for m in metrics],
        "Norm": args.norm,
    })
    summary_df = pd.DataFrame([summary])
    notes = pd.DataFrame([
        ["Assumption", "Paper does not publish exact CTI weights; values here are explicit and documented."],
        ["Mode", args.mode],
        ["Fit_Table3", "Yes — post-hoc weight search" if args.fit_table3 else "No — user/default weights"],
        ["Table3_source", "paper_claims.json / manuscript Table 3"],
        ["Integrity_note", "A good fit to Table 3 does not prove these were the original weights."],
    ], columns=["Key", "Value"])

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="CTI_per_row", index=False)
        summary_df.to_excel(writer, sheet_name="CTI_Summary", index=False)
        cmp.to_excel(writer, sheet_name="vs_Table3", index=False)
        weights_df.to_excel(writer, sheet_name="Weights", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"\nSaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
