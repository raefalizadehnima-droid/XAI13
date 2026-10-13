# Complete User Guide for `cti_audit.py`

Audit and reverse-engineering script for the Composite Trust Index (CTI)

---

## 1. What does this script do?

Given a table (Excel or CSV) with metric columns, the script builds an Excel workbook with live formulas that answers three questions:

1. **Can the CTI weights be recovered?** If you have the CTI of every row, yes, exactly. If you only have CTI summary statistics, only a post-hoc fit is possible.
2. **How reliable are the fitted weights?** The `Weight_Spread` sheet shows whether the data actually determine the weights.
3. **Are the model CTI/PMI tables consistent with the CTI distribution?** For example, no model's CTI can exceed the maximum of the distribution.

---

## 2. Installation

```bash
python -m pip install numpy pandas scipy openpyxl
```

Python 3.9 or newer is required. The `--recalc` option needs LibreOffice (the `soffice` command); it is optional.

---

## 3. Inputs

| Input | Required? | Description |
|---|---|---|
| Data file (`--data`) | Yes | CSV or XLSX with the metric columns |
| Column names (`--metrics`) | Yes | Exactly as written in the file headers |
| Per-row CTI column (`--cti-column`) | One of three | For **Mode A** (exact recovery) |
| CTI summary statistics (`--target` or `--target-from`) | One of three | For **Mode B** (post-hoc fit) |
| Model tables (`--models`) | No | JSON file for the `Model_Consistency` sheet |

Provide at least one of `--cti-column`, `--target` or `--target-from`, otherwise the script exits with an error.

---

## 4. All options

| Option | Default | Description |
|---|---|---|
| `--data` | – | Path to the data file (`.csv` or `.xlsx`) |
| `--sheet` | first sheet | Sheet name, Excel only |
| `--metrics` | – | Space-separated metric column names (quote names containing spaces) |
| `--invert` | none | Metrics where **lower is better** (e.g. Batch Failure Rate). After normalisation they become `1 − x` |
| `--norm` | `minmax` | `minmax`: each column's min/max in the data. `range100`: fixed 0–100 range for all columns |
| `--cti-column` | – | Name of the CTI column in the data file (Mode A) |
| `--target` | – | CTI summary as `mean=…,std=…,min=…,q25=…,q50=…,q75=…,max=…` (all seven keys required) |
| `--target-from` | – | An Excel file holding the CTI summary as text (like the `describe()` output in DADEH.xlsx) |
| `--models` | example tables | JSON file with the model tables |
| `--seeds` | 6 | Number of independent optimiser runs (Mode B) |
| `--maxiter` | 250 | Maximum optimiser iterations |
| `--spread-tol` | 5.0 | Loss tolerance for accepting "near-optimal" solutions in `Weight_Spread` |
| `--out` | `CTI_audit.xlsx` | Output file name |
| `--recalc` | off | Recalculate formulas with LibreOffice |

---

## 5. The two modes

### Mode A: exact recovery (you have per-row CTI)

```bash
python cti_audit.py --data data.xlsx --sheet Sheet1 \
  --metrics Accuracy FailureRate CostReduction Compliance Resilience \
  --invert FailureRate \
  --cti-column CTI --out audit.xlsx
```

The script runs two regressions on the normalised metrics:

- **OLS** with an intercept
- **NNLS** with non-negative weights and no intercept

If R² is close to 1, CTI is a linear combination of those metrics and the weights are exact. If R² is low, CTI has another form (e.g. geometric or nonlinear) or a different normalisation. This is the only mode that truly reaches the "origin".

### Mode B: post-hoc fit (summary statistics only)

```bash
python cti_audit.py --data benchmark.xlsx --sheet Data_120 \
  --metrics "AI_Model_Accuracy (%)" "Batch_Failure_Rate (%)" "Production_Cost_Reduction (%)" \
            "Regulatory_Compliance_Score" "System_Resilience_Score" \
  --invert "Batch_Failure_Rate (%)" \
  --target-from DADEH.xlsx --out CTI_audit.xlsx --recalc
```

Or by entering the statistics directly:

```bash
python cti_audit.py --data d.csv --metrics A B C \
  --target "mean=0.486,std=0.083,min=0.210,q25=0.438,q50=0.480,q75=0.547,max=0.675"
```

Four CTI forms are fitted:

| Form | Definition |
|---|---|
| `simple` | Weighted sum of normalised metrics, weights non-negative and summing to 1 |
| `equal` | All weights equal (reference for comparison) |
| `geo` | Weighted geometric mean |
| `affine` | `a + b × (weighted sum)` with free scale and offset |

For `--target-from`, the script looks after the text "Composite Trust Index" or "CTI" for lines like `mean 0.486293`. If all seven statistics are not found, it stops with a clear message.

### Combining both

If you give both `--cti-column` and `--target`, both analyses are run.

---

## 6. Output workbook

| Sheet | Contents |
|---|---|
| `Summary` | Summary, run mode, best form, warning if the data are labelled simulated |
| `CTI_Fit` | Target statistics vs each form, differences (Δ), sum of squares and max error, fitted weights (Mode B) |
| `Weight_Spread` | Weight ranges across near-optimal solutions. The "Identifiable?" column shows `CHECK` when the range exceeds 0.05 (Mode B) |
| `Exact_Recovery` | OLS and NNLS weights, R², max residual and verdict (Mode A) |
| `Model_Consistency` | Checks of the model CTI table and the PMI table |
| `Next_Steps` | What is needed to recover the true origin |

Colours: blue text = inputs or optimiser output, black text = formulas, green fill = check passed, red fill = check failed.

### Checks in `Model_Consistency`

- Does any model's CTI exceed the distribution maximum? (`EXCEEDS`)
- Does the mean of the model CTIs match the distribution mean? (difference below 0.02)
- Does the reported CI half-width match `z·SD/√n`? (difference below 0.01)
- In the PMI table: does the reported % reduction match the recomputed value, and is PMI ÷ energy-normalised PMI roughly constant?

---

## 7. Model tables JSON (`--models`)

```json
{
  "cti": [
    ["Baseline ML", 0.41, 0.04, 0],
    ["XAI",         0.44, 0.05, 1.3]
  ],
  "pmi": [
    ["Baseline ML", 42.3, 1.2, null, 0.229],
    ["XAI",         41.8, 1.1, 0.012, 0.224]
  ]
}
```

- `cti` rows: model name, CTI, CI half-width, rank stability (%)
- `pmi` rows: model name, PMI, CI half-width, reported % reduction (`null` for the baseline, otherwise a fraction such as `0.012`), energy-normalised PMI
- The first `pmi` row is the reference model; the other reductions are computed relative to it.

---

## 8. Interpreting the results

| Observation | Meaning |
|---|---|
| R² ≈ 1 in `Exact_Recovery` | CTI is a linear combination of these metrics and the weights are valid |
| Low R² in `Exact_Recovery` | CTI's form or normalisation differs |
| `simple` and `geo` far from target | Simple non-negative weights cannot generate the target statistics |
| `affine` close but `CHECK` in `Weight_Spread` | The match comes from free parameters; weights are not determined |
| `EXCEEDS` in the model table | That CTI cannot lie within the distribution's range |
| `INCONSISTENT` in the CI check | n or the CI method must be stated |

---

## 9. Scientific-integrity notes (please read)

1. **Mode B fitting is post-hoc.** Weights tuned only to hit desired statistics are not the original weights and must not be reported in a paper as a-priori weights.
2. If the status column (`Data_Status`) contains `SIMULATED`, the script warns in `Summary`. Results on simulated data are illustrative only.
3. The `affine` form has the number of metrics plus 2 parameters and fits only 7 statistics. A good match proves nothing.
4. The 0.05 threshold in `Weight_Spread` is a rule of thumb, not a statistical test.
5. Rows with empty or non-numeric values are dropped and a warning is printed.

---

## 10. Troubleshooting

| Message / problem | Fix |
|---|---|
| `Columns not found` | Column names must match the file exactly, including spaces and parentheses. The available columns are printed in the message |
| `Could not find CTI stats` | The `--target-from` file has no text statistics (mean, std, min, 25%, 50%, 75%, max). Use `--target` instead |
| `Provide --cti-column, --target, or --target-from` | Add one of these |
| Run is slow | Use `--seeds 3 --maxiter 150` |
| Formula cells are empty in a previewer | Run with `--recalc` or open the file in Excel |
| `--recalc` has no effect | `soffice` was not found on the system |
| Results differ slightly between runs | Seeds are fixed, so runs are reproducible; differences only appear when `--seeds` or `--maxiter` change |

---

## 11. What is needed to reach the "origin"?

- The CTI value of each row, or the CTI formula and code
- The original, non-simulated metric rows
- The n and CI method used for the model CTIs
- What the CTI histogram actually represents

With these, Mode A returns the weights exactly and the rest of the table can be recomputed with the same formula.

---

## 12. Added metrics: R², RMSE, ECE, explanation stability

| Metric | Where | Needs | How it is computed |
|---|---|---|---|
| R² and RMSE (row level) | `Metrics` (Mode A) | per-row CTI (`--cti-column`) | Live formulas on the `Row_Predictions` sheet (OLS and NNLS), plus K-fold out-of-fold R²/RMSE (`--cv`, default 5) |
| R² and RMSE (7 statistics) | `Metrics` (Mode B) | CTI summary statistics | Formulas over the Δ columns of `CTI_Fit`. They compare only the 7 summary statistics, not rows |
| ECE, MCE, Brier | `Metrics` | `--conf-column` (confidence in [0,1]) and `--correct-column` (0/1), `--ece-bins` (default 10) | Equal-width bins; ECE = Σ(count × |accuracy − confidence|) / n, as a live formula |
| Explanation stability | `Explanation_Stability` | Mode A: per-row CTI. Mode B: none | Mode A: `--boot` (default 500) row-bootstrap refits of NNLS weights; reports mean pairwise Spearman of weight ranks, mean cosine to the reference weights, top-feature agreement, per-weight SD, CV, 95 % interval and rank-match %. Mode B: the same measures over near-optimal optimiser solutions (this is optimiser stability, not data resampling) |

Notes:
- ECE cannot be computed from aggregate accuracy. Without the two per-sample columns the sheet says so instead of inventing a value.
- "Explanation stability" here means stability of the fitted CTI weights, not SHAP/LIME stability of an ML model.
- The Spearman (0.8) and cosine (0.95) thresholds are editable heuristics.
- In Mode B, `simple` and `geo` converge to one solution, so their stability is trivially 1. Only `affine` shows the real non-identifiability.

Example with calibration columns:
```bash
python cti_audit.py --data d.csv --metrics A B C D E --invert B --cti-column CTI \
  --conf-column confidence --correct-column correct --boot 1000 --out audit.xlsx
```
