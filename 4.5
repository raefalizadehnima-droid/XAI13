#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trust_simulation.py - transparent, seeded simulation that COMPUTES the quantities
reported for five model configurations (Baseline ML, XAI, HITL, Hybrid AI, Hybrid+UQ)
in a paper-style "Table 4" (PMI) and "Table 5" (CTI robustness).

WHAT THIS SCRIPT IS
-------------------
Everything here is SIMULATED. A synthetic batch-reactor "world" generates data; each
configuration is trained and evaluated on it; PMI, calibration, explanation stability and
a Composite Trust Index (CTI) are computed from those results. No number is typed in or
tuned to match a target; whatever the code produces is what is reported.

By construction the simulated process contains Arrhenius-type kinetics, so a model with
a kinetic term has a structural advantage in this world. Results therefore illustrate how
the assessment framework behaves; they are NOT evidence about real plants.

SIMULATED WORLD
---------------
Process variables x = (T, P, catalyst, solvent_polarity, energy). True yield (%) is a
smooth function with Arrhenius conversion, solvent selectivity optimum, high-energy
degradation, process noise, and a time-ordered drift in catalyst activity.
PMI = (reagent + solvent + catalyst mass) / (yield * theoretical product mass).

CONFIGURATIONS
--------------
 baseline : MLP regression on x (no UQ; constant Gaussian sigma = held-out validation RMSE)
 xai      : baseline, then permutation-importance pruning (share < 5%) and refit
 hitl     : baseline + expert correction y = yhat + eta*(y_expert - yhat) (Eq. 9) for a
            fraction of cases; expert values are noisy
 hybrid   : Arrhenius kinetic term (fitted by least squares) + MLP residual (Eq. 11-12)
 hybrid_uq: hybrid with MC-dropout residual (Eq. 4-6), variance recalibration on a
            validation split, and uncertainty-penalised selection (mean - kappa*sd)

PROCESS SETTING SELECTION (for PMI)
-----------------------------------
Each configuration scores N random candidate settings and picks the best
(score = predicted yield - energy penalty [- kappa*sd for UQ]). PMI, yield and energy are then
evaluated on the TRUE (noise-free) simulator at that setting.

METRICS (per repetition, per configuration)
-------------------------------------------
 R2, RMSE (blocked-CV pooled test predictions); ECE_reg (regression analogue of Eq. 7:
 mean |observed - nominal| central-interval coverage over levels 0.1..0.9);
 mean predictive variance U (Eq. 6); explanation stability E (Eq. 8, on
 permutation-importance shares, scaled so the mean share = 1); PMI; energy-adjusted PMI.
 NOTE: permutation importance is used instead of SHAP (no extra dependency).

CTI
---
CTI = sum_k w_k z_k, with z_k min-max normalised over ALL repetitions and configurations
(pooled), 'lower is better' components inverted. Components: R2, RMSE, ECE_reg, E, PMI.
Default weights equal (0.2 each). Weight robustness: n_weight_samples Dirichlet(1) vectors.

STATISTICS
----------
Paired t-tests and paired Cohen's d (Hybrid+UQ vs each other configuration, on CTI and
PMI) across repetitions, Bonferroni at alpha/m; one-way ANOVA on CTI; bootstrap of the
CTI difference (Hybrid+UQ - Baseline); hypervolume (Monte-Carlo) in the normalised
5-objective space; partial correlation CTI-PMI controlling for yield (pooled).
Mediation is NOT computed.

USAGE
-----
  python trust_simulation.py --out results_trust_simulation.xlsx          (default n_rep=30)
  python trust_simulation.py --n-rep 5 --epochs 200 --out quick.xlsx      (quick check)
Requires: numpy pandas scipy openpyxl.   Runtime: a few minutes at defaults.
"""
import argparse, json, time
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import least_squares

CONFIGS = ["Baseline ML", "XAI", "HITL", "Hybrid AI", "Hybrid + UQ"]
DEFAULTS = dict(seed=42, n_batches=120, folds=5, n_rep=30, n_candidates=300, mc_samples=50,
                dropout=0.1, epochs=500, hidden=32, lr=0.01, wd=1e-4, kappa=1.0,
                energy_penalty=0.8, expert_eta=0.6, expert_noise=1.5, expert_fraction=0.2,
                expert_top_k=3, stability_bootstrap=20, n_weight_samples=10000,
                n_bootstrap_delta=1000, hv_mc_points=400000, alpha=0.01,
                process_noise=2.0, drift=0.10, prune_share=0.05)
LO = np.array([320., 1., 0.5, 0.2, 50.]); HI = np.array([380., 5., 3., 1.0, 150.])  # T,P,cat,polarity,energy
R_GAS = 8.314

# ----------------------------------------------------------------- simulated world
def true_yield(X, k_scale=1.0):
    T, P, c, s, E = X.T
    k = 2.0e7 * np.exp(-6.0e4 / (R_GAS * T)) * k_scale
    conv = 1 - np.exp(-k * (c ** 0.5) * (P ** 0.3) * 3.0)
    sel = 0.75 + 0.25 * np.exp(-((s - 0.65) / 0.25) ** 2)
    degr = 0.18 * np.clip(E - 120.0, 0, None)
    return np.clip(100.0 * conv * sel - degr, 0, 100)

def pmi_energy(X, Y):
    T, P, c, s, E = X.T
    m_reag = 1.0; m_solv = 8.0 + 14.0 * s; m_cat = 0.02 * c
    prod = np.maximum(Y, 1.0) / 100.0
    return (m_reag + m_solv + m_cat) / prod, (m_reag + m_solv + m_cat + 0.02 * E) / prod

def make_batches(n, rng, p):
    X = LO + (HI - LO) * rng.random((n, 5))
    drift = np.where(np.arange(n) >= int(0.75 * n), 1.0 - p["drift"], 1.0)   # time-ordered shift
    y = true_yield(X, drift) + rng.normal(0, p["process_noise"], n)
    return X, np.clip(y, 0, 100)

# ----------------------------------------------------------------- numpy MLP (+dropout)
class MLP:
    def __init__(self, d, p, rng, dropout=0.0):
        h = p["hidden"]; self.p = p; self.rng = rng; self.dr = dropout
        s = lambda a, b: rng.normal(0, np.sqrt(2.0 / a), (a, b))
        self.W = [s(d, h), s(h, h), s(h, 1)]; self.b = [np.zeros(h), np.zeros(h), np.zeros(1)]
    def _fwd(self, X, drop):
        a = [X]; masks = []
        for i in range(2):
            z = a[-1] @ self.W[i] + self.b[i]; h = np.maximum(z, 0)
            if drop and self.dr > 0:
                m = (self.rng.random(h.shape) > self.dr) / (1 - self.dr); h = h * m; masks.append(m)
            else: masks.append(None)
            a.append(h)
        return a[-1] @ self.W[2] + self.b[2], a, masks
    def fit(self, X, y):
        p = self.p; self.mx, self.sx = X.mean(0), X.std(0) + 1e-9; self.my, self.sy = y.mean(), y.std() + 1e-9
        Xs = (X - self.mx) / self.sx; ys = ((y - self.my) / self.sy)[:, None]; n = len(Xs)
        params = self.W + self.b; m = [np.zeros_like(q) for q in params]; v = [np.zeros_like(q) for q in params]
        for t in range(1, p["epochs"] + 1):
            out, a, masks = self._fwd(Xs, True); g = 2 * (out - ys) / n
            gW2 = a[2].T @ g + p["wd"] * self.W[2]; gb2 = g.sum(0)
            d2 = (g @ self.W[2].T) * (a[2] > 0) * (masks[1] if masks[1] is not None else 1)
            gW1 = a[1].T @ d2 + p["wd"] * self.W[1]; gb1 = d2.sum(0)
            d1 = (d2 @ self.W[1].T) * (a[1] > 0) * (masks[0] if masks[0] is not None else 1)
            gW0 = Xs.T @ d1 + p["wd"] * self.W[0]; gb0 = d1.sum(0)
            grads = [gW0, gW1, gW2, gb0, gb1, gb2]
            for i, (q, gq) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gq; v[i] = 0.999 * v[i] + 0.001 * gq ** 2
                q -= p["lr"] * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        return self
    def predict(self, X):
        out, _, _ = self._fwd((X - self.mx) / self.sx, False); return out[:, 0] * self.sy + self.my
    def predict_mc(self, X, T):
        Xs = (X - self.mx) / self.sx
        return np.stack([self._fwd(Xs, True)[0][:, 0] * self.sy + self.my for _ in range(T)])

# ----------------------------------------------------------------- kinetic term (hybrid)
class Kinetic:
    def fit(self, X, y):
        Tn = X[:, 0]
        def f(th, T): return th[0] * (1 - np.exp(-np.exp(th[1] - th[2] * 1e4 / (R_GAS * T))))
        best = None
        for lnA in (5., 15., 25.):
            for Ea in (3., 6., 9.):
                try: r = least_squares(lambda th: f(th, Tn) - y, [90., lnA, Ea], bounds=([50, -10, 0.5], [100, 60, 20]))
                except Exception: continue
                if best is None or r.cost < best.cost: best = r
        self.th = best.x; self.f = f; return self
    def predict(self, X): return self.f(self.th, X[:, 0])

class Hybrid:
    def __init__(self, p, rng, dropout=0.0): self.p, self.rng, self.dr = p, rng, dropout
    def fit(self, X, y):
        self.kin = Kinetic().fit(X, y); self.res = MLP(X.shape[1], self.p, self.rng, self.dr).fit(X, y - self.kin.predict(X)); return self
    def predict(self, X): return self.kin.predict(X) + self.res.predict(X)
    def predict_mc(self, X, T): return self.kin.predict(X)[None, :] + self.res.predict_mc(X, T)

# ----------------------------------------------------------------- helpers
def ece_reg(y, mu, sd):
    sd = np.maximum(sd, 1e-6); lv = np.linspace(0.1, 0.9, 9)
    z = np.abs(y - mu) / sd
    return float(np.mean([abs(np.mean(z <= stats.norm.ppf((1 + q) / 2)) - q) for q in lv]))

def perm_importance(pred_fn, X, y, rng, idx_boot):
    base = np.mean((y - pred_fn(X)) ** 2); out = []
    for idx in idx_boot:
        Xb, yb = X[idx], y[idx]; b0 = np.mean((yb - pred_fn(Xb)) ** 2); imp = []
        for j in range(X.shape[1]):
            Xp = Xb.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); imp.append(max(np.mean((yb - pred_fn(Xp)) ** 2) - b0, 0))
        imp = np.array(imp); out.append(imp / (imp.sum() + 1e-12) * X.shape[1])
    return np.array(out)       # shares scaled so mean share = 1

def stability_E(imps): return float(1 - np.mean(np.var(imps, axis=0)))   # Eq. 8

def select_setting(score_fn, rng, p):
    C = LO + (HI - LO) * rng.random((p["n_candidates"], 5)); return C, score_fn(C)

# ----------------------------------------------------------------- one repetition
def run_rep(rep, p, master):
    rng = np.random.default_rng(master + rep)
    X, y = make_batches(p["n_batches"], rng, p); n = len(y)
    folds = np.array_split(np.arange(n), p["folds"])          # contiguous = time-blocked
    preds = {c: np.zeros(n) for c in CONFIGS}; sds = {c: np.zeros(n) for c in CONFIGS}
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        nv = max(int(0.2 * len(tr)), 10); va, tr2 = tr[-nv:], tr[:-nv]      # inner validation split (sigma estimation / UQ scaling)
        rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))) + 1e-6
        base = MLP(5, p, rng).fit(X[tr], y[tr])
        res_b = rm(y[va], MLP(5, p, rng).fit(X[tr2], y[tr2]).predict(X[va]))   # held-out residual sd (no in-sample optimism)
        preds["Baseline ML"][f] = base.predict(X[f]); sds["Baseline ML"][f] = res_b
        # XAI: prune by permutation importance then refit
        imp = perm_importance(base.predict, X[tr], y[tr], rng, [np.arange(len(tr))]).mean(0) / 5
        keep = imp >= p["prune_share"]; keep = keep if keep.any() else np.ones(5, bool)
        class Sub:   # wrapper on kept columns
            def __init__(s, m, k): s.m, s.k = m, k
            def predict(s, Z): return s.m.predict(Z[:, s.k])
        xm = MLP(int(keep.sum()), p, rng).fit(X[tr][:, keep], y[tr])
        preds["XAI"][f] = xm.predict(X[f][:, keep]); sds["XAI"][f] = rm(y[va], MLP(int(keep.sum()), p, rng).fit(X[tr2][:, keep], y[tr2]).predict(X[va][:, keep]))
        # HITL: expert corrects a random fraction of the test cases (Eq. 9)
        ph = preds["Baseline ML"][f].copy(); sdh = np.full(len(f), res_b)
        rev = rng.random(len(f)) < p["expert_fraction"]
        yexp = true_yield(X[f]) + rng.normal(0, p["expert_noise"], len(f))
        ph[rev] = ph[rev] + p["expert_eta"] * (yexp[rev] - ph[rev])
        sdh[rev] = np.sqrt((1 - p["expert_eta"]) ** 2 * res_b ** 2 + p["expert_eta"] ** 2 * p["expert_noise"] ** 2)
        preds["HITL"][f] = ph; sds["HITL"][f] = sdh
        # Hybrid
        hy = Hybrid(p, rng).fit(X[tr], y[tr])
        preds["Hybrid AI"][f] = hy.predict(X[f]); sds["Hybrid AI"][f] = rm(y[va], Hybrid(p, rng).fit(X[tr2], y[tr2]).predict(X[va]))
        # Hybrid + UQ: MC dropout, variance scaling on a validation split
        hu = Hybrid(p, rng, p["dropout"]).fit(X[tr2], y[tr2])
        mcv = hu.predict_mc(X[va], p["mc_samples"]); s0 = mcv.std(0) + 1e-6
        scale = max(np.sqrt(np.mean((y[va] - mcv.mean(0)) ** 2 / (s0 ** 2 + 1e-6))), 0.1)
        hu = Hybrid(p, rng, p["dropout"]).fit(X[tr], y[tr]); mc = hu.predict_mc(X[f], p["mc_samples"])
        preds["Hybrid + UQ"][f] = mc.mean(0); sds["Hybrid + UQ"][f] = (mc.std(0) + 1e-6) * scale
    rows = {}
    # final models on all batches -> explanation stability and setting selection
    boot = [rng.integers(0, n, n) for _ in range(p["stability_bootstrap"])]
    final = {"Baseline ML": MLP(5, p, rng).fit(X, y), "Hybrid AI": Hybrid(p, rng).fit(X, y), "Hybrid + UQ": Hybrid(p, rng, p["dropout"]).fit(X, y)}
    imp_b = perm_importance(final["Baseline ML"].predict, X, y, rng, boot)
    keepF = imp_b.mean(0) / 5 >= p["prune_share"]; keepF = keepF if keepF.any() else np.ones(5, bool)
    xf = MLP(int(keepF.sum()), p, rng).fit(X[:, keepF], y)
    final["XAI"] = xf
    def pred_of(c):
        if c == "XAI": return lambda Z: xf.predict(Z[:, keepF])
        if c == "HITL": return final["Baseline ML"].predict
        return final[c].predict
    for c in CONFIGS:
        yh, sd = preds[c], sds[c]
        U = float(np.mean(sd ** 2))
        imps = perm_importance(pred_of(c), X, y, rng, boot)
        # selection
        C = LO + (HI - LO) * rng.random((p["n_candidates"], 5)); En = (C[:, 4] - LO[4]) / (HI[4] - LO[4])
        pen = p["energy_penalty"] * 10 * En
        if c == "Hybrid + UQ":
            m = final[c].predict_mc(C, p["mc_samples"]); score = m.mean(0) - p["kappa"] * m.std(0) - pen
        else:
            score = pred_of(c)(C) - pen
        if c == "HITL":
            top = np.argsort(-score)[:p["expert_top_k"]]; mu = pred_of(c)(C[top])
            ye = true_yield(C[top]) + rng.normal(0, p["expert_noise"], len(top))
            corr = mu + p["expert_eta"] * (ye - mu) - pen[top]; pick = top[np.argmax(corr)]
        else: pick = np.argmax(score)
        Xs = C[pick:pick + 1]; Ys = true_yield(Xs); pm, pme = pmi_energy(Xs, Ys)
        rows[c] = dict(rep=rep, config=c, R2=1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2),
                       RMSE=float(np.sqrt(np.mean((y - yh) ** 2))), ECE=ece_reg(y, yh, sd), U=U, E=stability_E(imps),
                       PMI=float(pm[0]), PMI_energy=float(pme[0]), yield_sel=float(Ys[0]), energy_sel=float(Xs[0, 4]))
    return list(rows.values())

# ----------------------------------------------------------------- aggregation
def pooled_components(df):
    z = pd.DataFrame(index=df.index)
    for col, inv in [("R2", False), ("RMSE", True), ("ECE", True), ("E", False), ("PMI", True)]:
        v = df[col]; lo, hi = v.min(), v.max(); zz = (v - lo) / (hi - lo + 1e-12); z[col] = 1 - zz if inv else zz
    return z

def hv_contrib(points, mc, rng):
    U = rng.random((mc, points.shape[1]))
    def dom(P): return np.any(np.all(U[:, None, :] <= P[None, :, :], axis=2), axis=1)
    full = dom(points).mean(); out = []
    for i in range(len(points)): out.append(full - dom(np.delete(points, i, axis=0)).mean())
    return full, out

def analyse(df, p, master):
    rng = np.random.default_rng(master + 99999)
    Z = pooled_components(df); w0 = np.full(5, 0.2); df = df.copy(); df["CTI"] = Z.values @ w0
    comp = ["R2", "RMSE", "ECE", "E", "PMI"]
    t4 = []; base = df[df.config == "Baseline ML"].PMI.mean(); basee = df[df.config == "Baseline ML"].PMI_energy.mean()
    for c in CONFIGS:
        d = df[df.config == c]; ci = stats.t.ppf(.975, len(d) - 1) * d.PMI.std() / np.sqrt(len(d))
        # partial correlation of CTI and PMI controlling for yield is computed pooled below
        t4.append(dict(Model=c, PMI_mean=d.PMI.mean(), PMI_CI95=ci, Reduction_vs_Baseline_pct=100 * (base - d.PMI.mean()) / base,
                       PMI_energy_mean=d.PMI_energy.mean(), Energy_adj_reduction_pct=100 * (basee - d.PMI_energy.mean()) / basee,
                       Yield_selected=d.yield_sel.mean()))
    t4 = pd.DataFrame(t4)
    # partial r (CTI, PMI | yield) pooled and per config
    def partial(a, b, c):
        ra = a - np.polyval(np.polyfit(c, a, 1), c); rb = b - np.polyval(np.polyfit(c, b, 1), c); return stats.pearsonr(ra, rb)
    r_all = stats.pearsonr(df.CTI, df.PMI); pr_all = partial(df.CTI.values, df.PMI.values, df.yield_sel.values)
    # robustness over weights (mean component values per config)
    M = np.array([Z[df.config == c].mean().values for c in CONFIGS])
    W = rng.dirichlet(np.ones(5), p["n_weight_samples"]); S = W @ M.T
    first = np.bincount(S.argmax(1), minlength=len(CONFIGS)) / len(W) * 100
    # hypervolume in normalised space
    full, hvc = hv_contrib(M, p["hv_mc_points"], rng)
    ctis = {c: df[df.config == c].CTI.values for c in CONFIGS}
    t5 = pd.DataFrame(dict(Model=CONFIGS, CTI_mean=[ctis[c].mean() for c in CONFIGS],
        CTI_CI95=[stats.t.ppf(.975, len(ctis[c]) - 1) * ctis[c].std() / np.sqrt(len(ctis[c])) for c in CONFIGS],
        First_place_pct_over_weights=first, HV_exclusive_contribution=hvc))
    comp_tab = df.groupby("config")[comp + ["U", "PMI", "PMI_energy", "yield_sel", "CTI"]].mean().loc[CONFIGS].reset_index()
    # tests: Hybrid+UQ vs others (paired over reps)
    ref = ctis["Hybrid + UQ"]; refp = df[df.config == "Hybrid + UQ"].PMI.values; tests = []
    m = 2 * (len(CONFIGS) - 1); a = p["alpha"] / m
    for c in CONFIGS[:-1]:
        for name, x, y_ in [("CTI", ref, ctis[c]), ("PMI", refp, df[df.config == c].PMI.values)]:
            d = x - y_; t, pv = stats.ttest_rel(x, y_); dz = d.mean() / (d.std(ddof=1) + 1e-12)
            tests.append(dict(Comparison=f"Hybrid + UQ vs {c}", Metric=name, mean_diff=d.mean(), t=t, p=pv, cohen_dz=dz, significant_bonferroni=bool(pv < a)))
    tests = pd.DataFrame(tests)
    F, pF = stats.f_oneway(*[ctis[c] for c in CONFIGS])
    # bootstrap of CTI difference
    d = ref - ctis["Baseline ML"]; bs = np.array([rng.choice(d, len(d)).mean() for _ in range(p["n_bootstrap_delta"])])
    extra = dict(ANOVA_F_CTI=F, ANOVA_p_CTI=pF, delta_CTI_mean=d.mean(), delta_CTI_boot_sd=bs.std(), delta_CTI_boot_ci95=[np.percentile(bs, 2.5), np.percentile(bs, 97.5)],
                 corr_CTI_PMI_r=r_all[0], corr_CTI_PMI_p=r_all[1], partial_r_controlling_yield=pr_all[0], partial_p=pr_all[1],
                 hypervolume_total=full, bonferroni_alpha_per_test=a, n_rep=int(df.rep.nunique()))
    return df, t4, t5, comp_tab, tests, extra

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results_trust_simulation.xlsx")
    for k, v in DEFAULTS.items(): ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(); p = {k: getattr(a, k) for k in DEFAULTS}; t0 = time.time(); rows = []
    for r in range(p["n_rep"]):
        rows += run_rep(r, p, p["seed"])
        if (r + 1) % 5 == 0: print(f"rep {r + 1}/{p['n_rep']}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df, t4, t5, comp, tests, extra = analyse(df, p, p["seed"])
    with pd.ExcelWriter(a.out) as w:
        pd.DataFrame([{"NOTE": "ALL VALUES ARE SIMULATED (see script docstring). Not evidence about real plants."}]).to_excel(w, sheet_name="README", index=False)
        pd.DataFrame(list(p.items()), columns=["parameter", "value"]).to_excel(w, sheet_name="Parameters", index=False)
        t4.to_excel(w, sheet_name="Table4_PMI", index=False); t5.to_excel(w, sheet_name="Table5_CTI", index=False)
        comp.to_excel(w, sheet_name="Component_means", index=False); tests.to_excel(w, sheet_name="Paired_tests", index=False)
        pd.DataFrame([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in extra.items()}]).T.reset_index().to_excel(w, sheet_name="Extra_stats", index=False, header=["stat", "value"])
        df.to_excel(w, sheet_name="Per_repetition", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(comp.round(3)); print(t4.round(3)); print(t5.round(3)); print(tests.round(4)); print(json.dumps(extra, indent=1, default=float))
if __name__ == "__main__": main()
