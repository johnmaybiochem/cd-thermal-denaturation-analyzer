"""
Two-state and three-state thermal unfolding models for circular dichroism data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings

R = 8.314  # J/(mol·K)


def read_data_file(uploaded_file):
    """Read CSV or Excel file into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


# ── Two-state model ────────────────────────────────────────────────────────────

def two_state_sloping(T_C, Tm, dH, mN, bN, mU, bU):
    """
    Two-state unfolding model with linearly sloping baselines.

        CD_obs(T) = CD_N(T)·(1 - f_U) + CD_U(T)·f_U

    where:
        CD_N(T) = mN·T + bN          (native baseline)
        CD_U(T) = mU·T + bU          (unfolded baseline)
        f_U     = K / (1 + K)
        K       = exp(−ΔG / RT)
        ΔG(T)   = ΔH·(1 − T/Tm)     [Van't Hoff, T in Kelvin]
    """
    T_K  = T_C + 273.15
    Tm_K = Tm  + 273.15
    dG   = dH * (1.0 - T_K / Tm_K)
    K    = np.exp(-dG / (R * T_K))
    fu   = K / (1.0 + K)
    return (mN * T_C + bN) * (1 - fu) + (mU * T_C + bU) * fu


def fraction_unfolded(T_C, Tm, dH):
    """Fraction unfolded at each temperature given Tm and dH (two-state)."""
    T_K  = T_C + 273.15
    Tm_K = Tm  + 273.15
    dG   = dH * (1.0 - T_K / Tm_K)
    K    = np.exp(-dG / (R * T_K))
    return K / (1.0 + K)


def fit_replicate(T, y, n_baseline_pts=15):
    """Fit a single replicate (two-state). Returns dict of parameters and diagnostics."""
    pN = np.polyfit(T[:n_baseline_pts], y[:n_baseline_pts], 1)
    pU = np.polyfit(T[-n_baseline_pts:], y[-n_baseline_pts:], 1)
    mid_signal = (np.polyval(pN, T) + np.polyval(pU, T)) / 2
    Tm_g = T[np.argmin(np.abs(y - mid_signal))]

    p0 = [Tm_g, 300_000.0, pN[0], pN[1], pU[0], pU[1]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(two_state_sloping, T, y,
                            p0=p0, maxfev=100_000, method="lm")

    Tm_fit, dH_fit, mN, bN, mU, bU = popt
    Tm_K   = Tm_fit + 273.15
    dS_fit = dH_fit / Tm_K

    y_pred = two_state_sloping(T, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return {
        "Tm_C":    Tm_fit,
        "dH_kJ":   dH_fit / 1000.0,
        "dS_J":    dS_fit,
        "mN": mN, "bN": bN, "mU": mU, "bU": bU,
        "R2":      1.0 - ss_res / ss_tot,
        "RMSE":    np.sqrt(np.mean((y - y_pred) ** 2)),
        "fu_data": fraction_unfolded(T, Tm_fit, dH_fit),
        "y_pred":  y_pred,
    }


def fit_condition(df, temp_col="T", n_baseline_pts=15):
    """
    Fit all replicates in a DataFrame (two-state model).
    Returns (per_rep_df, summary_df, fu_matrix, T_array).
    """
    T       = df[temp_col].values
    rep_cols = [c for c in df.columns if c != temp_col]

    records, fu_rows, errors = [], [], []

    for rep in rep_cols:
        y = df[rep].dropna()
        T_rep = T[df[rep].notna().values]
        try:
            res = fit_replicate(T_rep, y.values, n_baseline_pts)
            if len(T_rep) == len(T):
                fu_full = res["fu_data"]
            else:
                fu_full = np.interp(T, T_rep, res["fu_data"])
            records.append({
                "Replicate":    rep,
                "Tm (°C)":      res["Tm_C"],
                "ΔH (kJ/mol)":  res["dH_kJ"],
                "ΔS (J/mol·K)": res["dS_J"],
                "R²":           res["R2"],
                "RMSE":         res["RMSE"],
            })
            fu_rows.append(fu_full)
            errors.append(None)
        except Exception as e:
            errors.append(str(e))
            records.append({
                "Replicate":    rep,
                "Tm (°C)":      np.nan,
                "ΔH (kJ/mol)":  np.nan,
                "ΔS (J/mol·K)": np.nan,
                "R²":           np.nan,
                "RMSE":         np.nan,
            })
            fu_rows.append(np.full(len(T), np.nan))

    per_rep_df = pd.DataFrame(records)
    per_rep_df["Fit OK"] = [e is None for e in errors]
    per_rep_df["Error"]  = errors

    ok       = per_rep_df["Fit OK"]
    num_cols = ["Tm (°C)", "ΔH (kJ/mol)", "ΔS (J/mol·K)", "R²", "RMSE"]
    summary  = per_rep_df.loc[ok, num_cols].agg(["mean", "std"])
    summary.index = ["Mean", "SD"]
    summary_df = summary.T.reset_index().rename(columns={"index": "Parameter"})

    fu_matrix = np.array(fu_rows)
    return per_rep_df, summary_df, fu_matrix, T


# ── Three-state model ──────────────────────────────────────────────────────────
#
# N ⇌ I ⇌ U   (two sequential equilibria)
#
# K1(T) = exp(−ΔG1/RT)   ΔG1 = ΔH1·(1 − T/Tm1)   N→I
# K2(T) = exp(−ΔG2/RT)   ΔG2 = ΔH2·(1 − T/Tm2)   I→U
#
# Populations:
#   fN = 1 / (1 + K1 + K1·K2)
#   fI = K1·fN
#   fU = K1·K2·fN
#
# Observed CD:
#   CD_obs = CD_N(T)·fN + CD_I(T)·fI + CD_U(T)·fU
#
# Baselines (all linear in T):
#   CD_N(T) = mN·T + bN
#   CD_I(T) = mI·T + bI
#   CD_U(T) = mU·T + bU
#
# Free parameters: Tm1, dH1, Tm2, dH2, mN, bN, mI, bI, mU, bU  (10 total)

def three_state_sloping(T_C, Tm1, dH1, Tm2, dH2, mN, bN, mI, bI, mU, bU):
    """
    Three-state sequential unfolding model (N ⇌ I ⇌ U) with sloping baselines.
    T_C in °C; Tm1, Tm2 in °C; dH1, dH2 in J/mol.
    """
    T_K   = T_C + 273.15
    Tm1_K = Tm1 + 273.15
    Tm2_K = Tm2 + 273.15

    dG1 = dH1 * (1.0 - T_K / Tm1_K)
    dG2 = dH2 * (1.0 - T_K / Tm2_K)

    K1 = np.exp(-dG1 / (R * T_K))
    K2 = np.exp(-dG2 / (R * T_K))

    denom = 1.0 + K1 + K1 * K2
    fN = 1.0 / denom
    fI = K1 / denom
    fU = K1 * K2 / denom

    CD_N = mN * T_C + bN
    CD_I = mI * T_C + bI
    CD_U = mU * T_C + bU

    return CD_N * fN + CD_I * fI + CD_U * fU


def populations_three_state(T_C, Tm1, dH1, Tm2, dH2):
    """
    Return (fN, fI, fU) arrays for the three-state model.
    dH1, dH2 in J/mol.
    """
    T_K   = T_C + 273.15
    Tm1_K = Tm1 + 273.15
    Tm2_K = Tm2 + 273.15

    dG1 = dH1 * (1.0 - T_K / Tm1_K)
    dG2 = dH2 * (1.0 - T_K / Tm2_K)

    K1 = np.exp(-dG1 / (R * T_K))
    K2 = np.exp(-dG2 / (R * T_K))

    denom = 1.0 + K1 + K1 * K2
    fN = 1.0 / denom
    fI = K1 / denom
    fU = K1 * K2 / denom
    return fN, fI, fU


def progress_variable_3state(T_C, Tm1, dH1, Tm2, dH2,
                              mN, bN, mI, bI, mU, bU):
    """
    Spectroscopically-weighted unfolding progress variable for the three-state model.

    Rather than assigning an arbitrary weight of 0.5 to the intermediate, the
    weight w_I is derived from where the intermediate baseline sits relative to
    the native and unfolded baselines at the reference temperature T_ref
    (midpoint between Tm1 and Tm2):

        w_I(T_ref) = [CD_I(T_ref) - CD_N(T_ref)] / [CD_U(T_ref) - CD_N(T_ref)]

    Progress = w_I · fI + 1 · fU

    This rises monotonically from 0 (all N) to 1 (all U) across both transitions,
    with a slope at each transition proportional to the actual CD amplitude change.

    Falls back to w_I = 0.5 if the N/U baselines are indistinguishable (|ΔCD| < 1e-9).
    """
    fN, fI, fU = populations_three_state(T_C, Tm1, dH1, Tm2, dH2)

    T_ref   = (Tm1 + Tm2) / 2.0
    CD_N_ref = mN * T_ref + bN
    CD_I_ref = mI * T_ref + bI
    CD_U_ref = mU * T_ref + bU

    denom_ref = CD_U_ref - CD_N_ref
    if abs(denom_ref) < 1e-9:
        w_I = 0.5
    else:
        w_I = (CD_I_ref - CD_N_ref) / denom_ref
        # Clamp to [0, 1] in case the intermediate sits outside N/U range
        w_I = float(np.clip(w_I, 0.0, 1.0))

    return w_I * fI + fU, w_I   # return (progress, w_I) so caller can report w_I


def fit_replicate_3state(T, y, n_baseline_pts=15):
    """
    Fit a single replicate to the three-state model.
    Returns dict of parameters and diagnostics.
    """
    # Initial guesses
    pN = np.polyfit(T[:n_baseline_pts], y[:n_baseline_pts], 1)
    pU = np.polyfit(T[-n_baseline_pts:], y[-n_baseline_pts:], 1)

    T_mid = T.mean()
    T_range = T.max() - T.min()
    # Guess Tm1 at ~1/3 and Tm2 at ~2/3 of temperature range
    Tm1_g = T.min() + T_range * 0.33
    Tm2_g = T.min() + T_range * 0.67

    # Intermediate baseline: midpoint between native and unfolded
    mI_g = (pN[0] + pU[0]) / 2
    bI_g = (pN[1] + pU[1]) / 2

    p0 = [
        Tm1_g, 200_000.0,   # Tm1, dH1
        Tm2_g, 200_000.0,   # Tm2, dH2
        pN[0], pN[1],       # mN, bN
        mI_g,  bI_g,        # mI, bI
        pU[0], pU[1],       # mU, bU
    ]

    # Bounds: Tm values must stay within data range; dH > 0
    T_lo, T_hi = T.min() - 5, T.max() + 5
    lower = [T_lo, 1e3,  T_lo, 1e3,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    upper = [T_hi, 1e9,  T_hi, 1e9,   np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(
            three_state_sloping, T, y,
            p0=p0, bounds=(lower, upper),
            maxfev=200_000,
        )

    Tm1, dH1, Tm2, dH2, mN, bN, mI, bI, mU, bU = popt

    # Enforce Tm1 < Tm2 convention (swap if needed)
    if Tm1 > Tm2:
        Tm1, dH1, Tm2, dH2 = Tm2, dH2, Tm1, dH1
        mN, bN, mU, bU = mU, bU, mN, bN
        popt = [Tm1, dH1, Tm2, dH2, mN, bN, mI, bI, mU, bU]

    Tm1_K = Tm1 + 273.15
    Tm2_K = Tm2 + 273.15
    dS1 = dH1 / Tm1_K
    dS2 = dH2 / Tm2_K

    y_pred = three_state_sloping(T, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    _, fI_arr, fU_arr = populations_three_state(T, Tm1, dH1, Tm2, dH2)

    return {
        "Tm1_C":   Tm1,
        "dH1_kJ":  dH1 / 1000.0,
        "dS1_J":   dS1,
        "Tm2_C":   Tm2,
        "dH2_kJ":  dH2 / 1000.0,
        "dS2_J":   dS2,
        "mN": mN, "bN": bN, "mI": mI, "bI": bI, "mU": mU, "bU": bU,
        "R2":      1.0 - ss_res / ss_tot,
        "RMSE":    np.sqrt(np.mean((y - y_pred) ** 2)),
        "fu_data": fU_arr,   # fU for fraction-unfolded plot
        "fi_data": fI_arr,
        "y_pred":  y_pred,
    }


def fit_condition_3state(df, temp_col="T", n_baseline_pts=15):
    """
    Fit all replicates in a DataFrame (three-state model).
    Returns (per_rep_df, summary_df, fu_matrix, fi_matrix, T_array).
    """
    T        = df[temp_col].values
    rep_cols = [c for c in df.columns if c != temp_col]

    records, fu_rows, fi_rows, errors = [], [], [], []

    for rep in rep_cols:
        y = df[rep].dropna()
        T_rep = T[df[rep].notna().values]
        try:
            res = fit_replicate_3state(T_rep, y.values, n_baseline_pts)
            if len(T_rep) == len(T):
                fu_full = res["fu_data"]
                fi_full = res["fi_data"]
            else:
                fu_full = np.interp(T, T_rep, res["fu_data"])
                fi_full = np.interp(T, T_rep, res["fi_data"])

            records.append({
                "Replicate":     rep,
                "Tm1 (°C)":      res["Tm1_C"],
                "ΔH1 (kJ/mol)":  res["dH1_kJ"],
                "ΔS1 (J/mol·K)": res["dS1_J"],
                "Tm2 (°C)":      res["Tm2_C"],
                "ΔH2 (kJ/mol)":  res["dH2_kJ"],
                "ΔS2 (J/mol·K)": res["dS2_J"],
                "mN": res["mN"], "bN": res["bN"],
                "mI": res["mI"], "bI": res["bI"],
                "mU": res["mU"], "bU": res["bU"],
                "R²":            res["R2"],
                "RMSE":          res["RMSE"],
            })
            fu_rows.append(fu_full)
            fi_rows.append(fi_full)
            errors.append(None)
        except Exception as e:
            errors.append(str(e))
            records.append({
                "Replicate":     rep,
                "Tm1 (°C)":      np.nan,
                "ΔH1 (kJ/mol)":  np.nan,
                "ΔS1 (J/mol·K)": np.nan,
                "Tm2 (°C)":      np.nan,
                "ΔH2 (kJ/mol)":  np.nan,
                "ΔS2 (J/mol·K)": np.nan,
                "R²":            np.nan,
                "RMSE":          np.nan,
            })
            fu_rows.append(np.full(len(T), np.nan))
            fi_rows.append(np.full(len(T), np.nan))

    per_rep_df = pd.DataFrame(records)
    per_rep_df["Fit OK"] = [e is None for e in errors]
    per_rep_df["Error"]  = errors

    ok       = per_rep_df["Fit OK"]
    num_cols = ["Tm1 (°C)", "ΔH1 (kJ/mol)", "ΔS1 (J/mol·K)",
                "Tm2 (°C)", "ΔH2 (kJ/mol)", "ΔS2 (J/mol·K)", "R²", "RMSE"]
    summary  = per_rep_df.loc[ok, num_cols].agg(["mean", "std"])
    summary.index = ["Mean", "SD"]
    summary_df = summary.T.reset_index().rename(columns={"index": "Parameter"})

    fu_matrix = np.array(fu_rows)
    fi_matrix = np.array(fi_rows)
    return per_rep_df, summary_df, fu_matrix, fi_matrix, T
