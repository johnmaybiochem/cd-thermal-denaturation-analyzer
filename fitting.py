"""
Two-state thermal unfolding model for circular dichroism data.
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
    """Fraction unfolded at each temperature given Tm and dH."""
    T_K  = T_C + 273.15
    Tm_K = Tm  + 273.15
    dG   = dH * (1.0 - T_K / Tm_K)
    K    = np.exp(-dG / (R * T_K))
    return K / (1.0 + K)


def fit_replicate(T, y, n_baseline_pts=15):
    """Fit a single replicate. Returns dict of parameters and diagnostics."""
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
    Fit all replicates in a DataFrame.
    Returns (per_rep_df, summary_df, fu_matrix, T_array).
    """
    T       = df[temp_col].values
    rep_cols = [c for c in df.columns if c != temp_col]

    records, fu_rows, errors = [], [], []

    for rep in rep_cols:
        y = df[rep].dropna()
        # align T to non-NaN y indices
        T_rep = T[df[rep].notna().values]
        try:
            res = fit_replicate(T_rep, y.values, n_baseline_pts)
            # interpolate fu back to full T grid if lengths differ
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
