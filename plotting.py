"""
Plotly figure generation for CD thermal unfolding analysis.
"""

import numpy as np
import plotly.graph_objects as go
from fitting import fraction_unfolded

MARKER_SYMBOLS = {
    "Circle":        "circle",
    "Square":        "square",
    "Diamond":       "diamond",
    "Triangle Up":   "triangle-up",
    "Triangle Down": "triangle-down",
    "Cross":         "cross",
    "X":             "x",
    "Star":          "star",
}

LINE_STYLES = {
    "Solid":    "solid",
    "Dashed":   "dash",
    "Dotted":   "dot",
    "Dash-Dot": "dashdot",
}

LEGEND_POSITIONS = {
    "Top-right":    dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
    "Top-left":     dict(x=0.01, y=0.99, xanchor="left",  yanchor="top"),
    "Bottom-right": dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
    "Bottom-left":  dict(x=0.01, y=0.01, xanchor="left",  yanchor="bottom"),
    "Custom":       None,
}

FONT_FAMILIES = {
    "Arial (sans-serif)":      "Arial, sans-serif",
    "Helvetica":               "Helvetica, sans-serif",
    "Times New Roman (serif)": "Times New Roman, serif",
    "Georgia (serif)":         "Georgia, serif",
    "Courier New (mono)":      "Courier New, monospace",
    "Inter":                   "Inter, Arial, sans-serif",
}

GRID_DASHES = {
    "Solid":      "solid",
    "Dashed":     "dash",
    "Dotted":     "dot",
    "Dash-Dot":   "dashdot",
    "Long Dash":  "longdash",
}


def _fmt_axis_title(text, bold=False, italic=False):
    """Wrap axis title text in HTML bold/italic tags for Plotly."""
    if bold and italic:
        return f"<b><i>{text}</i></b>"
    if bold:
        return f"<b>{text}</b>"
    if italic:
        return f"<i>{text}</i>"
    return text


def hex_to_rgba(hex_color, opacity=1.0):
    """Convert hex color string and opacity to rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity:.2f})"


def build_figure(conditions, settings):
    """
    Build a Plotly figure for one or more fitted conditions.

    Parameters
    ----------
    conditions : list of dicts
        Each dict contains label, T, fu_matrix, per_rep_df,
        color, marker_symbol, line_style, show_individuals.
    settings : dict
        All layout / appearance options (see Graph Options tab).
    """
    s = settings
    g = s.get   # shorthand

    fig = go.Figure()

    # ── Shared trace style values ──────────────────────────────────────────────
    marker_size       = g("marker_size",        7)
    marker_border_w   = g("marker_border_width", 1.0)
    marker_border_col = g("marker_border_color", "#ffffff")
    marker_opacity    = g("marker_opacity",      1.0)
    error_bar_w       = g("error_bar_width",     1.5)
    error_bar_cap     = g("error_bar_cap",        4)
    fit_line_w        = g("fit_line_width",       2.5)
    annot_size        = g("annotation_size",      11)
    font_color        = g("font_color",           "#1e293b")

    # ── Per-condition traces ───────────────────────────────────────────────────
    for cond in conditions:
        label      = cond["label"]
        T          = cond["T"]
        fu_matrix  = cond["fu_matrix"]
        per_rep    = cond["per_rep_df"]
        color      = cond["color"]
        marker_sym = MARKER_SYMBOLS.get(cond["marker_symbol"], "circle")
        line_dash  = LINE_STYLES.get(cond["line_style"],       "solid")
        show_indiv = cond.get("show_individuals", False)

        ok_mask = per_rep["Fit OK"].values
        fu_ok   = fu_matrix[ok_mask]
        if fu_ok.shape[0] == 0:
            continue

        fu_mean = np.nanmean(fu_ok, axis=0)
        fu_std  = (np.nanstd(fu_ok, axis=0, ddof=1)
                   if fu_ok.shape[0] > 1 else np.zeros_like(fu_mean))

        # Individual replicate ghosts
        if show_indiv:
            for i, rep_lbl in enumerate(per_rep.loc[ok_mask, "Replicate"].values):
                fig.add_trace(go.Scatter(
                    x=T, y=fu_ok[i], mode="markers",
                    marker=dict(symbol=marker_sym, color=color,
                                size=max(4, marker_size - 2),
                                opacity=0.35, line=dict(width=0)),
                    name=f"{label} – {rep_lbl}",
                    legendgroup=label, showlegend=False,
                    hovertemplate=(f"{label} {rep_lbl}<br>"
                                   "T: %{x:.1f} °C<br>f_U: %{y:.3f}<extra></extra>"),
                ))

        # Shaded ±SD band
        if g("show_error_band", True) and fu_ok.shape[0] > 1:
            fig.add_trace(go.Scatter(
                x=np.concatenate([T, T[::-1]]),
                y=np.concatenate([fu_mean + fu_std, (fu_mean - fu_std)[::-1]]),
                fill="toself", fillcolor=color, opacity=0.12,
                line=dict(width=0), hoverinfo="skip",
                showlegend=False, legendgroup=label,
            ))

        # Mean data points + error bars
        n_ok = int(ok_mask.sum())
        fig.add_trace(go.Scatter(
            x=T, y=fu_mean, mode="markers",
            marker=dict(
                symbol=marker_sym, color=color,
                size=marker_size, opacity=marker_opacity,
                line=dict(width=marker_border_w, color=marker_border_col),
            ),
            error_y=dict(
                type="data", array=fu_std,
                visible=g("show_error_bars", True),
                color=color, thickness=error_bar_w, width=error_bar_cap,
            ),
            name=f"{label} (n={n_ok})",
            legendgroup=label,
            hovertemplate=(f"{label}<br>T: %{{x:.1f}} °C<br>"
                           "f_U: %{y:.3f}<extra></extra>"),
        ))

        # Fit line (mean parameters)
        ok_rows = per_rep[ok_mask]
        Tm_mean = ok_rows["Tm (°C)"].mean()
        dH_mean = ok_rows["ΔH (kJ/mol)"].mean() * 1000.0
        T_fine  = np.linspace(T.min(), T.max(), 600)
        fu_fit  = fraction_unfolded(T_fine, Tm_mean, dH_mean)

        if g("show_fit_line", True):
            fig.add_trace(go.Scatter(
                x=T_fine, y=fu_fit, mode="lines",
                line=dict(color=color, dash=line_dash, width=fit_line_w),
                name=f"{label} fit  Tm={Tm_mean:.1f} °C",
                legendgroup=label,
                hovertemplate=(f"{label} fit<br>T: %{{x:.1f}} °C<br>"
                               "f_U: %{y:.3f}<extra></extra>"),
            ))

        # Tm vertical marker
        if g("show_tm_line", True):
            fig.add_vline(
                x=Tm_mean,
                line=dict(color=color, dash="dot", width=1),
                annotation_text=f"Tm = {Tm_mean:.1f} °C",
                annotation_font=dict(color=color, size=annot_size),
                annotation_position="top right",
            )

    # f = 0.5 reference line
    if g("show_half_line", True):
        fig.add_hline(
            y=0.5,
            line=dict(color=g("grid_color", "#e2e8f0"), dash="dash", width=1),
        )

    # ── Axis title formatting ──────────────────────────────────────────────────
    x_lbl = _fmt_axis_title(
        g("x_label", "Temperature (°C)"),
        g("axis_title_bold", False), g("axis_title_italic", False),
    )
    y_lbl = _fmt_axis_title(
        g("y_label", "Fraction Unfolded"),
        g("axis_title_bold", False), g("axis_title_italic", False),
    )

    # ── Tick direction ─────────────────────────────────────────────────────────
    tick_dir_map = {
        "Outside": "outside", "Inside": "inside",
        "Both": "outside",    "None": "",
    }
    ticks_val = tick_dir_map.get(g("tick_direction", "Outside"), "outside")

    # ── Legend ─────────────────────────────────────────────────────────────────
    leg_pos_name = g("legend_position", "Top-right")
    if leg_pos_name == "Custom":
        leg_xy = dict(x=g("legend_x", 0.99), y=g("legend_y", 0.99),
                      xanchor="auto", yanchor="auto")
    else:
        leg_xy = LEGEND_POSITIONS.get(leg_pos_name, LEGEND_POSITIONS["Top-right"])

    leg_bg  = hex_to_rgba(g("legend_bgcolor", "#ffffff"),
                           g("legend_bg_opacity", 0.9))
    leg_ori = "h" if g("legend_orientation", "Vertical") == "Horizontal" else "v"

    # ── Font family ────────────────────────────────────────────────────────────
    font_fam   = FONT_FAMILIES.get(g("font_family", "Arial (sans-serif)"),
                                    "Arial, sans-serif")
    title_font = dict(size=g("axis_title_size", 14),
                      color=g("axis_title_color", "#1e293b"),
                      family=font_fam)

    # ── Axis kwargs (shared base) ──────────────────────────────────────────────
    def axis_base(show_grid):
        d = dict(
            showline=g("show_axis_line", True),
            linecolor=g("axis_line_color", "#94a3b8"),
            linewidth=g("axis_line_width", 1),
            mirror=g("show_mirror", False),
            zeroline=g("show_zeroline", False),
            zerolinecolor=g("zeroline_color", "#cbd5e1"),
            zerolinewidth=1,
            showgrid=show_grid,
            gridcolor=g("grid_color", "#e2e8f0"),
            gridwidth=g("grid_width", 1.0),
            griddash=GRID_DASHES.get(g("grid_dash", "Solid"), "solid"),
            ticks=ticks_val,
            ticklen=g("tick_len", 5),
            tickwidth=g("tick_width", 1),
            tickcolor=g("tick_color", "#94a3b8"),
            tickfont=dict(
                size=g("tick_font_size", 12),
                color=g("tick_font_color", "#475569"),
                family=font_fam,
            ),
        )
        x_dt = g("x_dtick", 0)
        y_dt = g("y_dtick", 0)
        return d, x_dt, y_dt

    base, x_dt, y_dt = axis_base(True)

    x_axis = {
        **base,
        "showgrid": g("show_x_grid", True),
        "title":    dict(text=x_lbl, font=title_font),
        "type":     g("x_type", "linear"),
        "range":    g("x_range", None),
    }
    if x_dt and x_dt > 0:
        x_axis["dtick"] = x_dt

    y_axis = {
        **base,
        "showgrid": g("show_y_grid", True),
        "title":    dict(text=y_lbl, font=title_font),
        "type":     g("y_type", "linear"),
        "range":    g("y_range", [-0.05, 1.08]),
    }
    if y_dt and y_dt > 0:
        y_axis["dtick"] = y_dt

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig_w = g("fig_width", None)   # None → responsive in Streamlit
    fig_h = g("fig_height", 550)

    fig.update_layout(
        width=fig_w,
        height=fig_h,
        margin=dict(
            l=g("margin_l", 70), r=g("margin_r", 40),
            t=g("margin_t", 70), b=g("margin_b", 60),
        ),
        title=dict(
            text=g("plot_title", "Thermal Unfolding — Two-State Model"),
            font=dict(size=g("plot_title_size", 18),
                      color=font_color, family=font_fam),
            x=0.5,
        ),
        xaxis=x_axis,
        yaxis=y_axis,
        plot_bgcolor=g("plot_bg",   "#ffffff"),
        paper_bgcolor=g("paper_bg", "#ffffff"),
        font=dict(family=font_fam,
                  size=g("global_font_size", 12),
                  color=font_color),
        showlegend=g("show_legend", True),
        legend=dict(
            **leg_xy,
            orientation=leg_ori,
            font=dict(size=g("legend_font_size", 12), color=font_color),
            bgcolor=leg_bg,
            bordercolor=g("legend_border_color", "#e2e8f0"),
            borderwidth=g("legend_border_width", 1),
        ),
        hovermode="closest",
    )

    return fig
