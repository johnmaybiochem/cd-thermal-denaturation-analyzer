"""
CD Thermal Denaturation Analyzer
Streamlit app · Two-state and three-state Van't Hoff models · Sloping baselines
"""

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.io as pio

from version  import __version__, __app_name__
from fitting  import (fit_condition, fit_condition_3state, read_data_file,
                      fraction_unfolded, populations_three_state)
from plotting import (build_figure, MARKER_SYMBOLS, LINE_STYLES,
                      FONT_FAMILIES, GRID_DASHES, hex_to_rgba)

# ── Upload limits ──────────────────────────────────────────────────────────────
MAX_FILE_MB    = 10
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_ROWS       = 50_000

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=__app_name__,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .section-header {
        font-size:.72rem; font-weight:700; letter-spacing:.09em;
        text-transform:uppercase; color:#64748b; margin:18px 0 6px 0;
    }
    .swatch-row { display:flex; gap:5px; margin:4px 0 10px 0; flex-wrap:wrap; }
    .swatch { width:22px; height:22px; border-radius:4px; border:1px solid rgba(0,0,0,.12); }
    [data-testid="metric-container"] { background:#f8fafc; border-radius:8px; padding:8px; }
    .eq-box { background:#f8fafc; border:1px solid #e2e8f0;
               border-radius:10px; padding:16px 24px; margin:6px 0; }
    .dim-caption { font-size:.75rem; color:#94a3b8; margin-top:4px; }
    .preset-note { font-size:.75rem; color:#6366f1; font-style:italic; margin:4px 0 0 0; }
    .model-badge-2 {
        display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.05em;
        background:#dbeafe; color:#1d4ed8; border-radius:4px;
        padding:1px 7px; margin-left:6px; vertical-align:middle;
    }
    .model-badge-3 {
        display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:.05em;
        background:#fef3c7; color:#b45309; border-radius:4px;
        padding:1px 7px; margin-left:6px; vertical-align:middle;
    }
</style>
""", unsafe_allow_html=True)

# ── Color palettes ─────────────────────────────────────────────────────────────
PALETTES = {
    "Colorblind (default)": "colorblind",
    "Deep":   "deep",   "Muted":  "muted",  "Pastel": "pastel",
    "Bright": "bright", "Dark":   "dark",   "Tab10":  "tab10",
}

def palette_hex(name, n=10):
    rgb = sns.color_palette(PALETTES[name], n)
    return [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in rgb]

def swatch_html(colors):
    divs = "".join(f'<div class="swatch" style="background:{c}"></div>' for c in colors)
    return f'<div class="swatch-row">{divs}</div>'

# ── Theme presets ──────────────────────────────────────────────────────────────
THEMES = {
    "Light": dict(
        plot_bg="#ffffff",    paper_bg="#ffffff",    grid_color="#e2e8f0",
        font_color="#1e293b", axis_line_color="#94a3b8", tick_color="#94a3b8",
        tick_font_color="#475569", axis_title_color="#1e293b",
        zeroline_color="#cbd5e1", legend_bgcolor="#ffffff",
        legend_border_color="#e2e8f0", marker_border_color="#ffffff",
        show_x_grid=True, show_y_grid=True, show_axis_line=True,
    ),
    "Dark": dict(
        plot_bg="#1e293b",    paper_bg="#0f172a",    grid_color="#334155",
        font_color="#f1f5f9", axis_line_color="#475569", tick_color="#475569",
        tick_font_color="#94a3b8", axis_title_color="#e2e8f0",
        zeroline_color="#475569", legend_bgcolor="#1e293b",
        legend_border_color="#475569", marker_border_color="#1e293b",
        show_x_grid=True, show_y_grid=True, show_axis_line=True,
    ),
    "Minimal": dict(
        plot_bg="#ffffff",    paper_bg="#ffffff",    grid_color="#ffffff",
        font_color="#000000", axis_line_color="#000000", tick_color="#000000",
        tick_font_color="#000000", axis_title_color="#000000",
        zeroline_color="#ffffff", legend_bgcolor="#ffffff",
        legend_border_color="#cccccc", marker_border_color="#ffffff",
        show_x_grid=False, show_y_grid=False, show_axis_line=True,
    ),
}

# ── Figure presets ─────────────────────────────────────────────────────────────
PRESETS = {
    "Custom": {},
    "Publication — 3.5″ (single column)": dict(
        unit="in",    fig_width_px=336,  fig_height_px=252,
        theme="Minimal", export_scale=3,
        font_family="Arial (sans-serif)",
        global_font_size=10, plot_title_size=11,
        axis_title_size=11,  tick_font_size=10,
        legend_font_size=10, annotation_size=9,
        marker_size=4,       marker_border_width=0.5,
        fit_line_width=1.5,  error_bar_width=0.75,  error_bar_cap=3,
        axis_title_bold=True, plot_title_text="", show_mirror=False,
    ),
    "Publication — 7″ (double column)": dict(
        unit="in",    fig_width_px=672,  fig_height_px=504,
        theme="Minimal", export_scale=3,
        font_family="Arial (sans-serif)",
        global_font_size=10, plot_title_size=11,
        axis_title_size=11,  tick_font_size=10,
        legend_font_size=10, annotation_size=9,
        marker_size=5,       marker_border_width=0.5,
        fit_line_width=1.5,  error_bar_width=0.75,  error_bar_cap=3,
        axis_title_bold=True, plot_title_text="", show_mirror=False,
    ),
    "Poster": dict(
        unit="in",    fig_width_px=960,  fig_height_px=720,
        theme="Light", export_scale=2,
        font_family="Arial (sans-serif)",
        global_font_size=18, plot_title_size=22,
        axis_title_size=20,  tick_font_size=18,
        legend_font_size=18, annotation_size=14,
        marker_size=10,      marker_border_width=1.0,
        fit_line_width=3.0,  error_bar_width=2.0,   error_bar_cap=6,
        axis_title_bold=True,
    ),
    "PowerPoint": dict(
        unit="in",    fig_width_px=960,  fig_height_px=540,
        theme="Light", export_scale=2,
        font_family="Arial (sans-serif)",
        global_font_size=18, plot_title_size=24,
        axis_title_size=20,  tick_font_size=18,
        legend_font_size=18, annotation_size=16,
        marker_size=12,      marker_border_width=1.0,
        fit_line_width=3.0,  error_bar_width=2.0,   error_bar_cap=6,
        axis_title_bold=True,
    ),
}

# ── Units ──────────────────────────────────────────────────────────────────────
UNIT_PX   = {"px": 1.0, "in": 96.0, "cm": 37.795}
UNIT_STEP = {"px": 10,  "in": 0.25, "cm": 0.5}
UNIT_FMT  = {"px": 0,   "in": 2,    "cm": 2}

# ── Aspect ratio presets ───────────────────────────────────────────────────────
AR_PRESETS = {
    "Custom": None, "16 : 9": 9/16, "4 : 3": 3/4,
    "3 : 2": 2/3,  "1 : 1": 1.0,   "2 : 3": 3/2,
}

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("conditions", []), ("fit_done", False),
              ("raw_conditions", []), ("raw_plot_done", False),
              ("manual_entries", []), ("editor_counter", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"<h2 style='text-align:center;margin:0 0 1px 0;font-size:1.05rem;'>"
        f"{__app_name__}</h2>"
        f"<p style='text-align:center;font-size:.72rem;color:#94a3b8;margin:0 0 10px 0;'>"
        f"Two-state &amp; three-state Van't Hoff models</p>",
        unsafe_allow_html=True,
    )

    # ── Palette ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Color Palette</div>', unsafe_allow_html=True)
    palette_name = st.selectbox("Palette", list(PALETTES.keys()),
                                index=0, label_visibility="collapsed")
    pal_colors = palette_hex(palette_name)
    st.markdown(swatch_html(pal_colors[:8]), unsafe_allow_html=True)

    # ── File upload ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Upload Data Files</div>', unsafe_allow_html=True)
    st.caption("CSV or Excel (.xlsx). One file per condition. "
               "Temperature column + one column per replicate.")
    uploaded_files = st.file_uploader(
        "Files", type=["csv", "xlsx", "xls"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    file_configs = []
    if uploaded_files:
        st.markdown('<div class="section-header">Condition Settings</div>',
                    unsafe_allow_html=True)
        for i, f in enumerate(uploaded_files):
            size_mb = f.size / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                st.warning(
                    f"⚠️ **{f.name}** skipped — "
                    f"{size_mb:.1f} MB exceeds the {MAX_FILE_MB} MB limit."
                )
                continue
            default_color = pal_colors[i % len(pal_colors)]
            with st.expander(f"📄 {f.name}", expanded=True):
                label    = st.text_input("Condition name",
                                         value=f.name.rsplit(".", 1)[0], key=f"lbl_{i}")
                temp_col = st.text_input("Temperature column", value="T", key=f"tc_{i}")
                c1, c2   = st.columns(2)
                with c1:
                    color = st.color_picker("Color", value=default_color,
                                            key=f"col_{i}_{palette_name}")
                with c2:
                    marker = st.selectbox("Marker", list(MARKER_SYMBOLS.keys()),
                                          key=f"mk_{i}")
                line_style = st.selectbox("Fit line (2-state)", list(LINE_STYLES.keys()),
                                          key=f"ls_{i}")
                show_indiv = st.checkbox("Show individual replicates",
                                         value=False, key=f"si_{i}")
                use_3state = st.checkbox("Three-state model (N ⇌ I ⇌ U)",
                                         value=False, key=f"3s_{i}",
                                         help="Fit to a sequential three-state model with "
                                              "an intermediate (I) between native (N) and "
                                              "unfolded (U). Yields Tm1, Tm2, ΔH1, ΔH2.")
                if use_3state:
                    st.caption("3-state curve line styles")
                    ls1, ls2, ls3 = st.columns(3)
                    ls_fu   = ls1.selectbox("fU",       list(LINE_STYLES.keys()),
                                            index=0, key=f"ls3_fu_{i}",
                                            help="Line style for the fU (fully unfolded) curve.")
                    ls_fi   = ls2.selectbox("fI",       list(LINE_STYLES.keys()),
                                            index=1, key=f"ls3_fi_{i}",
                                            help="Line style for the fI (intermediate) curve.")
                    ls_prog = ls3.selectbox("Progress", list(LINE_STYLES.keys()),
                                            index=0, key=f"ls3_prog_{i}",
                                            help="Line style for the unfolding progress curve.")
                else:
                    ls_fu = ls_fi = ls_prog = "Solid"
                file_configs.append(dict(
                    file=f, label=label, temp_col=temp_col, color=color,
                    marker_symbol=marker, line_style=line_style,
                    show_individuals=show_indiv,
                    model="3state" if use_3state else "2state",
                    line_style_fu=ls_fu, line_style_fi=ls_fi, line_style_prog=ls_prog,
                ))

    # ── Manual entry ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Paste Data Manually</div>',
                unsafe_allow_html=True)
    with st.expander("📋 Open spreadsheet entry", expanded=False):
        st.caption("Paste from Excel or any spreadsheet. "
                   "First column = Temperature (°C). Remaining = replicates.")
        man_name = st.text_input("Condition name", key="man_name",
                                 placeholder="e.g. Wild-type +ligand")
        man_temp = st.text_input("Temperature column header", value="T", key="man_tc")
        empty_df = pd.DataFrame(
            np.full((80, 11), np.nan),
            columns=["T"] + [f"Y{j}" for j in range(1, 11)],
        )
        edited_df = st.data_editor(
            empty_df, key=f"man_editor_{st.session_state.editor_counter}",
            num_rows="dynamic", use_container_width=True, height=260,
        )
        if st.button("➕ Add this condition", use_container_width=True):
            if not man_name.strip():
                st.warning("Please enter a condition name.")
            else:
                clean = edited_df.dropna(how="all").dropna(axis=1, how="all")
                clean = clean.dropna(subset=[clean.columns[0]])
                if clean.empty:
                    st.warning("No data detected — paste your data first.")
                else:
                    clean.columns = [man_temp] + list(clean.columns[1:])
                    st.session_state.manual_entries.append(
                        {"name": man_name.strip(),
                         "df": clean.reset_index(drop=True),
                         "temp_col": man_temp}
                    )
                    st.session_state.editor_counter += 1
                    st.success(f"Added: {man_name.strip()}")

        if st.session_state.manual_entries:
            st.markdown("**Added conditions:**")
            for idx, entry in enumerate(st.session_state.manual_entries):
                r1, r2 = st.columns([3, 1])
                r1.markdown(f"• {entry['name']} "
                            f"({entry['df'].shape[0]} pts, "
                            f"{entry['df'].shape[1]-1} reps)")
                if r2.button("✕", key=f"rm_{idx}"):
                    st.session_state.manual_entries.pop(idx)
                    st.rerun()

    # Manual condition styles
    man_configs = []
    if st.session_state.manual_entries:
        st.markdown('<div class="section-header">Manual Condition Styles</div>',
                    unsafe_allow_html=True)
        n_file = len(file_configs)
        for mi, entry in enumerate(st.session_state.manual_entries):
            default_color = pal_colors[(n_file + mi) % len(pal_colors)]
            with st.expander(f"⌨️ {entry['name']}", expanded=False):
                mc1, mc2 = st.columns(2)
                with mc1:
                    m_color = st.color_picker("Color", value=default_color,
                                              key=f"mc_{mi}_{palette_name}")
                with mc2:
                    m_marker = st.selectbox("Marker", list(MARKER_SYMBOLS.keys()),
                                            key=f"mm_{mi}")
                m_line  = st.selectbox("Fit line (2-state)", list(LINE_STYLES.keys()),
                                       key=f"ml_{mi}")
                m_indiv = st.checkbox("Show individual replicates",
                                      value=False, key=f"ms_{mi}")
                m_3state = st.checkbox("Three-state model (N ⇌ I ⇌ U)",
                                       value=False, key=f"m3s_{mi}",
                                       help="Fit to a sequential three-state model.")
                if m_3state:
                    st.caption("3-state curve line styles")
                    mls1, mls2, mls3 = st.columns(3)
                    m_ls_fu   = mls1.selectbox("fU",       list(LINE_STYLES.keys()),
                                               index=0, key=f"mls3_fu_{mi}",
                                               help="Line style for the fU curve.")
                    m_ls_fi   = mls2.selectbox("fI",       list(LINE_STYLES.keys()),
                                               index=1, key=f"mls3_fi_{mi}",
                                               help="Line style for the fI curve.")
                    m_ls_prog = mls3.selectbox("Progress", list(LINE_STYLES.keys()),
                                               index=0, key=f"mls3_prog_{mi}",
                                               help="Line style for the progress curve.")
                else:
                    m_ls_fu = m_ls_fi = m_ls_prog = "Solid"
            man_configs.append(dict(
                label=entry["name"], temp_col=entry["temp_col"],
                df=entry["df"], color=m_color, marker_symbol=m_marker,
                line_style=m_line, show_individuals=m_indiv,
                model="3state" if m_3state else "2state",
                line_style_fu=m_ls_fu, line_style_fi=m_ls_fi, line_style_prog=m_ls_prog,
            ))

    # ── Fitting options ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Fitting Options</div>', unsafe_allow_html=True)
    n_baseline = st.slider("Baseline region (points each end)",
                            min_value=5, max_value=25, value=15)

    has_data = bool(uploaded_files) or bool(st.session_state.manual_entries)

    raw_btn = st.button(
        "📊  Plot Raw Data", use_container_width=True,
        disabled=not has_data,
        help="Plot CD signal vs temperature for all conditions without fitting. "
             "Useful for deciding whether data looks two-state or three-state.",
    )

    run_btn = st.button(
        "▶  Run Analysis", type="primary", use_container_width=True,
        disabled=not has_data,
    )

    # ── Version footer ─────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='text-align:center;font-size:.68rem;color:#cbd5e1;"
        f"margin-top:24px;'>v{__version__}</p>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# RUN FITTING
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    st.session_state.conditions = []
    all_srcs = ([(c, "file")   for c in file_configs] +
                [(c, "manual") for c in man_configs])
    prog = st.progress(0, text="Fitting…")
    for i, (cfg, src) in enumerate(all_srcs):
        if src == "file":
            cfg["file"].seek(0)
            df = read_data_file(cfg["file"])
        else:
            df = cfg["df"]

        if len(df) > MAX_ROWS:
            st.error(
                f"⚠️ **{cfg['label']}** has {len(df):,} rows — "
                f"maximum allowed is {MAX_ROWS:,}. Skipping."
            )
            prog.progress((i + 1) / len(all_srcs))
            continue

        is_3state = (cfg.get("model", "2state") == "3state")

        if is_3state:
            per_rep_df, summary_df, fu_matrix, fi_matrix, T = fit_condition_3state(
                df, temp_col=cfg["temp_col"], n_baseline_pts=n_baseline,
            )
            st.session_state.conditions.append({
                **cfg, "df": df, "per_rep_df": per_rep_df,
                "summary_df": summary_df, "fu_matrix": fu_matrix,
                "fi_matrix": fi_matrix, "T": T,
            })
        else:
            per_rep_df, summary_df, fu_matrix, T = fit_condition(
                df, temp_col=cfg["temp_col"], n_baseline_pts=n_baseline,
            )
            st.session_state.conditions.append({
                **cfg, "df": df, "per_rep_df": per_rep_df,
                "summary_df": summary_df, "fu_matrix": fu_matrix,
                "fi_matrix": None, "T": T,
            })

        prog.progress((i + 1) / len(all_srcs), text=f"Fitted: {cfg['label']}")
    st.session_state.fit_done = True
    prog.empty()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD RAW DATA (no fitting)
# ══════════════════════════════════════════════════════════════════════════════
if raw_btn:
    st.session_state.raw_conditions = []
    all_srcs = ([(c, "file")   for c in file_configs] +
                [(c, "manual") for c in man_configs])
    for cfg, src in all_srcs:
        if src == "file":
            cfg["file"].seek(0)
            df = read_data_file(cfg["file"])
        else:
            df = cfg["df"]
        st.session_state.raw_conditions.append({**cfg, "df": df})
    st.session_state.raw_plot_done = True

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## {__app_name__}")
st.caption("Two-state and three-state Van't Hoff models with sloping baselines")

conditions = st.session_state.conditions

if st.session_state.fit_done and conditions:

    # Refresh per-condition styles from live sidebar widgets
    cond_plot = []
    for i, cond in enumerate(conditions):
        is_manual = i >= len(file_configs)
        mi = i - len(file_configs)
        if not is_manual:
            color  = st.session_state.get(f"col_{i}_{palette_name}", cond["color"])
            marker = st.session_state.get(f"mk_{i}",  cond["marker_symbol"])
            line   = st.session_state.get(f"ls_{i}",  cond["line_style"])
            indiv  = st.session_state.get(f"si_{i}",  cond["show_individuals"])
            ls_fu  = st.session_state.get(f"ls3_fu_{i}",   cond.get("line_style_fu",   "Solid"))
            ls_fi  = st.session_state.get(f"ls3_fi_{i}",   cond.get("line_style_fi",   "Dashed"))
            ls_prog= st.session_state.get(f"ls3_prog_{i}", cond.get("line_style_prog", "Solid"))
        else:
            color  = st.session_state.get(f"mc_{mi}_{palette_name}", cond["color"])
            marker = st.session_state.get(f"mm_{mi}", cond["marker_symbol"])
            line   = st.session_state.get(f"ml_{mi}", cond["line_style"])
            indiv  = st.session_state.get(f"ms_{mi}", cond["show_individuals"])
            ls_fu  = st.session_state.get(f"mls3_fu_{mi}",   cond.get("line_style_fu",   "Solid"))
            ls_fi  = st.session_state.get(f"mls3_fi_{mi}",   cond.get("line_style_fi",   "Dashed"))
            ls_prog= st.session_state.get(f"mls3_prog_{mi}", cond.get("line_style_prog", "Solid"))
        cond_plot.append({**cond, "color": color, "marker_symbol": marker,
                          "line_style": line, "show_individuals": indiv,
                          "line_style_fu": ls_fu, "line_style_fi": ls_fi,
                          "line_style_prog": ls_prog})

    show_raw_tab = st.session_state.raw_plot_done and bool(st.session_state.raw_conditions)
    tab_labels = ["📈 Analysis", "📊 Graph Options"]
    if show_raw_tab:
        tab_labels = ["🔬 Raw Data", "📈 Analysis", "📊 Graph Options"]

    tabs_main = st.tabs(tab_labels)
    if show_raw_tab:
        tab_raw, tab_analysis, tab_opts = tabs_main
    else:
        tab_analysis, tab_opts = tabs_main

    # ══════════════════════════════════════════════════════════════════════════
    # TAB: GRAPH OPTIONS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_opts:
        st.markdown("All settings update the figure instantly when you switch "
                    "back to the Analysis tab.")

        # ── Figure Preset ──────────────────────────────────────────────────────
        preset_name = st.selectbox(
            "Figure preset",
            list(PRESETS.keys()), index=0,
            help="Loads size, font, and style defaults optimised for the target medium.",
        )
        pd_  = PRESETS.get(preset_name) or {}
        pk   = preset_name.lower()[:30].replace(" ", "_").replace("—","").replace('"',"")

        def pv(key, fallback):
            return pd_.get(key, fallback)

        if preset_name != "Custom":
            st.markdown(
                '<p class="preset-note">ℹ Preset loaded — adjust any option below to customise.</p>',
                unsafe_allow_html=True,
            )

        theme_name = st.selectbox(
            "Color theme",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index(pv("theme", "Light")),
            key=f"theme_{pk}",
        )
        tv = THEMES[theme_name]
        tk = f"{pk}_{theme_name}"

        # ── 1. Canvas, Layout, and Export ──────────────────────────────────────
        with st.expander("📐  Canvas, Layout, and Export", expanded=False):
            default_unit = pv("unit", "px")
            unit = st.radio(
                "Dimension units", ["px", "in", "cm"], horizontal=True,
                index=["px", "in", "cm"].index(default_unit),
                key=f"unit_{pk}",
            )
            ppu = UNIT_PX[unit]

            default_w_px = pv("fig_width_px", 900)
            default_w    = round(default_w_px / ppu, UNIT_FMT[unit])
            col_w, col_ar = st.columns(2)
            with col_w:
                fill_container = st.checkbox("Fill container width",
                                             value=(preset_name == "Custom"),
                                             key=f"fill_{pk}")
                if not fill_container:
                    fig_w_val  = st.number_input(
                        f"Width ({unit})", value=float(default_w),
                        step=float(UNIT_STEP[unit]), min_value=0.1,
                        key=f"fw_{pk}_{unit}",
                    )
                    fig_width_px = max(100, int(round(fig_w_val * ppu)))
                else:
                    fig_width_px = None
                    fig_w_val    = None

            with col_ar:
                ar_preset = st.selectbox("Aspect ratio", list(AR_PRESETS.keys()),
                                         key=f"ar_{pk}")
            default_h_px = pv("fig_height_px", 550)
            if ar_preset != "Custom" and fig_width_px is not None:
                fig_height_px = max(100, int(round(fig_width_px * AR_PRESETS[ar_preset])))
                h_in_units    = round(fig_height_px / ppu, UNIT_FMT[unit])
                st.caption(f"Height: {h_in_units} {unit}  →  {fig_height_px} px")
            else:
                default_h = round(default_h_px / ppu, UNIT_FMT[unit])
                fig_h_val = st.number_input(
                    f"Height ({unit})", value=float(default_h),
                    step=float(UNIT_STEP[unit]), min_value=0.1,
                    key=f"fh_{pk}_{unit}",
                )
                fig_height_px = max(100, int(round(fig_h_val * ppu)))

            export_scale_preview = pv("export_scale", 2)
            if fig_width_px:
                w_in  = fig_width_px / 96
                h_in  = fig_height_px / 96
                eff_dpi = 96 * export_scale_preview
                st.markdown(
                    f'<p class="dim-caption">'
                    f'Screen: {fig_width_px} × {fig_height_px} px '
                    f'({w_in:.2f}″ × {h_in:.2f}″ at 96 dpi) &nbsp;|&nbsp; '
                    f'At {export_scale_preview}× export: '
                    f'{fig_width_px*export_scale_preview} × {fig_height_px*export_scale_preview} px '
                    f'(≈{eff_dpi} dpi effective)</p>',
                    unsafe_allow_html=True,
                )

            export_scale = st.select_slider(
                "Export resolution",
                options=[1, 2, 3], value=pv("export_scale", 2),
                format_func=lambda x: f"{x}×  (≈{96*x} dpi effective)",
                key=f"escale_{pk}",
            )

            export_format = st.selectbox(
                "Export format (Camera icon)",
                options=["png", "svg", "jpeg", "webp"],
                index=0,
                help="Sets the file type downloaded when clicking the Camera icon.",
            )

            st.markdown("**Margins (px)**")
            mg1, mg2, mg3, mg4 = st.columns(4)
            margin_l = mg1.number_input("Left",   value=70,  step=5, min_value=0, key=f"ml_{pk}")
            margin_r = mg2.number_input("Right",  value=40,  step=5, min_value=0, key=f"mr_{pk}")
            margin_t = mg3.number_input("Top",    value=70,  step=5, min_value=0, key=f"mt_{pk}")
            margin_b = mg4.number_input("Bottom", value=60,  step=5, min_value=0, key=f"mb_{pk}")

        # ── 2. Theme & Background ──────────────────────────────────────────────
        with st.expander("🎨  Theme & Background", expanded=False):
            tb1, tb2 = st.columns(2)
            plot_bg  = tb1.color_picker("Plot area background",
                                         value=tv["plot_bg"],  key=f"plot_bg_{tk}")
            paper_bg = tb2.color_picker("Figure background",
                                         value=tv["paper_bg"], key=f"paper_bg_{tk}")

        # ── 3. Axes & Scales ───────────────────────────────────────────────────
        with st.expander("📏  Axes & Scales", expanded=False):
            T0 = conditions[0]["T"]
            ax1, ax2 = st.columns(2)
            with ax1:
                st.markdown("**X axis**")
                x_type = st.selectbox("Scale", ["Linear", "Log"], key=f"xtype_{pk}")
                xmin   = st.number_input("X min", value=float(T0.min()), step=1.0, key=f"xmin_{pk}")
                xmax   = st.number_input("X max", value=float(T0.max()), step=1.0, key=f"xmax_{pk}")
            with ax2:
                st.markdown("**Y axis**")
                y_type = st.selectbox("Scale", ["Linear", "Log"], key=f"ytype_{pk}")
                ymin   = st.number_input("Y min", value=-0.05, step=0.05, key=f"ymin_{pk}")
                ymax   = st.number_input("Y max", value=1.08,  step=0.05, key=f"ymax_{pk}")

            st.markdown("**Axis lines**")
            al1, al2, al3 = st.columns(3)
            show_axis_line = al1.checkbox("Show axis lines", value=tv["show_axis_line"],
                                           key=f"sal_{tk}")
            show_mirror    = al2.checkbox("Full box (mirror)", value=pv("show_mirror", False),
                                           key=f"mir_{pk}")
            show_zeroline  = al3.checkbox("Zero line", value=False, key=f"zl_{pk}")
            alc1, alc2, alc3 = st.columns(3)
            axis_line_color = alc1.color_picker("Axis color", value=tv["axis_line_color"],
                                                  key=f"alc_{tk}")
            axis_line_width = alc2.number_input("Axis width", value=1,
                                                  min_value=1, max_value=5, key=f"alw_{pk}")
            zeroline_color  = alc3.color_picker("Zero line color", value=tv["zeroline_color"],
                                                  key=f"zlc_{tk}")

            st.markdown("**Data display toggles**")
            dta, dtb, dtc, dtd, dte = st.columns(5)
            show_error_bars  = dta.checkbox("Error bars",    value=True, key=f"eb_{pk}")
            show_error_band  = dtb.checkbox("Error band",    value=True, key=f"ebd_{pk}")
            show_fit_line    = dtc.checkbox("Fit line",      value=True, key=f"fl_{pk}")
            show_tm_line     = dtd.checkbox("Tm markers",    value=True, key=f"tm_{pk}")
            show_half_line   = dte.checkbox("f=0.5 line",    value=True, key=f"hl_{pk}")
            st.markdown("**Three-state curve overlays**")
            dtf, dtg = st.columns(2)
            show_intermediate = dtf.checkbox(
                "fI curve (intermediate population)",
                value=True, key=f"fi_{pk}",
                help="Show the intermediate population curve for three-state fits "
                     "(dashed, lighter shade of condition color).",
            )
            show_nonnative = dtg.checkbox(
                "Unfolding progress curve",
                value=True, key=f"fnn_{pk}",
                help="Shows the spectroscopically-weighted unfolding progress: "
                     "wI·fI + fU, where wI = (CD_I − CD_N)/(CD_U − CD_N) at the "
                     "midpoint between Tm1 and Tm2. Rises 0→1 across both transitions "
                     "with amplitude weighting from the actual fitted baselines. "
                     "Displayed dotted in the condition color.",
            )

        # ── 4. Tick Marks ──────────────────────────────────────────────────────
        with st.expander("〰  Tick Marks", expanded=False):
            tk1, tk2, tk3 = st.columns(3)
            tick_direction = tk1.selectbox("Direction",
                                            ["Outside", "Inside", "Both", "None"],
                                            key=f"tdir_{pk}")
            tick_len    = tk2.slider("Length (px)", 2, 20, 5, key=f"tlen_{pk}")
            tick_width  = tk3.slider("Width (px)",  1,  5, 1, key=f"twid_{pk}")
            tk4, tk5    = st.columns(2)
            tick_color       = tk4.color_picker("Tick color",  value=tv["tick_color"],
                                                 key=f"tclr_{tk}")
            tick_font_color  = tk5.color_picker("Label color", value=tv["tick_font_color"],
                                                 key=f"tlclr_{tk}")
            tk6, tk7, tk8   = st.columns(3)
            tick_font_size   = tk6.slider("Label size", 6, 24,
                                           pv("tick_font_size", 12), key=f"tfs_{pk}")
            x_dtick  = tk7.number_input("X tick interval (0=auto)", value=0.0,
                                          min_value=0.0, step=1.0, key=f"xdt_{pk}")
            y_dtick  = tk8.number_input("Y tick interval (0=auto)", value=0.0,
                                          min_value=0.0, step=0.05, key=f"ydt_{pk}")

        # ── 5. Fonts & Labels ──────────────────────────────────────────────────
        with st.expander("🔤  Fonts & Labels", expanded=False):
            fl1, fl2, fl3 = st.columns(3)
            font_family = fl1.selectbox(
                "Font family", list(FONT_FAMILIES.keys()),
                index=list(FONT_FAMILIES.keys()).index(pv("font_family","Arial (sans-serif)")),
                key=f"ff_{pk}",
            )
            global_font_size = fl2.slider("Base font size", 6, 24,
                                           pv("global_font_size", 12), key=f"gfs_{pk}")
            plot_title_size  = fl3.slider("Title size", 8, 36,
                                           pv("plot_title_size", 18), key=f"pts_{pk}")

            fl4, fl5, fl6 = st.columns(3)
            axis_title_size  = fl4.slider("Axis title size", 6, 28,
                                           pv("axis_title_size", 14), key=f"ats_{pk}")
            axis_title_color = fl5.color_picker("Axis title color",
                                                  value=tv["axis_title_color"],
                                                  key=f"atc_{tk}")
            annotation_size  = fl6.slider("Tm label size", 6, 20,
                                           pv("annotation_size", 11), key=f"ans_{pk}")

            fl7, fl8 = st.columns(2)
            plot_title_text  = fl7.text_input("Plot title",
                                               value=pv("plot_title_text",
                                                        "CD Thermal Denaturation"),
                                               key=f"ptt_{pk}")
            fl_c1, fl_c2     = fl8.columns(2)
            axis_title_bold   = fl_c1.checkbox("Bold",   value=pv("axis_title_bold", False),
                                                key=f"atb_{pk}")
            axis_title_italic = fl_c2.checkbox("Italic", value=False, key=f"ati_{pk}")

            fla, flb = st.columns(2)
            x_label     = fla.text_input("X-axis label", value="Temperature (°C)",
                                          key=f"xl_{pk}")
            y_label     = flb.text_input("Y-axis label", value="Fraction Unfolded",
                                          key=f"yl_{pk}")
            font_color  = st.color_picker("General font & tick color",
                                           value=tv["font_color"], key=f"fc_{tk}")

        # ── 6. Gridlines ───────────────────────────────────────────────────────
        with st.expander("▦  Gridlines", expanded=False):
            gr1, gr2 = st.columns(2)
            show_x_grid = gr1.checkbox("Show X gridlines",
                                        value=tv["show_x_grid"], key=f"xg_{tk}")
            show_y_grid = gr2.checkbox("Show Y gridlines",
                                        value=tv["show_y_grid"], key=f"yg_{tk}")
            gr3, gr4, gr5 = st.columns(3)
            grid_color = gr3.color_picker("Grid color", value=tv["grid_color"],
                                           key=f"gc_{tk}")
            grid_width = gr4.number_input("Grid width (px)", value=1.0,
                                           min_value=0.5, max_value=5.0, step=0.5,
                                           key=f"gw_{pk}")
            grid_dash  = gr5.selectbox("Grid line style", list(GRID_DASHES.keys()),
                                        key=f"gd_{pk}")

        # ── 7. Legend ──────────────────────────────────────────────────────────
        with st.expander("📖  Legend", expanded=False):
            lg1, lg2, lg3 = st.columns(3)
            show_legend        = lg1.checkbox("Show legend",    value=True,  key=f"sl_{pk}")
            legend_orientation = lg2.selectbox("Orientation", ["Vertical", "Horizontal"],
                                                key=f"lo_{pk}")
            legend_position    = lg3.selectbox(
                "Position",
                ["Top-right", "Top-left", "Bottom-right", "Bottom-left", "Custom"],
                key=f"lpos_{pk}",
            )
            legend_x, legend_y = 0.99, 0.99
            if legend_position == "Custom":
                lx1, lx2 = st.columns(2)
                legend_x = lx1.number_input("X (0–1)", value=0.99, min_value=0.0,
                                             max_value=1.0, step=0.01, key=f"lx_{pk}")
                legend_y = lx2.number_input("Y (0–1)", value=0.99, min_value=0.0,
                                             max_value=1.0, step=0.01, key=f"ly_{pk}")
            lg4, lg5, lg6 = st.columns(3)
            legend_bgcolor     = lg4.color_picker("Background",
                                                   value=tv["legend_bgcolor"],
                                                   key=f"lbg_{tk}")
            legend_bg_opacity  = lg5.slider("Bg opacity", 0.0, 1.0, 0.9,
                                             step=0.05, key=f"lbgo_{pk}")
            legend_font_size   = lg6.slider("Font size", 6, 24,
                                             pv("legend_font_size", 12), key=f"lfs_{pk}")
            lg7, lg8 = st.columns(2)
            legend_border_color = lg7.color_picker("Border color",
                                                    value=tv["legend_border_color"],
                                                    key=f"lbc_{tk}")
            legend_border_width = lg8.number_input("Border width", value=1,
                                                    min_value=0, max_value=5,
                                                    key=f"lbw_{pk}")

        # ── 8. Lines & Markers ─────────────────────────────────────────────────
        with st.expander("✏️  Lines & Markers", expanded=False):
            lm1, lm2, lm3 = st.columns(3)
            marker_size         = lm1.slider("Marker size", 3, 20,
                                              pv("marker_size", 7), key=f"ms_{pk}")
            marker_border_width = lm2.number_input(
                "Marker border width", value=float(pv("marker_border_width", 1.0)),
                min_value=0.0, max_value=5.0, step=0.5, key=f"mbw_{pk}",
            )
            marker_border_color = lm3.color_picker("Marker border color",
                                                    value=tv["marker_border_color"],
                                                    key=f"mbc_{tk}")
            lm4, lm5, lm6 = st.columns(3)
            marker_opacity   = lm4.slider("Marker opacity", 0.1, 1.0, 1.0,
                                           step=0.05, key=f"mo_{pk}")
            error_bar_width  = lm5.number_input(
                "Error bar thickness",
                value=float(pv("error_bar_width", 1.5)),
                min_value=0.5, max_value=5.0, step=0.5, key=f"ebw_{pk}",
            )
            error_bar_cap    = lm6.number_input(
                "Error bar cap (px)",
                value=int(pv("error_bar_cap", 4)),
                min_value=0, max_value=20, key=f"ebc_{pk}",
            )
            fit_line_width = st.slider("Fit line width", 0.5, 6.0,
                                        float(pv("fit_line_width", 2.5)),
                                        step=0.5, key=f"flw_{pk}")

    # ── Build settings dict ────────────────────────────────────────────────────
    settings = dict(
        plot_title=plot_title_text, x_label=x_label, y_label=y_label,
        fig_width=fig_width_px, fig_height=fig_height_px,
        margin_l=int(margin_l), margin_r=int(margin_r),
        margin_t=int(margin_t), margin_b=int(margin_b),
        plot_bg=plot_bg, paper_bg=paper_bg, font_color=font_color,
        x_type=x_type.lower(), y_type=y_type.lower(),
        x_range=[xmin, xmax], y_range=[ymin, ymax],
        show_axis_line=show_axis_line, show_mirror=show_mirror,
        show_zeroline=show_zeroline,
        axis_line_color=axis_line_color, axis_line_width=int(axis_line_width),
        zeroline_color=zeroline_color,
        show_error_bars=show_error_bars, show_error_band=show_error_band,
        show_fit_line=show_fit_line, show_tm_line=show_tm_line,
        show_half_line=show_half_line, show_intermediate=show_intermediate,
        show_nonnative=show_nonnative,
        tick_direction=tick_direction, tick_len=tick_len, tick_width=tick_width,
        tick_color=tick_color, tick_font_color=tick_font_color,
        tick_font_size=tick_font_size,
        x_dtick=float(x_dtick), y_dtick=float(y_dtick),
        font_family=font_family, global_font_size=global_font_size,
        plot_title_size=plot_title_size, axis_title_size=axis_title_size,
        axis_title_color=axis_title_color, annotation_size=annotation_size,
        axis_title_bold=axis_title_bold, axis_title_italic=axis_title_italic,
        show_x_grid=show_x_grid, show_y_grid=show_y_grid,
        grid_color=grid_color, grid_width=float(grid_width), grid_dash=grid_dash,
        show_legend=show_legend, legend_position=legend_position,
        legend_x=float(legend_x), legend_y=float(legend_y),
        legend_orientation=legend_orientation,
        legend_bgcolor=legend_bgcolor, legend_bg_opacity=float(legend_bg_opacity),
        legend_font_size=legend_font_size,
        legend_border_color=legend_border_color,
        legend_border_width=int(legend_border_width),
        marker_size=marker_size, marker_border_width=float(marker_border_width),
        marker_border_color=marker_border_color, marker_opacity=float(marker_opacity),
        error_bar_width=float(error_bar_width), error_bar_cap=int(error_bar_cap),
        fit_line_width=float(fit_line_width),
    )

    fig = build_figure(cond_plot, settings)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB: RAW DATA
    # ══════════════════════════════════════════════════════════════════════════
    if show_raw_tab:
        with tab_raw:
            raw_settings = {**settings,
                "plot_title": settings.get("plot_title") or "Raw CD Data",
                "y_label":    "CD Signal",
                "y_range":    None,
                "show_fit_line":    False,
                "show_tm_line":     False,
                "show_half_line":   False,
                "show_intermediate": False,
                "show_nonnative":   False,
            }

            # Build raw condition list — same styles, but cd_matrix instead of fu_matrix
            raw_cond_plot = []
            for i, rc in enumerate(st.session_state.raw_conditions):
                is_manual = i >= len(file_configs)
                mi = i - len(file_configs)
                if not is_manual:
                    color  = st.session_state.get(f"col_{i}_{palette_name}", rc["color"])
                    marker = st.session_state.get(f"mk_{i}",  rc["marker_symbol"])
                    indiv  = st.session_state.get(f"si_{i}",  rc["show_individuals"])
                else:
                    color  = st.session_state.get(f"mc_{mi}_{palette_name}", rc["color"])
                    marker = st.session_state.get(f"mm_{mi}", rc["marker_symbol"])
                    indiv  = st.session_state.get(f"ms_{mi}", rc["show_individuals"])

                df       = rc["df"]
                temp_col = rc["temp_col"]
                T_raw    = df[temp_col].values
                rep_cols = [c for c in df.columns if c != temp_col]

                cd_rows = []
                rep_labels = []
                for col in rep_cols:
                    vals = df[col].values.astype(float)
                    cd_rows.append(vals)
                    rep_labels.append(col)
                cd_matrix = np.array(cd_rows)   # shape (n_reps, n_temps)

                raw_cond_plot.append({
                    "label":          rc["label"],
                    "T":              T_raw,
                    "fu_matrix":      cd_matrix,   # reuse fu_matrix slot for CD signal
                    "per_rep_df":     pd.DataFrame({
                        "Replicate": rep_labels,
                        "Fit OK":    [True] * len(rep_labels),
                    }),
                    "color":          color,
                    "marker_symbol":  marker,
                    "line_style":     rc.get("line_style", "Solid"),
                    "show_individuals": indiv,
                    "model":          "2state",    # suppress 3-state overlays
                })

            raw_fig = build_figure(raw_cond_plot, raw_settings)
            raw_plotly_config = {
                'toImageButtonOptions': {
                    'format':   export_format,
                    'filename': 'cd_raw_data',
                    'height':   fig_height_px if not fill_container else 600,
                    'width':    fig_width_px  if not fill_container else 900,
                    'scale':    export_scale,
                },
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            }
            st.plotly_chart(raw_fig, use_container_width=fill_container,
                            config=raw_plotly_config)
            st.caption(
                "📊 Raw CD signal vs temperature — no fitting applied. "
                "Use this to assess whether your data looks two-state (single sigmoidal) "
                "or three-state (shoulder or two distinct transitions)."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB: ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_analysis:

        plotly_config = {
            'toImageButtonOptions': {
                'format':   export_format,
                'filename': plot_title_text if plot_title_text else 'cd_denaturation_analysis',
                'height':   fig_height_px if not fill_container else 600,
                'width':    fig_width_px  if not fill_container else 900,
                'scale':    export_scale,
            },
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        }

        st.plotly_chart(fig, use_container_width=fill_container, config=plotly_config)
        st.caption(
            f"💡 **Export Tip:** Hover over the top-right of the chart and click the "
            f"**Camera Icon (📷)** to download your figure as **{export_format.upper()}** "
            f"at your custom dimensions."
        )

        # ── Fitting equations ──────────────────────────────────────────────────
        with st.expander("📐  Fitting Equations", expanded=False):
            st.markdown('<div class="eq-box">', unsafe_allow_html=True)

            has_3state = any(c.get("model") == "3state" for c in conditions)
            has_2state = any(c.get("model", "2state") == "2state" for c in conditions)

            if has_2state:
                st.markdown("#### Two-State Model")
                eq1, eq2, eq3 = st.columns(3)
                with eq1:
                    st.markdown("**Equilibrium**")
                    st.latex(r"\mathrm{N} \rightleftharpoons \mathrm{U}")
                    st.markdown("**Free energy (Van't Hoff)**")
                    st.latex(r"\Delta G(T) = \Delta H \!\left(1 - \frac{T}{T_m}\right)")
                with eq2:
                    st.markdown("**Equilibrium constant**")
                    st.latex(r"K(T) = \exp\!\left(\frac{-\Delta G(T)}{RT}\right)")
                    st.markdown("**Fraction unfolded**")
                    st.latex(r"f_U = \frac{K}{1 + K}")
                with eq3:
                    st.markdown("**Observed CD signal**")
                    st.latex(r"CD_\mathrm{obs} = CD_N(T)(1-f_U) + CD_U(T)\,f_U")
                    st.markdown("**Sloping baselines**")
                    st.latex(r"CD_{N,U}(T) = m_{N,U}\,T + b_{N,U}")
                st.markdown(
                    "<p style='font-size:.8rem;color:#64748b;margin:10px 0 4px 0;'>"
                    "Free parameters: <b>T<sub>m</sub></b>, <b>ΔH</b>, "
                    "<b>m<sub>N</sub></b>, <b>b<sub>N</sub></b>, "
                    "<b>m<sub>U</sub></b>, <b>b<sub>U</sub></b>.  "
                    "ΔS = ΔH / T<sub>m</sub>.  T in Kelvin; R = 8.314 J mol⁻¹ K⁻¹.</p>",
                    unsafe_allow_html=True,
                )

            if has_3state:
                if has_2state:
                    st.divider()
                st.markdown("#### Three-State Model")
                eq1, eq2, eq3 = st.columns(3)
                with eq1:
                    st.markdown("**Sequential equilibria**")
                    st.latex(r"\mathrm{N} \rightleftharpoons \mathrm{I} \rightleftharpoons \mathrm{U}")
                    st.markdown("**Free energies**")
                    st.latex(r"\Delta G_i(T) = \Delta H_i \!\left(1 - \frac{T}{T_{m,i}}\right)")
                with eq2:
                    st.markdown("**Equilibrium constants**")
                    st.latex(r"K_i = \exp\!\left(\frac{-\Delta G_i}{RT}\right)")
                    st.markdown("**Populations**")
                    st.latex(r"f_N = \frac{1}{1+K_1+K_1 K_2}")
                    st.latex(r"f_I = K_1 f_N \quad f_U = K_1 K_2 f_N")
                with eq3:
                    st.markdown("**Observed CD signal**")
                    st.latex(
                        r"CD_\mathrm{obs} = CD_N f_N + CD_I f_I + CD_U f_U"
                    )
                    st.markdown("**Sloping baselines**")
                    st.latex(r"CD_{N,I,U}(T) = m_{N,I,U}\,T + b_{N,I,U}")
                st.markdown(
                    "<p style='font-size:.8rem;color:#64748b;margin:10px 0 4px 0;'>"
                    "Free parameters: <b>T<sub>m1</sub></b>, <b>ΔH<sub>1</sub></b>, "
                    "<b>T<sub>m2</sub></b>, <b>ΔH<sub>2</sub></b>, "
                    "<b>m<sub>N</sub></b>, <b>b<sub>N</sub></b>, "
                    "<b>m<sub>I</sub></b>, <b>b<sub>I</sub></b>, "
                    "<b>m<sub>U</sub></b>, <b>b<sub>U</sub></b> (10 total).  "
                    "T<sub>m1</sub> &lt; T<sub>m2</sub> by convention.  "
                    "T in Kelvin; R = 8.314 J mol⁻¹ K⁻¹.</p>"
                    "<p style='font-size:.8rem;color:#64748b;margin:4px 0 0 0;'>"
                    "<b>Unfolding progress</b> = w<sub>I</sub>·f<sub>I</sub> + f<sub>U</sub>, "
                    "where w<sub>I</sub> = (CD<sub>I</sub> − CD<sub>N</sub>) / "
                    "(CD<sub>U</sub> − CD<sub>N</sub>) evaluated at "
                    "T<sub>ref</sub> = (T<sub>m1</sub> + T<sub>m2</sub>)/2 "
                    "using the fitted baselines. Rises 0→1 across both transitions "
                    "with weights proportional to the actual CD amplitude changes.</p>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Per-condition results ──────────────────────────────────────────────
        for cond in conditions:
            label      = cond["label"]
            per_rep_df = cond["per_rep_df"]
            summary_df = cond["summary_df"]
            T          = cond["T"]
            fu_matrix  = cond["fu_matrix"]
            ok         = per_rep_df["Fit OK"]
            ok_rows    = per_rep_df[ok]
            is_3state  = (cond.get("model", "2state") == "3state")

            badge = (
                '<span class="model-badge-3">3-state</span>' if is_3state
                else '<span class="model-badge-2">2-state</span>'
            )
            st.markdown(f"### {label} {badge}", unsafe_allow_html=True)

            if not ok_rows.empty:
                if not is_3state:
                    # ── Two-state metrics ──────────────────────────────────────
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Mean Tm",
                               f"{ok_rows['Tm (°C)'].mean():.2f} °C",
                               delta=f"±{ok_rows['Tm (°C)'].std():.2f}")
                    m2.metric("Mean ΔH",
                               f"{ok_rows['ΔH (kJ/mol)'].mean():.1f} kJ/mol",
                               delta=f"±{ok_rows['ΔH (kJ/mol)'].std():.1f}")
                    m3.metric("Mean ΔS",
                               f"{ok_rows['ΔS (J/mol·K)'].mean():.1f} J/mol·K",
                               delta=f"±{ok_rows['ΔS (J/mol·K)'].std():.1f}")
                    m4.metric("Mean R²", f"{ok_rows['R²'].mean():.5f}")
                else:
                    # ── Three-state metrics ────────────────────────────────────
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Mean Tm1",
                               f"{ok_rows['Tm1 (°C)'].mean():.2f} °C",
                               delta=f"±{ok_rows['Tm1 (°C)'].std():.2f}")
                    m2.metric("Mean ΔH1",
                               f"{ok_rows['ΔH1 (kJ/mol)'].mean():.1f} kJ/mol",
                               delta=f"±{ok_rows['ΔH1 (kJ/mol)'].std():.1f}")
                    m3.metric("Mean Tm2",
                               f"{ok_rows['Tm2 (°C)'].mean():.2f} °C",
                               delta=f"±{ok_rows['Tm2 (°C)'].std():.2f}")
                    m4.metric("Mean ΔH2",
                               f"{ok_rows['ΔH2 (kJ/mol)'].mean():.1f} kJ/mol",
                               delta=f"±{ok_rows['ΔH2 (kJ/mol)'].std():.1f}")
                    m5.metric("Mean R²", f"{ok_rows['R²'].mean():.5f}")

            if not is_3state:
                display_cols = ["Replicate", "Tm (°C)", "ΔH (kJ/mol)",
                                "ΔS (J/mol·K)", "R²", "RMSE"]
                fmt_map = {"Tm (°C)":"{:.2f}", "ΔH (kJ/mol)":"{:.2f}",
                           "ΔS (J/mol·K)":"{:.2f}", "R²":"{:.5f}", "RMSE":"{:.4f}"}
            else:
                display_cols = ["Replicate",
                                "Tm1 (°C)", "ΔH1 (kJ/mol)", "ΔS1 (J/mol·K)",
                                "Tm2 (°C)", "ΔH2 (kJ/mol)", "ΔS2 (J/mol·K)",
                                "R²", "RMSE"]
                fmt_map = {
                    "Tm1 (°C)":"{:.2f}", "ΔH1 (kJ/mol)":"{:.2f}", "ΔS1 (J/mol·K)":"{:.2f}",
                    "Tm2 (°C)":"{:.2f}", "ΔH2 (kJ/mol)":"{:.2f}", "ΔS2 (J/mol·K)":"{:.2f}",
                    "R²":"{:.5f}", "RMSE":"{:.4f}",
                }

            tab_labels = ["Per-replicate", "Summary (mean ± SD)",
                          "Fraction unfolded data", "Failed fits"]
            if is_3state:
                tab_labels.append("Intermediate population data")

            tabs = st.tabs(tab_labels)

            with tabs[0]:
                st.dataframe(ok_rows[display_cols].style.format(fmt_map),
                             use_container_width=True, hide_index=True)
            with tabs[1]:
                st.dataframe(
                    summary_df.style.format({"Mean":"{:.3f}", "SD":"{:.3f}"}),
                    use_container_width=True, hide_index=True,
                )
            with tabs[2]:
                st.caption("Mean fraction unfolded (fU) ± SD from per-replicate fits.")
                fu_ok   = fu_matrix[ok.values]
                fu_mean = np.nanmean(fu_ok, axis=0)
                fu_std  = (np.nanstd(fu_ok, axis=0, ddof=1)
                           if fu_ok.shape[0] > 1 else np.zeros_like(fu_mean))
                fu_df   = pd.DataFrame({"T (°C)": T,
                                        "Mean f_U": fu_mean, "SD f_U": fu_std})
                st.dataframe(fu_df.style.format("{:.4f}"),
                             use_container_width=True, hide_index=True, height=240)
            with tabs[3]:
                failed = per_rep_df[~ok]
                if failed.empty:
                    st.success("All replicates converged.")
                else:
                    st.dataframe(failed[["Replicate", "Error"]],
                                 use_container_width=True, hide_index=True)

            if is_3state and len(tabs) > 4:
                with tabs[4]:
                    fi_matrix = cond.get("fi_matrix")
                    if fi_matrix is not None:
                        st.caption("Mean intermediate population (fI) ± SD.")
                        fi_ok   = fi_matrix[ok.values]
                        fi_mean = np.nanmean(fi_ok, axis=0)
                        fi_std  = (np.nanstd(fi_ok, axis=0, ddof=1)
                                   if fi_ok.shape[0] > 1 else np.zeros_like(fi_mean))
                        fi_df   = pd.DataFrame({"T (°C)": T,
                                                "Mean f_I": fi_mean, "SD f_I": fi_std})
                        st.dataframe(fi_df.style.format("{:.4f}"),
                                     use_container_width=True, hide_index=True, height=240)

            # ── Downloads ──────────────────────────────────────────────────────
            st.markdown("##### ⬇ Download Results")
            dc1, dc2, dc3 = st.columns(3)
            dc1.download_button(
                "⬇ Per-replicate CSV",
                data=ok_rows[display_cols].to_csv(index=False).encode(),
                file_name=f"{label}_per_replicate.csv", mime="text/csv",
                key=f"dl_rep_{label}",
            )
            dc2.download_button(
                "⬇ Summary CSV",
                data=summary_df.to_csv(index=False).encode(),
                file_name=f"{label}_summary.csv", mime="text/csv",
                key=f"dl_sum_{label}",
            )
            fu_ok2   = fu_matrix[ok.values]
            fu_mean2 = np.nanmean(fu_ok2, axis=0)
            fu_std2  = (np.nanstd(fu_ok2, axis=0, ddof=1)
                        if fu_ok2.shape[0] > 1 else np.zeros_like(fu_mean2))
            fu_dl    = pd.DataFrame({"T (°C)": T,
                                     "Mean_fU": fu_mean2, "SD_fU": fu_std2})
            dc3.download_button(
                "⬇ Fraction Unfolded CSV",
                data=fu_dl.to_csv(index=False).encode(),
                file_name=f"{label}_fraction_unfolded.csv", mime="text/csv",
                key=f"dl_fu_{label}",
            )

            if is_3state and cond.get("fi_matrix") is not None:
                fi_matrix2 = cond["fi_matrix"]
                fi_ok2   = fi_matrix2[ok.values]
                fi_mean2 = np.nanmean(fi_ok2, axis=0)
                fi_std2  = (np.nanstd(fi_ok2, axis=0, ddof=1)
                            if fi_ok2.shape[0] > 1 else np.zeros_like(fi_mean2))
                fi_dl    = pd.DataFrame({"T (°C)": T,
                                         "Mean_fI": fi_mean2, "SD_fI": fi_std2})
                st.download_button(
                    "⬇ Intermediate Population CSV",
                    data=fi_dl.to_csv(index=False).encode(),
                    file_name=f"{label}_intermediate_population.csv", mime="text/csv",
                    key=f"dl_fi_{label}",
                )

            st.divider()

        # ── Cross-condition comparison ─────────────────────────────────────────
        if len(conditions) > 1:
            st.markdown("### Comparison Across Conditions")
            rows = []
            for cond in conditions:
                ok_r = cond["per_rep_df"][cond["per_rep_df"]["Fit OK"]]
                if ok_r.empty:
                    continue
                is_3 = (cond.get("model", "2state") == "3state")
                row  = {
                    "Condition": cond["label"],
                    "Model":     "3-state" if is_3 else "2-state",
                    "n":         int(ok_r.shape[0]),
                    "Mean R²":   ok_r["R²"].mean(),
                }
                if is_3:
                    row.update({
                        "Tm1 mean (°C)":    ok_r["Tm1 (°C)"].mean(),
                        "Tm1 SD":           ok_r["Tm1 (°C)"].std(),
                        "ΔH1 mean (kJ/mol)": ok_r["ΔH1 (kJ/mol)"].mean(),
                        "ΔH1 SD":           ok_r["ΔH1 (kJ/mol)"].std(),
                        "Tm2 mean (°C)":    ok_r["Tm2 (°C)"].mean(),
                        "Tm2 SD":           ok_r["Tm2 (°C)"].std(),
                        "ΔH2 mean (kJ/mol)": ok_r["ΔH2 (kJ/mol)"].mean(),
                        "ΔH2 SD":           ok_r["ΔH2 (kJ/mol)"].std(),
                    })
                else:
                    row.update({
                        "Tm mean (°C)":     ok_r["Tm (°C)"].mean(),
                        "Tm SD":            ok_r["Tm (°C)"].std(),
                        "ΔH mean (kJ/mol)": ok_r["ΔH (kJ/mol)"].mean(),
                        "ΔH SD":            ok_r["ΔH (kJ/mol)"].std(),
                        "ΔS mean (J/mol·K)": ok_r["ΔS (J/mol·K)"].mean(),
                        "ΔS SD":            ok_r["ΔS (J/mol·K)"].std(),
                    })
                rows.append(row)
            comp_df = pd.DataFrame(rows).fillna("—")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Comparison CSV",
                data=pd.DataFrame(rows).to_csv(index=False).encode(),
                file_name="comparison_summary.csv", mime="text/csv",
            )

elif st.session_state.raw_plot_done and st.session_state.raw_conditions:
    # ── Raw data plotted but fitting not yet run ────────────────────────────────
    st.info("📊 Raw data plotted below. Choose a model per condition and click "
            "**▶ Run Analysis** when ready.")

    raw_conditions_solo = st.session_state.raw_conditions
    # Build a minimal settings dict with sensible defaults (no graph options panel yet)
    raw_settings_solo = dict(
        plot_title="Raw CD Data", x_label="Temperature (°C)", y_label="CD Signal",
        fig_width=None, fig_height=550,
        margin_l=70, margin_r=40, margin_t=70, margin_b=60,
        plot_bg="#ffffff", paper_bg="#ffffff", font_color="#1e293b",
        x_type="linear", y_type="linear", x_range=None, y_range=None,
        show_axis_line=True, show_mirror=False, show_zeroline=False,
        axis_line_color="#94a3b8", axis_line_width=1, zeroline_color="#cbd5e1",
        show_error_bars=True, show_error_band=True,
        show_fit_line=False, show_tm_line=False, show_half_line=False,
        show_intermediate=False, show_nonnative=False,
        tick_direction="Outside", tick_len=5, tick_width=1,
        tick_color="#94a3b8", tick_font_color="#475569", tick_font_size=12,
        x_dtick=0, y_dtick=0,
        font_family="Arial (sans-serif)", global_font_size=12,
        plot_title_size=18, axis_title_size=14, axis_title_color="#1e293b",
        annotation_size=11, axis_title_bold=False, axis_title_italic=False,
        show_x_grid=True, show_y_grid=True,
        grid_color="#e2e8f0", grid_width=1.0, grid_dash="Solid",
        show_legend=True, legend_position="Top-right",
        legend_x=0.99, legend_y=0.99, legend_orientation="Vertical",
        legend_bgcolor="#ffffff", legend_bg_opacity=0.9, legend_font_size=12,
        legend_border_color="#e2e8f0", legend_border_width=1,
        marker_size=7, marker_border_width=1.0, marker_border_color="#ffffff",
        marker_opacity=1.0, error_bar_width=1.5, error_bar_cap=4,
        fit_line_width=2.5,
    )

    solo_cond_plot = []
    for i, rc in enumerate(raw_conditions_solo):
        df       = rc["df"]
        temp_col = rc["temp_col"]
        T_raw    = df[temp_col].values
        rep_cols = [c for c in df.columns if c != temp_col]
        cd_rows  = [df[col].values.astype(float) for col in rep_cols]
        cd_matrix = np.array(cd_rows)
        solo_cond_plot.append({
            "label":         rc["label"],
            "T":             T_raw,
            "fu_matrix":     cd_matrix,
            "per_rep_df":    pd.DataFrame({
                "Replicate": rep_cols,
                "Fit OK":    [True] * len(rep_cols),
            }),
            "color":         rc["color"],
            "marker_symbol": rc["marker_symbol"],
            "line_style":    rc.get("line_style", "Solid"),
            "show_individuals": rc.get("show_individuals", False),
            "model":         "2state",
        })

    raw_fig_solo = build_figure(solo_cond_plot, raw_settings_solo)
    st.plotly_chart(raw_fig_solo, use_container_width=True,
                    config={"displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    st.caption(
        "📊 Raw CD signal vs temperature — no fitting applied. "
        "Use this to assess whether your data looks two-state (single sigmoidal) "
        "or three-state (shoulder or two distinct transitions)."
    )
    with st.expander("Expected data format"):
        st.markdown("""
One CSV or Excel file per condition (wild-type, mutant, ±ligand, etc.):

| T | Y1 | Y2 | Y3 |
|---|----|----|-----|
| 25.0 | -38.81 | -38.82 | -24.80 |
| 26.1 | -38.66 | -38.86 | -24.54 |

- Temperature column named `T` by default (configurable per file)
- Replicate columns can have any name
- Upload one file per condition, or paste data via the manual entry panel
- Check **Three-state model** per condition if you expect an intermediate state
        """)

else:
    # ── Landing ────────────────────────────────────────────────────────────────
    st.info("👈  Upload files or paste data in the sidebar, then click **▶ Run Analysis** "
            "to fit, or **📊 Plot Raw Data** to preview first.")
    with st.expander("Expected data format"):
        st.markdown("""
One CSV or Excel file per condition (wild-type, mutant, ±ligand, etc.):

| T | Y1 | Y2 | Y3 |
|---|----|----|-----|
| 25.0 | -38.81 | -38.82 | -24.80 |
| 26.1 | -38.66 | -38.86 | -24.54 |

- Temperature column named `T` by default (configurable per file)
- Replicate columns can have any name
- Upload one file per condition, or paste data via the manual entry panel
- Check **Three-state model** per condition if you expect an intermediate state
        """)

# ── Privacy notice ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🔒 **Data privacy:** Files uploaded to this app transit and are processed on "
    "Streamlit's servers (hosted on AWS). Streamlit encrypts data in transit (TLS) "
    "and at rest (AES-256), but this app makes no guarantees about server-side "
    "logging or retention. Files are limited to 10 MB and 50,000 rows. "
    "See the [Streamlit Trust & Security page](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/trust-and-security) "
    "and [Snowflake Privacy Notice](https://www.snowflake.com/en/legal/privacy/privacy-policy/) for details."
)
