# CD Thermal Denaturation Analyzer
**Version 1.2.0**

A Streamlit web app for fitting circular dichroism thermal denaturation data
to two-state or three-state unfolding models with sloping baselines.

---

## Models

### Two-State Model (N ⇌ U)

The simplest unfolding model assumes a single cooperative transition directly
from the native (N) to the unfolded (U) state:

$$\mathrm{N} \rightleftharpoons \mathrm{U}$$

**Free energy (Van't Hoff):**

$$\Delta G(T) = \Delta H \left(1 - \frac{T}{T_m}\right)$$

**Equilibrium constant and fraction unfolded:**

$$K(T) = \exp\!\left(\frac{-\Delta G}{RT}\right), \quad f_U = \frac{K}{1+K}$$

**Observed CD signal with sloping baselines:**

$$CD_{obs}(T) = (m_N T + b_N)(1 - f_U) + (m_U T + b_U)\,f_U$$

Free parameters: **T**_m, **ΔH**, **m**_N, **b**_N, **m**_U, **b**_U (6 total)  
Derived: **ΔS** = ΔH / T_m

---

### Three-State Model (N ⇌ I ⇌ U)

For proteins that populate a folding intermediate, a sequential two-transition
model is used:

$$\mathrm{N} \rightleftharpoons \mathrm{I} \rightleftharpoons \mathrm{U}$$

Each transition is described by an independent Van't Hoff equilibrium:

$$\Delta G_i(T) = \Delta H_i \left(1 - \frac{T}{T_{m,i}}\right), \quad
  K_i(T) = \exp\!\left(\frac{-\Delta G_i}{RT}\right)$$

**Species populations:**

$$f_N = \frac{1}{1 + K_1 + K_1 K_2}, \quad
  f_I = K_1\,f_N, \quad
  f_U = K_1 K_2\,f_N$$

**Observed CD signal with three independent sloping baselines:**

$$CD_{obs}(T) = (m_N T + b_N)\,f_N + (m_I T + b_I)\,f_I + (m_U T + b_U)\,f_U$$

Free parameters: **T**_m1, **ΔH**_1, **T**_m2, **ΔH**_2,
**m**_N, **b**_N, **m**_I, **b**_I, **m**_U, **b**_U (10 total)  
Convention: T_m1 < T_m2 (enforced post-fit by swapping if needed)  
Derived: **ΔS**_i = ΔH_i / T_m,i

#### Unfolding Progress Variable

Because f_U alone does not complete until after the second transition, a
**spectroscopically-weighted progress variable** is used to visualise the
overall unfolding process in a single curve that rises monotonically from 0 to 1
across both transitions:

$$\text{Progress}(T) = w_I \cdot f_I(T) + f_U(T)$$

The intermediate weight **w**_I is derived from the fitted CD baselines evaluated
at a reference temperature T_ref = (T_m1 + T_m2) / 2:

$$w_I = \frac{CD_I(T_{ref}) - CD_N(T_{ref})}{CD_U(T_{ref}) - CD_N(T_{ref})}$$

This places the intermediate at its true spectroscopic position on the N→U
scale rather than assigning an arbitrary weight. w_I is reported in the legend
and clamped to [0, 1].

---

## Installation

```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda create -n cd_analyzer python=3.11
conda activate cd_analyzer
pip install -r requirements.txt
```

---

## Running

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Input Format

One CSV or Excel (.xlsx) file per condition:

| T     | Y1      | Y2      | Y3      |
|-------|---------|---------|---------|
| 25.00 | -38.807 | -38.818 | -24.803 |
| 26.11 | -38.656 | -38.860 | -24.545 |

- Default temperature column name: `T` (configurable per file)
- Any number of replicate columns
- Upload one file per condition; select two-state or three-state model per condition

---

## Features

- **📊 Plot Raw Data** button — preview CD signal vs temperature before fitting,
  to assess whether data looks two-state (single sigmoid) or three-state
  (shoulder or two distinct transitions)
- Multi-condition upload (CSV or Excel) and simultaneous plotting
- Manual data entry by paste from any spreadsheet
- **Per-condition model selection**: two-state or three-state, independently per condition
- Per-replicate and summary (mean ± SD) fit tables with R² and RMSE
- Three-state results include Tm1, ΔH1, ΔS1, Tm2, ΔH2, ΔS2 per replicate
- Three-state plot overlays: fU curve, fI (intermediate) curve, and
  spectroscopically-weighted unfolding progress curve — each with independent
  line style control (solid, dashed, dotted, dash-dot)
- Interactive Plotly figure with hover, zoom, pan
- **Figure presets**: Publication (3.5″ or 7″), Poster, PowerPoint
- **8 graph option categories**: Canvas, Theme, Axes, Ticks, Fonts, Gridlines, Legend, Markers
- **Dimension units**: px, inches, cm with live pixel conversion and effective DPI display
- Download figure as PNG / SVG / JPEG / WebP with configurable export resolution
- Download results as CSV (per-replicate, summary, fraction unfolded vs T,
  intermediate population vs T for three-state fits)
- Cross-condition comparison table (handles mixed two-state / three-state conditions)

---

## Files

```
cd_analyzer/
├── app.py           # Streamlit UI
├── fitting.py       # Two-state and three-state models, scipy curve_fit logic
├── plotting.py      # Plotly figure builder (shared for fitted and raw data)
├── version.py       # Version and authorship metadata
├── requirements.txt
└── README.md
```

---

## Citation

If you use this tool in published research, please cite it.
A `CITATION.cff` template is provided below — fill in your details
and add it to the repository root before pushing to GitHub.

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it."
authors:
  - family-names: May
    given-names: John
title: "CD Thermal Denaturation Analyzer"
version: 1.2.0
date-released: 2026-06-23
url: https://github.com/johnmaybiochem/cd-thermal-denaturation-analyzer
```

To mint a citable DOI:
1. Push to a public GitHub repo
2. Connect the repo to [Zenodo](https://zenodo.org)
3. Tag a release (`v1.2.0`) — Zenodo auto-archives and issues a DOI

---

## Deployment (private access)

**Streamlit Community Cloud (easiest)**
1. Push this folder to a private or public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Under *Advanced settings → Viewers*, add allowed email addresses

**Local network sharing**
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Share your machine's local IP (e.g. `http://192.168.1.x:8501`).

---

## Version history

| Version | Notes |
|---------|-------|
| 1.2.0   | Three-state model (N ⇌ I ⇌ U) with fU, fI, and spectroscopically-weighted progress curves; Plot Raw Data button; per-condition 3-state line style controls; marker cap fix |
| 1.1.1   | Updated figure download options and removed kaleido requirement for download |
| 1.1.0   | Figure presets (Publication/Poster/PowerPoint), in/cm units, version tracking |
| 1.0.0   | Initial release |
