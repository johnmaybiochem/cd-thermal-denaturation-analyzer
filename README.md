# CD Thermal Denaturation Analyzer
**Version 1.1.0**

A Streamlit web app for fitting circular dichroism thermal denaturation data
to a two-state unfolding model with sloping baselines. Code developed with assistance from Claude AI.

---

## Model

$$\Delta G(T) = \Delta H \left(1 - \frac{T}{T_m}\right)$$

$$K(T) = \exp\!\left(\frac{-\Delta G}{RT}\right), \quad f_U = \frac{K}{1+K}$$

$$CD_{obs}(T) = (m_N T + b_N)(1 - f_U) + (m_U T + b_U)\,f_U$$

Free parameters: **Tm**, **ΔH**, **mN**, **bN**, **mU**, **bU**  
Derived: **ΔS** = ΔH / Tm

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

---

## Features

- Multi-condition upload (CSV or Excel) and simultaneous plotting
- Manual data entry by paste from any spreadsheet
- Per-replicate and summary (mean ± SD) fit tables with R² and RMSE
- Interactive Plotly figure with hover, zoom, pan
- **Figure presets**: Publication (3.5″ or 7″), Poster, PowerPoint
- **8 graph option categories**: Canvas, Theme, Axes, Ticks, Fonts, Gridlines, Legend, Markers
- **Dimension units**: px, inches, cm with live pixel conversion and effective DPI display
- Download figure as PNG / PDF / SVG with configurable export resolution
- Download results as CSV (per-replicate, summary, fraction unfolded vs T)
- Cross-condition comparison table

---

## Files

```
cd_analyzer/
├── app.py           # Streamlit UI
├── fitting.py       # Two-state model and scipy curve_fit logic
├── plotting.py      # Plotly figure builder
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
  - family-names: Your Last Name
    given-names: Your First Name
    orcid: https://orcid.org/0000-0000-0000-0000
title: "CD Thermal Denaturation Analyzer"
version: 1.1.0
date-released: 2025-06-15
url: https://github.com/yourname/cd-analyzer
```

To mint a citable DOI:
1. Push to a public GitHub repo
2. Connect the repo to [Zenodo](https://zenodo.org)
3. Tag a release (`v1.1.0`) — Zenodo auto-archives and issues a DOI

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
| 1.1.0   | Figure presets (Publication/Poster/PowerPoint), in/cm units, version tracking, denaturation curve sidebar illustration |
| 1.0.0   | Initial release |
