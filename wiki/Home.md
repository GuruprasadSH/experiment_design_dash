# DOE Assistant v2 — Home

Welcome to the official wiki for **DOE Assistant v2**, a browser-based application that guides engineers, scientists, and analysts through the full Design of Experiments (DOE) workflow — from choosing a design type, through fitting a statistical model, to predicting and optimising outcomes — all with AI-generated interpretation at every step.

---

## What is DOE Assistant v2?

Design of Experiments is a structured method for understanding how input variables (factors) affect one or more outputs (responses). DOE Assistant v2 takes that methodology and wraps it in an interactive Dash application so you never have to write a line of R or Python to run a rigorous DOE analysis.

The application covers three phases:

| Phase | Tab | What you do |
|---|---|---|
| **Plan** | Design | Choose a design type, define factors and ranges, generate the experimental matrix, optionally get an AI recommendation |
| **Analyse** | Analysis | Enter results, fit a regression model, view ANOVA, Pareto charts, residual diagnostics, and AI interpretation |
| **Predict** | Prediction & Optimisation | Explore response surfaces, run numerical optimisation, find factor settings that maximise, minimise, or hit a target |

The AI layer uses Anthropic Claude models throughout. Tabular results are interpreted by a fast model (claude-haiku-4-5); chart images are sent to a vision model (claude-sonnet-4-6) which reads the actual plot and reports what it sees. You always see which model generated a given paragraph.

---

## Who is this for?

- **Process engineers** running screening studies to find the vital few factors before optimising.
- **R&D scientists** designing formulation experiments (mixture designs) or factorial studies.
- **Quality engineers** applying Taguchi methods or Plackett-Burman screening.
- **Students and practitioners** learning DOE concepts with immediate statistical feedback.
- **Data analysts** who need to share polished HTML reports with stakeholders who have no Python environment.

No statistics package subscription is required. Everything runs locally once you have an Anthropic API key.

---

## v2 Highlights

The table below summarises the major features added or improved since v1.

| Feature | v1 | v2 |
|---|---|---|
| Design types | 5 | 9 (added General Factorial, Taguchi, Simplex Lattice, Simplex Centroid) |
| AI assistance | Design tab only | All 4 analysis accordion sections + Help panel |
| AI vision | No | Yes — Effects and Residuals sections pass PNG images to claude-sonnet-4-6 |
| AI model routing | Single model | Three models matched to task complexity |
| Analysis accordion | Single scrolling page | 4 collapsible sections, each with its own Interpret button |
| Residual diagnostics | 2 plots | 4-panel figure + Shapiro-Wilk p-value badge |
| Save Report | CSV export only | Self-contained HTML (CDN Plotly, no server needed) |
| Help panel | None | Floating ? button, context-aware chat, knows active tab and model state |
| Coded vs actual equations | Coded only | Both, side by side |
| Kaleido compatibility | Pinned old version | Dual-path approach works with kaleido 0.1.x and 0.2.x |
| Center-point tracking | None | Point Type column, badges in Design Summary |

---

## Technology Stack

| Component | Library / Service |
|---|---|
| Web framework | [Plotly Dash](https://dash.plotly.com/) |
| UI components | [dash-bootstrap-components](https://dash-bootstrap-components.opensource.faculty.ai/) |
| DOE matrix generation | [pyDOE3](https://github.com/relf/pyDOE3) |
| Statistical modelling | [statsmodels](https://www.statsmodels.org/) |
| Optimisation | [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) |
| AI interpretation | [Anthropic Claude API](https://docs.anthropic.com/) |
| Chart image export | [kaleido](https://github.com/plotly/Kaleido) |

---

## Wiki Navigation Map

Use the links below to jump to any topic. Each page is self-contained but cross-links to related pages.

### Getting started
- [Installation and Setup](Installation-and-Setup) — prerequisites, clone, pip install, `.env`, running the app, troubleshooting startup

### Using the application
- [Design Types](Design-Types) — full reference for all 9 design types with comparison table and usage rules
- [Design Tab](Design-Tab) — step-by-step guide to generating a design matrix and using the AI Design Assistant
- [Analysis Tab](Analysis-Tab) — loading data, fitting models, reading the 4 accordion sections, saving the HTML report
- [AI Interpretation Guide](AI-Interpretation-Guide) — how the AI sections work, what data is passed, decision tree for next steps
- [Prediction and Optimisation](Prediction-and-Optimisation) — response surface plots, numerical optimisation, confirmatory experiments

### Reference
- [Troubleshooting](Troubleshooting) — symptom-by-symptom fixes for common problems
- [Developer Guide](Developer-Guide) — architecture, file map, callback conventions, how to add new design types or AI sections

---

## Application Layout at a Glance

```
┌─────────────────────────────────────────────────────────┐
│  DOE Assistant v2                              [?] Help  │
├───────────────┬─────────────────┬───────────────────────┤
│   Design      │    Analysis     │  Prediction & Optim.  │
└───────────────┴─────────────────┴───────────────────────┘
│                                                         │
│  [Tab content renders here]                             │
│                                                         │
│  Design tab:                                            │
│    Design type selector → factor table → options →      │
│    Generate button → design matrix table                │
│    [▼ AI Design Assistant] (collapsible)                │
│                                                         │
│  Analysis tab:                                          │
│    Data entry panel → response/factor selectors →       │
│    Fit Model button                                     │
│    Accordion:                                           │
│      1. Design Summary      [Interpret]                 │
│      2. ANOVA & Model Stats [Interpret]                 │
│      3. Effects & Interaction Analysis [Interpret]      │
│      4. Residual Analysis   [Interpret]                 │
│    [Save Report as HTML]                                │
│                                                         │
│  Prediction tab:                                        │
│    Coefficient table → regression equations             │
│    Response surface plot controls                       │
│    Numerical optimisation panel                         │
│    [Save Report as HTML]                                │
└─────────────────────────────────────────────────────────┘
```

---

## Screenshot Placeholders

> **Note:** Replace the image links below with actual screenshots once the application is running. Use the format `![caption](images/filename.png)` and commit the images to `wiki/images/`.

| Page / feature | Placeholder |
|---|---|
| Design tab with CCD selected | `![Design tab](images/design-tab-ccd.png)` |
| Design matrix table | `![Design matrix](images/design-matrix.png)` |
| AI Design Assistant panel | `![AI Design Assistant](images/ai-design-assistant.png)` |
| Analysis accordion — open | `![Analysis accordion](images/analysis-accordion.png)` |
| Pareto chart | `![Pareto chart](images/pareto-chart.png)` |
| Residual 4-panel diagnostic | `![Residual diagnostics](images/residual-4panel.png)` |
| AI interpretation card | `![AI interpretation](images/ai-interpretation-card.png)` |
| Response surface 3D plot | `![Response surface](images/response-surface-3d.png)` |
| Optimisation result panel | `![Optimisation result](images/optimisation-result.png)` |
| Help panel chat | `![Help panel](images/help-panel.png)` |

---

## Quick-Start (30-second version)

```bash
git clone https://github.com/your-org/doe-assistant.git
cd doe-assistant
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python app.py
# open http://127.0.0.1:8050 in your browser
```

For full setup instructions including troubleshooting, see [Installation and Setup](Installation-and-Setup).

---

## Statistical Standards

DOE Assistant v2 applies rules from two authoritative sources:

- **NIST/SEMATECH e-Handbook of Statistical Methods** ([itl.nist.gov/div898/handbook](https://www.itl.nist.gov/div898/handbook/))
- **Montgomery, D.C. — *Design and Analysis of Experiments*, 9th edition**

All thresholds (R² ≥ 0.90, Adj R² gap ≤ 0.10, Pred R² ≥ 0.70, error df ≥ 6, etc.) are sourced from these references. See [AI Interpretation Guide](AI-Interpretation-Guide) for the full rule set.

---

## Licence and Contributing

Refer to `README.md` in the repository root for licence information. Bug reports and pull requests are welcome. Before adding a new design type or AI section, read the [Developer Guide](Developer-Guide) to understand the expected conventions.
