# DOE Assistant — v2

A self-contained Design of Experiments workbench built with Plotly Dash and Claude AI. It covers the full experimental workflow — design generation → data entry → statistical analysis → response-surface optimisation — and augments every stage with AI interpretation grounded in NIST and Montgomery best practices.

---

## What's new in v2

| Area | Change |
|---|---|
| **Per-section AI interpretation** | Each analysis section (Design, ANOVA, Effects, Residuals) has its own independent "🤖 AI Interpretation" button instead of one monolithic button |
| **Multimodal AI** | Effects and Residual sections pass base64-encoded PNG images of the actual plots to Claude — the model reads the charts, not just numbers |
| **Save Report as HTML** | One-click export of the full analysis (tables + interactive Plotly charts + AI text) as a self-contained `.html` file — no server needed to open it |
| **AI Assistant moved to Design tab** | Collapsed panel at the bottom of the Design tab; no longer a separate tab |
| **Rename response button** | Moved next to "Add Response Column" for easier access |
| **Paste-data fix** | Pasting response values into the table no longer appends extra rows |
| **Interaction plot lines fix** | Lines now correctly connect across sparse factor-level combinations in RSM/CCD designs |
| **Quadratic quick-select** | "+ Quadratic" button in Model Terms for one-click RSM term selection |
| **kaleido compatibility** | Works with both kaleido 0.2.x and kaleido 1.x (auto-detects the correct rendering path) |

---

## Features

### Nine design types

| Category | Designs |
|---|---|
| **Screening** | Plackett-Burman (N = 12, 20, 24, …) |
| **Two-level factorial** | Full 2ᵏ and fractional 2ᵏ⁻ᵖ (Resolution III – V) |
| **Response surface** | Central Composite (CCC, CCI, CCF) and Box-Behnken |
| **General factorial** | Mixed-level; any number of levels per factor |
| **Taguchi** | Standard orthogonal arrays (L4 – L18) |
| **Mixture** | Simplex Lattice and Simplex Centroid |

---

### Design tab

- Factor definition: numeric or categorical, with named levels
- Structural options: replicates, center points, blocks, randomisation seed
- CCD sub-types and α choices (rotatable, orthogonal, face-centred, custom)
- Instant design matrix with run-order and point-type columns
- Copy to clipboard or download as CSV
- **🤖 AI Design Assistant** (collapsible panel at the bottom):
  - 7-question guided interview covering process, factors, run budget, and goals
  - NIST-grounded design recommendation with justification
  - **Apply Recommended Design →** button auto-configures all app settings

---

### Analysis tab

- Load data by: transferring from Design tab, uploading CSV/Excel, or pasting tab-separated text from Excel
- Add and rename response columns without losing previously entered data
- **Model setup**: factor checklist, model-term picker with quick-select buttons (Main only / +2FI / Full factorial / +Quadratic)
- **Fit Model** runs full OLS regression via statsmodels

Four accordion sections, each with an independent AI interpretation button:

| Section | Content | AI model |
|---|---|---|
| **Design Summary** | Factor table, run count, block/replicate/center-point badges, design type badge | claude-haiku (text) |
| **ANOVA & Model Stats** | Full ANOVA table, R², Adj R², Pred R², RMSE | claude-haiku (text) |
| **Effects & Interaction Analysis** | Pareto chart, half-normal plot, main effects plot, interaction plot | claude-sonnet (vision — reads the actual plots) |
| **Residual Analysis** | 4-panel residual diagnostic (Q-Q, vs Fitted, vs Run Order, Histogram) + Shapiro-Wilk stats | claude-sonnet (vision — reads the actual plots) |

Each AI interpretation follows the same format:
- One-line verdict (✅ / ⚠️ / ❌) with a summary sentence
- 3–5 bullet points citing specific numbers or visual patterns
- One "Next step:" recommendation

**💾 Save Report as HTML** — generates a self-contained `.html` file containing:
- All interactive Plotly charts (no server required)
- ANOVA and coefficient tables
- Regression equations (coded and actual units)
- Full experimental data table
- Any AI interpretation text generated during the session

---

### Prediction & Optimisation tab

- Regression equation in coded and actual units
- Coefficient table with standard errors and p-values
- Contour and 3D response-surface plots (factor X vs factor Y, others held at midpoint)
- Numerical optimisation: Maximize / Minimize / Target, with predicted response and optimal factor settings
- **💾 Save Report as HTML** also available here

---

### Help panel

- Floating **?** button (bottom-right corner) on every tab
- Context-aware chat powered by `claude-haiku-4-5`; knows which tab is active and the current model state
- Maintains full conversation history within the session

---

## Architecture

```
doe-assistant/
├── DOE/
│   ├── app.py              ← Dash layout + all callbacks
│   ├── analysis.py         ← Statistical engine: model fitting, all plot functions,
│   │                          residual diagnostics, fig_to_b64 image export
│   ├── doe_generators.py   ← Design matrix generators (pyDOE3 wrappers)
│   └── tests/              ← pytest suite (engine + Montgomery regression tests)
├── agent/
│   ├── knowledge.py        ← Hardcoded NIST/Montgomery decision rules (no RAG)
│   ├── interviewer.py      ← 7-question design-selection interview loop
│   ├── interpreter.py      ← Four per-section AI interpretation methods
│   ├── recommender.py      ← Rule-based next-step logic (no API call)
│   └── app_controller.py   ← HTTP bridge for apply-recommendation endpoint
├── rag/                    ← Retained for reproducibility; not used at runtime
├── ingest/                 ← NIST handbook scrape + embed pipeline (offline)
├── data/                   ← Curated NIST chunks + evaluation set
├── requirements.txt
└── .env.example
```

### AI model routing

| Task | Model | Reason |
|---|---|---|
| Design interview | claude-sonnet-4-6 | Multi-turn reasoning, design trade-off discussion |
| Design interpretation | claude-haiku-4-5 | Text-only, guideline-driven, fast |
| ANOVA interpretation | claude-haiku-4-5 | Text-only, table reading |
| Effects interpretation | claude-sonnet-4-6 | Vision required (Pareto + interaction plots) |
| Residuals interpretation | claude-sonnet-4-6 | Vision required (Q-Q + vs-Fitted panels) |
| Help panel chat | claude-haiku-4-5 | Short, contextual answers |

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd doe-assistant
pip install -r requirements.txt
```

> **kaleido note**: required for server-side PNG export used by the AI interpretation of Effects and Residual plots. Both kaleido 0.2.x and 1.x are supported — the app auto-detects the correct rendering path.

### 2. Configure API key

```bash
cp .env.example .env
# Open .env and add:
#   ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run

```bash
python DOE/app.py
# Open http://127.0.0.1:8050
```

---

## Typical workflow

### Path A — AI-assisted (recommended for new experiments)

1. Open the **Design** tab and expand the **🤖 AI Design Assistant** panel at the bottom.
2. Answer the 7 questions (process goal, factors, run budget, noise sources, etc.).
3. The assistant recommends a design type with NIST justification.
4. Click **Apply Recommended Design →** to auto-configure all settings.
5. Click **Generate Design** and download or copy the run sheet.
6. Run the experiments; enter response values in the **Analysis** tab.
7. Select the response variable, choose model terms, click **Fit Model**.
8. Expand each accordion section and click **🤖 AI Interpretation** for instant expert commentary.
9. Click **💾 Save Report as HTML** for a shareable report.

### Path B — Manual (you already know what design you need)

1. **Design tab** → pick a design type → define factors → Generate.
2. **Analysis tab** → enter or upload responses → Fit Model → interpret each section.
3. **Prediction & Optimisation tab** → run optimisation → inspect surface and contour plots.

---

## Sample AI Design Assistant session

```
Assistant: What process are you trying to improve, and what will you measure?

User:      Injection moulding cycle time (s). We control Melt temperature
           (200-250 C), Injection speed (50-100 mm/s), and Cooling time (10-30 s).
Assistant: How many experimental runs can you afford?

User:      Around 20 runs maximum.

...(5 more questions)...

Assistant: I recommend a 2^3 Full Factorial design with 4 center points
           (12 runs total). With 3 numeric factors and a 20-run budget,
           the full factorial estimates all main effects AND all two-factor
           interactions independently. Center points detect curvature at no
           cost to the factorial structure. If you see significant curvature,
           augment to a CCD in a follow-up experiment.
           [Apply Recommended Design button appears]
```

---

## Dependencies

| Package | Purpose |
|---|---|
| dash >= 2.14 | Web framework |
| dash-bootstrap-components >= 1.5 | UI components |
| plotly >= 5.18 | Interactive charts |
| pandas >= 2.0 | Data manipulation |
| numpy >= 1.24 | Numerics |
| scipy >= 1.11 | Shapiro-Wilk, stats |
| statsmodels >= 0.14 | OLS regression, ANOVA |
| pyDOE3 >= 1.0 | Design matrix generation |
| anthropic >= 0.25 | Claude API client |
| kaleido >= 0.2 | Server-side PNG export for AI vision features |
| markdown >= 3.5 | Markdown-to-HTML for report export |
| python-dotenv >= 1.0 | .env loading |
| openpyxl >= 3.1 | Excel upload/download |

---

## Security

- **Never commit `.env`** -- it is in `.gitignore`.
- The `.env.example` file contains only blank placeholder values.
- Before every push, verify no secrets have been staged:

  ```bash
  git diff --cached | grep -i "api_key"
  ```

- If a key is accidentally staged: `git reset HEAD <file>` immediately and do **not** push.

---

## Running the tests

```bash
cd DOE
pytest tests/ -v
```

The test suite covers:
- All nine design generators (dimensions, column names, run counts)
- Montgomery Example 6.2 regression (known coefficient values verified to 3 d.p.)
- Categorical factor encoding and term picker behaviour

---

## Contributing

Pull requests welcome. Please open an issue first for significant changes.
Follow the existing code style (Black-compatible, type hints on new functions).

---

*Powered by [Anthropic Claude](https://anthropic.com) and grounded in the [NIST/SEMATECH Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/) and Montgomery, D.C. (2017) Design and Analysis of Experiments, 9th ed.*
