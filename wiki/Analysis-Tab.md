# Analysis Tab

The Analysis tab is where you enter your experimental results, fit a statistical model, and explore the four accordion sections that explain what your data is telling you. Each section includes an **Interpret** button that sends the relevant data or charts to Claude for a plain-English explanation.

---

## Overview of the Analysis Tab Layout

```
┌────────────────────────────────────────────────────────────┐
│  Data Entry                                                │
│  [ Transfer from Design Tab ] [ Upload CSV/Excel ]         │
│  [ Paste from Excel ]                                      │
│  [Editable data table]                                     │
│  [ + Add Response Column ] [ Rename Column ]               │
├────────────────────────────────────────────────────────────┤
│  Model Setup                                               │
│  Response:  [Yield ▼]                                      │
│  Factors:   [✓ Temp] [✓ Pressure] [✓ Time] [ Catalyst]    │
│  Terms:     [Main] [+2FI] [Full Factorial] [+Quadratic]    │
│  Model term checklist: [Temp] [Pressure] [Time]            │
│                        [Temp×Pressure] [Temp×Time] ...     │
│  [ Fit Model ]                                             │
├────────────────────────────────────────────────────────────┤
│  Results accordion                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Design Summary                        [Interpret]│   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ 2. ANOVA & Model Stats                   [Interpret]│   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ 3. Effects & Interaction Analysis        [Interpret]│   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ 4. Residual Analysis                     [Interpret]│   │
│  └─────────────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────┤
│  [ Save Report as HTML ]                                   │
└────────────────────────────────────────────────────────────┘
```

---

## Loading Your Data

You have three ways to get experimental data into the analysis table.

### Method 1 — Transfer from the Design tab

This is the fastest method when you generated your design in the same session.

1. Make sure you have already generated a design matrix on the Design tab (see [Design Tab](Design-Tab)).
2. Switch to the Analysis tab.
3. Click **Transfer from Design Tab**.
4. The factor columns, Std Order, Run Order, Block, and Point Type columns will populate the editable table automatically.
5. The Response column will be added as an empty editable column — type your measured values directly into the cells.

> **Note:** Transferring from the Design tab preserves the Run Order, Point Type, and Block columns. These are used by the model-fitting code to assign center-point badges and block terms correctly.

### Method 2 — Upload CSV or Excel

1. Click **Upload CSV/Excel**.
2. Select a `.csv` or `.xlsx` file from your computer.
3. The file is parsed and loaded into the editable table.

Expected file format:

- The first row must be column headers.
- Factor and response columns can be in any order.
- Numeric factor columns should contain numbers only (no units in the cell).
- The Point Type column is optional but recommended — values should be `Factorial`, `Axial`, or `Center`.

If your file does not have a Run Order column, the app will assign run order as the row sequence (1, 2, 3, ...).

> **Tip:** If you are loading data from a physical lab notebook or LIMS, prepare it as an Excel file with clean headers matching your factor names exactly. Column name matching is case-insensitive but spaces and special characters must be consistent.

### Method 3 — Paste from Excel

1. Open your data in Excel or Google Sheets.
2. Select all cells including the header row.
3. Copy (Ctrl+C or Cmd+C).
4. Click the **Paste from Excel** button in the app.
5. The app parses the clipboard content as tab-separated values and populates the table.

> **Warning (v2 fix):** In v1, pasting sometimes added extra blank rows at the bottom. This was fixed in v2 — the app strips rows where all factor and response columns are empty. If you are on an older build, delete blank rows manually before fitting.

---

## Managing Response Columns

After loading data, you can add or rename response columns.

### Adding a response column

Click **+ Add Response Column**. A new column appears at the right of the table with a default name (`Response_2`, `Response_3`, etc.). You can rename it immediately.

You can have multiple response columns (e.g., Yield, Purity, Viscosity) and fit a separate model for each. Switch between responses using the **Response** dropdown in the Model Setup panel.

### Renaming a column

Double-click any column header in the editable table. The header becomes an input field — type the new name and press Enter.

### Deleting a column

Click the column header to select it, then click the **Delete Column** button that appears in the toolbar. This removes the column from the table but does not affect other columns.

---

## Model Setup

### Selecting the response

Use the **Response** dropdown to select which column is your output variable. The dropdown lists all columns that are not factor columns, Block, or order columns.

### Selecting factors

The **Factors** checklist shows all columns identified as factor columns (numeric or categorical). Check the factors you want to include in the model. Uncheck any factors that are nuisance variables or that you have decided to exclude after screening.

> **Tip:** Start with all factors checked. After seeing the Pareto chart (section 3 of the accordion), re-fit with only the significant factors.

### Picking model terms

The **model term picker** is a checklist that lists all possible terms for the selected factors: main effects, two-factor interactions (2FIs), quadratic terms, and higher-order terms for RSM designs.

Use the quick-select buttons to populate the checklist:

| Button | Terms included |
|---|---|
| **Main** | All main effects only (intercept + A + B + C + ...) |
| **+2FI** | Main effects + all two-factor interactions (A×B, A×C, B×C, ...) |
| **Full Factorial** | All main effects + all interactions up to the highest order |
| **+Quadratic** | Main effects + 2FIs + quadratic terms (A², B², C², ...) for numeric factors |

For response surface designs (CCD, Box-Behnken), start with **+Quadratic**. For screening designs (Plackett-Burman, fractional factorials), start with **Main**.

After clicking a quick-select button, you can refine the selection by checking or unchecking individual terms in the checklist. For example, you might start with **+2FI** and then uncheck specific interaction terms that are clearly inactive.

### Fitting the model

Click **Fit Model**. The app:

1. Encodes numeric factors to coded units (-1 to +1).
2. Builds the design matrix X for the selected terms.
3. Fits an OLS regression using statsmodels (`sm.OLS(y, X).fit()`).
4. Computes the ANOVA table, R² statistics, residuals, and Cook's distance.
5. Unlocks all four accordion sections and the Interpret buttons.
6. Enables the Prediction & Optimisation tab.

Fitting takes less than one second for typical DOE data (< 200 runs).

> **Warning:** If the **Fit Model** button is disabled, check that you have selected at least one response and at least one factor, and that all response cells contain numeric values (no blank cells, no text).

---

## The Four Accordion Sections

After fitting the model, the Results accordion becomes active. Each section can be expanded independently by clicking its header.

---

### Section 1 — Design Summary

**What it shows:**

- **Factor table:** Name, type, low level, high level, units for each factor in the model.
- **Run count badge:** Total number of runs.
- **Block badge:** Number of blocks (if > 1).
- **Replicate badge:** Number of replicates (if > 1).
- **Center-point badge:** Number of center points detected (using the Point Type column).
- **Design type badge:** The design family used (transferred from the Design tab, or inferred from the data structure if loaded from CSV).

**How to read it:**

The Design Summary section confirms that the app has correctly understood your experimental structure. Check that the factor ranges match what you actually ran, the center-point count matches the number of runs at the midpoint, and the design type is correct.

If anything looks wrong, fix the data in the editable table and re-fit.

**AI Interpret button:**

Uses **claude-haiku-4-5**. Passes the factor table and badge values as structured text. The AI produces a summary of the design structure, flags any potential issues (e.g., too few center points, no pure error df), and notes whether the error degrees of freedom meet the NIST minimum of 6.

See [AI Interpretation Guide](AI-Interpretation-Guide) for the full output format.

---

### Section 2 — ANOVA & Model Stats

**What it shows:**

**ANOVA table** with columns:

| Column | Meaning |
|---|---|
| Source | Model, each factor/interaction term, Residual, Lack of Fit, Pure Error, Total |
| df | Degrees of freedom |
| SS | Sum of squares |
| MS | Mean square (SS / df) |
| F | F-ratio (MS_term / MS_error) |
| p-value | Probability that the F-ratio is this large by chance |

**Model fit statistics** displayed as labelled badges:

| Statistic | Good threshold | Meaning |
|---|---|---|
| R² | > 0.90 | Fraction of total variation explained by the model |
| Adj R² | Close to R²; gap < 0.10 | R² adjusted for the number of model terms |
| Pred R² | > 0.70 | How well the model predicts new observations (leave-one-out) |
| RMSE | Smaller is better (relative to response scale) | Root mean square error — average prediction error in response units |

**How to read the ANOVA table:**

- **Model p-value:** Should be < 0.05 for the model to be considered statistically significant.
- **Individual term p-values:** Terms with p < 0.05 are statistically significant. Terms with p > 0.10 are candidates for removal to reduce model complexity.
- **Lack of Fit (LOF) p-value:** Should be > 0.10 (i.e., not significant). A significant LOF (p < 0.10) means the fitted model does not adequately describe the true response surface.
- **Pure Error:** Requires either center-point replicates or whole-design replicates. If pure error df = 0, LOF cannot be computed.

> **Warning:** A large gap between Adj R² and Pred R² (> 0.10) is a sign of overfitting. Remove non-significant terms and re-fit.

**AI Interpret button:**

Uses **claude-haiku-4-5**. Passes the full ANOVA table and the R² / RMSE statistics as text. The AI applies the statistical rules and tells you whether the model passes each criterion, with specific numbers from your table.

---

### Section 3 — Effects & Interaction Analysis

**What it shows:**

Four charts arranged in a 2×2 grid:

**Top-left — Pareto chart of standardised effects:**
Horizontal bar chart. Each bar represents one model term, sorted from largest to smallest absolute standardised effect (t-value). A vertical red reference line marks the t-critical value at α = 0.05. Bars crossing the line are statistically significant.

**Top-right — Half-normal plot:**
Plots the absolute standardised effects against half-normal quantiles. Inactive (noise) effects should fall along a straight line through the origin. Active effects appear as outliers above the line in the upper right.

**Bottom-left — Main effects plot:**
For each factor, shows the mean response at the low level (-1) and high level (+1). A steep slope indicates a large main effect.

**Bottom-right — Interaction plot:**
For each two-factor interaction in the model, plots the mean response at the four combinations of two factors. Parallel lines indicate no interaction; crossing or diverging lines indicate an interaction.

> **Note (v2 fix):** In v1, the interaction plot sometimes showed only data points without connecting lines for RSM designs with sparse combinations. This is fixed in v2 using Plotly's `connectgaps=True` setting.

**AI Interpret button:**

Uses **claude-sonnet-4-6** (vision model). The app exports all four charts as PNG images (via kaleido) and passes them to the model alongside the coefficient table. The AI reads the actual plots and comments on what it sees — naming specific factors, describing crossing vs. parallel lines, and flagging unusual patterns.

---

### Section 4 — Residual Analysis

**What it shows:**

A 4-panel diagnostic figure:

**Top-left — Normal Q-Q plot:**
Plots residuals against the theoretical quantiles of a normal distribution. If residuals are normally distributed, the points should fall approximately on the 45° diagonal line.

**Top-right — Residuals vs Fitted values:**
Plots residuals on the y-axis against fitted (predicted) values on the x-axis. You want a random scatter around zero with no pattern. Funnel-shaped spread indicates heteroscedasticity. A U-shape indicates missing quadratic terms.

**Bottom-left — Residuals vs Run Order:**
Plots residuals in run order. A trend indicates a time-dependent lurking variable.

**Bottom-right — Histogram of residuals:**
A simple frequency histogram of the residuals. Should be approximately bell-shaped.

**Shapiro-Wilk test badge:**
Below the 4-panel figure, the Shapiro-Wilk test statistic and p-value are displayed. A p-value > 0.05 means you cannot reject the normality assumption.

**How to read residual plots:**

| Pattern | Diagnosis | Action |
|---|---|---|
| Random scatter around zero | Assumptions met | Proceed with model |
| Q-Q plot curves away from diagonal | Non-normal residuals | Transform response (log, sqrt) or investigate outliers |
| Funnel in Residuals vs Fitted | Heteroscedasticity | Transform response (log, sqrt) |
| Arch or U-shape in Residuals vs Fitted | Missing quadratic terms | Add quadratic terms, augment to RSM design |
| Trend in Residuals vs Run Order | Time-related nuisance | Block on time period in next experiment |
| Large outliers (> 3 standard deviations) | Influential observation | Check data entry, investigate the run |

**AI Interpret button:**

Uses **claude-sonnet-4-6** (vision model). The 4-panel figure and the Shapiro-Wilk result are passed as a PNG image. The AI describes what it sees in each panel and applies the standard residual diagnostic rules.

---

## Save Report as HTML

The **Save Report as HTML** button at the bottom of the Analysis tab generates a self-contained HTML file that contains:

- All interactive Plotly charts (Pareto, half-normal, main effects, interaction, residual 4-panel, response surface)
- The full ANOVA table
- The coefficient table with standard errors, t-values, and p-values
- Regression equations in both coded units and actual (decoded) units
- The raw data table
- AI interpretation text for any sections where you clicked Interpret before saving

**The file is self-contained** — it embeds Plotly.js via CDN link and all chart data as JSON. Recipients can open it in any browser without Python, Dash, or an internet connection to a server.

### How to save

1. Make sure you have fitted a model.
2. Optionally, click **Interpret** in all four accordion sections to include AI text in the report.
3. Click **Save Report as HTML**.
4. A file download dialog appears. Save the file with a descriptive name (e.g., `yield_optimisation_report.html`).

---

## Working with Multiple Responses

If your experiment has more than one response (e.g., Yield and Purity), you can fit and report on each separately:

1. Select the first response in the Response dropdown.
2. Fit the model, review all four sections, click Interpret buttons.
3. Click **Save Report as HTML** (saves the first response report).
4. Select the second response in the Response dropdown.
5. Repeat the fitting and interpretation process.
6. Save a second HTML report.

The prediction tab updates to show the currently selected response's model.

---

## Cross-References

- For an explanation of what each AI section passes to Claude and how to interpret the output, see [AI Interpretation Guide](AI-Interpretation-Guide).
- For using the fitted model to predict responses and find optimum settings, see [Prediction and Optimisation](Prediction-and-Optimisation).
- For common problems (LOF not shown, interaction plot issues, Shapiro-Wilk rejection), see [Troubleshooting](Troubleshooting).
