# Prediction and Optimisation

The Prediction & Optimisation tab becomes active after you fit a model on the Analysis tab. It gives you two tools: an interactive response surface explorer and a numerical optimiser that finds the factor settings predicted to maximise, minimise, or hit a target response value.

---

## Prerequisites

Before using the Prediction tab, you need:

1. A fitted model (click **Fit Model** on the Analysis tab).
2. At least two numeric factors in the model (response surface plots require an X-axis and a Y-axis factor).

If the tab appears greyed out or shows a "No model fitted" message, return to the Analysis tab and fit a model.

---

## Coefficient Table

At the top of the Prediction tab you will find the **Coefficient Table**, which lists every term in the fitted model alongside its estimated coefficient, standard error, t-value, and p-value.

| Column | Meaning |
|---|---|
| Term | The model term (Intercept, A, B, AB, A², etc.) |
| Coefficient (coded) | The estimated effect size in coded units (-1 to +1) |
| Std Error | The standard error of the coefficient estimate |
| t-value | Coefficient / Std Error |
| p-value | Two-tailed p-value for the t-test on this coefficient |

### Why coded units?

The model is fitted in coded units so that coefficients are directly comparable — a coefficient of 5.2 for factor A and 1.8 for factor B means factor A has approximately three times the effect of factor B, regardless of the original measurement scales (e.g., Temperature in °C vs Pressure in bar).

In actual (original) units, the coefficients depend on the measurement scale and cannot be directly compared across factors.

---

## Regression Equations

Below the coefficient table, DOE Assistant v2 displays two regression equations side by side.

### Coded equation

The coded equation uses coded factor values (x_A, x_B, etc.) where -1 = low level, +1 = high level:

```
Yield = 78.4 + 5.2·x_A + 1.8·x_B - 0.9·x_A·x_B + 3.1·x_A² + 0.4·x_B²
```

Use the coded equation to:
- Compare the relative importance of terms (largest absolute coefficient = most important).
- Understand the direction of each effect (positive coefficient = response increases as factor increases).
- Identify the sign of interactions (negative coefficient = as A increases, the effect of B decreases).

### Actual equation

The actual equation uses original factor units (e.g., Temperature, Pressure):

```
Yield = -142.7 + 2.34·Temperature + 18.6·Pressure
        - 0.045·Temperature·Pressure + 0.00775·Temperature² + 0.207·Pressure²
```

Use the actual equation to:
- Make predictions without having to convert to coded units.
- Report to non-statistician audiences who prefer original units.
- Enter the equation into a spreadsheet for prediction without running the app.

### Why the coefficients differ between coded and actual equations

The intercept and all coefficients change when switching between coded and actual units because the equation is reparameterised to use different variable scales. The predicted response at any specific set of conditions is **identical** from both equations — the equations are mathematically equivalent, just expressed differently.

> **Tip:** Always report both equations in your technical documentation. The coded equation communicates relative effect sizes; the actual equation communicates how to use the model operationally.

---

## Response Surface Plots

The response surface section lets you visualise how the predicted response changes over a two-dimensional slice of the factor space.

### Choosing factors for the axes

Use the **Factor X** and **Factor Y** dropdowns to select which two factors form the horizontal and vertical axes of the plot. All other numeric factors are held at their midpoint (coded value = 0, actual value = (Low + High) / 2) unless you adjust the sliders (see below).

### Hold-at-midpoint sliders

For experiments with three or more factors, a set of sliders appears — one per factor not chosen for the axes. Each slider lets you set the factor value at which the surface is sliced, from the low level to the high level. This is called a **slice plot**.

For example, in a 3-factor experiment with Temperature, Pressure, and Time:
- If Factor X = Temperature, Factor Y = Pressure, then the Time slider controls the fixed value of Time.
- Moving the Time slider from its midpoint to its high level shows how the surface changes at high Time.

> **Tip:** Start with all hold-constant sliders at their midpoints (the default). This gives you the central slice, which is the most informative view for identifying where the optimum lies. Then vary the sliders one at a time to understand how the surface changes.

### Contour plot vs 3D surface

Use the **View** toggle to switch between:

**Contour plot (2D):**
- Overhead view of the response surface.
- Response value is shown by colour (colour scale on the right) and by contour lines (iso-response curves).
- Easier to read exact response values and identify ridge lines.
- Easier to mark the predicted optimum point.

**3D surface plot:**
- Three-dimensional perspective view with the response on the Z-axis.
- Easier to visualise peaks, valleys, and saddle points intuitively.
- Interactive: click and drag to rotate, scroll to zoom.

Both views are interactive Plotly charts that you can rotate (3D), zoom, and hover over for exact values.

### Reading the response surface

Look for these features in the response surface:

| Feature | What it means |
|---|---|
| A circular bowl or mound shape | A single interior optimum exists within the design space |
| A ridge (elongated peak) | Multiple factor combinations achieve a near-optimum response; the optimum is not well-defined |
| A saddle point | The optimum lies at or near the boundary of the design space |
| A monotone slope (colour gradient all in one direction) | The optimum may be outside the design space; consider extending factor ranges |
| A flat surface | No significant factors were identified; expand ranges or add factors |

> **Warning:** The response surface is a model prediction, not a measurement. It is valid within the factor ranges used in your experiment. Extrapolating beyond those ranges (especially beyond the axial points for a CCD) is unreliable.

---

## Numerical Optimisation

The optimisation panel uses `scipy.optimize.minimize` (with the SLSQP method) to find the factor settings that optimise the model prediction.

### Optimisation goal

Choose one of three goals:

| Goal | Description |
|---|---|
| **Maximise** | Find factor settings that give the highest predicted response |
| **Minimise** | Find factor settings that give the lowest predicted response |
| **Target** | Find factor settings that bring the predicted response as close as possible to a specified target value |

For Target optimisation, enter the desired value in the **Target value** field that appears when this goal is selected.

### Running optimisation

Click **Run Optimisation**. The optimiser:

1. Sets up the objective function (the negative of the predicted response for maximisation, the predicted response for minimisation, or squared deviation from target).
2. Uses the coded factor ranges [-1, +1] as bounds for each numeric factor.
3. Runs from multiple starting points (a grid of starting values) to reduce the chance of finding only a local optimum.
4. Reports the best solution found.

Optimisation typically completes in under one second.

### Reading the optimisation result

The result panel shows:

**Predicted optimum factor settings:**
The factor values (in both coded and actual units) that achieve the optimum predicted response.

```
Optimal factor settings:
  Temperature: 118.3 °C  (coded: +0.915)
  Pressure:      3.7 bar (coded: +0.350)
  Time:         45.0 min (coded:  0.000)

Predicted response: Yield = 87.4%
95% prediction interval: [84.1%, 90.7%]
```

**Predicted response:**
The model's prediction at the optimal settings.

**95% prediction interval:**
The range within which a single future observation at these settings is expected to fall 95% of the time. This accounts for both model uncertainty and process variability. The interval is wider than a confidence interval because it predicts an individual observation, not the mean.

> **Warning:** If the predicted optimum lands exactly on a boundary (coded value = -1 or +1 for any factor), the true optimum may lie outside your design space. Consider expanding that factor's range and running an augmented design.

### Multiple responses

If you have fitted models for multiple responses, the optimisation panel lets you set goals for each response independently. For example:

- Maximise Yield
- Minimise Viscosity
- Target Purity = 99.5%

The app uses a desirability function approach: each response is converted to a desirability score (0 to 1), and the geometric mean of all desirability scores is maximised. The combined desirability is shown as an overall score in the result panel.

> **Note:** Multi-response optimisation requires a fitted model for each response. If a response has no model, it is excluded from the desirability calculation.

---

## Confirmatory Experiments

> **This step is mandatory before implementing any process change based on an optimisation result.**

The model is a simplified mathematical approximation of reality. Before acting on the predicted optimum, you must verify it experimentally.

### Why you need confirmation runs

- The model may be overfitted or have residual structure not captured in the residual plots.
- The optimal settings may fall in a region of the design space that was not well-sampled.
- Process variability may mean that the actual response at the optimum differs from the prediction.
- Factor interactions with variables not included in your DOE (equipment, raw material variation) may shift the true optimum.

### How many confirmation runs to run

Run **3 to 5 experiments at the predicted optimum settings**. This gives you:

- A mean response to compare against the predicted value.
- A standard deviation to compare against the model's RMSE.
- Enough replicates to run a one-sample t-test if needed.

### Interpreting confirmation results

Compare your confirmation run mean against the model's 95% prediction interval.

| Outcome | Interpretation | Action |
|---|---|---|
| Confirmation mean is inside the 95% PI | Model is validated | Implement the new settings with confidence |
| Confirmation mean is outside the 95% PI but close | Model is slightly off | Run additional confirmation runs; investigate lurking variables |
| Confirmation mean is outside the 95% PI by a wide margin | Model is not predictive | Return to Analysis tab; check for missing terms, outliers, or domain-specific constraints |
| Confirmation runs show high variability (large standard deviation) | Process is noisy at these settings | Investigate noise sources; consider a nested design to separate process and measurement variability |

### Documenting the confirmation

Include your confirmation results in the final HTML report by adding them as a note in the Analysis tab data table (you can add a "Confirmation" value in the Point Type column). This creates a traceable record of validation.

---

## Save Report as HTML

The Prediction tab has its own **Save Report as HTML** button. The report generated from the Prediction tab includes:

- The coefficient table and regression equations (coded + actual)
- The response surface plot (the current view — 3D or contour, with current axis and slider settings)
- The optimisation result (if optimisation has been run)
- The prediction interval for the optimum
- Desirability scores for multi-response optimisation

If you save from the Prediction tab after having already generated AI interpretations on the Analysis tab, those interpretations are included in the Prediction tab report as well.

> **Tip:** For the most complete report, save from the Prediction tab after clicking all four Interpret buttons on the Analysis tab. This gives you a single HTML file with the full analysis and the optimisation results.

---

## Cross-References

- [Analysis Tab](Analysis-Tab) — fitting the model before using the Prediction tab
- [AI Interpretation Guide](AI-Interpretation-Guide) — when the AI recommends moving to Prediction & Optimisation
- [Troubleshooting](Troubleshooting) — if the response surface does not render or the optimiser returns unexpected results
