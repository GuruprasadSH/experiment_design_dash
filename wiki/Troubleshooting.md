# Troubleshooting

This page is organised by symptom. Find the problem you are experiencing, follow the steps, and check the "Still not resolved?" guidance at the end of each section.

---

## kaleido image export fails

**Symptom:** Clicking **Interpret** in the Effects & Interaction Analysis or Residual Analysis section produces a red error card with text like `kaleido subprocess failed`, `kaleido returned a non-zero exit code`, or `kaleido not found`. The Pareto chart and other plots render correctly in the browser, but the AI interpretation cannot be generated.

**Why it happens:** The AI vision interpretation requires exporting Plotly charts as PNG images. DOE Assistant v2 uses the `fig_to_b64` utility function (in `utils/image_export.py`) to try both the new kaleido 0.2.x API and the old 0.1.x API. If kaleido is not installed or is in a broken state, both paths fail.

**Fix — Step 1:** Check whether kaleido is installed at all.

```bash
pip show kaleido
```

If the command returns `WARNING: Package(s) not found`, install it:

```bash
pip install kaleido
```

**Fix — Step 2:** If kaleido is installed but export still fails, try a specific version that is known to work.

```bash
pip uninstall kaleido -y
pip install "kaleido==0.2.1"
```

Restart `app.py` after reinstalling.

**Fix — Step 3 (Apple Silicon / M1/M2 Mac):** The pre-built kaleido wheel may not work on ARM architecture. Try the pre-release wheel:

```bash
pip uninstall kaleido -y
pip install kaleido --pre
```

**Fix — Step 4:** Verify that the kaleido binary is executable:

```bash
python -c "import kaleido; print(kaleido.__version__)"
python -c "
import plotly.graph_objects as go
fig = go.Figure(go.Scatter(x=[1,2], y=[1,2]))
fig.write_image('/tmp/test_kaleido.png')
print('kaleido OK')
"
```

If the second command raises an error, the binary is not working. Uninstall and reinstall as above.

**Still not resolved?** Open a GitHub issue with the output of `pip show kaleido`, your OS version (`uname -a` or `winver`), and the full traceback from the Dash server console.

---

## AI Interpret button stays disabled

**Symptom:** The **Interpret** button in one or more accordion sections is greyed out and cannot be clicked.

**Why it happens:** The Interpret buttons are disabled until a model has been successfully fitted. This is enforced by the callback that checks `model_store` in the Dash state.

**Fix — Step 1:** Make sure you have clicked **Fit Model** on the Analysis tab and that it completed without errors. Look for a green success badge or the accordion sections becoming active.

**Fix — Step 2:** Check that all cells in the response column contain numeric values. A single empty cell or a cell with text (like "N/A" or "-") will prevent the model from fitting.

**Fix — Step 3:** If you transferred data from the Design tab but the response column is still empty (you have not entered experimental results yet), the Fit Model button will fail silently because the response has no variance. Enter at least some response values before fitting.

**Fix — Step 4:** Open the browser developer tools (F12) and check the Console tab for JavaScript errors. A Dash callback error will appear there. Copy the error and search for it in the GitHub issues.

**Still not resolved?** Check the Dash server console (the terminal where `app.py` is running) for a Python traceback. The error there is usually more informative than the browser console.

---

## Paste from Excel adds extra blank rows

**Symptom:** After clicking **Paste from Excel**, the data table has one or more empty rows at the bottom. When you try to fit the model, it fails because the empty rows have no response values.

**Why it happens:** This was a known bug in v1 caused by trailing newlines in the clipboard content. It is fixed in v2.

**Fix (v2):** This should not happen in v2. If it does, it means you are running an older commit. Update:

```bash
git pull origin main
pip install -r requirements.txt
```

**Workaround (if on older build):** After pasting, click the empty row in the table, then click **Delete Row** (the trash-can icon in the table toolbar). Alternatively, add a filter to the table and filter out blank rows before fitting.

---

## Lack of Fit is not shown in the ANOVA table

**Symptom:** The ANOVA table has a Residual row but no separate Lack of Fit and Pure Error rows. You cannot see the LOF p-value.

**Why it happens:** The LOF test requires pure error degrees of freedom. Pure error df comes from replicated observations — runs that are identical in all factor settings but differ in response value. The sources of replication are:

- Center-point replicates (runs where all factors = midpoint)
- Whole-design replicates (two or more complete copies of the design)
- Genuine replicates in an uploaded CSV (two rows with identical factor settings)

If none of these are present, pure error df = 0, and the LOF test cannot be computed.

**Fix:** Add center points (3–5 recommended) when generating your design on the Design tab. If you have already run the experiment and have no replicates, you cannot retrospectively compute a valid LOF test from a single-replicate design.

**Alternative:** If you have center points but the LOF row is still missing, check that the Point Type column is present and has the value `Center` (not `centre`, `cp`, or any other variant) for those rows. The app uses the Point Type column to identify center points.

---

## Interaction plot shows only points, not connecting lines

**Symptom:** The interaction plot in section 3 (Effects & Interaction Analysis) shows scattered data points but no lines connecting the points within each interaction panel.

**Why it happens:** This was a bug in v1 for RSM designs (CCD, Box-Behnken) where the combination of factor levels at axial points (values like ±1.414) does not cleanly pair with the factorial level values (±1) for plotting interaction lines.

**Fix:** This is fixed in v2 using `connectgaps=True` in the Plotly trace configuration. Update to the latest version:

```bash
git pull origin main
```

If you are on v2 and the bug persists, check whether your data has more than two values per factor (axial points). The interaction plot in the app is designed for the two-level factor values only (Low and High) and ignores axial points for the interaction panel. If the data contains only axial and center points (no factorial points), the interaction plot cannot be drawn and will show a message.

---

## Pred R² is much lower than Adj R²

**Symptom:** In the ANOVA & Model Stats section, Adj R² is 0.92 but Pred R² is 0.61. The AI Interpret button gives a ⚠️ verdict.

**Why it happens:** This gap indicates overfitting — the model fits the observed data well (high Adj R²) but does not generalise to new observations (low Pred R²). Overfitting occurs when the model has too many terms relative to the number of runs.

**Fix — Step 1:** Remove non-significant terms. On the Analysis tab, uncheck any terms in the model term checklist that have p > 0.10 in the ANOVA table. Re-fit the model.

**Fix — Step 2:** If removing terms is not sufficient, consider whether the design has enough runs to support the model. The rule of thumb is: number of terms ≤ number of runs / 4. A 13-run CCD with a full quadratic model for 3 factors has 10 terms — that is borderline (runs / terms ≈ 1.3). Consider adding center points or augmenting the design.

**Fix — Step 3:** If the gap persists after removing non-significant terms, check for outliers. A single influential observation (high Cook's distance) can inflate Adj R² while dramatically lowering Pred R². Look at the Residuals vs Fitted plot in section 4 for points far from zero.

---

## Shapiro-Wilk rejects normality (p < 0.05)

**Symptom:** In section 4 (Residual Analysis), the Shapiro-Wilk badge shows a red colour and p < 0.05. The AI Interpret button gives a ⚠️ or ❌ verdict mentioning non-normality.

**Why it happens:** The residuals do not follow a normal distribution. This can be caused by: a non-linear response that the model does not capture (missing quadratic terms), non-constant variance (heteroscedasticity), outliers, or a response variable that is inherently non-normal (counts, proportions, times-to-event).

**Fix — Option 1: Transform the response.** Apply a log or square-root transformation to the response column before fitting. In the Analysis tab editable table, add a new column (e.g., `log_Yield`) and enter the transformed values manually. Then select the new column as your response and re-fit.

**Fix — Option 2: Investigate outliers.** If one or two points have very large residuals (> 3 × RMSE), check whether those runs had data entry errors or unusual conditions. If so, correct or exclude those runs and re-fit.

**Fix — Option 3: Add missing terms.** A U-shape or S-curve in the Q-Q plot often indicates missing quadratic terms. On the Analysis tab, click **+Quadratic** to add quadratic terms to the model and re-fit.

**Fix — Option 4: Accept the result if the deviation is mild.** The Shapiro-Wilk test has high power for large sample sizes — it will reject normality for minor, practically inconsequential deviations. If your Q-Q plot looks approximately linear (only slight deviations at the tails), the violation may not materially affect your conclusions. Consult a statistician if uncertain.

**Still rejected after transformation?** Consider whether a generalised linear model (GLM) with a non-normal error distribution would be more appropriate. The current version of DOE Assistant v2 uses only OLS; GLM support is planned for a future release.

---

## API rate limit or timeout errors

**Symptom:** After clicking **Interpret** or sending a message to the AI Design Assistant, a red error card appears after a long wait with text like `RateLimitError`, `APITimeoutError`, or `Error code: 429`.

**Why it happens:**

- **Rate limit (429):** Your Anthropic account has exceeded its tier limit for requests per minute or tokens per minute. This is common on free or low-credit accounts.
- **Timeout:** The API call took longer than the timeout configured in the Anthropic client (default: 600 seconds). This is unusual and typically indicates API-side load.

**Fix — Rate limit:**

1. Wait one minute and try again. Rate limit windows typically reset every 60 seconds.
2. If you consistently hit the limit, upgrade your Anthropic usage tier at [console.anthropic.com](https://console.anthropic.com).
3. Avoid clicking all four Interpret buttons in rapid succession. Space them out by a few seconds.

**Fix — Timeout:**

1. Try again — transient timeouts usually resolve on retry.
2. Check [status.anthropic.com](https://status.anthropic.com) for any ongoing API incidents.
3. If timeouts are persistent, reduce the `max_tokens` value in the app's API calls. See `ai/claude_client.py` in the source code. Reducing from 1024 to 512 tokens speeds up responses at the cost of shorter output.

---

## Port 8050 is already in use

**Symptom:** `app.py` fails to start with `OSError: [Errno 48] Address already in use` (macOS/Linux) or `WinError 10048` (Windows).

**Fix (macOS/Linux):**

```bash
# Find the process using port 8050
lsof -ti :8050

# Kill it (replace PID with the actual number shown)
kill -9 $(lsof -ti :8050)

# Or start on a different port
python app.py --port 8051
```

**Fix (Windows):**

```powershell
netstat -ano | findstr :8050
# Note the PID in the rightmost column
taskkill /PID <PID> /F
```

**Alternative:** Change the default port in `app.py`:

```python
if __name__ == "__main__":
    app.run(debug=False, port=8051)  # Change 8050 to any free port
```

---

## The app loads but all tabs are blank

**Symptom:** The browser shows the app navigation header with the three tabs, but the tab content area is blank white.

**Fix — Step 1:** Open Developer Tools (F12) → Console. Look for JavaScript errors. A common cause is a CDN resource failing to load (Bootstrap CSS or Plotly JS).

**Fix — Step 2:** If you see CDN errors, try turning off any browser extensions that block ads or trackers (uBlock Origin, Privacy Badger). These sometimes block CDN resources.

**Fix — Step 3:** Hard-refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows/Linux).

**Fix — Step 4:** Try a different browser. Dash is tested on Chrome and Firefox.

**Fix — Step 5:** Check the Dash server console for Python errors. A crash in a layout callback will cause the layout to not render. Look for a traceback and search for it in the GitHub issues.

---

## Design matrix generates zero rows

**Symptom:** After clicking **Generate Design**, the table appears but has zero rows, or the "Estimated runs" counter shows 0.

**Why it happens:** This can occur with the Simplex Lattice design if the component lower bounds sum to more than 1, making the feasible simplex empty.

**Fix:** Lower the component lower bounds in the factor table so they sum to less than 1. For example, if you have 3 components with lower bounds 0.4, 0.4, 0.4, their sum is 1.2 > 1, which is infeasible. Reduce at least one bound.

For other design types, this should not occur. If it does, file a bug report with your factor settings.

---

## Cannot upload Excel file — "unsupported format" error

**Symptom:** Clicking **Upload CSV/Excel** and selecting an `.xlsx` file produces an error saying the format is not supported.

**Fix:** Check that `openpyxl` is installed:

```bash
pip install openpyxl
pip show openpyxl
```

If you are uploading an `.xls` (old Excel 97-2003 format) rather than `.xlsx`, the parser may not support it. Resave the file in Excel as `.xlsx` (File → Save As → Excel Workbook .xlsx).

---

## Cross-References

- [Installation and Setup](Installation-and-Setup) — dependency installation and API key configuration
- [Analysis Tab](Analysis-Tab) — how to use the Fit Model button and accordion sections
- [AI Interpretation Guide](AI-Interpretation-Guide) — how AI sections work and what errors mean
- [Developer Guide](Developer-Guide) — `fig_to_b64` kaleido dual-path implementation details
