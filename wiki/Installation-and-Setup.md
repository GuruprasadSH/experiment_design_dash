# Installation and Setup

This page walks you through every step needed to get DOE Assistant v2 running on your machine, from cloning the repository to verifying that all features work correctly.

---

## Prerequisites

Before you start, confirm that the following are available on your system.

| Requirement | Minimum version | How to check |
|---|---|---|
| Python | 3.10 | `python --version` |
| pip | 22.0 | `pip --version` |
| Git | any recent | `git --version` |
| Anthropic API key | — | [console.anthropic.com](https://console.anthropic.com) |
| Internet access (at runtime) | — | Claude API calls go outbound |

> **Windows note:** The instructions below use POSIX shell syntax. On Windows, replace `export` with `set` (Command Prompt) or `$env:VAR = "value"` (PowerShell). Using WSL2 with Ubuntu is the path of least resistance on Windows.

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/doe-assistant.git
cd doe-assistant
```

If you downloaded a ZIP instead of cloning, extract it and `cd` into the extracted folder.

---

## Step 2 — Create a virtual environment (strongly recommended)

Running in a virtual environment isolates the application's dependencies from your system Python and prevents version conflicts.

```bash
# Create the environment
python -m venv .venv

# Activate it (macOS / Linux)
source .venv/bin/activate

# Activate it (Windows Command Prompt)
.venv\Scripts\activate.bat

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

Your shell prompt should now show `(.venv)` at the start. Every `pip install` and `python` command you run from this point forward affects only this environment.

---

## Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file pins the following key packages:

| Package | Role |
|---|---|
| `dash` | Web framework |
| `dash-bootstrap-components` | Bootstrap-themed Dash components |
| `plotly` | Interactive charts |
| `pandas` | Data manipulation |
| `numpy` | Numerical arrays |
| `statsmodels` | OLS regression, ANOVA |
| `pyDOE3` | DOE matrix generation |
| `scipy` | Numerical optimisation |
| `anthropic` | Claude API client |
| `kaleido` | Static image export (PNG for AI vision) |
| `python-dotenv` | Loads `.env` file at startup |
| `openpyxl` | Excel file upload support |

Installation typically takes two to four minutes. If you see a network error, check your proxy settings or try `pip install --no-cache-dir -r requirements.txt`.

---

## Step 4 — Configure your Anthropic API key

DOE Assistant v2 reads the API key from an environment variable called `ANTHROPIC_API_KEY`. The recommended approach is a `.env` file in the project root.

```bash
# Create the file (macOS / Linux)
echo "ANTHROPIC_API_KEY=sk-ant-api03-REPLACE-WITH-YOUR-KEY" > .env
```

On Windows:

```powershell
"ANTHROPIC_API_KEY=sk-ant-api03-REPLACE-WITH-YOUR-KEY" | Out-File -Encoding utf8 .env
```

The `.env` file is listed in `.gitignore` — it will never be committed. Do not share it or commit it manually.

### Obtaining an API key

1. Go to [console.anthropic.com](https://console.anthropic.com).
2. Sign in or create an account.
3. Navigate to **API Keys** → **Create Key**.
4. Copy the key immediately — it is shown only once.
5. Paste it into your `.env` file.

### Verifying the key is loaded

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('ANTHROPIC_API_KEY', 'NOT SET')[:20])"
```

You should see the first 20 characters of your key (e.g., `sk-ant-api03-ABCDE…`). If you see `NOT SET`, check that the `.env` file is in the project root and that `python-dotenv` is installed.

---

## Step 5 — Run the application

```bash
python app.py
```

You should see output similar to:

```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app'
 * Debug mode: off
```

Open `http://127.0.0.1:8050` in your browser. The application should load within two to three seconds.

### Running in development (debug) mode

Debug mode enables hot-reload — the server restarts automatically when you save a Python file.

```bash
python app.py --debug
```

Or set the flag directly in `app.py`:

```python
if __name__ == "__main__":
    app.run(debug=True, port=8050)
```

> **Warning:** Never expose debug mode to a network. Use it only on `localhost`.

---

## Step 6 — Verify that all features work

Work through this checklist after the first launch.

### Design tab

- [ ] Open the app in your browser.
- [ ] Select **Central Composite Design** from the design type dropdown.
- [ ] Set k = 2 factors, enter factor names and ranges.
- [ ] Click **Generate Design**. A matrix table should appear with columns for Std Order, Run Order, Point Type, and your factors.
- [ ] Expand the **AI Design Assistant** panel at the bottom. Type a brief description of your experiment and click **Send**. You should receive a response within 10 seconds.

### Analysis tab

- [ ] Switch to the **Analysis** tab.
- [ ] Click **Transfer from Design Tab**. The factor columns from your generated design should populate the data entry table.
- [ ] Type some fictional response values into the Response column.
- [ ] Select the response column and all factor columns, then click **Fit Model**.
- [ ] The ANOVA accordion section should expand with a table.
- [ ] Click **Interpret** in the **ANOVA & Model Stats** section. An AI card should appear with a ✅/⚠️/❌ verdict.

### Prediction tab

- [ ] Switch to the **Prediction & Optimisation** tab.
- [ ] A coefficient table and regression equations should be visible (if a model was fitted in the Analysis tab).
- [ ] Select Factor X and Factor Y, then click **Plot Surface**. A 3D or contour plot should render.
- [ ] Click **Run Optimisation** with **Maximise** selected. A predicted optimum panel should appear.

### Image export (kaleido)

- [ ] In the Analysis tab with a fitted model, click **Interpret** in the **Effects & Interaction Analysis** section.
- [ ] If the Pareto chart and half-normal plot render correctly and an AI card appears, kaleido is working.

If any step fails, see the Troubleshooting section below.

---

## Troubleshooting

### Port 8050 is already in use

**Symptom:** `OSError: [Errno 48] Address already in use` when starting `app.py`.

**Fix:** Find and kill the existing process, or use a different port.

```bash
# macOS / Linux — find what is using port 8050
lsof -i :8050

# Kill by PID (replace 12345 with the actual PID)
kill -9 12345

# Or start on a different port
python app.py --port 8051
```

On Windows:

```powershell
netstat -ano | findstr :8050
# Note the PID in the last column, then:
taskkill /PID <PID> /F
```

---

### kaleido not found or image export fails

**Symptom:** The **Effects** or **Residuals** AI Interpret buttons produce an error, or a red alert appears saying image export failed.

**Cause:** kaleido is either not installed, or the installed version conflicts with the Plotly version.

**Fix:**

```bash
pip uninstall kaleido -y
pip install kaleido==0.2.1
```

Then restart `app.py`. DOE Assistant v2 uses a dual-path approach (`fig_to_b64`) that handles both kaleido 0.1.x and 0.2.x, but the library must be installed. See the [Developer Guide](Developer-Guide) for implementation details.

If kaleido installation itself fails (common on Apple Silicon Macs):

```bash
pip install kaleido --pre
```

---

### API key errors

**Symptom:** Clicking any **Interpret** button or sending a message to the AI Design Assistant produces a red error card saying `AuthenticationError` or `Invalid API Key`.

**Steps to resolve:**

1. Confirm the `.env` file exists in the project root: `ls -la .env`
2. Confirm the key starts with `sk-ant-`: `cat .env`
3. Confirm `python-dotenv` is installed: `pip show python-dotenv`
4. Confirm the key is valid by testing it directly:

```python
import anthropic, os
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic()
msg = client.messages.create(model="claude-haiku-4-5", max_tokens=10, messages=[{"role":"user","content":"hi"}])
print(msg.content)
```

If that raises `AuthenticationError`, your key is invalid or has been revoked — generate a new one at [console.anthropic.com](https://console.anthropic.com).

---

### API rate limits or timeout

**Symptom:** AI cards appear after a long pause, then show `RateLimitError` or `APITimeoutError`.

**Fix:**

- Rate limits: Add credits to your Anthropic account or wait for your rate limit window to reset (usually one minute).
- Timeouts: The default timeout in `anthropic` is 600 seconds, so a timeout usually means the API is under heavy load. Retry after a moment.
- If you are generating AI interpretations for all four sections in rapid succession, space them out by a few seconds.

---

### Missing openpyxl — Excel upload does not work

**Symptom:** Uploading an `.xlsx` file in the Analysis tab produces a `ModuleNotFoundError: No module named 'openpyxl'`.

**Fix:**

```bash
pip install openpyxl
```

---

### pandas version conflict

**Symptom:** Import errors at startup mentioning `DataFrame.append` or `FutureWarning` becoming `AttributeError`.

**Fix:** Ensure you are on pandas ≥ 2.0. The `requirements.txt` pins a compatible version.

```bash
pip install "pandas>=2.0" --upgrade
```

---

### Browser shows blank white page

**Symptom:** The page loads but is completely blank (no tabs, no layout).

**Steps:**

1. Open your browser's Developer Tools (F12) and check the Console tab for JavaScript errors.
2. If you see `CDN resource failed to load`, check your internet connection — Dash loads Bootstrap CSS from a CDN.
3. Try a hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (macOS).
4. Try a different browser — Dash works best in Chrome and Firefox.

---

## Updating the application

To pull the latest changes:

```bash
git pull origin main
pip install -r requirements.txt   # in case new dependencies were added
python app.py
```

---

## Next steps

With the application running, head to the [Design Types](Design-Types) page to understand which experimental design fits your situation, or go directly to the [Design Tab](Design-Tab) guide to start generating your first design matrix.
