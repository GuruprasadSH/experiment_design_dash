# Developer Guide

This page is for contributors and developers who want to understand the codebase, add new design types or AI sections, run the test suite, or follow the project's code conventions.

---

## Architecture Overview

DOE Assistant v2 is a single-process Python application. Plotly Dash handles the web server, routing, and reactive callback system. All computation (DOE matrix generation, model fitting, optimisation, AI calls) happens in Python on the server side.

```
Browser (HTML/JS/CSS)
      ↕  HTTP (WebSocket via Socket.IO)
Dash server (Flask + Dash)
      ├── Layout (app.py, layouts/)
      ├── Callbacks (callbacks/)
      │     ├── callbacks_A_design.py
      │     ├── callbacks_B_analysis.py
      │     ├── callbacks_C_prediction.py
      │     └── callbacks_D_shared.py
      ├── DOE generators (doe_generators.py)
      ├── Statistical engine (stats_engine.py)
      ├── AI client (ai/claude_client.py)
      ├── Image export (utils/image_export.py)
      └── Report builder (utils/report_builder.py)
```

---

## Key Files and Their Responsibilities

```
doe-assistant/
├── app.py                      Main entry point; creates Dash app, registers callbacks
├── layouts/
│   ├── design_layout.py        All Dash components for the Design tab
│   ├── analysis_layout.py      All Dash components for the Analysis tab
│   └── prediction_layout.py   All Dash components for the Prediction tab
├── callbacks/
│   ├── callbacks_A_design.py   Design tab callbacks (A-prefix)
│   ├── callbacks_B_analysis.py Analysis tab callbacks (B-prefix)
│   ├── callbacks_C_prediction.py Prediction tab callbacks (C-prefix)
│   └── callbacks_D_shared.py   Cross-tab callbacks (help panel, store updates)
├── doe_generators.py           DOE matrix generation for all 9 design types
├── stats_engine.py             Model fitting, ANOVA, R² stats, residuals, optimisation
├── ai/
│   ├── claude_client.py        Anthropic API wrapper; builds prompts for each section
│   └── prompts.py              Prompt templates and NIST/Montgomery rule strings
├── utils/
│   ├── image_export.py         fig_to_b64 dual-path kaleido image export
│   └── report_builder.py       HTML report generation
├── assets/
│   ├── custom.css              App-wide custom CSS
│   └── favicon.ico
├── tests/
│   ├── test_doe_generators.py
│   ├── test_stats_engine.py
│   └── test_image_export.py
├── requirements.txt
└── .env                        Not committed; contains ANTHROPIC_API_KEY
```

---

## Callback Naming Convention (A/B/C/D)

Callbacks are split across four files with a letter prefix that indicates which tab or scope the callback belongs to:

| Prefix | File | Scope |
|---|---|---|
| A | `callbacks_A_design.py` | Design tab only |
| B | `callbacks_B_analysis.py` | Analysis tab only |
| C | `callbacks_C_prediction.py` | Prediction tab only |
| D | `callbacks_D_shared.py` | Cross-tab (help panel, shared stores) |

Within each file, callback functions follow the naming pattern `cb_<letter>_<number>_<description>`:

```python
# Example from callbacks_B_analysis.py
@app.callback(...)
def cb_b_01_fit_model(n_clicks, response, factors, terms, data):
    ...

@app.callback(...)
def cb_b_02_interpret_design_summary(n_clicks, model_store):
    ...
```

The number (01, 02, ...) indicates the approximate order in which the callbacks would fire during a typical user session. This is a readability convention — Dash does not enforce execution order.

### Shared Dash stores

Cross-tab data is passed using `dcc.Store` components registered in `app.py`. The most important stores are:

| Store ID | Type | Contents |
|---|---|---|
| `design-store` | `memory` | Generated design matrix as a JSON-serialised DataFrame |
| `model-store` | `memory` | Fitted model parameters (coefficients, ANOVA, R² stats) as a dict |
| `factor-meta-store` | `memory` | Factor names, types, low/high levels, units |
| `ai-results-store` | `session` | AI interpretation text for each section (persists within session) |

Stores are cleared on page refresh. The `session` type (for AI results) persists across tab switches within the same browser tab.

---

## How to Add a New Design Type

Follow this checklist when adding a new DOE design type.

### Step 1 — Add a generator function to `doe_generators.py`

The function must have this signature:

```python
def generate_my_design(factors: list[dict], options: dict) -> pd.DataFrame:
    """
    Generate a my_design design matrix.

    Parameters
    ----------
    factors : list of dicts, each with keys:
        'name'  : str
        'type'  : 'numeric' | 'categorical'
        'low'   : float (numeric factors only)
        'high'  : float (numeric factors only)
        'levels': list[str] (categorical factors only)
        'units' : str
    options : dict with design-specific keys (e.g., 'n_runs', 'replicates')

    Returns
    -------
    pd.DataFrame with columns:
        'Std Order', 'Run Order', 'Point Type', plus one column per factor.
        Point Type values: 'Factorial', 'Axial', 'Center'.
        Factor columns in actual (decoded) units.
    """
    ...
```

The function must:
- Return a DataFrame with at least the three system columns plus factor columns.
- Decode coded values to actual units using factor `low` and `high`.
- Apply randomisation to Run Order if `options.get('randomise', True)` is True.
- Assign Point Type correctly.

### Step 2 — Register the design in the `DESIGNS` dict

In `doe_generators.py`, add an entry to the `DESIGNS` dictionary:

```python
DESIGNS = {
    ...
    'my_design': {
        'label': 'My New Design',        # Shown in the dropdown
        'generator': generate_my_design,  # Function reference
        'min_factors': 2,
        'max_factors': 10,
        'sub_types': [],                  # List of sub-type dicts if applicable
        'options': ['replicates', 'randomise'],  # Which option controls to show
        'description': 'One-line description for the AI Design Assistant.',
    },
    ...
}
```

### Step 3 — Add UI options in `layouts/design_layout.py`

If your design needs additional controls (e.g., a degree selector, an array size selector), add them to the `design_options_panel` function. Follow the pattern of existing option controls:

```python
# Example: add a degree dropdown (like Simplex Lattice uses)
dbc.Row([
    dbc.Col([
        html.Label("Degree"),
        dcc.Dropdown(
            id='my-design-degree-dropdown',
            options=[{'label': str(i), 'value': i} for i in [1, 2, 3]],
            value=2,
            clearable=False,
        ),
    ], width=3),
], id='my-design-options-row', style={'display': 'none'}),
```

Then add a callback in `callbacks_A_design.py` that shows/hides this row based on the selected design type.

### Step 4 — Update the design-generation callback

In `callbacks_A_design.py`, update `cb_a_02_generate_design` to pass the new option values to your generator function.

### Step 5 — Update the AI Design Assistant prompts

In `ai/prompts.py`, add your new design to the `DESIGN_DESCRIPTIONS` dict so the AI Design Assistant can recommend it:

```python
DESIGN_DESCRIPTIONS = {
    ...
    'my_design': (
        "My New Design: suitable for [use case]. "
        "Requires [min_factors] to [max_factors] factors. "
        "Run count: [formula]. "
        "Key trade-off: [trade-off]."
    ),
    ...
}
```

### Step 6 — Add tests

Add a test function to `tests/test_doe_generators.py`:

```python
def test_my_design_basic():
    factors = [
        {'name': 'A', 'type': 'numeric', 'low': 10, 'high': 50, 'units': ''},
        {'name': 'B', 'type': 'numeric', 'low': 1, 'high': 5, 'units': ''},
    ]
    options = {'replicates': 1, 'randomise': False}
    df = generate_my_design(factors, options)
    
    assert 'Std Order' in df.columns
    assert 'Run Order' in df.columns
    assert 'Point Type' in df.columns
    assert 'A' in df.columns
    assert 'B' in df.columns
    assert df['A'].between(10, 50).all()  # Values within factor range
    assert df['B'].between(1, 5).all()
```

---

## How to Add a New AI Interpretation Section

If you add a new accordion section to the Analysis tab and want it to have an AI Interpret button, follow these steps.

### Step 1 — Add the section to the accordion layout

In `layouts/analysis_layout.py`, add a new `dbc.AccordionItem` with an Interpret button and an output container:

```python
dbc.AccordionItem(
    [
        # Your section content here
        html.Div(id='my-section-content'),
        # AI interpret controls
        dbc.Button(
            "Interpret",
            id='interpret-my-section-btn',
            color='primary',
            size='sm',
            disabled=True,
            className='mt-2',
        ),
        dbc.Spinner(html.Div(id='my-section-ai-output'), size='sm'),
    ],
    title="5. My New Section",
    item_id='my-section',
),
```

### Step 2 — Add the Interpret callback in `callbacks_B_analysis.py`

```python
@app.callback(
    Output('my-section-ai-output', 'children'),
    Input('interpret-my-section-btn', 'n_clicks'),
    State('model-store', 'data'),
    prevent_initial_call=True,
)
def cb_b_07_interpret_my_section(n_clicks, model_store):
    if not n_clicks or not model_store:
        raise PreventUpdate
    
    # Build the data payload for the AI
    data_for_ai = build_my_section_data(model_store)
    
    # Decide which model to use
    # Use haiku for text-only; sonnet for vision (images)
    use_vision = False  # Set True if passing images
    
    ai_text = claude_client.interpret_section(
        section='my_section',
        data=data_for_ai,
        use_vision=use_vision,
    )
    
    return build_ai_card(ai_text)
```

### Step 3 — Add the prompt template to `ai/prompts.py`

```python
MY_SECTION_PROMPT = """
You are a DOE statistics expert. Apply NIST/Montgomery rules.
Output format: [✅/⚠️/❌] one-line verdict, then 3-5 bullets with specific numbers,
then "Next step: " one sentence.

Evaluate this [description of section]:
{data}
"""
```

### Step 4 — Register the prompt in `ai/claude_client.py`

Add a branch for `section='my_section'` in the `interpret_section` method of the `ClaudeClient` class.

### Step 5 — Enable the Interpret button after model fitting

In `cb_b_01_fit_model`, add `'interpret-my-section-btn'` to the list of component IDs whose `disabled` property is set to `False` on successful model fit.

---

## The `fig_to_b64` Dual-Path Kaleido Approach

Kaleido changed its Python API between version 0.1.x and 0.2.x. The 0.1.x API uses `plotly.io.to_image`; the 0.2.x API uses `kaleido.scopes.plotly.PlotlyScope` directly. DOE Assistant v2 supports both versions with a try/except dual-path:

```python
# utils/image_export.py

import base64
import io

def fig_to_b64(fig) -> str:
    """
    Export a Plotly figure to a base64-encoded PNG string.
    Supports kaleido 0.1.x and 0.2.x.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure

    Returns
    -------
    str
        Base64-encoded PNG, suitable for use in an <img src="data:image/png;base64,..."> tag
        or for passing to the Anthropic vision API.

    Raises
    ------
    RuntimeError
        If both kaleido paths fail.
    """
    # Path 1: kaleido 0.2.x
    try:
        from kaleido.scopes.plotly import PlotlyScope
        scope = PlotlyScope()
        img_bytes = scope.transform(fig.to_json(), format='png', width=900, height=600, scale=2)
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception:
        pass

    # Path 2: kaleido 0.1.x via plotly.io
    try:
        import plotly.io as pio
        img_bytes = pio.to_image(fig, format='png', width=900, height=600, scale=2)
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception:
        pass

    raise RuntimeError(
        "kaleido image export failed on both 0.1.x and 0.2.x paths. "
        "See the Troubleshooting wiki page for fixes."
    )
```

The `scale=2` parameter doubles the pixel density, producing clearer images for the vision model. Width 900 × height 600 at scale 2 gives a 1800×1200 pixel PNG, which is within Anthropic's supported image dimensions.

---

## Running the Test Suite

```bash
# Install test dependencies (if not already in requirements.txt)
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_doe_generators.py -v

# Run a specific test
pytest tests/test_doe_generators.py::test_ccd_ccc_basic -v
```

Tests do not require an Anthropic API key — AI calls are mocked using `unittest.mock.patch` in the test fixtures.

### Test fixtures

The `tests/conftest.py` file provides shared fixtures:

```python
@pytest.fixture
def two_factor_numeric():
    return [
        {'name': 'Temperature', 'type': 'numeric', 'low': 80, 'high': 120, 'units': 'C'},
        {'name': 'Pressure',    'type': 'numeric', 'low': 1,  'high': 5,   'units': 'bar'},
    ]

@pytest.fixture
def mock_anthropic(monkeypatch):
    """Patch the Anthropic client to avoid real API calls in tests."""
    # ... mock implementation
```

---

## Code Style Conventions

DOE Assistant v2 follows these conventions. New code should match them; pull requests that diverge will be asked to conform before merging.

### Formatting

- **Black-compatible formatting.** Run `black .` before committing. The CI pipeline checks Black formatting.
- **Line length:** 100 characters maximum (Black default is 88 — we override to 100 in `pyproject.toml`).
- **Imports:** sorted with `isort --profile black`.

### Type hints

- All new **public functions** must have type hints on parameters and return values.
- Internal (leading-underscore) helper functions should have type hints where it adds clarity.
- Use `from __future__ import annotations` at the top of files that use PEP 604 union syntax (`X | Y`) for Python 3.9 compatibility.

```python
# Good
def generate_ccd(factors: list[dict], options: dict) -> pd.DataFrame:
    ...

# Acceptable for simple internals
def _clip_to_range(value, lo, hi):
    return max(lo, min(hi, value))
```

### Dash component IDs

- Component IDs use kebab-case: `'design-type-dropdown'`, `'fit-model-btn'`.
- Interpret buttons follow the pattern `'interpret-<section-name>-btn'`.
- AI output containers follow the pattern `'<section-name>-ai-output'`.
- Stores follow the pattern `'<name>-store'`.

### Callback structure

- Each callback function does one thing. Do not combine design generation and factor table rendering into a single callback.
- Callbacks that call the Anthropic API must catch `anthropic.APIError` and return a user-friendly error card (use the `build_error_card(message)` utility in `utils/components.py`).
- Use `prevent_initial_call=True` on all callbacks that respond to button clicks.

### Adding docstrings

Public functions in `doe_generators.py` and `stats_engine.py` use NumPy-style docstrings:

```python
def fit_ols_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Fit an OLS regression model and return all statistics needed by the app.

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix in coded units, with an 'Intercept' column.
    y : pd.Series
        Response variable.

    Returns
    -------
    dict with keys:
        'coefficients'   : pd.DataFrame (Term, Coef, StdErr, t, p)
        'anova'          : pd.DataFrame (Source, df, SS, MS, F, p)
        'r2'             : float
        'adj_r2'         : float
        'pred_r2'        : float
        'rmse'           : float
        'residuals'      : np.ndarray
        'fitted'         : np.ndarray
        'shapiro_stat'   : float
        'shapiro_p'      : float
    """
```

---

## Adding a Dependency

If your contribution requires a new Python package:

1. Add it to `requirements.txt` with a version specifier: `my_package>=1.2,<2.0`.
2. Add a brief comment explaining why it is needed.
3. Test that `pip install -r requirements.txt` works in a fresh virtual environment.
4. If the package has a C extension or a binary (like kaleido), document any platform-specific installation caveats in the [Installation and Setup](Installation-and-Setup) wiki page.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key. Set in `.env` or the shell environment. |
| `DOE_DEBUG` | No | Set to `1` to enable verbose logging of AI prompt/response pairs to the console. |
| `DOE_PORT` | No | Override the default port (8050). Useful in containerised deployments. |

---

## Deployment Notes

DOE Assistant v2 is designed for local use. For multi-user deployment:

- Use a production WSGI server (`gunicorn`): `gunicorn app:server -b 0.0.0.0:8050`.
- The Dash app object exposes a Flask `server` attribute — use `app.server` as the WSGI entry point.
- Each user's data (design matrix, model, AI results) is stored in `dcc.Store` components in the browser, not on the server. This means the app is stateless on the server side — safe for multi-user deployment.
- The Anthropic API key is a server-side secret. Never expose it to the client.
- For HTTPS, terminate TLS at a reverse proxy (nginx, Caddy) in front of gunicorn.

---

## Cross-References

- [Installation and Setup](Installation-and-Setup) — setting up the development environment
- [Design Types](Design-Types) — understanding the nine design families before adding a new one
- [Troubleshooting](Troubleshooting) — kaleido issues and the `fig_to_b64` function in context
