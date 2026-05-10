"""
DOE Statistical Analysis Engine

Reference:
  Montgomery, D.C. (2017). Design and Analysis of Experiments, 9th ed. Wiley.
  Lenth, R.V. (1989). Quick and easy analysis of unreplicated factorials.
      Technometrics, 31(4), 469-473.
  Daniel, C. (1959). Use of half-normal plots in interpreting factorial two-level
      experiments. Technometrics, 1(4), 311-341.
"""

import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Columns that come from the design structure, not factors
ADMIN_COLS = {"Std Order", "Run Order", "Block", "Replicate", "Run", "Point Type"}

ACCENT  = "#2C7BE5"
C_GRAY  = "#adb5bd"
C_RED   = "#e03131"
C_GREEN = "#0ca678"
C_ORG   = "#e67700"

PLOT_TEMPLATE = "plotly_white"
FONT = dict(family="system-ui, -apple-system, sans-serif", size=12)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe(name: str) -> str:
    """Sanitise a column name to a formula-safe identifier."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    return ("x" + s) if s[0].isdigit() else s


def encode_factors(df: pd.DataFrame, factor_cols: list):
    """
    Linearly encode factors to [-1, +1] (coded scale).

    Returns
    -------
    coded_df  : DataFrame with same index as df, columns = factor_cols
    encoding  : dict  col → {low, high, mid, half}
    """
    coded, enc = {}, {}
    for col in factor_cols:
        vals = df[col].dropna()
        lo, hi = float(vals.min()), float(vals.max())
        mid  = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 if (hi - lo) != 0 else 1.0
        coded[col] = (df[col] - mid) / half
        enc[col]   = dict(low=lo, high=hi, mid=mid, half=half)
    return pd.DataFrame(coded, index=df.index), enc


def _decode_term(term: str, rev: dict) -> str:
    """Translate a safe-name formula term back to human-readable text."""
    m = re.match(r"I\((\w+)\*\*2\)", term)
    if m:
        return f"{rev.get(m.group(1), m.group(1))}²"
    parts = term.split(":")
    return " × ".join(rev.get(p.strip(), p.strip()) for p in parts)


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_model(df: pd.DataFrame,
              response_col: str,
              factor_cols: list,
              model_type: str = "main",
              custom_terms: list = None,
              block_col: str | None = None) -> dict:
    """
    Fit an OLS model in coded (-1/+1) factor space.

    Parameters
    ----------
    df           : DataFrame containing design + response columns
    response_col : name of the response column
    factor_cols  : list of factor column names
    model_type   : 'main'  — first-order main-effects only
                   'twfi'  — main effects + all 2-factor interactions
                   'quad'  — full quadratic (main + 2FI + pure quadratic)
    custom_terms : if provided, overrides model_type. Each entry is one of:
                   "A"       — main effect for factor A
                   "A*B"     — 2-way interaction (any arity: A*B*C, etc.)
                   "A^2"     — pure quadratic on A
    block_col    : name of a blocking column; included as C(block_safe) in
                   the formula so the block contribution is separated from
                   the factor residual.

    Returns
    -------
    fit_info dict used by all downstream functions.
    """
    if block_col is not None and block_col not in df.columns:
        raise ValueError(f"block_col '{block_col}' not found in DataFrame.")

    cols_needed = factor_cols + [response_col]
    if block_col is not None:
        cols_needed = cols_needed + [block_col]
    df_fit = df[cols_needed].dropna().copy()
    n, k   = len(df_fit), len(factor_cols)

    if n < k + 2:
        raise ValueError(
            f"Only {n} complete observations — need at least {k + 2} to fit."
        )

    coded_df, encoding = encode_factors(df_fit, factor_cols)

    safe_map = {col: f"x{i}" for i, col in enumerate(factor_cols)}
    rev_map  = {v: k for k, v in safe_map.items()}
    xs       = list(safe_map.values())

    if custom_terms is not None:
        # Validate and translate each term
        terms = []
        for t in custom_terms:
            t = t.strip()
            if "^2" in t:
                # Pure quadratic: "A^2"
                factor_name = t.replace("^2", "").strip()
                if factor_name not in safe_map:
                    raise ValueError(f"Unknown factor '{factor_name}' in term '{t}'")
                x = safe_map[factor_name]
                # Check that this factor has center points or 3+ levels
                coded_vals = coded_df[factor_name].round(8).unique()
                pm1_only = all(v in (-1.0, 1.0) for v in coded_vals)
                if pm1_only:
                    raise ValueError(
                        f"Pure quadratic on '{factor_name}' requires center "
                        "points or a factor with 3+ levels."
                    )
                terms.append(f"I({x}**2)")
            elif "*" in t:
                # Interaction: "A*B" or "A*B*C" etc.
                parts = [p.strip() for p in t.split("*")]
                for p in parts:
                    if p not in safe_map:
                        raise ValueError(f"Unknown factor '{p}' in term '{t}'")
                terms.append(":".join(safe_map[p] for p in parts))
            else:
                # Main effect
                if t not in safe_map:
                    raise ValueError(f"Unknown factor '{t}' in term '{t}'")
                terms.append(safe_map[t])
    else:
        if model_type == "main":
            terms = xs
        elif model_type == "twfi":
            terms = xs + [f"{a}:{b}" for a, b in combinations(xs, 2)]
        elif model_type == "quad":
            # Guard: refuse if no factor has center points or 3+ levels
            has_curvature = False
            for col in factor_cols:
                coded_vals = coded_df[col].round(8).unique()
                if not all(v in (-1.0, 1.0) for v in coded_vals):
                    has_curvature = True
                    break
            if not has_curvature:
                raise ValueError(
                    "Pure quadratic on 'all factors' requires center "
                    "points or a factor with 3+ levels."
                )
            terms = (xs
                     + [f"{a}:{b}" for a, b in combinations(xs, 2)]
                     + [f"I({x}**2)" for x in xs])
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

    # Saturation check applies only to factor terms (block df excluded).
    # Allow exactly-saturated models (n_params == n, df_resid == 0) — they
    # are valid for unreplicated designs and are needed for Lenth PSE.
    n_params = 1 + len(terms)
    if n_params > n:
        raise ValueError(
            f"Model has {n_params} parameters but only {n} observations. "
            "Reduce model complexity or add more runs / replicates."
        )

    fdf          = coded_df.rename(columns=safe_map)
    fdf["y"]     = df_fit[response_col].values

    # Add block column as categorical if requested
    block_safe = None
    if block_col is not None:
        block_safe = _safe(block_col)
        fdf[block_safe] = df_fit[block_col].astype(str).values
        formula = "y ~ C(" + block_safe + ") + " + " + ".join(terms)
    else:
        formula  = "y ~ " + " + ".join(terms)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = smf.ols(formula, data=fdf).fit()

    # Run order for residual-vs-sequence plot
    run_order = None
    for col in ("Run Order", "Std Order", "Run"):
        if col in df.columns:
            run_order = df.loc[df_fit.index, col].values
            break
    if run_order is None:
        run_order = np.arange(1, n + 1)

    return dict(
        results=results, formula_df=fdf, encoding=encoding,
        safe_map=safe_map, rev_map=rev_map,
        factor_cols=factor_cols, response_col=response_col,
        model_type=model_type if custom_terms is None else "custom",
        custom_terms=custom_terms,
        n=n, k=k,
        terms=terms, run_order=run_order,
        df_fit=df_fit, df_original=df,
        block_col=block_col,
        block_safe=block_safe,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANOVA Table  (Type III Sums of Squares)
# ─────────────────────────────────────────────────────────────────────────────

def _lof_rows(groups: dict, ss_res: float, df_res: int) -> list:
    """Return [LoF row, PE row] dicts if pure error can be computed, else []."""
    ss_pe = sum(np.var(v, ddof=1) * (len(v) - 1)
                for v in groups.values() if len(v) > 1)
    df_pe = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    if df_pe > 0 and df_res > df_pe:
        ss_lof = max(0.0, ss_res - ss_pe)
        df_lof = df_res - df_pe
        ms_pe  = ss_pe  / df_pe
        ms_lof = ss_lof / df_lof
        f_lof  = ms_lof / ms_pe
        p_lof  = float(stats.f.sf(f_lof, df_lof, df_pe))
        return [
            dict(Source="  Lack of Fit", SS=ss_lof, df=df_lof,
                 MS=ms_lof, F=f_lof, **{"p-value": p_lof}),
            dict(Source="  Pure Error",  SS=ss_pe,  df=df_pe,
                 MS=ms_pe,  F=np.nan,  **{"p-value": np.nan}),
        ]
    return []


def _curvature_rows(fi: dict, ss_res: float, df_res: int) -> list:
    """
    Center-point curvature test (Montgomery Eq. 6.30).

    Returns [Pure-quadratic row, Pure-Error-center-pts row] when:
      - fi['formula_df'] contains both factorial (all xi in {-1,+1}) and
        center-point (at least one xi == 0 or otherwise not ±1) rows, AND
      - the fitted model contains no explicit I(xi**2) terms.

    Returns [] otherwise.
    """
    # Guard — no quadratic terms in model
    if any(t.startswith("I(") for t in fi["terms"]):
        return []

    coded_cols = list(fi["safe_map"].values())
    fdf = fi["formula_df"]
    coded = fdf[coded_cols].round(8)
    y = fdf["y"].values

    def is_center(row):
        return all(abs(v) < 1e-8 for v in row)

    def is_factorial(row):
        return all(abs(abs(v) - 1.0) < 1e-8 for v in row)

    mask_center    = coded.apply(is_center,    axis=1)
    mask_factorial = coded.apply(is_factorial, axis=1)

    y_C = y[mask_center.values]
    y_F = y[mask_factorial.values]
    n_C, n_F = len(y_C), len(y_F)

    if n_C == 0 or n_F == 0:
        return []

    df_PE = n_C - 1
    if df_PE < 1:
        return []

    ybar_C = float(np.mean(y_C))
    ybar_F = float(np.mean(y_F))

    SS_PQ = (n_F * n_C * (ybar_F - ybar_C) ** 2) / (n_F + n_C)
    SS_PE = float(np.sum((y_C - ybar_C) ** 2))

    if SS_PE == 0:
        return []

    MS_PQ = SS_PQ / 1
    MS_PE = SS_PE / df_PE
    F_PQ  = MS_PQ / MS_PE if MS_PE > 0 else np.nan
    p_PQ  = float(stats.f.sf(F_PQ, 1, df_PE)) if not np.isnan(F_PQ) else np.nan

    return [
        dict(Source="Pure quadratic", SS=SS_PQ, df=1,
             MS=MS_PQ, F=F_PQ, **{"p-value": float(p_PQ)}),
        dict(Source="  Pure Error (center pts)", SS=SS_PE, df=df_PE,
             MS=MS_PE, F=np.nan, **{"p-value": np.nan}),
    ]


def get_anova_table(fi: dict) -> pd.DataFrame:
    """
    Type III ANOVA table.
    Columns: Source | SS | df | MS | F | p-value

    When fi['block_col'] is set, emits a 'Blocks' row before the factor rows.
    Detects Lack-of-Fit / Pure-Error from repeated factor-level tuples even
    when no 'Replicate' column is present.
    """
    res = fi["results"]
    rev = fi["rev_map"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            aov = anova_lm(res, typ=3)
        except Exception:
            aov = anova_lm(res, typ=1)

    # ── Separate block row from factor rows ───────────────────────────────────
    block_row = None
    term_rows = []
    for src, row in aov.iterrows():
        if src in ("Intercept", "Residual"):
            continue
        ss  = float(row["sum_sq"])
        df_ = int(row["df"])
        ms  = ss / df_ if df_ > 0 else np.nan
        f   = float(row.get("F",      np.nan))
        p   = float(row.get("PR(>F)", np.nan))
        if src.startswith("C("):
            # This is the block term — no F/p per standard fixed-block ANOVA
            block_row = dict(Source="Blocks", SS=ss, df=df_, MS=ms,
                             F=np.nan, **{"p-value": np.nan})
        else:
            term_rows.append(dict(
                Source=_decode_term(src, rev),
                SS=ss, df=df_, MS=ms, F=f, **{"p-value": p}
            ))

    # ── Model summary row ─────────────────────────────────────────────────────
    ss_model = float(res.ess)
    df_model = int(res.df_model)
    ms_model = ss_model / df_model if df_model > 0 else np.nan
    ss_res   = float(res.ssr)
    df_res   = int(res.df_resid)
    ms_res   = ss_res / df_res if df_res > 0 else np.nan
    f_model  = ms_model / ms_res if (ms_res and ms_res > 0) else np.nan
    p_model  = float(stats.f.sf(f_model, df_model, df_res)) if not np.isnan(f_model) else np.nan
    ss_tot   = float(res.centered_tss)
    df_tot   = fi["n"] - 1

    # ── Lack-of-Fit / Pure-Error ──────────────────────────────────────────────
    lof_rows = []
    df_orig  = fi["df_original"]

    # Build factor-level groups from coded columns (xi only, excludes block).
    # Exclude center-point rows (all coded values == 0): they are handled by
    # _curvature_rows and must not inflate the LoF pure-error estimate.
    coded    = fi["formula_df"][list(fi["safe_map"].values())]
    y        = fi["formula_df"]["y"].values
    is_ctr   = (coded.round(8) == 0).all(axis=1).values
    coded_lof = coded[~is_ctr]
    y_lof     = y[~is_ctr]
    keys  = [tuple(row) for row in coded_lof.round(8).values.tolist()]
    groups: dict = {}
    for key, yi in zip(keys, y_lof):
        groups.setdefault(key, []).append(float(yi))

    # Path A — explicit Replicate column
    if ("Replicate" in df_orig.columns and
            df_orig.loc[fi["df_fit"].index, "Replicate"].nunique() > 1):
        lof_rows = _lof_rows(groups, ss_res, df_res)
    # Path B — detect from repeated factor-level tuples
    elif any(len(v) > 1 for v in groups.values()):
        lof_rows = _lof_rows(groups, ss_res, df_res)

    # ── Center-point curvature split ─────────────────────────────────────────
    curv_rows = _curvature_rows(fi, ss_res, df_res)

    rows = (
        [dict(Source="Model",          SS=ss_model, df=df_model,
              MS=ms_model, F=f_model,  **{"p-value": p_model})]
        + ([block_row] if block_row else [])
        + term_rows
        + [dict(Source="Residual",     SS=ss_res,   df=df_res,
                MS=ms_res,  F=np.nan,  **{"p-value": np.nan})]
        + lof_rows
        + curv_rows
        + [dict(Source="Total (Corr.)",SS=ss_tot,   df=df_tot,
                MS=np.nan,  F=np.nan,  **{"p-value": np.nan})]
    )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Coefficients & Model Statistics
# ─────────────────────────────────────────────────────────────────────────────

def get_coefficients(fi: dict) -> pd.DataFrame:
    """
    Regression coefficients in coded variables.
    Effect = 2 × Coefficient (half-effect convention for 2-level factorials).
    """
    res, rev = fi["results"], fi["rev_map"]
    rows = []
    for term in res.params.index:
        decoded = "Intercept" if term == "Intercept" else _decode_term(term, rev)
        coef    = float(res.params[term])
        se      = float(res.bse[term])
        t       = float(res.tvalues[term])
        p       = float(res.pvalues[term])
        effect  = 2.0 * coef if term != "Intercept" else np.nan
        rows.append(dict(Term=decoded, Effect=effect,
                         Coefficient=coef, **{"Std Error": se,
                                               "t-value": t, "p-value": p}))
    return pd.DataFrame(rows)


def get_model_stats(fi: dict) -> dict:
    """R², Adj-R², Pred-R², PRESS, S (root mean-square error)."""
    res    = fi["results"]
    infl   = res.get_influence()
    e      = res.resid.values
    h      = infl.hat_matrix_diag
    press  = float(np.sum((e / (1.0 - h)) ** 2))
    ss_tot = float(res.centered_tss)
    return dict(
        R2      = float(res.rsquared),
        AdjR2   = float(res.rsquared_adj),
        PredR2  = 1.0 - press / ss_tot if ss_tot > 0 else np.nan,
        PRESS   = press,
        S       = float(np.sqrt(res.mse_resid)),
        n       = fi["n"],
        df_mod  = int(res.df_model),
        df_res  = int(res.df_resid),
    )


def get_equations(fi: dict) -> dict:
    """
    Return regression equations in both coded and actual units.

    Coded equation:  ŷ = b₀ + b₁x₁ + b₂x₂ + ...
    Actual equation: ŷ = b₀* + b₁*X₁ + b₂*X₂ + ...
      where  xᵢ = (Xᵢ − centᵢ) / halfᵢ
    Actual-unit back-substitution is fully symbolic for main-effects models
    and shown as coding key for 2FI / quadratic.
    """
    res      = fi["results"]
    rev      = fi["rev_map"]
    enc      = fi["encoding"]
    safe_map = fi["safe_map"]

    # Coded equation string
    parts = []
    for term, coef in res.params.items():
        decoded = "Intercept" if term == "Intercept" else _decode_term(term, rev)
        if term == "Intercept":
            parts.append(f"{coef:+.4f}")
        else:
            sign = "+" if coef >= 0 else "−"
            parts.append(f"  {sign} {abs(coef):.4f}·{decoded}")
    coded_eq = "ŷ = " + "".join(parts)

    # Actual-unit equation (main effects only — closed form)
    terms_are_main_only = all(":" not in t and "I(" not in t for t in fi["terms"])
    if fi["model_type"] == "main" or (fi["model_type"] == "custom" and terms_are_main_only):
        b0 = float(res.params["Intercept"])
        b0_act = b0
        act_parts = []
        for col in fi["factor_cols"]:
            s = safe_map[col]
            if s not in res.params:
                continue
            b    = float(res.params[s])
            e    = enc[col]
            b_a  = b / e["half"]
            b0_act -= b * e["mid"] / e["half"]
            sign = "+" if b_a >= 0 else "−"
            act_parts.append(f"  {sign} {abs(b_a):.4f}·{col}")
        actual_eq = "ŷ = " + f"{b0_act:+.4f}" + "".join(act_parts)
    else:
        # Coding reference table for interactive/quadratic terms
        coding_lines = [
            f"  {col}: x = (X − {enc[col]['mid']:.4f}) / {enc[col]['half']:.4f}"
            for col in fi["factor_cols"]
        ]
        actual_eq = coded_eq + "\n\nCoding (X = actual, x = coded):\n" + "\n".join(coding_lines)

    return dict(coded=coded_eq, actual=actual_eq)


# ─────────────────────────────────────────────────────────────────────────────
# Residuals
# ─────────────────────────────────────────────────────────────────────────────

def get_residuals(fi: dict) -> pd.DataFrame:
    """DataFrame: Run Order | Fitted | Residual | Std Residual | Leverage"""
    res  = fi["results"]
    infl = res.get_influence()
    return pd.DataFrame(dict(
        **{"Run Order":   fi["run_order"]},
        Fitted          = res.fittedvalues.values,
        Residual        = res.resid.values,
        **{"Std Residual": infl.resid_studentized_internal},
        Leverage        = infl.hat_matrix_diag,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Lenth's PSE  (for unreplicated factorial experiments)
# ─────────────────────────────────────────────────────────────────────────────

def lenth_pse(effects: np.ndarray) -> float:
    """
    Pseudo Standard Error via Lenth (1989).
    Steps: s₀ = 1.5·median(|cᵢ|);  PSE = 1.5·median(|cᵢ| for |cᵢ|<2.5s₀)
    """
    a  = np.abs(effects)
    s0 = 1.5 * np.median(a)
    m  = a[a < 2.5 * s0]
    return float(1.5 * np.median(m)) if len(m) > 0 else float(np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _layout(**kw):
    return dict(template=PLOT_TEMPLATE, font=FONT,
                margin=dict(l=65, r=25, t=55, b=50), **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Pareto Chart of Standardized Effects
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto(fi: dict) -> go.Figure:
    """
    Horizontal bar chart of |t-values| for each model term.
    Reference line:
      • df_error > 0 → t_{0.025, df_error}  (Bonferroni corrected if many terms)
      • df_error = 0 → Lenth ME = t_{0.025, m/3} × PSE  (m = number of effects)
    Bars exceeding the reference line are coloured with ACCENT blue; others gray.
    """
    res    = fi["results"]
    rev    = fi["rev_map"]
    df_res = int(res.df_resid)

    names, tvals = [], []
    for term in res.params.index:
        if term == "Intercept":
            continue
        names.append(_decode_term(term, rev))
        tvals.append(abs(float(res.tvalues[term])))

    if not names:
        return go.Figure().add_annotation(text="No effects to display", showarrow=False)

    order  = np.argsort(tvals)
    names  = [names[i] for i in order]
    tvals  = [tvals[i] for i in order]

    if df_res > 0:
        t_crit    = float(stats.t.ppf(0.975, df_res))
        ref_label = f"t₀.₀₂₅,{df_res} = {t_crit:.2f}"
        ref_val   = t_crit
    else:
        effects = np.array([float(res.params[t]) * 2.0
                            for t in res.params.index if t != "Intercept"])
        pse = lenth_pse(effects)
        m   = len(effects)
        df_lenth = max(1, m // 3)
        t_lenth  = float(stats.t.ppf(0.975, df_lenth))
        if not np.isnan(pse) and pse > 0:
            me = t_lenth * pse
            ref_val   = me          # on |effect| scale; convert tvals to same scale
            # Re-express bars on |effect| scale
            tvals = [abs(float(res.params[
                next(t for t in res.params.index
                     if t != "Intercept" and _decode_term(t, rev) == nm)
            ]) * 2.0) for nm in names]
            ref_label = f"Lenth ME = {me:.4f}"
        else:
            ref_val   = None
            ref_label = ""

    colors = [ACCENT if tv > (ref_val or 0) else C_GRAY for tv in tvals]

    fig = go.Figure(go.Bar(
        x=tvals, y=names, orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    if ref_val is not None:
        fig.add_vline(x=ref_val, line_dash="dash", line_color=C_RED,
                      annotation_text=ref_label,
                      annotation_position="top right",
                      annotation_font_size=10)
    fig.update_layout(**_layout(
        title=dict(text="Pareto Chart of Standardized Effects", font_size=14),
        xaxis_title="|Effect| or |t-value|",
        yaxis_title="Term",
        height=max(280, 55 + 28 * len(names)),
        showlegend=False,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Half-Normal Probability Plot of Effects
# ─────────────────────────────────────────────────────────────────────────────

def plot_half_normal(fi: dict, lenth_lines: bool = True) -> go.Figure:
    """
    Half-normal probability plot of absolute effects (Daniel 1959).
    Points that lie above the reference line are likely significant.
    Half-normal quantile for rank i of n:  Φ⁻¹((n + i − 0.5) / (2n + 0.5))

    When the model has zero residual df (unreplicated design) and
    lenth_lines=True, draws ME and SME reference lines using Lenth's PSE.
    """
    res = fi["results"]
    rev = fi["rev_map"]

    names, effects = [], []
    for term in res.params.index:
        if term == "Intercept":
            continue
        names.append(_decode_term(term, rev))
        effects.append(float(res.params[term]) * 2.0)

    if not names:
        return go.Figure().add_annotation(text="No effects to display", showarrow=False)

    n       = len(effects)
    abs_eff = np.abs(effects)
    order   = np.argsort(abs_eff)
    s_abs   = abs_eff[order]
    s_names = [names[i] for i in order]

    # Half-normal quantiles (Daniel / Birnbaum formula)
    qi = np.array([stats.norm.ppf((n + i + 0.5) / (2 * n + 1.0))
                   for i in range(n)])

    # Significance threshold and optional Lenth ME/SME lines
    df_res = int(res.df_resid)
    me_val  = None
    sme_val = None

    if df_res > 0:
        # Replicated design — t-based threshold from residual MS (unchanged)
        se_avg    = float(np.mean([res.bse[t] for t in res.bse.index if t != "Intercept"]))
        t_crit    = float(stats.t.ppf(0.975, df_res))
        threshold = 2.0 * t_crit * se_avg
    else:
        # Unreplicated design — Lenth PSE
        pse = lenth_pse(np.array(effects))
        m   = len(effects)
        if np.isnan(pse) or pse <= 0 or m < 3:
            threshold = None
        else:
            df_lenth  = max(1, m // 3)
            t_me      = float(stats.t.ppf(0.975, df_lenth))
            gamma     = (1 + 0.95 ** (1.0 / m)) / 2.0
            t_sme     = float(stats.t.ppf(gamma, df_lenth))
            me_val    = t_me  * pse    # on |effect| scale (2 × |coef|)
            sme_val   = t_sme * pse    # on |effect| scale
            threshold = me_val         # label points above ME

    # Point colouring: three levels when Lenth lines available, two otherwise
    if me_val is not None:
        colors = []
        for ae in s_abs:
            if ae > sme_val:
                colors.append(ACCENT)
            elif ae > me_val:
                colors.append(C_ORG)
            else:
                colors.append(C_GRAY)
    else:
        colors = [ACCENT if (threshold and ae > threshold) else C_GRAY
                  for ae in s_abs]

    # Reference line fitted to the lower ~50% of points (assumed inert)
    half = max(2, n // 2)
    if len(qi[:half]) >= 2:
        slope = float(np.polyfit(qi[:half], s_abs[:half], 1)[0])
    else:
        slope = float(np.polyfit(qi, s_abs, 1)[0]) if n >= 2 else 1.0
    x_line = np.array([0.0, qi[-1] * 1.05])

    # Label points that are ACCENT or C_ORG (above ME)
    label_colors = {ACCENT, C_ORG}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_line, y=slope * x_line,
        mode="lines", line=dict(color=C_RED, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=qi, y=s_abs,
        mode="markers+text",
        text=[nm if c in label_colors else "" for nm, c in zip(s_names, colors)],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(color=colors, size=9, line=dict(width=1, color="white")),
        hovertemplate=[f"{nm}<br>|Effect| = {ae:.4f}<extra></extra>"
                       for nm, ae in zip(s_names, s_abs)],
        showlegend=False,
    ))

    # Lenth ME / SME reference lines (horizontal, on |Effect| axis)
    if lenth_lines and me_val is not None:
        fig.add_hline(y=me_val,  line_dash="dot",  line_color=C_ORG,
                      annotation_text=f"ME = {me_val:.3f}",
                      annotation_position="right",
                      annotation_font_size=9)
        fig.add_hline(y=sme_val, line_dash="dash", line_color=C_RED,
                      annotation_text=f"SME = {sme_val:.3f}",
                      annotation_position="right",
                      annotation_font_size=9)

    fig.update_layout(**_layout(
        title=dict(text="Half-Normal Plot of Effects", font_size=14),
        xaxis_title="Half-Normal Score",
        yaxis_title="|Effect|",
        height=380,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Main Effects Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_main_effects(fi: dict) -> go.Figure:
    """
    Mean response at each distinct level of each factor.
    Grand mean shown as a dashed reference line per subplot.
    """
    df_fit   = fi["df_fit"]
    factors  = fi["factor_cols"]
    response = fi["response_col"]
    n_f      = len(factors)

    cols_per_row = min(n_f, 4)
    n_rows       = (n_f + cols_per_row - 1) // cols_per_row

    fig = make_subplots(
        rows=n_rows, cols=cols_per_row,
        subplot_titles=factors,
        horizontal_spacing=0.10,
        vertical_spacing=0.22,
    )
    grand_mean = float(df_fit[response].mean())

    for idx, col in enumerate(factors):
        r = idx // cols_per_row + 1
        c = idx  % cols_per_row + 1
        x_raw  = df_fit[col].values
        y_raw  = df_fit[response].values
        ux     = np.sort(np.unique(np.round(x_raw, 6)))
        means  = [float(np.mean(y_raw[np.round(x_raw, 6) == xv])) for xv in ux]

        fig.add_trace(go.Scatter(
            x=ux, y=means, mode="lines+markers",
            marker=dict(color=ACCENT, size=8),
            line=dict(color=ACCENT, width=2),
            hovertemplate=f"{col}=%{{x}}<br>Mean {response}=%{{y:.4f}}<extra></extra>",
            showlegend=False,
        ), row=r, col=c)
        fig.add_hline(y=grand_mean, line_dash="dot", line_color=C_GRAY,
                      row=r, col=c)
        fig.update_xaxes(title_text=col,      row=r, col=c)
        fig.update_yaxes(title_text=response, row=r, col=c)

    fig.update_layout(**_layout(
        title=dict(text="Main Effects Plot  (dashed = grand mean)", font_size=14),
        height=280 * n_rows,
        showlegend=False,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Interaction Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_interaction(fi: dict, factor_a: str, factor_b: str) -> go.Figure:
    """
    Interaction plot: mean response vs factor_a at each level of factor_b.
    Parallel lines → no interaction; crossing or diverging lines → interaction.
    """
    df_fit   = fi["df_fit"]
    response = fi["response_col"]

    xa = df_fit[factor_a].values
    xb = df_fit[factor_b].values
    y  = df_fit[response].values

    ux_a = np.sort(np.unique(np.round(xa, 6)))
    ux_b = np.sort(np.unique(np.round(xb, 6)))
    palette = [ACCENT, C_ORG, C_GREEN, "#ae3ec9", C_RED]

    fig = go.Figure()
    for j, bv in enumerate(ux_b):
        mask  = np.round(xb, 6) == bv
        means = [float(np.mean(y[mask & (np.round(xa, 6) == av)]))
                 if np.any(mask & (np.round(xa, 6) == av)) else np.nan
                 for av in ux_a]
        fig.add_trace(go.Scatter(
            x=ux_a, y=means, mode="lines+markers",
            name=f"{factor_b} = {bv}",
            line=dict(color=palette[j % len(palette)], width=2),
            marker=dict(size=9),
        ))
    fig.update_layout(**_layout(
        title=dict(text=f"Interaction Plot:  {factor_a}  ×  {factor_b}", font_size=14),
        xaxis_title=factor_a, yaxis_title=response,
        legend_title=factor_b, height=380,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Residual Analysis (2 × 2 panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals(fi: dict) -> go.Figure:
    """
    Four-panel residual diagnostic:
      [Normal Probability Plot]   [Residuals vs Fitted Values]
      [Residuals vs Run Order]    [Histogram of Residuals]
    """
    rdf    = get_residuals(fi)
    e      = rdf["Residual"].values
    fitted = rdf["Fitted"].values
    runs   = rdf["Run Order"].values
    n      = len(e)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Normal Probability Plot of Residuals",
            "Residuals vs Fitted Values",
            "Residuals vs Run Order",
            "Histogram of Residuals",
        ),
        horizontal_spacing=0.13,
        vertical_spacing=0.20,
    )

    # ── Panel 1: Normal Q-Q ───────────────────────────────────────────────────
    (osm, osr), (slope, intercept, _) = stats.probplot(e, dist="norm")
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color=ACCENT, size=6),
        hovertemplate="Theoretical=%{x:.3f}<br>Residual=%{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    xl = np.array([osm[0], osm[-1]])
    fig.add_trace(go.Scatter(
        x=xl, y=slope * xl + intercept, mode="lines",
        line=dict(color=C_RED, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    # Anderson-Darling test annotation
    ad_stat, ad_cv, _ = stats.anderson(e, dist="norm")
    ad_sig = ad_stat < ad_cv[-1]   # compare to 1% critical value
    ad_text = f"AD = {ad_stat:.3f}  {'(Normal OK)' if ad_sig else '(Non-normal?)'}"
    fig.add_annotation(x=0.03, y=0.97, xref="x domain", yref="y domain",
                       text=ad_text, showarrow=False, font=dict(size=8, color=C_GRAY),
                       align="left", row=1, col=1)

    # ── Panel 2: Residuals vs Fitted ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fitted, y=e, mode="markers",
        marker=dict(color=ACCENT, size=6),
        hovertemplate="Fitted=%{x:.4f}<br>Residual=%{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=C_GRAY, row=1, col=2)

    # ── Panel 3: Residuals vs Run Order ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=runs, y=e, mode="markers+lines",
        marker=dict(color=ACCENT, size=6),
        line=dict(color=ACCENT, width=0.8),
        hovertemplate="Run=%{x}<br>Residual=%{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=C_GRAY, row=2, col=1)

    # ── Panel 4: Histogram ────────────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=e, nbinsx=max(5, min(12, n // 2)),
        marker_color=ACCENT, opacity=0.85, showlegend=False,
        hovertemplate="Residual=%{x:.4f}<br>Count=%{y}<extra></extra>",
    ), row=2, col=2)

    fig.update_xaxes(title_text="Theoretical Quantile", row=1, col=1)
    fig.update_yaxes(title_text="Residual",             row=1, col=1)
    fig.update_xaxes(title_text="Fitted Value",         row=1, col=2)
    fig.update_yaxes(title_text="Residual",             row=1, col=2)
    fig.update_xaxes(title_text="Run Order",            row=2, col=1)
    fig.update_yaxes(title_text="Residual",             row=2, col=1)
    fig.update_xaxes(title_text="Residual",             row=2, col=2)
    fig.update_yaxes(title_text="Count",                row=2, col=2)

    fig.update_layout(**_layout(height=530, showlegend=False))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Prediction, Response Surface, Optimization
# ─────────────────────────────────────────────────────────────────────────────

def predict_response(fi: dict, point_actual: dict) -> float:
    """
    Predict the response at a single point given in actual (un-coded) units.

    Parameters
    ----------
    fi            : fit_info dict returned by fit_model
    point_actual  : {factor_name: value, …}

    Returns
    -------
    Predicted response (float)
    """
    encoding = fi["encoding"]
    safe_map = fi["safe_map"]

    coded_row = {}
    for col in fi["factor_cols"]:
        enc = encoding[col]
        x_a = float(point_actual.get(col, enc["mid"]))
        coded_row[safe_map[col]] = (x_a - enc["mid"]) / enc["half"]

    if fi.get("block_col"):
        block_safe = fi.get("block_safe") or _safe(fi["block_col"])
        block_levels = sorted(fi["formula_df"][block_safe].unique().tolist())
        coded_row[block_safe] = block_levels[0]

    pred_df = pd.DataFrame([coded_row])
    pred_df["y"] = 0.0
    return float(fi["results"].predict(pred_df).iloc[0])


def get_surface_data(fi: dict, fa: str, fb: str,
                     constants_actual: dict = None, n: int = 50):
    """
    Generate a grid of predicted responses for a response-surface plot.

    Parameters
    ----------
    fi                : fit_info dict
    fa, fb            : names of the two factors placed on the axes
    constants_actual  : {other_factor_name: hold_value, …}
                        defaults to the midpoint of each factor's range
    n                 : grid resolution (n × n points)

    Returns
    -------
    X : (n, n) array of fa actual values
    Y : (n, n) array of fb actual values
    Z : (n, n) array of predicted response values
    """
    encoding  = fi["encoding"]
    safe_map  = fi["safe_map"]
    constants = constants_actual or {}

    enc_a = encoding[fa]
    enc_b = encoding[fb]

    xa = np.linspace(enc_a["low"], enc_a["high"], n)
    xb = np.linspace(enc_b["low"], enc_b["high"], n)
    X, Y = np.meshgrid(xa, xb)

    rows = {}
    rows[safe_map[fa]] = (X.ravel() - enc_a["mid"]) / enc_a["half"]
    rows[safe_map[fb]] = (Y.ravel() - enc_b["mid"]) / enc_b["half"]

    for col in fi["factor_cols"]:
        if col in (fa, fb):
            continue
        enc   = encoding[col]
        val_a = float(constants.get(col, enc["mid"]))
        val_c = (val_a - enc["mid"]) / enc["half"]
        rows[safe_map[col]] = np.full(n * n, val_c)

    if fi.get("block_col"):
        block_safe = fi.get("block_safe") or _safe(fi["block_col"])
        block_levels = sorted(fi["formula_df"][block_safe].unique().tolist())
        rows[block_safe] = [block_levels[0]] * (n * n)

    pred_df = pd.DataFrame(rows)
    pred_df["y"] = 0.0
    Z = fi["results"].predict(pred_df).values.reshape(n, n)

    return X, Y, Z


def optimize_response(fi: dict, goal: str = "maximize",
                      target: float = None) -> tuple:
    """
    Numerical response optimisation using differential evolution.

    Parameters
    ----------
    fi      : fit_info dict
    goal    : 'maximize' | 'minimize' | 'target'
    target  : desired response value (required when goal == 'target')

    Returns
    -------
    (best_point_actual: dict, predicted_value: float)
    """
    from scipy.optimize import differential_evolution

    encoding    = fi["encoding"]
    factor_cols = fi["factor_cols"]
    safe_map    = fi["safe_map"]

    def objective(x_coded):
        pred_row = {safe_map[col]: x_coded[i]
                    for i, col in enumerate(factor_cols)}
        if fi.get("block_col"):
            block_safe = fi.get("block_safe") or _safe(fi["block_col"])
            block_levels = sorted(fi["formula_df"][block_safe].unique().tolist())
            pred_row[block_safe] = block_levels[0]
        pred_row["y"] = 0.0
        y = float(fi["results"].predict(pd.DataFrame([pred_row])).iloc[0])
        if goal == "maximize":
            return -y
        elif goal == "minimize":
            return y
        else:
            return (y - float(target or 0.0)) ** 2

    bounds = [(-1.0, 1.0)] * len(factor_cols)
    result = differential_evolution(
        objective, bounds, seed=42, tol=1e-10,
        maxiter=2000, popsize=20, polish=True,
    )

    best_actual = {}
    for i, col in enumerate(factor_cols):
        enc = encoding[col]
        best_actual[col] = enc["mid"] + result.x[i] * enc["half"]

    return best_actual, predict_response(fi, best_actual)
