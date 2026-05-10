"""
Rule-based next-step recommendation — no API call.
Implements the ANALYSIS_INTERPRETATION_RULES decision tree directly in Python.
"""


def recommend_next_step(
    model_stats: dict, anova_rows: list[dict]
) -> tuple[str, str]:
    """
    Return (recommendation_key, justification) based on ANOVA and model stats.

    recommendation_key is one of:
        'optimize'         – proceed to the Prediction & Optimization tab
        'add_runs'         – increase factor ranges or replicate to gain power
        'augment_design'   – move to CCD / Resolution V to handle curvature or interactions
        'run_confirmatory' – model looks good; run confirmation experiments at optimum
    """
    r2      = _safe_float(model_stats.get("R2"))
    adj_r2  = _safe_float(model_stats.get("AdjR2"))
    pred_r2 = _safe_float(model_stats.get("PredR2"))

    model_p = None
    lof_p   = None
    has_significant_interactions = False
    all_terms_insignificant      = True

    for row in (anova_rows or []):
        source = str(row.get("Source", "")).lower().strip()
        p      = _safe_float(row.get("p-value"))
        if p is None:
            continue

        if "model" in source and "lack" not in source:
            model_p = p
        elif "lack" in source or "lof" in source:
            lof_p = p
        elif source not in ("total", "residual", "error", "pure error"):
            # individual term row
            if p < 0.05:
                all_terms_insignificant = False
                src_raw = str(row.get("Source", ""))
                if "*" in src_raw or "×" in src_raw or ":" in src_raw:
                    has_significant_interactions = True

    # ── Decision tree ─────────────────────────────────────────────────────────

    # Curvature / LOF significant → augment design
    if lof_p is not None and lof_p < 0.05:
        return (
            "augment_design",
            f"Lack-of-Fit is significant (p = {lof_p:.4f}). "
            "The current model form is inadequate — consider augmenting to a "
            "Central Composite Design (CCD) to estimate quadratic effects, "
            "or add center-point replicates to confirm the curvature.",
        )

    # Overall model not significant → need more data / wider ranges
    if model_p is not None and model_p >= 0.05:
        return (
            "add_runs",
            f"The overall model is not statistically significant (p = {model_p:.4f}). "
            "Either the active factor ranges are too narrow, "
            "or additional runs are needed to increase statistical power. "
            "Consider widening factor levels or adding replicates.",
        )

    if model_p is not None and model_p < 0.05:
        # Possible overfitting: large gap between Adj R² and Pred R²
        if adj_r2 is not None and pred_r2 is not None and (adj_r2 - pred_r2) > 0.20:
            return (
                "add_runs",
                f"Model is significant (p = {model_p:.4f}) but the gap between "
                f"Adj R² ({adj_r2:.3f}) and Pred R² ({pred_r2:.3f}) exceeds 0.20, "
                "suggesting possible overfitting or an influential outlier. "
                "Inspect residual plots and consider removing insignificant terms.",
            )

        # Good fit — ready for optimisation and then confirmation
        if r2 is not None and r2 >= 0.90:
            if pred_r2 is not None and pred_r2 >= 0.70:
                return (
                    "run_confirmatory",
                    f"Model is significant (p = {model_p:.4f}), "
                    f"R² = {r2:.3f}, Pred R² = {pred_r2:.3f}. "
                    "Model quality is good. Use the Prediction & Optimization tab to find "
                    "the optimal settings, then run 3–5 confirmatory experiments at the "
                    "predicted optimum before implementing changes.",
                )
            # Good R² but lower Pred R²
            return (
                "optimize",
                f"Model is significant (p = {model_p:.4f}), R² = {r2:.3f}. "
                "Pred R² is modest; predictions may be imprecise for new runs. "
                "Proceed to optimization but treat the predicted optimum as approximate.",
            )

        # Significant but lower R²
        if r2 is not None and r2 < 0.90:
            return (
                "add_runs",
                f"Model is significant (p = {model_p:.4f}) but R² = {r2:.3f} is below 0.90. "
                "Noise is masking some effects. Add replicates to improve precision, "
                "or verify that factor ranges span the active region.",
            )

    # Interactions significant but possibly aliased
    if has_significant_interactions:
        return (
            "augment_design",
            "Significant two-factor interactions are present. "
            "If the current design aliases interactions with other effects, "
            "augment to Resolution V or a full factorial to estimate them independently.",
        )

    # Fallback
    return (
        "optimize",
        "Proceed to the Prediction & Optimization tab to find optimal factor settings.",
    )


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None if missing or non-numeric."""
    if val is None:
        return None
    try:
        f = float(val)
        import math
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None
