"""
End-to-end tests for categorical factor support.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np
import pytest

from doe_generators import generate_design, two_level_full_factorial, general_full_factorial
from analysis import fit_model, get_anova_table, predict_response, optimize_response


# ── Test 1: Mixed design (1 numeric, 1 categoric) ────────────────────────────

def test_two_level_with_categoric_generation():
    """Two-level full factorial with one categoric factor generates string levels."""
    factors = [
        {"name": "Temperature", "type": "numeric",   "low": 150, "high": 200},
        {"name": "Catalyst",    "type": "categoric", "levels": ["A", "B"]},
    ]
    df = two_level_full_factorial(factors)
    assert set(df["Catalyst"].unique()) == {"A", "B"}
    assert set(df["Temperature"].unique()) == {150.0, 200.0}


def test_general_factorial_with_categoric():
    """General factorial with 3-level categoric factor generates correct labels."""
    factors = [
        {"name": "Machine",  "type": "categoric", "levels": ["M1", "M2", "M3"], "num_levels": 3},
        {"name": "Operator", "type": "categoric", "levels": ["Op1", "Op2"],      "num_levels": 2},
    ]
    df = general_full_factorial(factors)
    assert set(df["Machine"].unique())  == {"M1", "M2", "M3"}
    assert set(df["Operator"].unique()) == {"Op1", "Op2"}
    assert len(df) == 6  # 3 × 2


def test_mixed_design_fit_model():
    """fit_model on mixed design: correct df for categoric factor (k-1 per factor)."""
    data = {
        "Temperature": [150, 200, 150, 200],
        "Catalyst":    ["A", "A", "B", "B"],
        "Yield":       [65.0, 70.0, 72.0, 80.0],
    }
    df = pd.DataFrame(data)
    fi = fit_model(df, "Yield", ["Temperature", "Catalyst"])
    aov = get_anova_table(fi)

    # Catalyst should appear in ANOVA (1 df for 2 levels → k-1 = 1)
    sources = list(aov["Source"])
    assert any("Catalyst" in s for s in sources), f"Catalyst not found in ANOVA sources: {sources}"
    cat_row = aov[aov["Source"].str.contains("Catalyst", na=False)]
    assert len(cat_row) == 1


def test_mixed_design_predict():
    """predict_response returns a float for mixed numeric+categoric design."""
    data = {
        "Temperature": [150, 200, 150, 200],
        "Catalyst":    ["A", "A", "B", "B"],
        "Yield":       [65.0, 70.0, 72.0, 80.0],
    }
    df = pd.DataFrame(data)
    fi = fit_model(df, "Yield", ["Temperature", "Catalyst"])
    pred = predict_response(fi, {"Temperature": 175, "Catalyst": "A"})
    assert isinstance(pred, float)
    assert 60 < pred < 90


def test_mixed_design_optimize():
    """optimize_response returns categorical level alongside numeric value."""
    data = {
        "Temperature": [150, 200, 150, 200],
        "Catalyst":    ["A", "A", "B", "B"],
        "Yield":       [65.0, 70.0, 72.0, 80.0],
    }
    df = pd.DataFrame(data)
    fi = fit_model(df, "Yield", ["Temperature", "Catalyst"])
    best_point, pred_val = optimize_response(fi, goal="maximize")
    assert "Catalyst" in best_point
    assert best_point["Catalyst"] in ("A", "B")
    assert "Temperature" in best_point
    assert isinstance(pred_val, float)


# ── Test 2: All-categoric design ─────────────────────────────────────────────

def test_all_categoric_fit():
    """fit_model with all-categorical factors — ANOVA shows correct df."""
    data = {
        "Machine":  ["M1", "M2", "M3", "M1", "M2", "M3"],
        "Operator": ["Op1", "Op2", "Op1", "Op2", "Op1", "Op2"],
        "Output":   [55.0, 60.0, 58.0, 62.0, 61.0, 59.0],
    }
    df = pd.DataFrame(data)
    fi = fit_model(df, "Output", ["Machine", "Operator"])
    aov = get_anova_table(fi)
    sources = list(aov["Source"])
    # Machine should have 2 df (3 levels - 1), Operator should have 1 df (2 levels - 1)
    machine_row  = aov[aov["Source"].str.contains("Machine",  na=False)]
    operator_row = aov[aov["Source"].str.contains("Operator", na=False)]
    assert len(machine_row)  >= 1
    assert len(operator_row) >= 1
    # Check df (may be combined or split in Type III)
    m_df = int(machine_row.iloc[0]["df"])
    o_df = int(operator_row.iloc[0]["df"])
    assert m_df == 2, f"Machine should have df=2, got {m_df}"
    assert o_df == 1, f"Operator should have df=1, got {o_df}"


def test_surface_plot_raises_for_categoric():
    """get_surface_data raises ValueError when a categorical factor is selected."""
    from analysis import get_surface_data
    data = {
        "Temperature": [150, 200, 150, 200],
        "Catalyst":    ["A", "A", "B", "B"],
        "Yield":       [65.0, 70.0, 72.0, 80.0],
    }
    df = pd.DataFrame(data)
    fi = fit_model(df, "Yield", ["Temperature", "Catalyst"])
    with pytest.raises(ValueError, match="numeric"):
        get_surface_data(fi, "Temperature", "Catalyst")
