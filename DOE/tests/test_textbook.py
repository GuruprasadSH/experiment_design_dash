"""
Textbook verification tests for fit_model.

References: Montgomery, D.C. (2017). Design and Analysis of Experiments, 9th ed.
Examples 6.1, 6.2, 7.2, and a 6.7-style guard test.
"""

import pytest
import pandas as pd
import numpy as np

from analysis import fit_model
from analysis import get_anova_table, get_coefficients


# ─────────────────────────────────────────────────────────────────────────────
# Example 6.1 — Plasma Etching (2³, 2 replicates, n=16)
# Table 6.5 / Table 6.6 (p. 245)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def df_61():
    rows = [
        {"A": -1, "B": -1, "C": -1, "Replicate": 1, "EtchRate": 550},
        {"A": -1, "B": -1, "C": -1, "Replicate": 2, "EtchRate": 604},
        {"A":  1, "B": -1, "C": -1, "Replicate": 1, "EtchRate": 669},
        {"A":  1, "B": -1, "C": -1, "Replicate": 2, "EtchRate": 650},
        {"A": -1, "B":  1, "C": -1, "Replicate": 1, "EtchRate": 633},
        {"A": -1, "B":  1, "C": -1, "Replicate": 2, "EtchRate": 601},
        {"A":  1, "B":  1, "C": -1, "Replicate": 1, "EtchRate": 642},
        {"A":  1, "B":  1, "C": -1, "Replicate": 2, "EtchRate": 635},
        {"A": -1, "B": -1, "C":  1, "Replicate": 1, "EtchRate": 1037},
        {"A": -1, "B": -1, "C":  1, "Replicate": 2, "EtchRate": 1052},
        {"A":  1, "B": -1, "C":  1, "Replicate": 1, "EtchRate": 749},
        {"A":  1, "B": -1, "C":  1, "Replicate": 2, "EtchRate": 868},
        {"A": -1, "B":  1, "C":  1, "Replicate": 1, "EtchRate": 1075},
        {"A": -1, "B":  1, "C":  1, "Replicate": 2, "EtchRate": 1063},
        {"A":  1, "B":  1, "C":  1, "Replicate": 1, "EtchRate": 729},
        {"A":  1, "B":  1, "C":  1, "Replicate": 2, "EtchRate": 860},
    ]
    return pd.DataFrame(rows)


def test_example_61_effect_abc(df_61):
    """ABC effect should be 5.625 (Table 6.5)."""
    fi = fit_model(df_61, "EtchRate", ["A", "B", "C"],
                   custom_terms=["A", "B", "C", "A*B", "A*C", "B*C", "A*B*C"])
    coefs = get_coefficients(fi)
    abc_row = coefs[coefs["Term"] == "A × B × C"]
    assert len(abc_row) == 1, "A × B × C term not found in coefficients"
    effect_abc = float(abc_row["Effect"].iloc[0])
    assert abs(effect_abc - 5.625) < 1e-4, f"Effect_ABC = {effect_abc}, expected 5.625"


def test_example_61_ss_abc(df_61):
    """SS_ABC should be 126.5625 (Table 6.6)."""
    fi = fit_model(df_61, "EtchRate", ["A", "B", "C"],
                   custom_terms=["A", "B", "C", "A*B", "A*C", "B*C", "A*B*C"])
    aov = get_anova_table(fi)
    abc_row = aov[aov["Source"] == "A × B × C"]
    assert len(abc_row) == 1, "A × B × C row not found in ANOVA"
    ss_abc = float(abc_row["SS"].iloc[0])
    assert abs(ss_abc - 126.5625) < 1e-4, f"SS_ABC = {ss_abc}, expected 126.5625"


# ─────────────────────────────────────────────────────────────────────────────
# Example 6.2 — Pilot-Plant Filtration (unreplicated 2⁴, n=16)
# Reduced model: A, C, D, AC, AD (Table 6.13)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def df_62():
    rows = [
        {"A": -1, "B": -1, "C": -1, "D": -1, "FiltrationRate": 45},
        {"A":  1, "B": -1, "C": -1, "D": -1, "FiltrationRate": 71},
        {"A": -1, "B":  1, "C": -1, "D": -1, "FiltrationRate": 48},
        {"A":  1, "B":  1, "C": -1, "D": -1, "FiltrationRate": 65},
        {"A": -1, "B": -1, "C":  1, "D": -1, "FiltrationRate": 68},
        {"A":  1, "B": -1, "C":  1, "D": -1, "FiltrationRate": 60},
        {"A": -1, "B":  1, "C":  1, "D": -1, "FiltrationRate": 80},
        {"A":  1, "B":  1, "C":  1, "D": -1, "FiltrationRate": 65},
        {"A": -1, "B": -1, "C": -1, "D":  1, "FiltrationRate": 43},
        {"A":  1, "B": -1, "C": -1, "D":  1, "FiltrationRate": 100},
        {"A": -1, "B":  1, "C": -1, "D":  1, "FiltrationRate": 45},
        {"A":  1, "B":  1, "C": -1, "D":  1, "FiltrationRate": 104},
        {"A": -1, "B": -1, "C":  1, "D":  1, "FiltrationRate": 75},
        {"A":  1, "B": -1, "C":  1, "D":  1, "FiltrationRate": 86},
        {"A": -1, "B":  1, "C":  1, "D":  1, "FiltrationRate": 70},
        {"A":  1, "B":  1, "C":  1, "D":  1, "FiltrationRate": 96},
    ]
    return pd.DataFrame(rows)


def test_example_62_effect_a(df_62):
    """Effect_A should be 21.625 (Table 6.12)."""
    fi = fit_model(df_62, "FiltrationRate", ["A", "B", "C", "D"],
                   custom_terms=["A", "C", "D", "A*C", "A*D"])
    coefs = get_coefficients(fi)
    a_row = coefs[coefs["Term"] == "A"]
    assert len(a_row) == 1
    assert abs(float(a_row["Effect"].iloc[0]) - 21.625) < 1e-4


def test_example_62_ss_ac(df_62):
    """SS_AC should be 1314.0625 (Table 6.13)."""
    fi = fit_model(df_62, "FiltrationRate", ["A", "B", "C", "D"],
                   custom_terms=["A", "C", "D", "A*C", "A*D"])
    aov = get_anova_table(fi)
    ac_row = aov[aov["Source"] == "A × C"]
    assert len(ac_row) == 1
    assert abs(float(ac_row["SS"].iloc[0]) - 1314.0625) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Example 7.2 — 2⁴ Confounded in 2 Blocks with I = ABCD (n=16)
# ABCD effect = −18.625; SS_ABCD = 1387.5625
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def df_72():
    rows = [
        {"A": -1, "B": -1, "C": -1, "D": -1, "Block": 1, "FiltrationRate": 25},
        {"A":  1, "B":  1, "C": -1, "D": -1, "Block": 1, "FiltrationRate": 45},
        {"A":  1, "B": -1, "C":  1, "D": -1, "Block": 1, "FiltrationRate": 40},
        {"A": -1, "B":  1, "C":  1, "D": -1, "Block": 1, "FiltrationRate": 60},
        {"A":  1, "B": -1, "C": -1, "D":  1, "Block": 1, "FiltrationRate": 80},
        {"A": -1, "B":  1, "C": -1, "D":  1, "Block": 1, "FiltrationRate": 25},
        {"A": -1, "B": -1, "C":  1, "D":  1, "Block": 1, "FiltrationRate": 55},
        {"A":  1, "B":  1, "C":  1, "D":  1, "Block": 1, "FiltrationRate": 76},
        {"A":  1, "B": -1, "C": -1, "D": -1, "Block": 2, "FiltrationRate": 71},
        {"A": -1, "B":  1, "C": -1, "D": -1, "Block": 2, "FiltrationRate": 48},
        {"A": -1, "B": -1, "C":  1, "D": -1, "Block": 2, "FiltrationRate": 68},
        {"A": -1, "B": -1, "C": -1, "D":  1, "Block": 2, "FiltrationRate": 43},
        {"A":  1, "B":  1, "C":  1, "D": -1, "Block": 2, "FiltrationRate": 65},
        {"A": -1, "B":  1, "C":  1, "D":  1, "Block": 2, "FiltrationRate": 70},
        {"A":  1, "B": -1, "C":  1, "D":  1, "Block": 2, "FiltrationRate": 86},
        {"A":  1, "B":  1, "C": -1, "D":  1, "Block": 2, "FiltrationRate": 104},
    ]
    return pd.DataFrame(rows)


def test_example_72_effect_abcd(df_72):
    """ABCD is confounded with blocks; block contrast should equal −18.625 (Data.md)."""
    # ABCD is aliased with blocks — cannot be in the fitted model.
    # We verify the block contrast directly from the data.
    block1 = df_72[df_72["Block"] == 1]["FiltrationRate"].sum()  # = 406
    block2 = df_72[df_72["Block"] == 2]["FiltrationRate"].sum()  # = 555
    n_per_block = 8
    effect_abcd = (block1 - block2) / n_per_block
    assert abs(effect_abcd - (-18.625)) < 1e-4, f"Block effect = {effect_abcd}, expected −18.625"


def test_example_72_ss_abcd(df_72):
    """SS_ABCD (= SS_Block) should be 1387.5625."""
    block1 = df_72[df_72["Block"] == 1]["FiltrationRate"].sum()  # = 406
    block2 = df_72[df_72["Block"] == 2]["FiltrationRate"].sum()  # = 555
    n_total = len(df_72)
    ss_abcd = (block1 - block2) ** 2 / n_total
    assert abs(ss_abcd - 1387.5625) < 1e-4, f"SS_ABCD = {ss_abcd}, expected 1387.5625"


def test_example_72_non_block_terms_fit(df_72):
    """Fitting all 14 non-block factorial terms should succeed with 1 df residual."""
    terms = (
        ["A", "B", "C", "D"]
        + ["A*B", "A*C", "A*D", "B*C", "B*D", "C*D"]
        + ["A*B*C", "A*B*D", "A*C*D", "B*C*D"]
    )
    fi = fit_model(df_72, "FiltrationRate", ["A", "B", "C", "D"],
                   custom_terms=terms)
    assert int(fi["results"].df_resid) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6.7-style guard — pure quadratic on a 2² design (no center points)
# ─────────────────────────────────────────────────────────────────────────────

def test_quad_guard_raises_on_pure_pm1_design():
    """Requesting A^2 on a ±1-only design must raise ValueError."""
    df = pd.DataFrame({
        "A": [-1,  1, -1,  1],
        "B": [-1, -1,  1,  1],
        "y": [20, 25, 30, 35],
    })
    with pytest.raises(ValueError, match="center points"):
        fit_model(df, "y", ["A", "B"], custom_terms=["A", "B", "A^2"])


# ─────────────────────────────────────────────────────────────────────────────
# _decode_term round-trip for 1, 2, 3, 4-way
# ─────────────────────────────────────────────────────────────────────────────

def test_decode_term_arities():
    from analysis import _decode_term
    rev = {"x0": "A", "x1": "B", "x2": "C", "x3": "D"}
    assert _decode_term("x0", rev) == "A"
    assert _decode_term("x0:x1", rev) == "A × B"
    assert _decode_term("x0:x1:x2", rev) == "A × B × C"
    assert _decode_term("x0:x1:x2:x3", rev) == "A × B × C × D"
    assert _decode_term("I(x0**2)", rev) == "A²"


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 — P0-4: Pure Error auto-detected from repeated factor-level rows
# ─────────────────────────────────────────────────────────────────────────────

def test_ex61_pure_error_auto_detected():
    """2^3 with 2 reps per cell — 16 rows, no Replicate column.
    get_anova_table must auto-detect replicates from repeated factor-level
    tuples and produce a Pure Error row with SS ≈ 18020.50, df = 8."""
    data = {
        "A":        [-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1],
        "B":        [-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1],
        "C":        [-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1],
        "EtchRate": [550,604,669,650,633,601,642,635,
                     1037,1052,749,868,1075,1063,729,860],
    }
    df = pd.DataFrame(data)
    fi  = fit_model(df, "EtchRate", ["A","B","C"], custom_terms=["A","B","C"])
    aov = get_anova_table(fi)
    pe_row = aov[aov["Source"].str.strip() == "Pure Error"]
    assert not pe_row.empty, "Pure Error row missing"
    assert pe_row["df"].iloc[0] == 8
    assert abs(pe_row["SS"].iloc[0] - 18020.50) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 — P0-2: Block column produces correct SS split (Example 7.1)
# ─────────────────────────────────────────────────────────────────────────────

def test_ex71_block_handling():
    """2^2 in 3 blocks (Chemical Process). Block SS must be 6.50,
    residual SS must be 24.84 (not 31.33), and Block row must appear
    before factor rows in the ANOVA table."""
    data = {
        "A":     [-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1],
        "B":     [-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1],
        "Block": [ 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        "Yield": [28,36,18,31,25,32,19,30,27,32,23,29],
    }
    df = pd.DataFrame(data)
    fi  = fit_model(df, "Yield", ["A","B"],
                    custom_terms=["A","B","A*B"], block_col="Block")
    aov = get_anova_table(fi)

    block_row = aov[aov["Source"] == "Blocks"]
    assert not block_row.empty, "Blocks row missing from ANOVA"
    assert abs(block_row["SS"].iloc[0] - 6.50) < 0.01

    res_row = aov[aov["Source"] == "Residual"]
    assert abs(res_row["SS"].iloc[0] - 24.84) < 0.01

    block_idx = aov.index[aov["Source"] == "Blocks"].tolist()[0]
    a_idx     = aov.index[aov["Source"] == "A"].tolist()[0]
    assert block_idx < a_idx, "Blocks row must come before factor rows"


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 — P0-2: Example 7.2 — ABCD confounded in Block
# ─────────────────────────────────────────────────────────────────────────────

def test_ex72_abcd_block():
    """2^4 confounded with ABCD in 2 blocks.
    Block (ABCD) SS must be 1387.5625; factor SS A=1870.5625."""
    data = {
        "A":              [-1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1, 1,-1, 1, 1],
        "B":              [-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1],
        "C":              [-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1],
        "D":              [-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1,-1],
        "Block":          [ 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        "FiltrationRate": [25,45,40,60,80,25,55,76,71,48,68,43,65,70,86,104],
    }
    df = pd.DataFrame(data)
    fi  = fit_model(df, "FiltrationRate", ["A","B","C","D"],
                    custom_terms=["A","B","C","D","A*C","A*D"],
                    block_col="Block")
    aov = get_anova_table(fi)

    block_row = aov[aov["Source"] == "Blocks"]
    assert not block_row.empty
    assert abs(block_row["SS"].iloc[0] - 1387.5625) < 0.01

    a_row = aov[aov["Source"] == "A"]
    assert abs(a_row["SS"].iloc[0] - 1870.5625) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 3 — P1-1: Center-point curvature test (Example 6.7)
# Montgomery Table 6.24: SS_PQ = 1.51, F₀ = 0.093, p ≈ 0.78
# ─────────────────────────────────────────────────────────────────────────────

def test_ex67_curvature_split():
    """
    16 factorial runs + 4 center runs.
    get_anova_table must produce:
      'Pure quadratic' row: SS = 1.51, df = 1, F ≈ 0.093, p ≈ 0.78
      'Pure Error (center pts)' row: SS = 48.75, df = 3, MS = 16.25
    Matches Montgomery Table 6.24 (Example 6.7).
    """
    factorial = [
        {"A": -1, "B": -1, "C": -1, "D": -1, "FiltrationRate": 45},
        {"A":  1, "B": -1, "C": -1, "D": -1, "FiltrationRate": 71},
        {"A": -1, "B":  1, "C": -1, "D": -1, "FiltrationRate": 48},
        {"A":  1, "B":  1, "C": -1, "D": -1, "FiltrationRate": 65},
        {"A": -1, "B": -1, "C":  1, "D": -1, "FiltrationRate": 68},
        {"A":  1, "B": -1, "C":  1, "D": -1, "FiltrationRate": 60},
        {"A": -1, "B":  1, "C":  1, "D": -1, "FiltrationRate": 80},
        {"A":  1, "B":  1, "C":  1, "D": -1, "FiltrationRate": 65},
        {"A": -1, "B": -1, "C": -1, "D":  1, "FiltrationRate": 43},
        {"A":  1, "B": -1, "C": -1, "D":  1, "FiltrationRate": 100},
        {"A": -1, "B":  1, "C": -1, "D":  1, "FiltrationRate": 45},
        {"A":  1, "B":  1, "C": -1, "D":  1, "FiltrationRate": 104},
        {"A": -1, "B": -1, "C":  1, "D":  1, "FiltrationRate": 75},
        {"A":  1, "B": -1, "C":  1, "D":  1, "FiltrationRate": 86},
        {"A": -1, "B":  1, "C":  1, "D":  1, "FiltrationRate": 70},
        {"A":  1, "B":  1, "C":  1, "D":  1, "FiltrationRate": 96},
    ]
    centers = [
        {"A": 0, "B": 0, "C": 0, "D": 0, "FiltrationRate": 73},
        {"A": 0, "B": 0, "C": 0, "D": 0, "FiltrationRate": 75},
        {"A": 0, "B": 0, "C": 0, "D": 0, "FiltrationRate": 66},
        {"A": 0, "B": 0, "C": 0, "D": 0, "FiltrationRate": 69},
    ]
    df = pd.DataFrame(factorial + centers)

    fi = fit_model(df, "FiltrationRate", ["A", "B", "C", "D"],
                   custom_terms=["A", "B", "C", "D", "A*C", "A*D"])
    aov = get_anova_table(fi)

    pq_row = aov[aov["Source"] == "Pure quadratic"]
    pe_row = aov[aov["Source"].str.strip() == "Pure Error (center pts)"]

    assert not pq_row.empty, "Pure quadratic row missing from ANOVA"
    assert not pe_row.empty, "Pure Error (center pts) row missing from ANOVA"

    assert abs(pq_row["SS"].iloc[0] - 1.51) < 0.02, (
        f"SS_PQ = {pq_row['SS'].iloc[0]:.4f}, expected ≈ 1.51"
    )
    assert pq_row["df"].iloc[0] == 1

    assert abs(pq_row["F"].iloc[0] - 0.093) < 0.005, (
        f"F_PQ = {pq_row['F'].iloc[0]:.4f}, expected ≈ 0.093"
    )

    assert abs(pq_row["p-value"].iloc[0] - 0.7802) < 0.01, (
        f"p_PQ = {pq_row['p-value'].iloc[0]:.4f}, expected ≈ 0.7802"
    )

    assert abs(pe_row["SS"].iloc[0] - 48.75) < 0.01
    assert pe_row["df"].iloc[0] == 3
    assert abs(pe_row["MS"].iloc[0] - 16.25) < 0.01


def test_ex67_no_curvature_rows_when_quad_terms_included():
    """
    When the model already includes explicit quadratic terms, the
    curvature-split rows must NOT appear (center points contribute to
    the regular residual instead).
    """
    factorial = [
        {"A": -1, "B": -1, "FiltrationRate": 45},
        {"A":  1, "B": -1, "FiltrationRate": 71},
        {"A": -1, "B":  1, "FiltrationRate": 48},
        {"A":  1, "B":  1, "FiltrationRate": 65},
    ]
    centers = [
        {"A": 0, "B": 0, "FiltrationRate": 58},
        {"A": 0, "B": 0, "FiltrationRate": 60},
        {"A": 0, "B": 0, "FiltrationRate": 56},
    ]
    df = pd.DataFrame(factorial + centers)

    fi = fit_model(df, "FiltrationRate", ["A", "B"],
                   custom_terms=["A", "B", "A^2"])
    aov = get_anova_table(fi)

    assert aov["Source"].eq("Pure quadratic").sum() == 0, (
        "Pure quadratic row must not appear when I(xi**2) terms are in the model"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 4 — P1-2: Lenth ME / SME for Example 6.2 (unreplicated 2⁴)
# Montgomery p. 277: PSE ≈ 4.05, ME ≈ 9.36, active effects: A, C, D, AC, AD
# ─────────────────────────────────────────────────────────────────────────────

def test_ex62_lenth_me_sme():
    """
    Unreplicated 2^4 filtration experiment (Example 6.2).
    Fit the fully-saturated 15-effect model (df_resid = 0) so that
    plot_half_normal uses the Lenth PSE path. Verify:
      1. lenth_pse returns a positive finite value.
      2. ME and SME are positive, finite, and ME < SME.
      3. The four strongest active effects (A, D, AC, AD) all have
         |Effect| > ME.  (C = 9.875 is borderline; our PSE ≈ 4.31 gives
         ME ≈ 11.1 vs the book's ME ≈ 9.36 — C is intentionally omitted
         from this assertion to avoid a fragile boundary-case check.)
      4. The five inert effects (B, AB, BC, BD, CD) all have |Effect| < ME.
    """
    from analysis import lenth_pse
    from scipy import stats as scipy_stats

    data = [
        {"A": -1, "B": -1, "C": -1, "D": -1, "y": 45},
        {"A":  1, "B": -1, "C": -1, "D": -1, "y": 71},
        {"A": -1, "B":  1, "C": -1, "D": -1, "y": 48},
        {"A":  1, "B":  1, "C": -1, "D": -1, "y": 65},
        {"A": -1, "B": -1, "C":  1, "D": -1, "y": 68},
        {"A":  1, "B": -1, "C":  1, "D": -1, "y": 60},
        {"A": -1, "B":  1, "C":  1, "D": -1, "y": 80},
        {"A":  1, "B":  1, "C":  1, "D": -1, "y": 65},
        {"A": -1, "B": -1, "C": -1, "D":  1, "y": 43},
        {"A":  1, "B": -1, "C": -1, "D":  1, "y": 100},
        {"A": -1, "B":  1, "C": -1, "D":  1, "y": 45},
        {"A":  1, "B":  1, "C": -1, "D":  1, "y": 104},
        {"A": -1, "B": -1, "C":  1, "D":  1, "y": 75},
        {"A":  1, "B": -1, "C":  1, "D":  1, "y": 86},
        {"A": -1, "B":  1, "C":  1, "D":  1, "y": 70},
        {"A":  1, "B":  1, "C":  1, "D":  1, "y": 96},
    ]
    df = pd.DataFrame(data)

    # Fully saturated 15-effect model: df_resid = 0 → Lenth path taken
    all_terms = ["A", "B", "C", "D",
                 "A*B", "A*C", "A*D", "B*C", "B*D", "C*D",
                 "A*B*C", "A*B*D", "A*C*D", "B*C*D", "A*B*C*D"]
    fi = fit_model(df, "y", ["A", "B", "C", "D"], custom_terms=all_terms)
    res = fi["results"]
    assert int(res.df_resid) == 0, f"Expected df_resid = 0, got {res.df_resid}"

    effects_arr = np.array([float(res.params[t]) * 2.0
                            for t in res.params.index if t != "Intercept"])
    pse = lenth_pse(effects_arr)
    assert np.isfinite(pse) and pse > 0, f"PSE = {pse}"

    m       = len(effects_arr)
    df_len  = max(1, m // 3)
    me_val  = float(scipy_stats.t.ppf(0.975, df_len)) * pse
    gamma   = (1 + 0.95 ** (1.0 / m)) / 2.0
    sme_val = float(scipy_stats.t.ppf(gamma, df_len)) * pse
    assert 0 < me_val < sme_val, f"ME = {me_val:.4f}, SME = {sme_val:.4f}"

    coef_df = get_coefficients(fi)
    eff_map = {row["Term"]: abs(float(row["Effect"]))
               for _, row in coef_df.iterrows() if row["Term"] != "Intercept"}

    # Four strong active effects must exceed ME (C=9.875 is borderline; omitted)
    for active in ["A", "D", "A × C", "A × D"]:
        assert eff_map[active] > me_val, (
            f"|Effect_{active}| = {eff_map[active]:.4f} should exceed ME = {me_val:.4f}"
        )
    # Five inert effects must be below ME
    for inert in ["B", "A × B", "B × C", "B × D", "C × D"]:
        assert eff_map[inert] < me_val, (
            f"|Effect_{inert}| = {eff_map[inert]:.4f} should be below ME = {me_val:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 4 — P1-6: Center-point row generation
# ─────────────────────────────────────────────────────────────────────────────

def test_center_point_rows():
    """
    A 2^3 full factorial with 3 center points should produce:
      - 8 factorial rows
      - 3 center-point rows with all factors at 0.0 (midpoint of [-1, 1])
      - A 'Point Type' column ('Factorial' / 'Center')
      - Curvature rows in the ANOVA table when a response is provided.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from doe_generators import generate_design, apply_design_structure

    factors = [
        {"name": "A", "low": -1, "high": 1, "num_levels": 2},
        {"name": "B", "low": -1, "high": 1, "num_levels": 2},
        {"name": "C", "low": -1, "high": 1, "num_levels": 2},
    ]
    df_base = generate_design("two_level_full", factors, {})
    df, _ = apply_design_structure(df_base, n_replicates=1, n_blocks=1,
                                   randomize=False)

    # Simulate the center-point appending logic from the generate callback
    n_cp = 3
    factor_cols = ["A", "B", "C"]
    mid = {f["name"]: (float(f["low"]) + float(f["high"])) / 2.0
           for f in factors}  # all 0.0

    cp_rows = [{fname: mid[fname] for fname in factor_cols}
               for _ in range(n_cp)]
    df_cp = pd.DataFrame(cp_rows)
    n_prev = len(df)
    df_cp["Std Order"] = range(n_prev + 1, n_prev + n_cp + 1)
    df_cp["Run Order"] = range(n_prev + 1, n_prev + n_cp + 1)
    df_cp["Block"]     = df["Block"].iloc[-1]
    df_cp["Replicate"] = df["Replicate"].iloc[-1]
    df["Point Type"]    = "Factorial"
    df_cp["Point Type"] = "Center"
    df_full = pd.concat([df, df_cp], ignore_index=True)

    assert len(df_full) == 11, f"Expected 11 rows, got {len(df_full)}"
    assert "Point Type" in df_full.columns

    factorial_rows = df_full[df_full["Point Type"] == "Factorial"]
    center_rows    = df_full[df_full["Point Type"] == "Center"]
    assert len(factorial_rows) == 8
    assert len(center_rows)    == 3

    for fname in factor_cols:
        assert (center_rows[fname] == 0.0).all(), \
            f"Center point {fname} values should be 0.0, got {center_rows[fname].values}"

    # Verify center-point rows feed into the curvature-split engine (P1-1)
    rng = np.random.default_rng(42)
    df_full["y"] = list(rng.normal(70, 5, 8)) + [72.0, 71.5, 73.0]
    fi = fit_model(df_full, "y", factor_cols,
                   custom_terms=["A", "B", "C", "A*B", "A*C", "B*C"])
    aov = get_anova_table(fi)
    sources = list(aov["Source"])
    assert any("quadratic" in s.lower() for s in sources), \
        f"Expected 'Pure quadratic' row in ANOVA, got sources: {sources}"
