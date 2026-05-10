"""
Curated NIST Statistical Handbook decision rules for the DoE agent.
Hardcoded intentionally — RAG retrieval not used in the agent pipeline.
"""

DESIGN_SELECTION_RULES = """
DESIGN SELECTION (from NIST Handbook of Statistical Methods):

SCREENING (5+ factors, limited budget):
- Plackett-Burman (N=12,20,24): screens up to N-1 factors in N runs; main effects only
- Resolution III fractional factorial: main effects estimable but aliased with 2FI
- Goal is to identify the vital few factors for follow-up study

TWO-LEVEL FACTORIAL (2-4 factors, interactions matter):
- Full 2^k: use when run budget allows and all interactions are needed
  k=2 → 4 runs, k=3 → 8 runs, k=4 → 16 runs, k=5 → 32 runs
- Resolution IV: main effects clear of 2FI; 2FI aliased with other 2FI
- Resolution V: all main effects and 2FI independently estimable

RESPONSE SURFACE (optimisation; use after screening/factorial):
- CCD (Central Composite Design): adds axial and center points to an existing factorial
  Total runs: 2^k + 2k + n_c (n_c = 4-6 center points recommended)
- Box-Behnken: three-level design; no corner points (useful when corners are infeasible)
- Use RSM when center-point test shows significant curvature OR when you need to find optimum

BLOCKING:
- Use when an uncontrollable nuisance factor exists (different operators, batches, days)
- Each block is one complete or partial replicate
- Block effects are estimated separately; they do not inflate factor error

REPLICATION vs REPETITION:
- Replication = complete re-randomisation of all factor settings → independent error estimate
- Repetition = multiple measurements at same run without re-randomising → NOT independent
- Only true replication gives a valid estimate of pure experimental error

CENTER POINTS:
- Add 3-5 center points to any 2-level factorial
- Purpose: detect curvature without committing to full RSM
- If center-point F-test is significant: curvature present, move to CCD or Box-Behnken

RESOLUTION GUIDE:
- Resolution III: fast screening; interactions confounded with main effects
- Resolution IV: good for 5-7 factors; some 2FI estimable with follow-up
- Resolution V: when 2FI are critical and budget allows
"""

ANALYSIS_INTERPRETATION_RULES = """
ANALYSIS INTERPRETATION (Montgomery, Design and Analysis of Experiments):

ANOVA TABLE:
- Model p < 0.05: overall model is statistically significant
- Term p < 0.05: that effect is significant at the 5% level
- Lack-of-Fit p < 0.05: model functional form may be inadequate; consider adding terms
- Lack-of-Fit p > 0.10: no evidence of misfit; model form is acceptable

MODEL FIT STATISTICS:
- R² > 0.90: good fit for most engineering/manufacturing purposes
- Adj R² vs R²: gap > 0.10 suggests overfitting (too many terms relative to runs)
- Pred R² > 0.70: model generalises to new observations
- Pred R² vs Adj R² gap > 0.20: possible overfitting or influential outlier

RESIDUAL DIAGNOSTICS:
- Normal probability plot: points should follow a straight line; S-curves indicate non-normality
- Residuals vs fitted: random scatter expected; fan shape indicates non-constant variance
- Residuals vs run order: no trend expected; trends indicate lurking time-related variables

INTERACTIONS:
- Significant A×B interaction: the effect of A changes depending on the level of B
- When a large interaction is present, do not interpret main effects of A and B in isolation
- Interaction plot: non-parallel lines confirm the interaction

NEXT STEPS DECISION TREE:
- Model significant, R² good, residuals random, no LOF → proceed to optimization
- All terms insignificant → increase factor ranges or add runs
- LOF significant → add quadratic terms (move to CCD) or collect additional center points
- Interactions significant but not estimable → augment to Resolution V or full factorial
- After finding optimum → always run 3-5 confirmatory runs at predicted optimum before implementing
"""

# Design type keys recognised by the Dash app (from app.py DESIGNS dict)
VALID_DESIGN_TYPES = [
    "two_level_full",
    "fractional",
    "plackett_burman",
    "ccd",
    "box_behnken",
    "general_factorial",
    "taguchi",
    "simplex_lattice",
    "simplex_centroid",
]
