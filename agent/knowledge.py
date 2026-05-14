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

DESIGN_REVIEW_GUIDELINES = """
EXPERIMENTAL DESIGN REVIEW (NIST Handbook of Statistical Methods):

APPROPRIATENESS:
- Two-level full factorial (2^k): ideal for k=2–4 factors when interactions must be quantified
- Fractional factorial (Res III–V): use for k≥5 factors or limited budget; note aliasing trade-offs
- Plackett-Burman: purely for screening ≥5 factors; main effects only, interactions fully confounded
- CCD (Central Composite): for optimisation when quadratic curvature is expected; needs 3+ levels per factor
- Box-Behnken: alternative RSM when extreme corners of the design space are infeasible

RUN COUNT / POWER:
- Error degrees of freedom ≥ 6 recommended for a reliable MSE estimate
- Fewer than 3 df_error makes F-tests unreliable; flag if df_error < 6
- Center points add error df without inflating factor runs — valuable in 2-level designs

BLOCKING AND REPLICATION:
- Blocking removes known nuisance variation (batches, operators, days) from the error term
- True replication (independent re-randomisation) is required for a valid pure-error estimate
- Multiple blocks increase precision if block effects are substantial
- Center points: 3–5 recommended per block for curvature detection

ERROR DEGREES OF FREEDOM FORMULA:
  df_error = total_runs - df_model - 1 (intercept) - df_block
"""

ANOVA_REVIEW_GUIDELINES = """
ANOVA AND MODEL FIT INTERPRETATION (Montgomery, Design and Analysis of Experiments):

SIGNIFICANCE THRESHOLDS:
- Overall model p < 0.05: model explains statistically significant variation
- Individual term p < 0.05: that effect is active at the 5% level
- Lack-of-Fit p < 0.10: model form may be inadequate; consider adding terms
- Lack-of-Fit p ≥ 0.10: no evidence of systematic misfit

MODEL FIT STATISTICS:
- R² > 0.90: good fit for engineering/manufacturing purposes
- Adj R² close to R²: gap > 0.10 suggests too many terms relative to runs (overfitting)
- Pred R² > 0.70: model will generalise to new observations within the design space
- Pred R² vs Adj R² gap > 0.20: likely overfitting or an influential outlier — inspect residuals
- RMSE (S): the estimated standard deviation of experimental error in response units

NEXT-STEP RULES:
- Good fit + LOF not significant → proceed to optimisation
- All terms insignificant → widen factor ranges or add runs to increase power
- LOF significant → add curvature terms (CCD) or additional center-point replicates
- Large Adj R²/Pred R² gap → remove non-significant terms; check for outliers
"""

EFFECTS_REVIEW_GUIDELINES = """
EFFECTS AND INTERACTION ANALYSIS (Montgomery, Design and Analysis of Experiments):

PARETO CHART:
- Bars represent standardised effects (|t-value| or |effect estimate|) sorted largest to smallest
- Bonferroni-corrected line: effects above this line are significant at experiment-wide α=0.05
- t-value reference line: effects above this but below Bonferroni are potentially active
- Effects below both lines: likely noise

HALF-NORMAL PLOT:
- Active effects appear as points deviating above the reference line through the origin
- Points following the line closely are noise effects
- A clear separation between noise cloud and outlying points confirms active factors
- Multiple outlying points may indicate interactions or quadratic effects

MAIN EFFECTS PLOT:
- Steep slope = large effect; flat line = negligible effect
- Slope direction shows whether increasing the factor increases or decreases the response
- Parallel lines in main effects (with non-parallel interaction lines) indicate masked effects

INTERACTION PLOT:
- Parallel lines = no interaction (interpret main effects independently)
- Crossing or diverging lines = significant interaction (interpret effects jointly)
- The greater the non-parallelism, the stronger the interaction
- Always check interaction plots before drawing conclusions from main effects alone

PRACTICAL VS STATISTICAL SIGNIFICANCE:
- A statistically significant effect may be practically unimportant if the magnitude is small
- Always interpret effect size in the context of engineering tolerances or process targets
"""

RESIDUAL_REVIEW_GUIDELINES = """
RESIDUAL DIAGNOSTICS (NIST Engineering Statistics Handbook, Chapter 4):

NORMAL PROBABILITY PLOT (Q-Q):
- Points should follow the diagonal reference line closely
- S-shaped curve → heavy tails or bimodal distribution (possible outliers)
- Systematic curve → skewed distribution; transformation of response may help
- A few isolated points far from the line → outliers worth investigating

RESIDUALS vs FITTED VALUES:
- Expected pattern: random horizontal scatter around zero with constant width
- Fan/funnel shape (variance increases with fitted value) → heteroscedasticity; try log or sqrt transform
- Curved pattern → missing quadratic term; consider augmenting to CCD
- Systematic band structure → possible discrete or bounded response variable

RESIDUALS vs RUN ORDER:
- Expected pattern: random scatter with no trend
- Upward or downward drift → lurking time-related variable (machine warm-up, reagent aging, operator learning)
- Cyclical pattern → periodic environmental effect; consider adding a blocking variable

OUTLIERS:
- Standardised residuals outside ±3 are potential outliers
- Investigate cause before removing: recording error, unusual condition, or genuine process event

SHAPIRO-WILK TEST:
- p ≥ 0.05: insufficient evidence to reject normality (residuals consistent with normal distribution)
- p < 0.05: normality rejected; examine Q-Q plot for the nature of departure
- Note: with small n (<10), the test has low power; use the plot primarily
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
