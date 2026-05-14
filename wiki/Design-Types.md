# Design Types

DOE Assistant v2 supports nine experimental design families. This page gives you the information you need to choose the right design before you open the Design tab.

---

## Quick Comparison Table

| Design | Primary use | Min factors | Max factors | Typical runs | Estimates interactions? | Estimates curvature? |
|---|---|---|---|---|---|---|
| Plackett-Burman | Screening (many factors) | 2 | 47 | 12, 20, 24, 28, 32… | No (aliased) | No |
| Two-level Full Factorial | Screening / characterisation | 2 | 7 | 2^k (4–128) | Yes (all) | No (add centre pts) |
| Two-level Fractional Factorial | Screening (moderate factors) | 3 | 7 | 2^(k−p) | Partially (Res III–V) | No |
| Central Composite (CCC/CCI/CCF) | Response surface / optimisation | 2 | 6 | 2^k + 2k + n_c | Yes (all 2FI + quadratic) | Yes |
| Box-Behnken | Response surface / optimisation | 3 | 6 | Varies (e.g. 15 for k=3) | Yes (all 2FI + quadratic) | Yes |
| General Factorial | Characterisation (mixed levels) | 2 | Any | Product of level counts | Yes | No |
| Taguchi Orthogonal Array | Robust design / screening | 2 | 13 | L4–L18 (4–18 rows) | Limited | No |
| Simplex Lattice | Mixture optimisation | 3 | 8 | (q+m−1)! / (m! (q−1)!) | Partial | Yes (with degree ≥ 2) |
| Simplex Centroid | Mixture optimisation | 3 | 7 | 2^q − 1 | Yes (all blends) | Partial |

> **Resolution** (for fractional and two-level full designs) indicates which effects are aliased. Res III = main effects aliased with 2FIs; Res IV = main effects clear, 2FIs aliased with each other; Res V = main effects and 2FIs clear. Higher is better for characterisation; Res III is acceptable for screening when many factors are tested.

---

## 1. Plackett-Burman Design

### Purpose

Plackett-Burman (PB) designs are highly efficient screening designs for identifying the vital few factors out of many. They use the minimum number of runs to estimate all main effects under the assumption that interactions are negligible.

### When to use

- You have 5 or more factors and want to screen them quickly.
- Budget or time limits the number of experiments you can run.
- You are willing to assume that two-factor interactions (2FIs) are small compared to main effects.
- This is the first step in a sequential experimentation programme.

### Factors and run counts

PB designs exist for N = 12, 20, 24, 28, 32, 36, 40, 44, 48 runs. Each supports up to N − 1 factors at 2 levels. The most common is N = 12 (up to 11 factors).

| N (runs) | Max factors |
|---|---|
| 12 | 11 |
| 20 | 19 |
| 24 | 23 |

### Key options in the app

- **N (run count):** Select from the supported sizes. Choose the smallest N that is ≥ k + 1.
- **Replicates:** Whole-design replicates add pure error degrees of freedom.
- **Randomise run order:** Recommended to protect against time-trend confounding.

### Statistical rules (NIST/Montgomery)

- A PB design has **complex aliasing**: every main effect is partially aliased with every 2FI not involving that factor. If interactions are suspected, augment to a full or fractional factorial after screening.
- Use the half-normal plot in the Analysis tab to identify active factors. See [Analysis Tab](Analysis-Tab).
- Error df = N − k − 1. With N = 12 and k = 11, error df = 0 — you must either use fewer factors, add replicates, or fold-over.

---

## 2. Two-Level Full Factorial (2^k)

### Purpose

A full factorial at two levels (-1 and +1 for each factor) estimates every main effect and every interaction — including the highest-order interaction. It is the gold standard for characterising a system when the number of factors is manageable.

### When to use

- You have 2 to 7 factors.
- You want unaliased estimates of all interactions.
- You have enough experimental budget for 2^k runs.

### Run counts

| k (factors) | Runs | Typical use |
|---|---|---|
| 2 | 4 | Pilot study, very fast |
| 3 | 8 | Small characterisation |
| 4 | 16 | Standard characterisation |
| 5 | 32 | Large characterisation |
| 6 | 64 | Budget-intensive; consider a fraction |
| 7 | 128 | Very expensive; almost always fractionated |

### Key options in the app

- **k (number of factors):** 2–7.
- **Center points:** Add 3–5 center points to estimate curvature and provide pure error df.
- **Replicates:** Whole-design or center-point replicates.
- **Blocks:** Divide into blocks of equal size to accommodate batch effects.

### Statistical rules

- Adding 3–5 center points is recommended by Montgomery to detect curvature without adding runs.
- If the curvature test (center points vs factorial points) is significant (p < 0.05), augment to a CCD or Box-Behnken design.
- With k = 7, consider fractionating to a 2^(7−2) = 32-run Resolution V design instead.

---

## 3. Two-Level Fractional Factorial (2^{k−p})

### Purpose

A fraction of a full factorial that deliberately aliases some high-order interactions so that you can estimate main effects and selected 2FIs with fewer runs.

### When to use

- You have 4 to 7 factors and cannot afford a full factorial.
- You are willing to accept that some effects will be aliased — typically high-order interactions.
- You want Resolution IV or V so that main effects and 2FIs are protected.

### Common fractions

| k | p | Design | Runs | Resolution |
|---|---|---|---|---|
| 4 | 1 | 2^(4−1) | 8 | IV |
| 5 | 1 | 2^(5−1) | 16 | V |
| 5 | 2 | 2^(5−2) | 8 | III |
| 6 | 1 | 2^(6−1) | 32 | VI |
| 6 | 2 | 2^(6−2) | 16 | IV |
| 7 | 1 | 2^(7−1) | 64 | VII |
| 7 | 2 | 2^(7−2) | 32 | V |
| 7 | 3 | 2^(7−3) | 16 | IV |
| 7 | 4 | 2^(7−4) | 8 | III |

### Key options in the app

- **k and p:** The app lists all standard fractions. The design resolution is shown next to each option.
- **Generator:** The app uses the minimum aberration generators from the pyDOE3 library.
- **Center points, replicates, blocks:** Same options as the full factorial.

### Statistical rules

- **Resolution III:** Main effects are aliased with 2FIs. Use only for initial screening.
- **Resolution IV:** Main effects are clear; 2FIs are aliased with each other. Adequate for characterisation if you can assume some 2FIs are negligible.
- **Resolution V:** Main effects and 2FIs are both clear; only 3FIs are aliased. Preferred for characterisation.
- Always check the alias table (shown in the Design Summary accordion) before interpreting effects.

---

## 4. Central Composite Design (CCD)

### Purpose

CCD is the most widely used response surface design. It extends a two-level factorial (or fraction) with axial (star) points and center points to allow estimation of quadratic (curvature) terms. This makes it suitable for finding an optimum.

### Sub-types

| Variant | Axial points position | α value | Use case |
|---|---|---|---|
| **CCC** (circumscribed) | Outside the factorial cube | Rotatable or custom | Factor ranges can be exceeded; pure optimisation |
| **CCI** (inscribed) | Inside the factorial cube | 1/rotatable | Factor ranges are hard limits; all runs within bounds |
| **CCF** (face-centred) | On the faces of the cube | 1.0 | Precision around edges; simpler; not rotatable |

### α options

The axial distance α determines where the star points are placed relative to the factorial points.

| α type | Formula | Property |
|---|---|---|
| Rotatable | α = (2^k)^0.25 | Variance of prediction is equal at all points equidistant from centre |
| Orthogonal | Derived from N, n_f, n_c | ANOVA columns are orthogonal |
| Face-centred | α = 1.0 | All points within the original factor range |
| Custom | User-specified | Full control |

For k = 2: rotatable α ≈ 1.414; for k = 3: α ≈ 1.682; for k = 4: α ≈ 2.000.

### Run counts

| k | Factorial points (2^k) | Axial points (2k) | Center points (recommended) | Total (approx.) |
|---|---|---|---|---|
| 2 | 4 | 4 | 5 | 13 |
| 3 | 8 | 6 | 6 | 20 |
| 4 | 16 | 8 | 7 | 31 |
| 5 | 32 | 10 | 6 | 48 |
| 6 | 64 | 12 | 9 | 85 |

### Key options in the app

- **Sub-type:** CCC, CCI, or CCF.
- **k:** 2–6 factors.
- **α:** Rotatable, orthogonal, face-centred, or custom.
- **Center points:** 3–9 (defaults to the standard recommendation).
- **Half-fraction:** Use a 2^(k−1) half-fraction for the factorial portion when k ≥ 5.

### Statistical rules

- The minimum center points for a rotatable CCD are given by: n_c ≥ 4√(2^k) − 2k.
- CCC has the best statistical properties but requires the widest factor range.
- CCI is conservative — no run exceeds the stated factor limits — but variance of prediction degrades at the edges.
- CCF requires the fewest extreme settings but cannot detect pure quadratic effects as precisely as CCC.

---

## 5. Box-Behnken Design

### Purpose

Box-Behnken (BBD) is an alternative response surface design that never tests factors simultaneously at their extreme values. It is more economical than CCC for k = 3 and avoids extreme corners that may be physically impossible or dangerous.

### When to use

- You have 3 to 6 factors.
- Running experiments at corners of the design space is impractical or unsafe (e.g., extreme temperature + extreme pressure simultaneously).
- You want fewer runs than a CCC for the same k.

### Run counts

| k | Runs (approx.) |
|---|---|
| 3 | 15 |
| 4 | 27 |
| 5 | 46 |
| 6 | 54 |

### Key options in the app

- **k:** 3–6.
- **Center points:** 3–5 recommended.
- **Replicates:** Whole-design replicates.

### Statistical rules

- BBD does not estimate the intercept as precisely as CCC for k = 3, but is more efficient.
- The corner vertices of the design space are never tested, so predictions at those points have higher variance.
- For optimisation problems where the optimum is expected near the centre, BBD is often sufficient.

---

## 6. General Factorial Design

### Purpose

A general factorial design crosses all levels of all factors, producing one run per combination. Unlike two-level designs, factors can have different numbers of levels (mixed-level design).

### When to use

- One or more factors are categorical (e.g., supplier A/B/C).
- Factors have more than two levels (e.g., temperature at 80 °C / 100 °C / 120 °C).
- You want a full characterisation of a system with inherently multi-level factors.

### Run counts

The total run count is the product of all level counts. For two factors at 3 and 4 levels: 3 × 4 = 12 runs per replicate.

### Key options in the app

- **Number of factors:** Any.
- **Levels per factor:** Enter as a comma-separated list (e.g., `3, 4, 2`).
- **Factor type:** Numeric (shown as coded values) or categorical (shown as level labels).
- **Replicates:** Multiply the base design by this factor.

### Statistical rules

- A full general factorial becomes large quickly. For k = 4 factors at 3 levels each: 81 runs per replicate.
- If run count is prohibitive, consider a Taguchi orthogonal array (below) as a fractional approach.
- Main effects and interactions are estimated with full power only if all cells are balanced (equal replicates per cell).

---

## 7. Taguchi Orthogonal Arrays

### Purpose

Taguchi orthogonal arrays (OAs) are highly fractionated designs optimised for robustness studies. They allow estimation of main effects with very few runs by using strong aliasing that Taguchi argued is acceptable in practice.

### When to use

- You are conducting a robust parameter design study (signal-to-noise ratio analysis).
- You have many factors (up to 13) at 2 or 3 levels and need a very small design.
- You accept that interaction effects will be confounded with main effects.

### Available arrays in the app

| Array | Runs | Max factors | Levels |
|---|---|---|---|
| L4 | 4 | 3 | 2 |
| L8 | 8 | 7 | 2 |
| L9 | 9 | 4 | 3 |
| L12 | 12 | 11 | 2 |
| L16 | 16 | 15 | 2 |
| L18 | 18 | 8 | 2–3 mixed |

### Key options in the app

- **Array:** Select from L4 to L18.
- **Factor-to-column assignment:** The app assigns factors to columns automatically using the standard linear graph for each array.
- **Signal-to-noise ratio type:** Smaller-is-better, larger-is-better, or nominal-is-best (for robust design analysis).

### Statistical rules

- Taguchi designs use strong aliasing. Unlike Resolution III fractional factorials, the aliasing in Taguchi OAs is well-structured and published in linear graphs.
- Taguchi methods are controversial in classical statistics for their treatment of interactions. Use the resulting factor settings as a starting point, not a final optimum.
- For optimisation (not just screening), switch to a CCD or Box-Behnken after identifying the key factors.

---

## 8. Simplex Lattice Design

### Purpose

Simplex Lattice (SLD) designs are used for **mixture experiments** where the factors are component proportions that sum to 1 (100%). You use these when changing one component necessarily changes others.

### When to use

- You are formulating a product (food, pharmaceutical, chemical, alloy).
- Factors are proportions or percentages of ingredients.
- All factor values must be ≥ 0 and their sum must equal 1.

### Structure

A {q, m} simplex lattice places m+1 equally spaced points on each edge and interior of the q-component simplex. The degree m determines how many mixture-model terms can be estimated.

| q (components) | m (degree) | Runs |
|---|---|---|
| 3 | 1 | 3 |
| 3 | 2 | 6 |
| 3 | 3 | 10 |
| 4 | 2 | 10 |
| 4 | 3 | 20 |

### Key options in the app

- **q:** 3–8 components.
- **m (degree):** 1, 2, or 3. Degree 2 is the minimum for fitting a quadratic mixture model.
- **Lower bounds:** If components cannot go below a threshold (e.g., at least 10% water), set lower bounds. The app will scale the design to the constrained simplex.

### Statistical rules

- A degree-1 lattice fits only linear blending effects. Use degree ≥ 2 to detect synergistic or antagonistic blending.
- If components have lower or upper bounds, the effective simplex is smaller. The app handles the L-pseudocomponent transformation automatically.
- Augment with centroid and check-blend points for better model validation.

---

## 9. Simplex Centroid Design

### Purpose

Simplex Centroid (SCD) designs test all pure blends (single components), all binary blends, all ternary blends, and so on up to the overall centroid. They are richer than a simplex lattice at low degrees but grow rapidly with q.

### When to use

- You have 3 to 7 components.
- You want to test all possible blend combinations up to the full mixture.
- You need to estimate the full Scheffé polynomial including all cross-product terms.

### Run counts

| q | Runs (SCD) |
|---|---|
| 3 | 7 |
| 4 | 15 |
| 5 | 31 |
| 6 | 63 |
| 7 | 127 |

The run count is 2^q − 1. For q ≥ 6, an augmented simplex lattice is usually more economical.

### Key options in the app

- **q:** 3–7 components.
- **Augment with axial points:** Adds q additional runs inside the simplex for better variance properties.
- **Lower bounds:** Same L-pseudocomponent handling as the simplex lattice.

### Statistical rules

- The SCD is supported by the Scheffé canonical polynomial, which has no intercept term.
- All fitted parameters in a mixture model represent the response at the corresponding pure component or blend — they are directly interpretable.
- If any component has a lower bound > 0, the SCD is applied to the transformed components and back-transformed for display.

---

## Choosing a Design — Decision Guide

```
Do your factors sum to a constant (mixture experiment)?
  Yes → Simplex Lattice or Simplex Centroid
  No  → Continue

Is your goal screening (find important factors)?
  Yes, ≤ 4 factors  → Two-level Full Factorial (2^k)
  Yes, 5–11 factors → Plackett-Burman or Two-level Fractional Factorial
  Yes, many factors, robust design → Taguchi OA
  No  → Continue

Is your goal characterisation (understand all interactions)?
  Yes, 2–7 factors, can afford 2^k runs → Two-level Full Factorial
  Yes, 4–7 factors, need fewer runs    → Fractional Factorial (Res IV or V)
  Yes, factors have > 2 levels         → General Factorial
  No  → Continue

Is your goal optimisation (find best factor settings)?
  Yes, avoid extreme corners → Box-Behnken
  Yes, widest factor range OK → CCC (circumscribed CCD)
  Yes, factor limits are hard constraints → CCI or CCF
```

For further guidance, use the [AI Design Assistant](Design-Tab#ai-design-assistant-panel) on the Design tab — it will ask you seven questions and recommend the most appropriate design.
