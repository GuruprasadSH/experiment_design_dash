# AI Interpretation Guide

DOE Assistant v2 uses Anthropic Claude models to generate plain-English explanations at four points in the analysis workflow and in the context-aware Help panel. This page explains exactly what each AI section does, what data is sent to Claude, which model is used and why, how to read the output, and how to act on the recommendations.

---

## Overview of AI-Assisted Sections

| Section | Location | Claude model | Input type | Key purpose |
|---|---|---|---|---|
| Design Assistant | Design tab | claude-sonnet-4-6 | Text (structured interview) | Recommend the right design |
| Design Summary | Analysis accordion, section 1 | claude-haiku-4-5 | Text (factor table + badges) | Confirm design structure, flag df issues |
| ANOVA & Model Stats | Analysis accordion, section 2 | claude-haiku-4-5 | Text (ANOVA table + R² stats) | Evaluate model significance and fit quality |
| Effects & Interaction Analysis | Analysis accordion, section 3 | claude-sonnet-4-6 | PNG images (4 charts) | Read and interpret the effect charts |
| Residual Analysis | Analysis accordion, section 4 | claude-sonnet-4-6 | PNG image (4-panel diagnostic) | Diagnose assumption violations |
| Help panel | Floating ? button | claude-haiku-4-5 | Text (context + chat history) | Answer questions about the active state |

---

## Model Routing Rationale

### Why two different models?

**claude-haiku-4-5** is a fast, cost-efficient model that is excellent at applying structured rules to structured data. Sections 1 and 2 pass tables and numbers — tasks well-suited to haiku's strengths.

**claude-sonnet-4-6** is a more capable model with vision capabilities. Sections 3 and 4 pass PNG images of charts. The model reads the visual structure of the Pareto chart, the curvature of the Q-Q plot, the crossing of interaction lines, etc. These tasks require visual reasoning that haiku cannot perform.

The Design Assistant uses sonnet because the free-form interview requires reasoning about experimental design trade-offs from natural language input — a more cognitively demanding task than rule application.

---

## Output Format

Every AI section except the Design Assistant produces output in a consistent three-part format:

```
✅ VERDICT LINE

• Bullet 1 — specific finding with a number from your data
• Bullet 2 — specific finding with a number from your data
• Bullet 3 — specific finding with a number from your data
• (up to 2 more bullets if relevant)

Next step: One sentence stating the single most important action.
```

### The verdict line

The verdict uses one of three icons:

| Icon | Meaning |
|---|---|
| ✅ | The analysis passes the key criterion for this section. Proceed with confidence. |
| ⚠️ | The analysis has a warning — something to watch or investigate, but not a blocking problem. |
| ❌ | The analysis has a significant problem that should be resolved before proceeding. |

For the ANOVA section, the verdict reflects the overall model quality (significance, R², Pred R², LOF). For the Residual section, it reflects whether the key assumptions are met.

### Bullet points

Bullets always reference specific numbers from your data (e.g., "R² = 0.947, which exceeds the 0.90 threshold" rather than "R² is high"). This means you can cross-check every AI statement against the tables and plots on screen.

### Next step sentence

The final sentence states one prioritised action. It is drawn from a fixed decision tree (described below) applied to the combination of findings in that section.

---

## What Data Is Sent to Claude

The app never sends raw file contents or session state to the API. It constructs a targeted prompt for each section.

### Design Summary prompt

```
System: You are a DOE statistics expert. Apply NIST/Montgomery rules.
        Output format: [verdict] + 3-5 bullets with numbers + "Next step:"

User: Evaluate this experimental design.
      Design type: Central Composite Design (CCC, rotatable)
      Factors: [table of factor names, types, low, high, units]
      Run count: 20
      Center points: 5
      Blocks: 1
      Error df: 8
      NIST rules: error df >= 6; center points 3-5 recommended
```

### ANOVA & Model Stats prompt

```
System: [same as above]

User: Evaluate this ANOVA table and model statistics.
      [Full ANOVA table as formatted text]
      R² = 0.947, Adj R² = 0.921, Pred R² = 0.883, RMSE = 1.24
      NIST rules:
        R² > 0.90 good
        Adj R² gap > 0.10 = overfitting
        Pred R² > 0.70 for prediction
        Model p < 0.05 significant
        LOF p < 0.10 = misfit
```

### Effects & Interaction Analysis prompt

```
System: You are a DOE statistics expert with vision capability.
        Analyse the attached chart images and provide:
        [verdict] + 3-5 bullets + "Next step:"

User: These four charts are from a DOE effects analysis.
      [coefficient table as text]
      [image: Pareto chart PNG]
      [image: half-normal plot PNG]
      [image: main effects plot PNG]
      [image: interaction plot PNG]
      
      Identify: which factors are significant, direction of effects,
      presence/absence of interactions, any unusual patterns.
```

### Residual Analysis prompt

```
System: [same as above]

User: Analyse this residual diagnostic figure.
      Shapiro-Wilk: W = 0.962, p = 0.183 (normality not rejected)
      [image: 4-panel residual figure PNG]
      
      Evaluate: normality (Q-Q), constant variance (Residuals vs Fitted),
      independence (Residuals vs Run Order), distribution shape (histogram).
```

---

## Decision Tree for Next Steps

The AI's "Next step" sentence is generated by applying the following decision tree to the combined results across all sections. You can use this tree yourself to anticipate what the AI will recommend.

```
Model p-value < 0.05?
  No  → ❌ "Consider adding more runs or broadening factor ranges."
  Yes → continue

R² > 0.90?
  No  → ⚠️  check: is the response inherently noisy? If Pred R² also low,
              "Consider adding missing terms or more runs."
  Yes → continue

Adj R² - Pred R² gap > 0.10?
  Yes → ⚠️  "Remove non-significant terms (p > 0.10) and re-fit."
  No  → continue

LOF p-value significant (< 0.10)?
  Yes → ⚠️  "Augment the design with axial or interior points to capture
              the response surface shape, or add quadratic terms."
  No  → continue

Shapiro-Wilk p < 0.05?
  Yes → ⚠️  "Apply a log or square-root transformation to the response,
              then re-fit."
  No  → continue

All checks pass?
  Pareto shows clear active factors?
    Yes → ✅ "Run 3-5 confirmatory experiments at the predicted optimum
               before implementing changes."
    No  → ⚠️  "No factors are clearly active — consider expanding factor
               ranges or adding more factors to the study."
```

### When to optimise

Proceed to the Prediction & Optimisation tab when all of the following are true:

- Model p < 0.05
- R² > 0.80 (higher preferred for optimisation)
- Pred R² > 0.70
- LOF p > 0.10 (or no LOF test possible because no replicates)
- No severe residual assumption violations

### When to add runs

Consider adding runs when:

- Error df < 6 (NIST minimum)
- LOF is significant and you suspect missing quadratic terms
- Pred R² < 0.70
- The half-normal plot shows no clear active effects (the factor ranges may be too narrow)

### When to augment the design

Augment the existing design (rather than starting fresh) when:

- You have a resolution III fractional factorial and want to de-alias main effects from 2FIs (add a fold-over)
- You have a two-level factorial and the curvature test is significant (augment to a CCD)
- You have a Plackett-Burman or Taguchi design and want to resolve aliasing for the top-ranked factors

### When to transform the response

Consider a transformation when:

- Shapiro-Wilk p < 0.05 and the Q-Q plot shows systematic deviation
- Residuals vs Fitted shows a funnel pattern (increasing variance)
- The response is a count, proportion, or ratio (Box-Cox transformations often appropriate)

Common transformations and when to use them:

| Transformation | When to use |
|---|---|
| log(y) | Response is a rate, ratio, or count; funnel pattern in residuals |
| sqrt(y) | Response is a count or area; mild funnel pattern |
| 1/y | Response is a time-to-event (cycle time, wait time) |
| arcsine(sqrt(y)) | Response is a proportion (0 to 1) |
| Box-Cox (λ) | General-purpose; the app reports the optimal λ in the ANOVA section |

### When to run confirmatory experiments

Always run 3–5 confirmatory experiments at the predicted optimum before implementing any process change. The rule is hard-coded into the AI's next-step logic for any ✅ verdict on the ANOVA section.

Confirmatory experiments verify that:

1. The model predicts the response correctly at the optimum point.
2. The process is stable under the new settings.
3. The prediction interval (shown on the Prediction tab) is acceptable for the application.

If the confirmatory results fall within the 95% prediction interval from the model, the optimisation is validated.

---

## Tips for Getting Better Interpretations

### Click Interpret after reviewing the plots yourself

The AI adds value by applying systematic rules and noticing patterns you might miss. But it works best as a check on your own reading, not as a replacement for looking at the plots. Review the Pareto chart and residual plots first, then click Interpret.

### Run all four sections before saving the report

The HTML report includes AI interpretation text only for sections where you clicked Interpret. If you want a complete report with AI commentary on every section, click Interpret in sections 1, 2, 3, and 4 before clicking Save Report as HTML.

### Use the verdicts as a checklist, not a final answer

The AI applies the same rules every time. If it gives a ⚠️ verdict on LOF, check the LOF p-value in the ANOVA table, check how many center points you have, and decide whether the warning is practically significant given your specific application.

### The AI does not know your application context

The AI knows your data (factors, levels, ANOVA table, chart images) but not your domain. It does not know that a 5% improvement in Yield is worth pursuing but a 2% improvement is not. Factor in your practical knowledge when deciding whether a "Next step" recommendation applies.

### Re-interpret after re-fitting

If you remove non-significant terms and re-fit the model, click Interpret again in each section. The AI interpretation is not automatically updated when you re-fit.

---

## Help Panel

The floating **?** button in the bottom-right corner opens a context-aware chat powered by **claude-haiku-4-5**.

**What the Help panel knows:**

- The currently active tab (Design, Analysis, or Prediction)
- Whether a model has been fitted
- The design type selected (if on the Design tab)
- The response and factor names (if a model is fitted)
- The full chat history within the current session

**Example questions you can ask:**

- "What does Pred R² mean and why is mine low?"
- "Should I use a CCC or Box-Behnken for my 4-factor optimisation?"
- "The interaction plot shows crossing lines — what should I do?"
- "My Shapiro-Wilk p-value is 0.03. Is that a problem?"
- "How do I interpret the axial points in my CCD?"

The Help panel does not perform calculations or re-fit models — it provides explanations and guidance in the context of your current session state.

---

## Cross-References

- [Analysis Tab](Analysis-Tab) — full description of the four accordion sections and Interpret buttons
- [Prediction and Optimisation](Prediction-and-Optimisation) — what to do after a ✅ ANOVA verdict
- [Troubleshooting](Troubleshooting) — what to do when specific AI Interpret buttons fail
