# Design Tab

The Design tab is where every DOE starts. You define what you want to test, and the app generates the randomised experimental matrix you hand to the lab. This page walks you through every control on the tab from top to bottom, then explains the AI Design Assistant panel in detail.

---

## Overview of the Design Tab Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Design Type                                                  │
│  [Central Composite Design ▼]   [CCC ▼]  [Rotatable ▼]       │
├──────────────────────────────────────────────────────────────┤
│  Factors                                                      │
│  [ + Add Factor ]                                             │
│  ┌──────────┬──────────┬──────────┬──────────┬─────────────┐ │
│  │ Name     │ Type     │ Low      │ High     │ Units       │ │
│  ├──────────┼──────────┼──────────┼──────────┼─────────────┤ │
│  │ Temp     │ Numeric  │ 80       │ 120      │ °C          │ │
│  │ Pressure │ Numeric  │ 1        │ 5        │ bar         │ │
│  └──────────┴──────────┴──────────┴──────────┴─────────────┘ │
├──────────────────────────────────────────────────────────────┤
│  Design Options                                               │
│  Center points: [5]  Replicates: [1]  Blocks: [1]            │
│  [✓] Randomise run order                                      │
├──────────────────────────────────────────────────────────────┤
│  [ Generate Design ]   Estimated runs: 13                    │
├──────────────────────────────────────────────────────────────┤
│  Design Matrix                                                │
│  [ Download CSV ]  [ Copy to Clipboard ]                      │
│  ┌────────┬──────────┬────────────┬──────┬──────────┐        │
│  │ Std    │ Run      │ Point Type │ Temp │ Pressure │        │
│  │ Order  │ Order    │            │      │          │        │
│  ├────────┼──────────┼────────────┼──────┼──────────┤        │
│  │ ...    │ ...      │ ...        │ ...  │ ...      │        │
│  └────────┴──────────┴────────────┴──────┴──────────┘        │
├──────────────────────────────────────────────────────────────┤
│  ▼ AI Design Assistant                                        │
│  [Chat interface]                                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Step 1 — Select a Design Type

At the top of the Design tab you will find the **Design Type** dropdown. It lists all nine design families. Selecting a design type may reveal additional sub-type and option dropdowns to the right.

| Design type | Additional dropdowns |
|---|---|
| Plackett-Burman | N (run count) |
| Two-level Full Factorial | — |
| Two-level Fractional Factorial | Resolution (III/IV/V) |
| Central Composite | Sub-type (CCC/CCI/CCF), α (rotatable/orthogonal/face-centred/custom) |
| Box-Behnken | — |
| General Factorial | — |
| Taguchi OA | Array (L4–L18) |
| Simplex Lattice | Degree (1/2/3) |
| Simplex Centroid | Augment with axials (Yes/No) |

If you are unsure which design to select, skip to the [AI Design Assistant](#ai-design-assistant-panel) section of this page, or read [Design Types](Design-Types) for a full comparison.

---

## Step 2 — Define Your Factors

### Adding factors

Click **+ Add Factor** to add a row to the factor table. Each row has the following fields:

| Column | Description |
|---|---|
| **Name** | The factor label that appears in the design matrix and all plots. Keep it short (≤ 20 characters). |
| **Type** | **Numeric** (continuous variable with low and high levels) or **Categorical** (discrete options). |
| **Low** | The low level (−1 in coded units) for numeric factors. For categorical factors, this column is disabled. |
| **High** | The high level (+1 in coded units) for numeric factors. |
| **Units** | Optional label shown in axis titles (e.g., °C, bar, rpm). |
| **Levels** | For categorical factors only: enter the level names as a comma-separated list (e.g., `A, B, C`). |

### Removing factors

Click the **×** icon at the right end of any factor row to remove it.

### Tips for defining factors

> **Tip:** Set your low and high levels to span the full range of practical variation — not just a comfortable middle range. A wide range gives more statistical power to detect effects.

> **Warning:** For mixture designs (Simplex Lattice, Simplex Centroid), the Low and High fields represent minimum and maximum proportions. The app checks that proportions are ≥ 0 and warns if the sum of minimums exceeds 1.

### Numeric vs categorical factors

Numeric factors are treated as continuous in the model. The model can estimate linear, quadratic, and interaction terms involving numeric factors.

Categorical factors are entered as indicator (dummy) variables in the regression model. A factor with L levels generates L − 1 indicator columns. The app handles this automatically.

> **Note:** Taguchi OA and General Factorial designs support mixed-level factors (some at 2 levels, others at 3 levels). For CCD and Box-Behnken, all factors must be numeric.

---

## Step 3 — Set Design Options

The **Design Options** panel appears below the factor table. The options shown depend on the selected design type.

### Center points

Center points are runs where all numeric factors are set to their midpoint ((Low + High) / 2). In coded units this is 0.

- **Purpose:** Detect curvature (non-linear response) and provide pure error degrees of freedom.
- **NIST recommendation:** 3–5 center points for full and fractional factorials; use the formula-based default for CCD.
- **Mixture designs:** Center points correspond to the overall centroid (equal proportions of all components).

### Replicates

Whole-design replicates duplicate the entire design (including randomisation). Setting Replicates = 2 doubles the run count.

- **Purpose:** Increase power to detect small effects; provide more pure error df.
- **When to use:** When you expect the effect size to be small relative to measurement noise.

> **Tip:** Whole-design replication is different from center-point replication. Center-point replicates are cheaper (fewer runs) and still provide pure error df, but they only cover the centre of the design space.

### Blocks

Blocking divides the design into groups of runs that are performed under the same conditions of a nuisance variable (e.g., same day, same batch, same operator). The block variable is included in the ANOVA as a fixed effect, removing its contribution from the error term.

- **Number of blocks:** Must divide evenly into the total run count.
- **Block size:** (Total runs) / (Number of blocks).
- **What the app does:** Assigns a **Block** column to the design matrix. The analysis tab will automatically include the block term in the model.

### Randomise run order

When checked (default), the app generates a random permutation of the standard order and fills the Run Order column. Running experiments in random order protects against lurking time trends (equipment drift, ambient temperature change, etc.).

> **Warning:** Never skip randomisation in a real experiment unless there is a strong physical reason (e.g., a factor is expensive to change and a split-plot design is being used deliberately).

---

## Step 4 — Generate the Design

Click **Generate Design**. The app:

1. Calls the appropriate generator in `doe_generators.py` (using pyDOE3 or custom code for Taguchi/simplex designs).
2. Applies your factor levels to decode the coded values into actual units.
3. Assigns random run order (if selected).
4. Assigns block labels (if blocks > 1).
5. Adds the Point Type column.
6. Renders the design matrix table.

The **Estimated runs** counter above the button updates as you change options, before you click Generate.

---

## Step 5 — Reading the Design Matrix

The generated table has the following columns:

| Column | Description |
|---|---|
| **Std Order** | The standard (Yates) order — the canonical ordering of runs before randomisation. Used as a unique row identifier. |
| **Run Order** | The sequence in which you should perform the experiments. This is the randomised order. Always run in Run Order, not Std Order. |
| **Block** | Which block this run belongs to. Only present if blocks > 1. |
| **Point Type** | Describes the type of design point: `Factorial`, `Axial` (for CCD star points), or `Center`. Used by the Analysis tab to assign the correct badges and check center-point count. |
| **Factor columns** | One column per factor, showing the actual (decoded) setting. For numeric factors these are in your original units. For categorical factors these are the level labels. |

### Coded vs actual values

The design matrix displays values in **actual units** (e.g., Temperature = 80 °C, 100 °C, 120 °C). The statistical model in the Analysis tab works with **coded values** (−1, 0, +1) for numeric factors. The transformation is:

```
coded = 2 × (actual − midpoint) / (high − low)
```

where `midpoint = (high + low) / 2`.

This is handled automatically — you enter actual values in the data table on the Analysis tab.

---

## Step 6 — Download or Copy the Design

Two buttons appear above the design matrix table:

- **Download CSV:** Saves the design matrix as a comma-separated file. You can open this in Excel and distribute it to the lab.
- **Copy to Clipboard:** Copies the table as tab-separated text. Paste directly into Excel or a LIMS.

> **Tip:** Before you send the design to the lab, print or export the table with the Run Order column visible, and highlight that column in red or yellow. Operators must follow the Run Order column, not the Std Order column.

---

## AI Design Assistant Panel

The AI Design Assistant is a collapsible panel at the bottom of the Design tab. It uses **claude-sonnet-4-6** to conduct a structured interview and recommend the most appropriate design for your experiment.

### How to open the panel

Click the **▼ AI Design Assistant** header. The panel expands to reveal a chat interface with an initial greeting from the assistant.

### The 7-question interview flow

The assistant asks up to seven questions. You do not need to answer them in sequence — the assistant adapts based on your answers. The questions it covers are:

1. **Goal:** Are you screening (finding important factors), characterising (understanding interactions), or optimising (finding the best settings)?
2. **Factors:** How many factors do you have, and what type are they (continuous or categorical)?
3. **Factor ranges:** What are the practical low and high settings for each factor?
4. **Run budget:** How many experiments can you afford to run?
5. **Prior knowledge:** Do you expect interactions between factors? Do you suspect non-linear (curved) responses?
6. **Constraints:** Are there any factor combinations that are impossible or dangerous to test?
7. **Mixture constraint:** Do your factor values need to sum to a constant (mixture experiment)?

You can describe your experiment in plain language — for example: *"I'm formulating a coating and want to know which of 6 ingredients most affects viscosity. I can run about 20 experiments."* The assistant extracts the structured information from your description.

### Reading the recommendation

After gathering enough information, the assistant produces a recommendation in this format:

```
Recommended design: Two-level Fractional Factorial (2^(6-2)), Resolution IV

Reasoning:
• With 6 factors and a budget of 20 runs, a 2^(6-2) = 16-run design fits your budget.
• Resolution IV ensures all main effects are free from 2FI aliasing.
• Your prior knowledge suggests interactions are possible, so Resolution III would be risky.
• Adding 4 center points (total 20 runs) will detect curvature if present.

Settings to apply:
  Design type: Two-level Fractional Factorial
  k = 6, p = 2
  Center points: 4
  Randomise: Yes

⚠ Note: If the center-point curvature test is significant after analysis,
augment this design to a CCD to characterise the response surface.
```

The recommendation includes the design type, key parameters, and the reasoning so you can evaluate whether it fits your situation.

### Apply Recommended Design button

After the assistant makes a recommendation, an **Apply Recommended Design** button appears below the chat. Clicking it:

1. Sets the Design Type dropdown to the recommended type.
2. Fills in the sub-type, resolution, and other options.
3. Populates the factor table with any factor names and ranges you mentioned.
4. Updates the center-point count.

You can then review the pre-filled settings, adjust anything that does not match your intent, and click **Generate Design**.

> **Tip:** You can have a back-and-forth conversation with the assistant before accepting the recommendation. For example, you can say "I can actually run 32 experiments" and it will revise its recommendation.

### When the AI Design Assistant does not apply

- For **mixture experiments** (Simplex designs), the assistant will recommend the design but cannot automatically set proportions for constrained components — enter those manually in the factor table.
- For **Taguchi designs**, the assistant will recommend the array but column assignment is done automatically by the app.
- The assistant does not have access to your specific domain (e.g., it does not know that Temperature above 150 °C will damage your sample). Provide constraints explicitly in the chat.

---

## After Generating the Design

Once you have a design matrix:

1. Export it (Download CSV or Copy to Clipboard).
2. Perform the experiments in Run Order.
3. Record the response values.
4. Return to the app and switch to the **Analysis tab** to enter your results.

See [Analysis Tab](Analysis-Tab) for full instructions on loading data and fitting a model.
