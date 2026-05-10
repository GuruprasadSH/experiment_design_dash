# DOE Assistant

A Plotly Dash application combining a full-featured statistical DOE engine (nine design types, ANOVA, response-surface optimisation) with a Claude-powered AI consultant. The AI interviews you about your experimental goals, recommends the right design using NIST Statistical Handbook rules, automatically configures the app, and interprets ANOVA results in plain English.

---

## Architecture

```
User -> Agent Interview -> Design Config -> Dash App -> Data Entry -> Agent Interpretation
```

Detailed flow:

```
User
 |
 +-- AI Assistant tab --> Interviewer (claude-sonnet-4-6 + NIST rules)
 |                            |
 |                      Design recommendation
 |                            |
 |                     Configure App button
 |                            |
 v                            v
Dash App <---- /agent/configure  <---- agent/app_controller.py
 |
 +-- Design tab       (generate design matrix)
 +-- Analysis tab     (Fit Model -> Interpret Results)
 |        |
 |   Interpreter (claude-sonnet-4-6 + ANOVA table)
 |        |
 |   Rule-based next-step recommendation (no API call)
 |
 +-- Prediction & Optimization tab
```

---

## Setup

```bash
git clone <repo-url>
cd doe-assistant
pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your keys (see .env.example for variable names)

python DOE/app.py
# Open http://127.0.0.1:8050
```

---

## Two modes of use

### Agentic mode (AI Assistant tab first)

1. Go to the **AI Assistant** tab.
2. Answer 7 questions about your process, factors, and run budget.
3. Receive a NIST-grounded design recommendation with justification.
4. Click **Configure App from Recommendation** to auto-populate the Design tab.
5. Generate the matrix, run experiments, enter response values in **Analysis**.
6. Fit the model, then click **Interpret Results** for plain-English ANOVA summary.

### Manual mode (Design tab first)

- Pick a design type in the left sidebar (Factorial, Screening, RSM, Taguchi, Mixture).
- Define factors and structural options (replicates, blocks, randomise).
- Generate the design, enter responses, fit the model, optimise.

The floating **?** button (bottom-right) opens a context-aware help panel
powered by `claude-haiku-4-5` on every tab.

---

## Sample conversation

```
Assistant: What process are you trying to improve, and what will you measure?

User:      Reaction yield (%). We control Temperature (150-200 C),
           Pressure (10-30 bar), and Catalyst loading (1-5 g/L).

Assistant: How many experimental runs can you afford?

User:      About 16 runs.

...(7 questions)...

Assistant: I recommend a Resolution IV fractional factorial (2^3-1) with
           2 center points — 10 runs total. This estimates all main effects
           independently and detects curvature without committing to a full
           RSM. If you see significant curvature, augment to a CCD.
           [Configure App from Recommendation button appears]
```

---

## Repository layout

```
doe-assistant/
├── DOE/
│   ├── analysis.py        <- statistical engine (unchanged)
│   ├── app.py             <- Dash app + agent bridge
│   ├── doe_generators.py  <- design generators (unchanged)
│   └── tests/             <- existing test suite
├── agent/
│   ├── knowledge.py       <- hardcoded NIST rules (no RAG)
│   ├── interviewer.py     <- 7-question interview loop
│   ├── interpreter.py     <- ANOVA plain-English interpretation
│   ├── recommender.py     <- rule-based next-step logic
│   └── app_controller.py  <- HTTP wrapper for Flask endpoints
├── rag/                   <- retained for reproducibility; not used by agent
├── ingest/                <- NIST handbook scrape + embed pipeline
├── data/                  <- curated NIST chunks + eval set
├── scripts/
│   └── test_api_bridge.py <- Sprint 1 verification script
├── .env.example
└── requirements.txt
```

---

## Security

- **Never commit `.env`** — it is in `.gitignore`.
- **Never commit any PDF** — `.gitignore` blocks `*.pdf`.
- Before every push, verify:

  ```bash
  git grep -r "ANTHROPIC_API_KEY=[^=]"    # must return only .env.example blank line
  git grep -r '.pdf'                       # must return empty (gitignore excluded)
  ```

- `.env.example` contains only blank placeholder values.
- If an API key is accidentally staged, run `git reset HEAD` immediately
  and do **not** push.
