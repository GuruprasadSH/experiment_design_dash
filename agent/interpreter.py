"""
Plain-English interpretation of ANOVA results using claude-sonnet-4-6.
Reads the state dict produced by the Dash fit-info-store.
"""

from anthropic import Anthropic
from agent.knowledge import ANALYSIS_INTERPRETATION_RULES


SYSTEM_PROMPT = f"""You are a Design of Experiments analysis expert.
You will receive ANOVA results from a fitted statistical model and explain them clearly.

{ANALYSIS_INTERPRETATION_RULES}

Rules for your response:
- Reference the actual numbers from the table (p-values, R², F-statistics)
- Explain what each significant effect means in practical terms for the user's process
- Flag any residual diagnostic concerns if data is provided
- End with a single clear recommendation for next steps (one of: optimize / add runs / augment design / run confirmatory experiments)
- Write for a practitioner, not a statistician — avoid jargon where possible, define it where necessary
- Do not fabricate numbers. If a field is missing, say so."""


class Interpreter:
    def __init__(self):
        self._client = Anthropic()

    def interpret(
        self,
        anova_data: list[dict],
        model_stats: dict,
        factor_cols: list[str],
        response_col: str,
        user_question: str | None = None,
    ) -> str:
        anova_str = _format_anova(anova_data)
        context = (
            f"RESPONSE VARIABLE: {response_col}\n"
            f"FACTORS: {', '.join(factor_cols)}\n\n"
            f"ANOVA TABLE:\n{anova_str}\n\n"
            f"MODEL STATISTICS:\n"
            f"  R²      = {model_stats.get('R2', 'N/A')}\n"
            f"  Adj R²  = {model_stats.get('AdjR2', 'N/A')}\n"
            f"  Pred R² = {model_stats.get('PredR2', 'N/A')}\n"
            f"  S (σ̂)  = {model_stats.get('S', 'N/A')}\n"
            f"  n runs  = {model_stats.get('n', 'N/A')}\n"
        )
        user_msg = user_question or "Please interpret these results and tell me what to do next."
        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context + "\n\n" + user_msg}],
        )
        return response.content[0].text


def _format_anova(rows: list[dict]) -> str:
    if not rows:
        return "(no ANOVA data)"
    header = f"{'Source':<25} {'SS':>10} {'df':>5} {'MS':>10} {'F':>8} {'p-value':>10}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row.get('Source', '')):<25} "
            f"{str(row.get('SS',      '')):>10} "
            f"{str(row.get('df',      '')):>5} "
            f"{str(row.get('MS',      '')):>10} "
            f"{str(row.get('F',       '')):>8} "
            f"{str(row.get('p-value', '')):>10}"
        )
    return "\n".join(lines)
