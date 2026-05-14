"""
Plain-English interpretation of ANOVA results using claude-sonnet-4-6.
Reads the state dict produced by the Dash fit-info-store.
"""

import json

from anthropic import Anthropic
from dotenv import load_dotenv
from agent.knowledge import (
    ANALYSIS_INTERPRETATION_RULES,
    DESIGN_REVIEW_GUIDELINES,
    ANOVA_REVIEW_GUIDELINES,
    EFFECTS_REVIEW_GUIDELINES,
    RESIDUAL_REVIEW_GUIDELINES,
)

load_dotenv(override=True)


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


    # ── per-section methods ──────────────────────────────────────────────────

    def interpret_design(self, design_data: dict) -> str:
        """Assess design quality. Text-only. Uses claude-haiku-4-5."""
        system = (
            f"{DESIGN_REVIEW_GUIDELINES}\n\n"
            "Use markdown. Start with a one-line verdict (✅ / ⚠️ / ❌ + single sentence). "
            "Then 3–5 bullet points citing specific numbers. "
            "End with one 'Next step:' sentence. No preamble."
        )
        user_msg = (
            f"Experimental design summary:\n{json.dumps(design_data, indent=2)}\n\n"
            "Assess whether this design is appropriate and adequate."
        )
        resp = self._client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text

    def interpret_anova(
        self,
        anova_data: list[dict],
        model_stats: dict,
        factor_cols: list[str],
        response_col: str,
    ) -> str:
        """Interpret ANOVA + model fit. Text-only. Uses claude-haiku-4-5."""
        system = (
            f"{ANOVA_REVIEW_GUIDELINES}\n\n"
            "Use markdown. Start with a one-line verdict (✅ / ⚠️ / ❌ + single sentence). "
            "Then 3–5 bullet points citing specific numbers from the table. "
            "End with one 'Next step:' sentence. No preamble."
        )
        anova_str = _format_anova(anova_data)
        user_msg = (
            f"Response: {response_col}   Factors: {', '.join(factor_cols)}\n\n"
            f"ANOVA TABLE:\n{anova_str}\n\n"
            f"MODEL STATISTICS:\n"
            f"  R²      = {model_stats.get('R2', 'N/A')}\n"
            f"  Adj R²  = {model_stats.get('AdjR2', 'N/A')}\n"
            f"  Pred R² = {model_stats.get('PredR2', 'N/A')}\n"
            f"  RMSE    = {model_stats.get('S', 'N/A')}\n"
            f"  n       = {model_stats.get('n', 'N/A')}\n"
        )
        resp = self._client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text

    def interpret_effects(
        self,
        plot_images: list[str],
        coef_json: str,
        response_col: str,
        factor_cols: list[str],
        ia_factor_a: str | None = None,
        ia_factor_b: str | None = None,
    ) -> str:
        """Interpret effects plots via vision. Multimodal. Uses claude-sonnet-4-6."""
        system = (
            f"{EFFECTS_REVIEW_GUIDELINES}\n\n"
            "Use markdown. Start with a one-line verdict (✅ / ⚠️ / ❌ + single sentence). "
            "Then 3–5 bullet points, each referencing what you see in a specific plot. "
            "End with one 'Next step:' sentence. No preamble."
        )
        image_labels = ["Pareto chart", "Half-normal plot", "Main effects plot",
                        "Interaction plot"]
        content: list[dict] = [
            {"type": "text",
             "text": (f"Response: {response_col}   Factors: {', '.join(factor_cols)}\n"
                      + (f"Interaction plot shows {ia_factor_a} × {ia_factor_b}.\n"
                         if ia_factor_a and ia_factor_b else "")
                      + "Interpret all provided plots together.")},
        ]
        for label, img_b64 in zip(image_labels, plot_images):
            if img_b64:
                content.append({"type": "text",  "text": f"{label}:"})
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                })
        content.append({"type": "text",
                        "text": f"Coefficient estimates (for exact numbers):\n{coef_json}"})
        resp = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text

    def interpret_residuals(
        self,
        plot_image: str,
        residual_stats: dict,
        response_col: str,
    ) -> str:
        """Assess model adequacy from residual plots + stats. Multimodal. Uses claude-sonnet-4-6."""
        system = (
            f"{RESIDUAL_REVIEW_GUIDELINES}\n\n"
            "Use markdown. Start with a one-line verdict (✅ / ⚠️ / ❌ + single sentence). "
            "Then 3–5 bullet points referencing specific panels or statistics. "
            "End with one 'Next step:' sentence. No preamble."
        )
        stats = residual_stats
        stats_text = (
            f"n = {stats['n_obs']} | RMSE = {stats['rmse']} | "
            f"Skewness = {stats['skewness']} | "
            f"Shapiro-Wilk: W={stats['shapiro_wilk_stat']}, p={stats['shapiro_wilk_p']} "
            f"({'normality not rejected' if stats['normality_ok'] else 'normality REJECTED'}) | "
            f"Max |std residual| = {stats['max_std_residual']} | "
            f"Outliers >3σ: {stats['n_outliers_3sigma']}"
        )
        content: list[dict] = [
            {"type": "text",
             "text": f"Residual diagnostic plots for '{response_col}' model ({stats_text}):"},
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png", "data": plot_image}},
            {"type": "text", "text": "Assess all four diagnostic panels."},
        ]
        resp = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text


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
