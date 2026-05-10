"""
Structured DoE problem elicitation via multi-turn conversation.
Uses claude-sonnet-4-6 with NIST decision rules in system prompt.
Session state is held in the Interviewer instance; persistence across
Dash callbacks is managed by a module-level session dict in app.py.
"""

import json
from anthropic import Anthropic
from agent.knowledge import DESIGN_SELECTION_RULES, VALID_DESIGN_TYPES

SYSTEM_PROMPT = f"""You are an expert Design of Experiments consultant.
Your job is to interview the user, understand their experimental problem,
and recommend the most appropriate design from the NIST Statistical Handbook.

{DESIGN_SELECTION_RULES}

INTERVIEW SEQUENCE — ask exactly ONE question per turn, in this order.
Do not ask two questions in the same message. Wait for the answer before proceeding.

1. What process or product are you trying to improve, and what will you measure as your response?
2. What factors (inputs or settings) do you want to study? For each, what are the lowest and highest values you want to test?
3. Roughly how many experimental runs can you afford?
4. Are there any conditions you cannot control that might vary — like different operators, different batches of material, or tests spread across multiple days?
5. Do you need to understand how factors interact with each other, or is finding the most important factors enough for now?
6. Do you expect the response to behave in a curved or nonlinear way across your factor ranges?
7. Can you repeat any experimental conditions more than once (true replication)?

After all seven questions are answered, provide:
- Your recommended design type (use exact names: {', '.join(VALID_DESIGN_TYPES)})
- Why you chose it, citing specific criteria from the NIST rules above
- Expected number of runs
- What effects will and will not be estimable
- Any important assumptions or limitations

Be conversational. Explain technical terms when you first use them.
Do not use bullet points in your questions — ask naturally, as a consultant would."""


class Interviewer:
    def __init__(self):
        self._client = Anthropic()
        self._history: list[dict] = []
        self._config_extracted: dict | None = None

    def start(self) -> str:
        """Return the opening message without adding a user turn."""
        opening = (
            "Hello! I'm here to help you design your experiment. "
            "Let's start with the basics — what process or product are you trying to improve, "
            "and what will you measure as your outcome?"
        )
        self._history.append({"role": "assistant", "content": opening})
        return opening

    def chat(self, user_message: str) -> str:
        self._history.append({"role": "user", "content": user_message})
        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=self._history,
        )
        reply = response.content[0].text
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def extract_design_config(self) -> dict:
        """
        Call after the interview recommendation turn.
        Extracts a structured config dict for app_controller.configure_design().
        Caches the result so it can be called multiple times safely.
        """
        if self._config_extracted:
            return self._config_extracted

        extraction_prompt = (
            "Based on our conversation, extract the agreed design configuration as JSON. "
            "Use only these design_type values: " + ", ".join(VALID_DESIGN_TYPES) + ". "
            "Return only valid JSON, no other text:\n"
            "{\n"
            '  "design_type": "...",\n'
            '  "factors": [{"name": "...", "low": <number>, "high": <number>}],\n'
            '  "options": {\n'
            '    "resolution": <3|4|5 or null>,\n'
            '    "replicates": <integer>,\n'
            '    "blocks": <integer>,\n'
            '    "randomize": true,\n'
            '    "center_points": <integer>\n'
            "  },\n"
            '  "justification": "<one sentence>"\n'
            "}"
        )
        extract_messages = self._history + [{"role": "user", "content": extraction_prompt}]
        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system="Extract structured design configuration from conversation. Return JSON only, no other text.",
            messages=extract_messages,
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if model wraps in them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        self._config_extracted = json.loads(raw.strip())
        return self._config_extracted

    def has_recommendation(self) -> bool:
        """Heuristic: True if the agent has already given a design recommendation."""
        for msg in self._history:
            if msg["role"] == "assistant" and any(
                dt in msg["content"] for dt in VALID_DESIGN_TYPES
            ):
                return True
        return False
