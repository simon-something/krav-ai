"""Claude coaching module for structured technique feedback."""

import json
import re

import anthropic

from backend.prompts.claude_system import TECHNIQUE_RUBRIC, get_claude_prompt


class ClaudeCoach:
    def __init__(
        self, api_key: str, model: str = "claude-sonnet-4-20250514"
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def _build_prompt(
        self,
        strike_type: str,
        observations: dict,
        rag_chunks: list[dict],
        session_history: list[dict],
    ) -> str:
        """Build the user message for Claude."""
        return get_claude_prompt(strike_type, observations, rag_chunks, session_history)

    def _parse_json_response(self, text: str) -> dict:
        """Extract and parse JSON from Claude's response text.

        Handles cases where the model wraps JSON in markdown code blocks.
        """
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
        return json.loads(text)

    def get_feedback(
        self,
        strike_type: str,
        observations: dict,
        rag_chunks: list[dict],
        session_history: list[dict],
    ) -> dict:
        """Get structured coaching feedback from Claude.

        Args:
            strike_type: One of "jab", "cross", or "jab_cross".
            observations: Structured observations dict from Gemini analysis.
            rag_chunks: Relevant RAG chunks (list of dicts with "text" key).
            session_history: Previous feedback dicts for progress tracking.

        Returns:
            Parsed feedback dict with keys: summary, positives, corrections.

        Raises:
            RuntimeError: If both attempts fail.
        """
        user_message = self._build_prompt(
            strike_type, observations, rag_chunks, session_history
        )

        last_error: Exception | None = None

        for _attempt in range(2):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=2048,
                    system=TECHNIQUE_RUBRIC,
                    messages=[{"role": "user", "content": user_message}],
                )
                text = response.content[0].text
                return self._parse_json_response(text)

            except (anthropic.APIError, json.JSONDecodeError, KeyError) as exc:
                last_error = exc

        raise RuntimeError(
            f"Claude coaching failed after 2 attempts: {last_error}"
        ) from last_error
