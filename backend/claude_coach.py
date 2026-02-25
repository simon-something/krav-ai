"""Claude coaching module for structured technique feedback via OpenRouter."""

import json
import re

import httpx

from backend.prompts.claude_system import TECHNIQUE_RUBRIC, get_claude_prompt

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "anthropic/claude-sonnet-4"


class ClaudeCoach:
    def __init__(
        self, api_key: str, model: str = _DEFAULT_MODEL
    ) -> None:
        self._api_key = api_key
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
        """Get structured coaching feedback from Claude via OpenRouter.

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

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": TECHNIQUE_RUBRIC},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": 2048,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None

        for _attempt in range(2):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(
                        _OPENROUTER_URL,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()

                data = response.json()
                text = data["choices"][0]["message"]["content"]
                return self._parse_json_response(text)

            except (httpx.HTTPStatusError, json.JSONDecodeError, KeyError) as exc:
                last_error = exc

        raise RuntimeError(
            f"Claude coaching failed after 2 attempts: {last_error}"
        ) from last_error
