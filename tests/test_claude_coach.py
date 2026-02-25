"""Tests for the Claude coaching module."""

import json
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from backend.claude_coach import ClaudeCoach

MOCK_COACHING = {
    "summary": "Solid jab foundation, focus on guard retention",
    "positives": ["Good stance width", "Proper fist alignment"],
    "corrections": [
        {
            "issue": "Rear hand drops during jab",
            "why_it_matters": "Leaves face exposed to counters",
            "how_to_fix": "Focus on keeping right hand glued to cheekbone",
            "priority": "high",
        }
    ],
}


@pytest.fixture
def coach():
    with patch("backend.claude_coach.anthropic.Anthropic"):
        return ClaudeCoach(api_key="test_key")


def test_build_prompt_includes_observations(coach):
    observations = {"stance": {"width": "narrow"}}
    prompt = coach._build_prompt("jab", observations, [], [])
    assert "narrow" in prompt


def test_build_prompt_includes_rag(coach):
    chunks = [{"text": "Keep guard up", "metadata": {}}]
    prompt = coach._build_prompt("jab", {}, chunks, [])
    assert "Keep guard up" in prompt


def test_coach_returns_structured_feedback(coach):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=json.dumps(MOCK_COACHING))]
    coach._client.messages.create.return_value = mock_msg

    result = coach.get_feedback("jab", {}, [], [])

    assert result["summary"] == "Solid jab foundation, focus on guard retention"
    assert "Good stance width" in result["positives"]
    assert len(result["corrections"]) == 1
    assert result["corrections"][0]["priority"] == "high"


def test_coach_retries_on_failure(coach):
    # First call raises API error, second succeeds
    mock_success = MagicMock()
    mock_success.content = [MagicMock(text=json.dumps(MOCK_COACHING))]

    coach._client.messages.create.side_effect = [
        anthropic.APIConnectionError(request=MagicMock()),
        mock_success,
    ]

    result = coach.get_feedback("jab", {}, [], [])

    assert result["summary"] == "Solid jab foundation, focus on guard retention"
    assert coach._client.messages.create.call_count == 2
