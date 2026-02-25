"""Tests for the Claude coaching module."""

import json
from unittest.mock import MagicMock, patch

import httpx
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
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(MOCK_COACHING)}}]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("backend.claude_coach.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = coach.get_feedback("jab", {}, [], [])

    assert result["summary"] == "Solid jab foundation, focus on guard retention"
    assert "Good stance width" in result["positives"]
    assert len(result["corrections"]) == 1
    assert result["corrections"][0]["priority"] == "high"


def test_coach_retries_on_failure(coach):
    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.raise_for_status = MagicMock()
    mock_success.json.return_value = {
        "choices": [{"message": {"content": json.dumps(MOCK_COACHING)}}]
    }

    mock_fail = MagicMock()
    mock_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=MagicMock(status_code=500)
    )

    mock_client = MagicMock()
    mock_client.post.side_effect = [mock_fail, mock_success]

    with patch("backend.claude_coach.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = coach.get_feedback("jab", {}, [], [])

    assert result["summary"] == "Solid jab foundation, focus on guard retention"
    assert mock_client.post.call_count == 2
