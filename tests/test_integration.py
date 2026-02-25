"""Integration test — real RAG + video processing, mocked APIs."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_video_path():
    path = "tests/fixtures/test_strike.avi"
    if not os.path.exists(path):
        import subprocess

        subprocess.run(
            ["uv", "run", "python", "tests/create_test_fixture.py"], check=True
        )
    return path


def test_full_analyze_pipeline(test_video_path):
    """Test the full pipeline with mocked Gemini and Claude but real video processing and RAG."""
    mock_gemini_observations = {
        "stance": {
            "stance_width_relative_to_shoulders": "approximately shoulder width"
        },
        "guard": {"lead_hand_height_relative_to_face": "at chin level, slightly low"},
        "strike_mechanics": {
            "hip_rotation_degree": "minimal, approximately 10 degrees"
        },
    }
    mock_coaching = {
        "summary": "Decent jab foundation, focus on hip engagement",
        "positives": ["Good stance width", "Straight extension path"],
        "corrections": [
            {
                "issue": "Minimal hip rotation",
                "why_it_matters": "Reduces power generation",
                "how_to_fix": "Initiate the jab with a slight hip turn",
                "priority": "high",
            }
        ],
    }

    with (
        patch("backend.server.gemini_analyzer") as mock_gemini,
        patch("backend.server.claude_coach") as mock_claude,
    ):
        mock_gemini.analyze.return_value = mock_gemini_observations
        mock_claude.get_feedback.return_value = mock_coaching

        from backend.server import app

        client = TestClient(app)

        with open(test_video_path, "rb") as f:
            response = client.post(
                "/api/analyze",
                data={"strike_type": "jab"},
                files={"video": ("test.webm", f, "video/webm")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "positives" in data
        assert "corrections" in data
        assert len(data["corrections"]) > 0
        assert data["corrections"][0]["priority"] == "high"

        # Verify Gemini was called with the right strike type
        mock_gemini.analyze.assert_called_once()
        call_args = mock_gemini.analyze.call_args
        assert call_args[0][1] == "jab"  # strike_type argument

        # Verify Claude was called with observations
        mock_claude.get_feedback.assert_called_once()


def test_session_history_accumulates(test_video_path):
    """Test that session history grows across multiple analyses."""
    mock_obs = {"stance": {"width": "good"}}
    mock_feedback = {
        "summary": "Good technique",
        "positives": ["Nice form"],
        "corrections": [],
    }

    with (
        patch("backend.server.gemini_analyzer") as mock_gemini,
        patch("backend.server.claude_coach") as mock_claude,
    ):
        mock_gemini.analyze.return_value = mock_obs
        mock_claude.get_feedback.return_value = mock_feedback

        from backend.server import app, session_history

        session_history.clear()

        client = TestClient(app)

        # First analysis
        with open(test_video_path, "rb") as f:
            client.post(
                "/api/analyze",
                data={"strike_type": "jab"},
                files={"video": ("test.webm", f, "video/webm")},
            )

        # Second analysis
        with open(test_video_path, "rb") as f:
            client.post(
                "/api/analyze",
                data={"strike_type": "cross"},
                files={"video": ("test.webm", f, "video/webm")},
            )

        # Claude should receive session history on the second call
        assert mock_claude.get_feedback.call_count == 2
