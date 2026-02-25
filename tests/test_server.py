"""Tests for the FastAPI server."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.server import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_rejects_missing_video():
    response = client.post("/api/analyze", data={"strike_type": "jab"})
    assert response.status_code == 422


def test_analyze_rejects_invalid_strike_type():
    response = client.post(
        "/api/analyze",
        data={"strike_type": "roundhouse"},
        files={"video": ("test.webm", b"fake_video_data", "video/webm")},
    )
    assert response.status_code == 400
    assert "roundhouse" in response.json()["detail"]


@patch("backend.server.rag_engine")
@patch("backend.server.claude_coach")
@patch("backend.server.gemini_analyzer")
@patch("backend.server.get_video_duration", return_value=3.0)
@patch("backend.server.extract_frames", return_value=[MagicMock()] * 15)
def test_full_pipeline_with_mocks(
    mock_extract, mock_duration, mock_gemini, mock_claude, mock_rag
):
    mock_gemini.analyze.return_value = {
        "stance": {"stance_width_relative_to_shoulders": "shoulder width"},
        "guard": {"lead_hand_height_relative_to_face": "chin level"},
    }
    mock_rag.query.return_value = [
        {"text": "Keep your guard up", "metadata": {"source": "test"}, "distance": 0.1}
    ]
    mock_claude.get_feedback.return_value = {
        "summary": "Good technique overall",
        "positives": ["Nice stance width"],
        "corrections": [
            {
                "issue": "Guard drops during jab",
                "why_it_matters": "Leaves face exposed",
                "how_to_fix": "Keep rear hand at cheekbone",
                "priority": "high",
            }
        ],
    }

    with open("tests/fixtures/test_strike.avi", "rb") as f:
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
    assert data["summary"] == "Good technique overall"
    assert len(data["corrections"]) == 1

    # Verify services were called
    mock_gemini.analyze.assert_called_once()
    mock_rag.query.assert_called_once()
    mock_claude.get_feedback.assert_called_once()
