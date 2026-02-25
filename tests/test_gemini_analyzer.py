"""Tests for the Gemini analyzer module."""

import json
from unittest.mock import MagicMock, mock_open, patch

import httpx
import pytest

from backend.gemini_analyzer import GeminiAnalyzer

MOCK_OBSERVATIONS = {
    "stance": {"stance_width_relative_to_shoulders": "approximately shoulder width"},
    "guard": {"lead_hand_height_relative_to_face": "at chin level, slightly low"},
    "strike_mechanics": {"hip_rotation_degree": "minimal, approximately 10 degrees"},
}


@pytest.fixture
def analyzer():
    return GeminiAnalyzer(api_key="test_key")


def test_build_messages_includes_video(analyzer):
    fake_video = b"fake_video_bytes"
    fake_files = {"video.webm": fake_video}

    def side_effect(path, mode="r"):
        if path in fake_files:
            return mock_open(read_data=fake_files[path])()
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=side_effect):
        messages = analyzer._build_messages("video.webm", "jab", [])

    assert len(messages) == 1
    content = messages[0]["content"]
    # Should have text part + video part
    assert len(content) == 2
    video_part = content[1]
    assert video_part["type"] == "image_url"
    assert "base64" in video_part["image_url"]["url"]
    assert "video" in video_part["image_url"]["url"]


def test_build_messages_includes_reference_images(analyzer):
    fake_video = b"fake_video_bytes"
    fake_img1 = b"fake_img1_bytes"
    fake_img2 = b"fake_img2_bytes"
    fake_files = {
        "video.webm": fake_video,
        "ref1.jpg": fake_img1,
        "ref2.png": fake_img2,
    }

    def side_effect(path, mode="r"):
        if path in fake_files:
            return mock_open(read_data=fake_files[path])()
        raise FileNotFoundError(path)

    with patch("builtins.open", side_effect=side_effect):
        messages = analyzer._build_messages(
            "video.webm", "cross", ["ref1.jpg", "ref2.png"]
        )

    content = messages[0]["content"]
    # text + video + 2 reference images = 4 parts
    assert len(content) == 4
    assert content[2]["type"] == "image_url"
    assert content[3]["type"] == "image_url"
    assert "image" in content[2]["image_url"]["url"]


def test_analyze_returns_parsed_observations(analyzer):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(MOCK_OBSERVATIONS)}}]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with (
        patch("backend.gemini_analyzer.httpx.Client", return_value=mock_client),
        patch("builtins.open", mock_open(read_data=b"fake_video")),
    ):
        result = analyzer.analyze("test.webm", "jab", [])

    assert isinstance(result, dict)
    assert "stance" in result
    assert "guard" in result
    assert result["stance"]["stance_width_relative_to_shoulders"] == "approximately shoulder width"


def test_analyze_retries_on_failure(analyzer):
    # First call raises HTTP error, second succeeds
    fail_response = MagicMock()
    fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=MagicMock(status_code=500)
    )

    success_response = MagicMock()
    success_response.raise_for_status = MagicMock()
    success_response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(MOCK_OBSERVATIONS)}}]
    }

    mock_client = MagicMock()
    mock_client.post.side_effect = [fail_response, success_response]
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with (
        patch("backend.gemini_analyzer.httpx.Client", return_value=mock_client),
        patch("builtins.open", mock_open(read_data=b"fake_video")),
    ):
        result = analyzer.analyze("test.webm", "jab", [])

    assert result["stance"]["stance_width_relative_to_shoulders"] == "approximately shoulder width"
    assert mock_client.post.call_count == 2
