import pytest
from backend.video_processor import extract_frames, get_video_duration

FIXTURE = "tests/fixtures/test_strike.avi"


def test_extract_frames_returns_list_of_frames():
    frames = extract_frames(FIXTURE, num_frames=10)
    assert isinstance(frames, list)
    assert len(frames) == 10


def test_extract_frames_returns_numpy_arrays():
    frames = extract_frames(FIXTURE, num_frames=5)
    assert all(hasattr(f, "shape") for f in frames)


def test_extract_frames_respects_num_frames():
    frames_5 = extract_frames(FIXTURE, num_frames=5)
    frames_15 = extract_frames(FIXTURE, num_frames=15)
    assert len(frames_5) == 5
    assert len(frames_15) == 15


def test_get_video_duration():
    duration = get_video_duration(FIXTURE)
    assert 2.5 < duration < 3.5  # ~3 seconds


def test_extract_frames_invalid_file():
    with pytest.raises(FileNotFoundError):
        extract_frames("nonexistent.avi")
