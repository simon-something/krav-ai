"""Tests for Gemini observation prompt and Claude coaching rubric."""

from backend.prompts.gemini_system import OBSERVATION_SCHEMA, get_gemini_prompt
from backend.prompts.claude_system import TECHNIQUE_RUBRIC, get_claude_prompt


# --- Gemini prompt tests ---


def test_gemini_prompt_contains_strike_type():
    prompt = get_gemini_prompt("jab")
    assert "jab" in prompt.lower()


def test_gemini_prompt_contains_schema():
    prompt = get_gemini_prompt("cross")
    assert "stance" in prompt.lower()
    assert "guard" in prompt.lower()


def test_observation_schema_has_required_fields():
    assert "stance" in OBSERVATION_SCHEMA
    assert "guard" in OBSERVATION_SCHEMA
    assert "strike_mechanics" in OBSERVATION_SCHEMA
    assert "footwork" in OBSERVATION_SCHEMA
    assert "non_striking_hand" in OBSERVATION_SCHEMA
    assert "telegraphing" in OBSERVATION_SCHEMA
    assert "combination_flow" in OBSERVATION_SCHEMA


# --- Claude prompt tests ---


def test_claude_prompt_contains_strike_type():
    prompt = get_claude_prompt("jab", observations={}, rag_chunks=[], session_history=[])
    assert "jab" in prompt.lower()


def test_claude_prompt_includes_rag_chunks():
    chunks = ["Keep your guard up during the jab", "Rotate hips for power"]
    prompt = get_claude_prompt("jab", observations={}, rag_chunks=chunks, session_history=[])
    assert "Keep your guard up" in prompt
    assert "Rotate hips" in prompt


def test_claude_prompt_includes_session_history():
    history = [{"attempt": 1, "summary": "Good stance, work on guard"}]
    prompt = get_claude_prompt("jab", observations={}, rag_chunks=[], session_history=history)
    assert "Good stance" in prompt


def test_technique_rubric_covers_all_categories():
    rubric_lower = TECHNIQUE_RUBRIC.lower()
    assert "stance" in rubric_lower
    assert "guard" in rubric_lower
    assert "jab" in rubric_lower
    assert "cross" in rubric_lower
    assert "footwork" in rubric_lower
    assert "common errors" in rubric_lower
    assert "combination" in rubric_lower
