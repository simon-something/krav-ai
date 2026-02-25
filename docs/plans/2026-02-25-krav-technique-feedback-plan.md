# Krav Maga Technique Feedback App — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web app where a user records a short striking clip via webcam, and gets AI-powered coaching feedback on 25+ technique checkpoints for jab, cross, and jab-cross combinations.

**Architecture:** Modular Python backend (FastAPI) with separate vanilla JS frontend. Gemini 2.5 Pro analyzes video for body mechanics observations, RAG retrieves relevant coaching material from ChromaDB, Claude synthesizes coaching feedback. Three knowledge layers: system prompt rubric, RAG documents, reference images.

**Tech Stack:** Python 3.11+, FastAPI, OpenCV, ChromaDB, sentence-transformers, Anthropic SDK, OpenRouter (Gemini), vanilla HTML/CSS/JS.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/__init__.py`
- Create: `backend/prompts/__init__.py`
- Create: `.env.example`
- Create: `.gitignore`

**Step 1: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.env
backend/knowledge/chromadb/
venv/
.venv/
node_modules/
*.webm
*.mp4
```

**Step 2: Create requirements.txt**

```txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-dotenv==1.0.1
python-multipart==0.0.20
opencv-python-headless==4.11.0.86
anthropic==0.43.0
httpx==0.28.1
chromadb==0.6.3
sentence-transformers==3.4.1
```

**Step 3: Create .env.example**

```
ANTHROPIC_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

**Step 4: Create empty __init__.py files and directory structure**

```bash
mkdir -p backend/prompts backend/knowledge/references backend/knowledge/documents frontend tests
touch backend/__init__.py backend/prompts/__init__.py
```

**Step 5: Create venv and install deps**

```bash
cd /home/ubuntu/repos/perso/krav-ai
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

**Step 6: Commit**

```bash
git add .gitignore backend/requirements.txt backend/__init__.py backend/prompts/__init__.py .env.example
git commit -m "feat: project scaffolding with dependencies and directory structure"
```

---

## Task 2: Research & Curate Krav Maga Technique Knowledge Base

**Files:**
- Create: `backend/knowledge/documents/striking_fundamentals.md`
- Create: `backend/knowledge/documents/jab_technique.md`
- Create: `backend/knowledge/documents/cross_technique.md`
- Create: `backend/knowledge/documents/jab_cross_combination.md`
- Create: `backend/knowledge/documents/common_errors.md`
- Create: `backend/knowledge/documents/biomechanics.md`

**Context:** This task requires web research to gather authoritative Krav Maga technique information. Each document should be structured with clear sections that will chunk well for RAG (one concept per section, ~300-500 tokens per section).

**Step 1: Research and write striking_fundamentals.md**

Cover: fighting stance (width = shoulder width, staggered, bladed ~45 degrees), weight distribution (50/50 or slight front bias), guard position (hands at cheekbones, elbows tucked to ribs, chin tucked, shoulders slightly raised), rear heel raised, knees slightly bent. Source from IKMF/KMG curriculum, Imi Lichtenfeld principles.

Use web search to find authoritative sources. Structure as sections of ~300-500 tokens each with clear headers.

**Step 2: Research and write jab_technique.md**

Cover: lead hand straight-line extension, fist rotation to horizontal at impact, shoulder rises to cover chin, non-punching hand stays at guard, hip engagement (slight), snap-back retraction speed, full arm extension without locking elbow. Include coaching cues ("punch through the target", "hand returns faster than it goes out").

**Step 3: Research and write cross_technique.md**

Cover: rear hand drive, full hip rotation, rear foot pivot (ball of foot), weight transfer from rear to front, shoulder rotation through centerline, full extension, chin stays behind shoulder, return to guard.

**Step 4: Research and write jab_cross_combination.md**

Cover: transition fluidity, no pause/reset between strikes, guard maintenance during transition, stance recovery between strikes, no telegraph between strikes, maintaining forward pressure, rhythm.

**Step 5: Research and write common_errors.md**

Cover for each strike: dropping non-punching hand, telegraphing (pulling hand back before striking, shifting weight obviously), chicken-winging elbow, leaning forward/overcommitting, crossing feet, winding up, not rotating hips, punching with loose fist, not returning to guard.

**Step 6: Research and write biomechanics.md**

Cover: kinetic chain (ground → legs → hips → torso → shoulder → arm → fist), rotational mechanics, momentum transfer, balance and center of gravity, impact alignment (first two knuckles, straight wrist).

**Step 7: Commit**

```bash
git add backend/knowledge/documents/
git commit -m "feat: curate Krav Maga technique knowledge base documents"
```

---

## Task 3: Prompts Module — Gemini Observation Schema

**Files:**
- Create: `backend/prompts/gemini_system.py`
- Create: `tests/test_prompts.py`

**Step 1: Write the test**

```python
# tests/test_prompts.py
from backend.prompts.gemini_system import get_gemini_prompt, OBSERVATION_SCHEMA

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
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_prompts.py -v
```
Expected: FAIL — module not found.

**Step 3: Implement gemini_system.py**

```python
# backend/prompts/gemini_system.py

OBSERVATION_SCHEMA = {
    "stance": {
        "description": "Fighting stance observations",
        "fields": [
            "stance_width_relative_to_shoulders",
            "foot_stagger_depth",
            "body_angle_degrees",
            "weight_distribution_front_back",
            "knee_bend",
            "rear_heel_position",
        ],
    },
    "guard": {
        "description": "Guard position observations",
        "fields": [
            "lead_hand_height_relative_to_face",
            "rear_hand_height_relative_to_face",
            "elbow_tuck",
            "chin_position",
            "shoulder_position",
        ],
    },
    "strike_mechanics": {
        "description": "Mechanics of the executed strike(s)",
        "fields": [
            "extension_path",
            "fist_orientation_at_impact",
            "hip_rotation_degree",
            "shoulder_rotation",
            "rear_foot_pivot",
            "weight_transfer",
            "retraction_speed_vs_extension",
            "full_extension_achieved",
            "elbow_position_during_strike",
        ],
    },
    "non_striking_hand": {
        "description": "What the non-punching hand does during the strike",
        "fields": [
            "guard_maintained",
            "hand_drop_observed",
            "position_at_strike_peak",
        ],
    },
    "footwork": {
        "description": "Footwork and balance observations",
        "fields": [
            "feet_crossed",
            "overcommitment",
            "balance_throughout",
            "stance_recovery_after_strike",
        ],
    },
    "telegraphing": {
        "description": "Any telegraphing observed before strikes",
        "fields": [
            "hand_pullback_before_strike",
            "weight_shift_before_strike",
            "shoulder_dip",
            "other_tells",
        ],
    },
    "combination_flow": {
        "description": "Only for jab-cross combinations",
        "fields": [
            "transition_fluidity",
            "pause_between_strikes",
            "guard_during_transition",
            "rhythm",
        ],
    },
}


def _format_schema_for_prompt() -> str:
    lines = ["Provide your observations as JSON with these categories:\n"]
    for category, info in OBSERVATION_SCHEMA.items():
        lines.append(f"### {category}")
        lines.append(f"{info['description']}")
        for field in info["fields"]:
            lines.append(f"  - {field}: <describe what you observe>")
        lines.append("")
    return "\n".join(lines)


def get_gemini_prompt(strike_type: str) -> str:
    schema_text = _format_schema_for_prompt()

    return f"""You are an expert biomechanics analyst specializing in combat sports striking technique.

You are analyzing a video of a person performing a **{strike_type}** strike (Krav Maga context).

Reference images of correct technique are provided alongside the video. Compare the practitioner's form against these references.

## Your Task

Observe the practitioner's body mechanics in detail. Be FACTUAL and SPECIFIC — describe exactly what you see, with approximate angles, positions, and timing where possible. Do NOT provide coaching advice; only observations.

## Observation Categories

{schema_text}

## Important Notes

- If this is a single strike (jab or cross), skip the combination_flow category.
- For each field, describe what you actually observe. Use phrases like "approximately 45 degrees", "hand drops to chin level", "slight hip rotation ~20 degrees".
- Note the TIMING of movements — what happens first, what happens simultaneously, what is delayed.
- If you cannot clearly observe something (e.g., occluded by angle), say "not clearly visible" for that field.
- Compare against the reference images where applicable.

Respond with valid JSON only."""
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_prompts.py -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add backend/prompts/gemini_system.py tests/test_prompts.py
git commit -m "feat: Gemini observation prompt with structured schema"
```

---

## Task 4: Prompts Module — Claude Coaching Rubric

**Files:**
- Create: `backend/prompts/claude_system.py`
- Modify: `tests/test_prompts.py`

**Step 1: Write the test**

```python
# Append to tests/test_prompts.py
from backend.prompts.claude_system import get_claude_prompt, TECHNIQUE_RUBRIC

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
    assert "stance" in TECHNIQUE_RUBRIC.lower()
    assert "guard" in TECHNIQUE_RUBRIC.lower()
    assert "jab" in TECHNIQUE_RUBRIC.lower()
    assert "cross" in TECHNIQUE_RUBRIC.lower()
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_prompts.py::test_claude_prompt_contains_strike_type -v
```
Expected: FAIL.

**Step 3: Implement claude_system.py**

This file contains the full technique rubric with all 25+ checkpoints and the prompt builder. The rubric should be a detailed string covering:

- Stance & base (6 checkpoints)
- Guard (5 checkpoints)
- Jab mechanics (7 checkpoints)
- Cross mechanics (8 checkpoints)
- Jab-cross combination (5 checkpoints)
- Footwork & recovery (4 checkpoints)
- Common errors to watch for (8+ items)

The `get_claude_prompt()` function assembles: rubric + Gemini observations + RAG chunks + session history into a single coaching prompt.

Output format instructions should tell Claude to return JSON:
```json
{
  "summary": "One-line overall assessment",
  "positives": ["What's done well"],
  "corrections": [
    {
      "issue": "What's wrong",
      "why_it_matters": "Impact on technique/safety",
      "how_to_fix": "Specific actionable instruction",
      "priority": "high|medium|low"
    }
  ]
}
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_prompts.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add backend/prompts/claude_system.py tests/test_prompts.py
git commit -m "feat: Claude coaching prompt with full technique rubric"
```

---

## Task 5: Video Processor Module

**Files:**
- Create: `backend/video_processor.py`
- Create: `tests/test_video_processor.py`
- Create: `tests/fixtures/` (test video fixture)

**Step 1: Create a test video fixture**

```python
# tests/create_test_fixture.py — run once to generate a test video
import cv2
import numpy as np
import os

os.makedirs("tests/fixtures", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("tests/fixtures/test_strike.avi", fourcc, 30.0, (640, 480))
for i in range(90):  # 3 seconds at 30fps
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a moving circle to simulate motion
    x = int(100 + i * 4)
    cv2.circle(frame, (x, 240), 30, (0, 255, 0), -1)
    out.write(frame)
out.release()
```

```bash
python tests/create_test_fixture.py
```

**Step 2: Write the tests**

```python
# tests/test_video_processor.py
import os
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
```

**Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_video_processor.py -v
```
Expected: FAIL.

**Step 4: Implement video_processor.py**

```python
# backend/video_processor.py
import cv2
import numpy as np


def get_video_duration(video_path: str) -> float:
    if not __import__("os").path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps == 0:
        raise ValueError("Could not read video FPS")
    return frame_count / fps


def extract_frames(video_path: str, num_frames: int = 15) -> list[np.ndarray]:
    if not __import__("os").path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames")

    # Uniform sampling across the video
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_video_processor.py -v
```
Expected: all PASS.

**Step 6: Commit**

```bash
git add backend/video_processor.py tests/test_video_processor.py tests/create_test_fixture.py tests/fixtures/
git commit -m "feat: video processor with frame extraction"
```

---

## Task 6: RAG Engine + Knowledge Base Builder

**Files:**
- Create: `backend/rag_engine.py`
- Create: `backend/build_knowledge_base.py`
- Create: `tests/test_rag_engine.py`

**Step 1: Write the tests**

```python
# tests/test_rag_engine.py
import os
import shutil
import pytest
from backend.rag_engine import RAGEngine

TEST_DB_PATH = "tests/fixtures/test_chromadb"

@pytest.fixture
def rag_engine():
    engine = RAGEngine(db_path=TEST_DB_PATH)
    engine.add_document(
        text="Keep your guard hand at cheekbone height during the jab",
        metadata={"strike_type": "jab", "category": "guard", "source": "test"},
    )
    engine.add_document(
        text="Rotate hips fully when throwing the cross for maximum power",
        metadata={"strike_type": "cross", "category": "mechanics", "source": "test"},
    )
    engine.add_document(
        text="Fighting stance should be shoulder width with rear heel raised",
        metadata={"strike_type": "both", "category": "stance", "source": "test"},
    )
    yield engine
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)

def test_query_returns_relevant_chunks(rag_engine):
    results = rag_engine.query("guard position during jab", n_results=2)
    assert len(results) <= 2
    assert any("guard" in r["text"].lower() for r in results)

def test_query_with_strike_filter(rag_engine):
    results = rag_engine.query("hip rotation", n_results=3, strike_type="cross")
    assert len(results) > 0

def test_query_returns_metadata(rag_engine):
    results = rag_engine.query("stance width", n_results=1)
    assert "metadata" in results[0]
    assert "source" in results[0]["metadata"]

def test_reset_clears_data(rag_engine):
    rag_engine.reset()
    results = rag_engine.query("guard", n_results=5)
    assert len(results) == 0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_rag_engine.py -v
```
Expected: FAIL.

**Step 3: Implement rag_engine.py**

```python
# backend/rag_engine.py
import chromadb
from chromadb.utils import embedding_functions


class RAGEngine:
    def __init__(self, db_path: str = "backend/knowledge/chromadb"):
        self._client = chromadb.PersistentClient(path=db_path)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name="krav_technique", embedding_function=self._ef
        )
        self._doc_count = self._collection.count()

    def add_document(self, text: str, metadata: dict) -> None:
        self._doc_count += 1
        self._collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[f"doc_{self._doc_count}"],
        )

    def query(
        self, query_text: str, n_results: int = 5, strike_type: str | None = None
    ) -> list[dict]:
        where_filter = None
        if strike_type:
            where_filter = {
                "$or": [
                    {"strike_type": strike_type},
                    {"strike_type": "both"},
                ]
            }

        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
            where=where_filter,
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return chunks

    def reset(self) -> None:
        self._client.delete_collection("krav_technique")
        self._collection = self._client.get_or_create_collection(
            name="krav_technique", embedding_function=self._ef
        )
        self._doc_count = 0
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_rag_engine.py -v
```
Expected: all PASS.

**Step 5: Implement build_knowledge_base.py**

```python
# backend/build_knowledge_base.py
"""Reads all .md files from knowledge/documents/, chunks them, and indexes into ChromaDB."""

import os
import re
from backend.rag_engine import RAGEngine

DOCS_DIR = os.path.join(os.path.dirname(__file__), "knowledge", "documents")


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split markdown by ## headers into chunks with metadata."""
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    # Try to detect strike_type from filename
    strike_type = "both"
    source_lower = source.lower()
    if "jab_cross" in source_lower or "combination" in source_lower:
        strike_type = "jab_cross"
    elif "jab" in source_lower:
        strike_type = "jab"
    elif "cross" in source_lower:
        strike_type = "cross"

    for section in sections:
        section = section.strip()
        if len(section) < 50:  # skip tiny sections
            continue
        # Detect category from header
        category = "general"
        header_match = re.match(r"##\s*(.+)", section)
        if header_match:
            header = header_match.group(1).lower()
            for cat in ["stance", "guard", "mechanics", "footwork", "errors", "biomechanics"]:
                if cat in header:
                    category = cat
                    break

        chunks.append(
            {
                "text": section,
                "metadata": {
                    "strike_type": strike_type,
                    "category": category,
                    "source": source,
                },
            }
        )
    return chunks


def build():
    engine = RAGEngine()
    engine.reset()

    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath) as f:
            text = f.read()
        chunks = chunk_markdown(text, source=filename)
        for chunk in chunks:
            engine.add_document(text=chunk["text"], metadata=chunk["metadata"])
        print(f"Indexed {len(chunks)} chunks from {filename}")

    print("Knowledge base built successfully.")


if __name__ == "__main__":
    build()
```

**Step 6: Commit**

```bash
git add backend/rag_engine.py backend/build_knowledge_base.py tests/test_rag_engine.py
git commit -m "feat: RAG engine with ChromaDB and knowledge base builder"
```

---

## Task 7: Gemini Analyzer Module

**Files:**
- Create: `backend/gemini_analyzer.py`
- Create: `tests/test_gemini_analyzer.py`

**Step 1: Write the tests**

```python
# tests/test_gemini_analyzer.py
import json
import pytest
from unittest.mock import patch, MagicMock
from backend.gemini_analyzer import GeminiAnalyzer

MOCK_RESPONSE = json.dumps({
    "stance": {"stance_width_relative_to_shoulders": "approximately shoulder width"},
    "guard": {"lead_hand_height_relative_to_face": "at chin level, slightly low"},
    "strike_mechanics": {"hip_rotation_degree": "minimal, approximately 10 degrees"},
})

@pytest.fixture
def analyzer():
    return GeminiAnalyzer(api_key="test_key")

def test_build_messages_includes_video(analyzer):
    messages = analyzer._build_messages(
        video_path="test.webm",
        strike_type="jab",
        reference_images=[],
    )
    assert any("video" in str(m).lower() or "file" in str(m).lower() for m in messages)

@patch("backend.gemini_analyzer.httpx.Client")
def test_analyze_returns_parsed_observations(mock_client_cls, analyzer):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": MOCK_RESPONSE}}]
    }
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

    with patch("builtins.open", MagicMock()):
        with patch("base64.b64encode", return_value=b"fake_b64"):
            result = analyzer.analyze("test.webm", "jab", [])

    assert isinstance(result, dict)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_gemini_analyzer.py -v
```
Expected: FAIL.

**Step 3: Implement gemini_analyzer.py**

The module should:
- Read the video file and base64-encode it
- Load any reference images and base64-encode them
- Build the request payload for OpenRouter's Gemini 2.5 Pro endpoint
- Send via httpx to `https://openrouter.ai/api/v1/chat/completions`
- Parse the JSON response into a Python dict of observations
- Handle errors (retry once on failure)

Key implementation details:
- Model: `google/gemini-2.5-pro` via OpenRouter
- Video sent as base64 data URL in a multimodal message
- Reference images sent as additional image parts
- Response parsed as JSON from the model output

**Step 4: Run tests**

```bash
python -m pytest tests/test_gemini_analyzer.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add backend/gemini_analyzer.py tests/test_gemini_analyzer.py
git commit -m "feat: Gemini analyzer module with OpenRouter integration"
```

---

## Task 8: Claude Coach Module

**Files:**
- Create: `backend/claude_coach.py`
- Create: `tests/test_claude_coach.py`

**Step 1: Write the tests**

```python
# tests/test_claude_coach.py
import json
import pytest
from unittest.mock import patch, MagicMock
from backend.claude_coach import ClaudeCoach

MOCK_COACHING = json.dumps({
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
})

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

@patch("backend.claude_coach.anthropic.Anthropic")
def test_coach_returns_structured_feedback(mock_anthropic_cls, coach):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=MOCK_COACHING)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    coach._client = mock_client

    result = coach.get_feedback("jab", {}, [], [])

    assert "summary" in result
    assert "corrections" in result
    assert isinstance(result["corrections"], list)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_claude_coach.py -v
```
Expected: FAIL.

**Step 3: Implement claude_coach.py**

The module should:
- Use the Anthropic SDK
- Build the prompt using `get_claude_prompt()` from the prompts module
- Send to Claude (claude-sonnet-4-20250514 for speed, or claude-opus-4-0-20250115 for max quality — make configurable)
- Parse JSON response into structured feedback dict
- Handle errors (retry once)
- Accept session_history to enable progress tracking

**Step 4: Run tests**

```bash
python -m pytest tests/test_claude_coach.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add backend/claude_coach.py tests/test_claude_coach.py
git commit -m "feat: Claude coaching module with structured feedback"
```

---

## Task 9: FastAPI Server

**Files:**
- Create: `backend/server.py`
- Create: `tests/test_server.py`

**Step 1: Write the tests**

```python
# tests/test_server.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
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
    # Create a minimal file-like upload
    response = client.post(
        "/api/analyze",
        data={"strike_type": "uppercut"},
        files={"video": ("test.webm", b"fake_video_data", "video/webm")},
    )
    assert response.status_code == 422 or response.status_code == 400
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_server.py -v
```
Expected: FAIL.

**Step 3: Implement server.py**

The server should:
- Serve static files from `frontend/` at root
- `GET /api/health` returns `{"status": "ok"}`
- `POST /api/analyze` accepts multipart form: `video` (file) + `strike_type` (string: jab|cross|jab_cross)
- Orchestrate the full pipeline: save temp video → extract frames → call Gemini → query RAG → call Claude → return feedback
- Maintain in-memory session history (list of last 5 feedback results)
- Validate strike_type is one of the allowed values
- Validate video duration (1-15 seconds)
- CORS enabled for local development
- Load API keys from .env via python-dotenv

**Step 4: Run tests**

```bash
python -m pytest tests/test_server.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add backend/server.py tests/test_server.py
git commit -m "feat: FastAPI server with /api/analyze endpoint"
```

---

## Task 10: Frontend — HTML & CSS

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/style.css`

**Step 1: Create index.html**

Single-page layout with three states:
- Camera preview area with video element
- Strike type selector (dropdown: Jab / Cross / Jab-Cross)
- Record / Stop button
- Status/progress area
- Feedback display area (summary, positives, corrections)
- Session history sidebar/section

Use semantic HTML. No framework needed.

**Step 2: Create style.css**

Clean, dark-themed UI (appropriate for a training app):
- Dark background, light text
- Camera preview centered, max-width ~640px
- Red border on video when recording
- Feedback sections clearly delineated
- Corrections color-coded by priority (high=red, medium=orange, low=yellow)
- Responsive for different screen sizes
- Loading spinner/animation

**Step 3: Verify HTML loads**

```bash
cd /home/ubuntu/repos/perso/krav-ai && python -m http.server 8080 --directory frontend &
# Open http://localhost:8080 and verify layout renders
kill %1
```

**Step 4: Commit**

```bash
git add frontend/index.html frontend/style.css
git commit -m "feat: frontend HTML and CSS layout"
```

---

## Task 11: Frontend — JavaScript Application Logic

**Files:**
- Create: `frontend/app.js`

**Step 1: Implement camera and recording logic**

```javascript
// Core functionality needed:
// 1. Camera access via navigator.mediaDevices.getUserMedia
// 2. MediaRecorder for video capture (WebM format)
// 3. Record/Stop button state management
// 4. Timer display during recording
// 5. Duration validation (1-15 seconds)
```

**Step 2: Implement API communication**

```javascript
// 1. POST /api/analyze with FormData (video blob + strike_type)
// 2. Progress state display ("Extracting frames..." etc.)
// 3. Parse JSON response
// 4. Error handling and display
```

**Step 3: Implement feedback rendering**

```javascript
// 1. Render summary, positives, corrections from API response
// 2. Color-code corrections by priority
// 3. Append to session history
// 4. Auto-scroll to feedback after analysis
```

**Step 4: Manual test**

Start the backend server and test the full flow:
```bash
cd /home/ubuntu/repos/perso/krav-ai
source venv/bin/activate
uvicorn backend.server:app --reload --port 8000
# Open http://localhost:8000 in browser
# Test: select jab, record 3 seconds, stop, verify feedback appears
```

**Step 5: Commit**

```bash
git add frontend/app.js
git commit -m "feat: frontend JS with recording, API calls, and feedback display"
```

---

## Task 12: Integration Testing & Polish

**Files:**
- Create: `tests/test_integration.py`
- Modify: various files for bug fixes

**Step 1: Build the knowledge base**

```bash
cd /home/ubuntu/repos/perso/krav-ai
source venv/bin/activate
python -m backend.build_knowledge_base
```

Verify output shows chunks indexed from each document.

**Step 2: Write integration test**

```python
# tests/test_integration.py
"""Integration test using mock APIs but real RAG and video processing."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)

@patch("backend.server.gemini_analyzer")
@patch("backend.server.claude_coach")
def test_full_analyze_pipeline(mock_coach, mock_gemini):
    mock_gemini.analyze.return_value = {"stance": {"width": "good"}}
    mock_coach.get_feedback.return_value = {
        "summary": "Good technique",
        "positives": ["Nice stance"],
        "corrections": [],
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
    assert "corrections" in data
```

**Step 3: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all PASS.

**Step 4: End-to-end manual test**

Set up real `.env` with API keys and test the full flow with a real webcam recording.

**Step 5: Fix any issues found**

Address bugs found during manual testing.

**Step 6: Final commit**

```bash
git add -A
git commit -m "feat: integration tests and polish"
```

---

## Task Dependency Graph

```
Task 1 (scaffolding)
  ├── Task 2 (knowledge docs) ──────────────────┐
  ├── Task 3 (gemini prompts) ──┐                │
  ├── Task 4 (claude prompts) ──┤                │
  ├── Task 5 (video processor)  │                │
  └── Task 6 (RAG engine) ──────┤────────────────┘
       │                        │
       │  Task 7 (gemini) ◄─────┘ (needs prompts)
       │  Task 8 (claude) ◄──────── (needs prompts + RAG)
       │         │
       └─────────┴──► Task 9 (server) ◄── needs 5,7,8
                          │
                     Task 10 (HTML/CSS)
                          │
                     Task 11 (frontend JS) ◄── needs 9,10
                          │
                     Task 12 (integration) ◄── needs everything
```

**Parallelizable groups:**
- **Group A** (after Task 1): Tasks 2, 3, 4, 5, 6 — all independent
- **Group B** (after Group A): Tasks 7, 8, 10 — partially parallel
- **Group C** (after Group B): Tasks 9, 11
- **Group D** (after Group C): Task 12
