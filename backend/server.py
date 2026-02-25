"""FastAPI server orchestrating the video analysis pipeline."""

import json
import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.claude_coach import ClaudeCoach
from backend.gemini_analyzer import GeminiAnalyzer
from backend.rag_engine import RAGEngine
from backend.video_processor import extract_frames, get_video_duration

load_dotenv()

VALID_STRIKE_TYPES = {"jab", "cross", "jab_cross", "hook", "uppercut", "elbow", "hammer_fist", "knee"}
REFERENCES_DIR = os.path.join(os.path.dirname(__file__), "knowledge", "references")

app = FastAPI(title="Krav AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Service initialization ---

_openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
gemini_analyzer = GeminiAnalyzer(api_key=_openrouter_key)
claude_coach = ClaudeCoach(api_key=_openrouter_key)
rag_engine = RAGEngine()

session_history: list[dict] = []


def _get_reference_images() -> list[str]:
    """Collect reference image paths from the references directory."""
    if not os.path.isdir(REFERENCES_DIR):
        return []
    return [
        os.path.join(REFERENCES_DIR, f)
        for f in sorted(os.listdir(REFERENCES_DIR))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]


def _summarize_observations(observations: dict) -> str:
    """Build a text summary of Gemini observations for RAG querying."""
    parts = []
    for category, fields in observations.items():
        if isinstance(fields, dict):
            for field, value in fields.items():
                parts.append(f"{field}: {value}")
    return "; ".join(parts) if parts else "general striking technique"


# --- API routes (defined BEFORE static mount) ---


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    strike_type: str = Form(...),
):
    # Validate strike type
    if strike_type not in VALID_STRIKE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strike_type '{strike_type}'. Must be one of: {', '.join(sorted(VALID_STRIKE_TYPES))}",
        )

    # Save uploaded video to temp file
    suffix = os.path.splitext(video.filename or "video.webm")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await video.read())
        tmp.close()
        video_path = tmp.name

        # Validate duration
        duration = get_video_duration(video_path)
        if duration < 1 or duration > 15:
            raise HTTPException(
                status_code=400,
                detail=f"Video duration {duration:.1f}s is outside the allowed range (1-15 seconds).",
            )

        # Extract frames (for future use)
        extract_frames(video_path, num_frames=15)

        # Gemini analysis
        reference_images = _get_reference_images()
        try:
            observations = gemini_analyzer.analyze(
                video_path, strike_type, reference_images
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Video analysis failed: {exc}",
            ) from exc

        # RAG retrieval
        obs_summary = _summarize_observations(observations)
        rag_chunks = rag_engine.query(
            obs_summary, n_results=8, strike_type=strike_type
        )

        # Claude coaching
        try:
            feedback = claude_coach.get_feedback(
                strike_type, observations, rag_chunks, session_history
            )
        except RuntimeError:
            # Fallback: return raw observations if Claude fails
            feedback = {
                "summary": "Coaching unavailable — showing raw observations.",
                "positives": [],
                "corrections": [],
                "raw_observations": observations,
            }

        # Track session history (keep last 5)
        session_history.append(feedback)
        if len(session_history) > 5:
            session_history.pop(0)

        return feedback

    finally:
        # Clean up temp file
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# --- Static files (AFTER API routes) ---

_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="static")
