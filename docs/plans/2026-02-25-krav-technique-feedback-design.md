# Krav Maga Striking Technique Feedback App — Design

## Overview

A web app that gives post-strike coaching feedback on Krav Maga jab, cross, and jab-cross combinations. The user records a short clip via webcam, and the system analyzes technique across ~25+ checkpoints, returning prioritized coaching corrections.

## Decisions

- **Analysis mode**: Post-strike clip analysis (not real-time)
- **Input method**: Manual record (tap to start/stop)
- **Knowledge base**: All 3 layers — system prompt rubric + RAG + reference images
- **Platform**: MacBook web app (browser-based)
- **AI pipeline**: Gemini 2.5 Pro (vision/video) → Claude (coaching reasoning)
- **Feedback format**: Text only for v1
- **Technique depth**: Full breakdown (~25+ checkpoints)
- **Architecture**: Modular Python backend + separate frontend

## Architecture

### Project structure

```
krav-ai/
├── backend/
│   ├── server.py              # FastAPI app, routes
│   ├── video_processor.py     # Frame extraction, video handling
│   ├── gemini_analyzer.py     # Gemini API calls, structured observation
│   ├── claude_coach.py        # Claude API calls, coaching feedback
│   ├── rag_engine.py          # ChromaDB setup, retrieval
│   ├── prompts/
│   │   ├── gemini_system.py   # Gemini observation prompt + JSON schema
│   │   └── claude_system.py   # Claude coaching rubric + system prompt
│   ├── knowledge/
│   │   ├── references/        # Reference images of correct technique
│   │   └── documents/         # Source material for RAG (PDFs, text)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js                 # Recording, API calls, rendering
└── docs/
    └── plans/
```

### API endpoints

- `POST /api/analyze` — receives video blob + strike type, returns coaching feedback
- `GET /api/health` — health check

### Key dependencies

- FastAPI + uvicorn (server)
- OpenCV / ffmpeg-python (frame extraction)
- ChromaDB (vector store, in-process)
- Anthropic SDK (Claude API)
- OpenAI SDK or httpx (OpenRouter for Gemini)
- sentence-transformers (embeddings for RAG)

### Config

API keys via `.env`:
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`

## Knowledge Base

### Three layers

**Layer 1: System prompt rubric** — structured technique checkpoints always present in the Claude coaching prompt.

**Layer 2: RAG** — curated Krav Maga source material chunked, embedded, and stored in ChromaDB. Retrieved per analysis based on Gemini's observations.

**Layer 3: Reference images** — correct technique at key positions, sent to Gemini alongside the user's video for visual comparison.

### Technique checkpoints (~25+)

**Stance & base**: fighting stance width, weight distribution, rear heel raised, knees bent, bladed angle.

**Guard**: hands at cheekbone height, elbows tucked, chin tucked, shoulders raised.

**Jab mechanics**: lead hand extension path, fist rotation, non-punching hand guard, shoulder covers chin, hip engagement, snap-back retraction.

**Cross mechanics**: rear hand drive, full hip rotation, rear foot pivot, weight transfer, shoulder rotation, full extension.

**Jab-cross combination**: transition fluidity, no telegraph, guard maintenance during transition, stance recovery.

**Footwork & recovery**: no overcommitment, return to stance, balance throughout.

**Common errors**: dropping non-punching hand, telegraphing, chicken-winging, leaning forward, crossing feet.

### RAG pipeline

1. Collect sources (manuals, biomechanics papers, coaching guides)
2. Extract text, chunk into ~300-500 token segments (one concept per chunk)
3. Tag metadata: strike_type, category, source
4. Embed with sentence-transformers, store in ChromaDB
5. At analysis time: query with Gemini observations + strike type, retrieve top 5-8 chunks

### Bootstrap script

`build_knowledge_base.py` reads from `knowledge/documents/`, chunks, embeds, indexes into ChromaDB at `knowledge/chromadb/`. Re-runnable (wipes and rebuilds).

## Video Processing Pipeline

1. **Capture**: Browser MediaRecorder API, WebM format, 720p 30fps
2. **Frame extraction**: ~15 key frames (uniform sampling + peak extension detection)
3. **Gemini analysis**: Full video clip + reference images → structured JSON observations
4. **RAG retrieval**: Query ChromaDB with observations → 5-8 relevant chunks
5. **Claude coaching**: Observations + RAG chunks + rubric + session history → coaching feedback
6. **Display**: Structured text in frontend

### Expected latency

- Frame extraction: ~1-2s
- Gemini analysis: ~5-10s
- RAG retrieval: <1s
- Claude coaching: ~3-5s
- Total: ~10-18s

## Frontend UX

### Three states

**Ready**: Camera preview, record button, strike type selector (jab/cross/jab-cross), previous feedback below.

**Recording**: Red border indicator, elapsed timer, stop button.

**Analyzing**: Clip thumbnail, progress steps ("Extracting frames..." → "Analyzing technique..." → "Generating feedback...").

### Feedback display

- **Summary**: One-line overall assessment
- **What's good**: Positive reinforcement on correct mechanics
- **Corrections**: Prioritized list (most impactful first), each with what's wrong, why it matters, how to fix
- **Session history**: Previous attempts scrollable, enables progress tracking

### Session

No login/accounts. In-memory session history. Backend keeps last 3-5 feedbacks as context for Claude so it can track improvement across attempts.

## Error handling

- Gemini fails → retry once, fall back to Claude vision with extracted frames
- Claude fails → retry once, show raw Gemini observations
- Video too short/long → frontend validation (min 1s, max 15s)

## Phase 1: Research (before code)

Systematically gather:
- Imi Lichtenfeld / foundational Krav Maga striking principles
- IKMF/KMG P1-P2 curriculum on punches
- Biomechanics of the straight punch (sports science)
- Common instructor corrections and drills
- Visual references of correct technique at key positions
