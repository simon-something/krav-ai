# Krav AI

AI-powered technique feedback for Krav Maga strikes. Record a short clip of yourself throwing a strike, and get detailed coaching corrections from an AI pipeline that combines video analysis with a curated knowledge base of Krav Maga technique.

## How it works

1. **Record** a 1-15 second clip of yourself via webcam
2. **Gemini 2.5 Pro** analyzes the video for body mechanics (joint angles, stance, timing, rotation)
3. **RAG retrieval** pulls relevant technique material from a curated Krav Maga knowledge base (82 chunks across 11 documents)
4. **Claude Sonnet 4.6** synthesizes coaching feedback: what's good, what to correct, and how to fix it
5. **Session tracking** remembers your last 5 attempts so feedback can reference your progress

## Supported strikes

| Strike | Checkpoints |
|--------|------------|
| Jab | 7 |
| Cross | 8 |
| Jab-Cross combo | 5 |
| Hook (lead/rear) | 6 |
| Uppercut (lead/rear) | 6 |
| Elbow (horizontal/vertical/diagonal/backward) | 6 |
| Hammer Fist (downward/forward/sideways) | 5 |
| Knee (straight/round/clinch) | 6 |

60+ technique checkpoints covering stance, guard, strike mechanics, footwork, telegraphing, and common errors.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An [OpenRouter](https://openrouter.ai/) API key (powers both Gemini and Claude)

### Install

```bash
git clone https://github.com/simon-something/krav-ai.git
cd krav-ai
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### Build the knowledge base

```bash
uv run python -m backend.build_knowledge_base
```

### Run

```bash
uv run uvicorn backend.server:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

1. Allow camera access when prompted
2. Select your strike type from the dropdown
3. Position yourself so your upper body (and legs for knee strikes) is visible
4. Click **Record**, throw your strike, click **Stop**
5. Wait ~10-18 seconds for analysis
6. Read the feedback: summary, positives, and prioritized corrections
7. Adjust and repeat

## Architecture

```
Browser (webcam) ──> FastAPI server
                         │
                    ┌────┴────┐
                    │ Video   │
                    │Processor│
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │ Gemini  │  (video analysis via OpenRouter)
                    │2.5 Pro  │
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │  RAG    │  (ChromaDB + sentence-transformers)
                    │ Engine  │
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │ Claude  │  (coaching via OpenRouter)
                    │Sonnet4.6│
                    └─────────┘
```

## Project structure

```
krav-ai/
├── backend/
│   ├── server.py              # FastAPI app, routes, orchestration
│   ├── video_processor.py     # OpenCV frame extraction
│   ├── gemini_analyzer.py     # Gemini 2.5 Pro via OpenRouter
│   ├── claude_coach.py        # Claude Sonnet 4.6 via OpenRouter
│   ├── rag_engine.py          # ChromaDB vector store
│   ├── build_knowledge_base.py
│   ├── prompts/
│   │   ├── gemini_system.py   # Observation schema + prompt
│   │   └── claude_system.py   # Technique rubric (60+ checkpoints)
│   └── knowledge/
│       ├── documents/         # 11 curated technique docs
│       └── references/        # Reference images (add your own)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── tests/                     # 30 tests
```

## Tests

```bash
uv run pytest tests/ -v
```

## Adding reference images

For better analysis, add reference images of correct technique to `backend/knowledge/references/`. These are sent alongside your video to Gemini for visual comparison. Use `.jpg`, `.png`, or `.webp` format.

## License

MIT
