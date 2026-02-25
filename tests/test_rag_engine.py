import shutil
import uuid

import pytest

from backend.rag_engine import RAGEngine


@pytest.fixture
def rag_engine(tmp_path):
    db_path = str(tmp_path / f"test_chromadb_{uuid.uuid4().hex[:8]}")
    engine = RAGEngine(db_path=db_path)
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
    shutil.rmtree(db_path, ignore_errors=True)


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
