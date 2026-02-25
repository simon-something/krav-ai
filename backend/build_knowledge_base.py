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
            for cat in [
                "stance",
                "guard",
                "mechanics",
                "footwork",
                "errors",
                "biomechanics",
            ]:
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
