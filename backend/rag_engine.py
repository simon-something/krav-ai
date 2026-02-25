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
