"""Embedding generation and semantic search over extracted clauses (sentence-transformers + ChromaDB).

sentence-transformers is used instead of an Ollama embedding model because it
runs in-process with no server dependency: clause embedding/search works even
if Ollama isn't running (e.g. re-running search well after extraction already
completed and was cached). all-MiniLM-L6-v2 is small (~80MB), fast on CPU, and
is the de facto default embedding model for ChromaDB demos.
"""

import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = BASE_DIR / "outputs" / "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "contract_clauses"

CLAUSE_FIELDS = ("termination_clause", "confidentiality_clause", "liability_clause")
_FAILED_CLAUSE_MARKERS = {"NOT FOUND", "EXTRACTION_FAILED", ""}

_model = None
_client = None


def _get_model() -> SentenceTransformer:
    """Lazily load the sentence-transformers embedding model, reused across calls."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_chroma_client() -> "chromadb.ClientAPI":
    """Lazily create a persistent ChromaDB client, reused across calls."""
    global _client
    if _client is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return _client


def _get_collection(collection_name: str):
    """Get or create a ChromaDB collection configured for cosine similarity."""
    client = _get_chroma_client()
    return client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})


def embed_and_store(results: list[dict], collection_name: str = DEFAULT_COLLECTION_NAME) -> int:
    """Embed and store extracted clauses from pipeline results into ChromaDB.

    Iterates each contract's termination/confidentiality/liability clauses,
    skipping any that are empty or failed extraction ("NOT FOUND" /
    "EXTRACTION_FAILED"), and upserts the rest into a persistent ChromaDB
    collection with {contract_id, clause_type} metadata for semantic search.

    Args:
        results: Pipeline result dicts (see pipeline.py's run_pipeline), each
            with a contract_id and the three clause fields.
        collection_name: Name of the ChromaDB collection to write to.

    Returns:
        The number of clauses actually embedded and stored.
    """
    collection = _get_collection(collection_name)
    model = _get_model()

    ids, documents, metadatas = [], [], []
    for result in results:
        contract_id = result.get("contract_id", "unknown")
        for clause_type in CLAUSE_FIELDS:
            text = (result.get(clause_type) or "").strip()
            if not text or text.upper() in _FAILED_CLAUSE_MARKERS:
                continue

            ids.append(f"{contract_id}::{clause_type}")
            documents.append(text)
            metadatas.append({"contract_id": contract_id, "clause_type": clause_type})

    if not documents:
        logger.warning("No valid clauses found to embed across %d contract(s).", len(results))
        return 0

    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    logger.info("Stored %d clause(s) in collection '%s'.", len(documents), collection_name)
    return len(documents)


def search_clauses(
    query: str, top_k: int = 5, collection_name: str = DEFAULT_COLLECTION_NAME
) -> list[dict]:
    """Semantically search stored clauses for the closest matches to a query.

    Args:
        query: Natural-language search query (e.g. "indemnification obligations").
        top_k: Maximum number of matches to return.
        collection_name: Name of the ChromaDB collection to query.

    Returns:
        A list of dicts with "text", "contract_id", "clause_type", and "score"
        (cosine similarity, higher is more similar), ordered by descending score.
    """
    collection = _get_collection(collection_name)
    model = _get_model()

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    raw = collection.query(query_embeddings=query_embedding, n_results=top_k)

    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    return [
        {
            "text": text,
            "contract_id": metadata.get("contract_id"),
            "clause_type": metadata.get("clause_type"),
            "score": 1 - distance,  # cosine distance -> cosine similarity
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    example_queries = [
        "indemnification obligations",
        "early termination notice period",
        "limitation of liability cap",
    ]

    for query in example_queries:
        print(f'\n=== Query: "{query}" ===')
        matches = search_clauses(query, top_k=3)

        if not matches:
            print("No results. Run embed_and_store(results) on pipeline output first.")
            continue

        for i, match in enumerate(matches, start=1):
            print(f"{i}. [{match['clause_type']}] {match['contract_id']} (score={match['score']:.3f})")
            print(f"   {match['text'][:200]}")
