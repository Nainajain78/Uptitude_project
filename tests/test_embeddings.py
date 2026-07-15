"""Tests for src/embeddings.py's clause filtering and result-shaping logic.

Monkeypatches the embedding model and ChromaDB collection so these tests don't
need to download a real model or touch disk-backed vector storage.
"""

from src import embeddings


class _FakeArray(list):
    """Mimics the subset of numpy.ndarray's interface encode() callers rely on."""

    def tolist(self):
        return list(self)


class FakeModel:
    """Stands in for SentenceTransformer.encode(); vectors are irrelevant here
    since these tests only check filtering/shaping, not real similarity."""

    def encode(self, texts, normalize_embeddings=True):
        return _FakeArray([float(len(t))] for t in texts)


class FakeCollection:
    def __init__(self):
        self.upserted = None

    def upsert(self, ids, embeddings, documents, metadatas):
        self.upserted = {"ids": ids, "documents": documents, "metadatas": metadatas}

    def query(self, query_embeddings, n_results):
        return {
            "documents": [["a matched clause"]],
            "metadatas": [[{"contract_id": "c1", "clause_type": "termination_clause"}]],
            "distances": [[0.2]],
        }


def test_embed_and_store_skips_failed_and_not_found(monkeypatch):
    fake_collection = FakeCollection()
    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel())
    monkeypatch.setattr(embeddings, "_get_collection", lambda name: fake_collection)

    results = [
        {
            "contract_id": "c1",
            "termination_clause": "a real clause",
            "confidentiality_clause": "NOT FOUND",
            "liability_clause": "EXTRACTION_FAILED",
        },
    ]

    count = embeddings.embed_and_store(results)

    assert count == 1
    assert fake_collection.upserted["documents"] == ["a real clause"]
    assert fake_collection.upserted["ids"] == ["c1::termination_clause"]
    assert fake_collection.upserted["metadatas"][0] == {
        "contract_id": "c1", "clause_type": "termination_clause",
    }


def test_embed_and_store_returns_zero_when_nothing_valid(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel())
    monkeypatch.setattr(embeddings, "_get_collection", lambda name: FakeCollection())

    results = [{
        "contract_id": "c1",
        "termination_clause": "NOT FOUND",
        "confidentiality_clause": "EXTRACTION_FAILED",
        "liability_clause": "",
    }]

    assert embeddings.embed_and_store(results) == 0


def test_search_clauses_returns_expected_shape(monkeypatch):
    monkeypatch.setattr(embeddings, "_get_model", lambda: FakeModel())
    monkeypatch.setattr(embeddings, "_get_collection", lambda name: FakeCollection())

    matches = embeddings.search_clauses("termination notice period", top_k=1)

    assert matches == [{
        "text": "a matched clause",
        "contract_id": "c1",
        "clause_type": "termination_clause",
        "score": 0.8,
    }]
