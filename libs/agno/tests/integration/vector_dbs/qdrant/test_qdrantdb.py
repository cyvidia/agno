
import os
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from agno.filters import EQ, IN, GT, LT, AND, OR, NOT
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.search import SearchType
from agno.knowledge.document import Document


# --------------------------------------------------------------------
# Qdrant availability / configuration
# --------------------------------------------------------------------
# defaults to docker-compose setup
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def _qdrant_is_available() -> bool:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_is_available(),
    reason="Qdrant server is not available at configured host/port",
)

# TODO: reuse mock_embedder from libs/agno/tests/unit/vectordb/conftest.py instead of duping
@pytest.fixture(scope="module")
def mock_embedder():
    """Create a mock embedder with appropriate return values."""
    mock = MagicMock()

    # Mock dimensions property
    mock.dimensions = 1024

    # Create a fixed embedding vector of the correct size
    mock_embedding: List[float] = [0.1] * 1024

    # Mock the get_embedding method
    mock.get_embedding.return_value = mock_embedding

    # Mock the get_embedding_and_usage method
    mock_usage: Dict[str, Any] = {"prompt_tokens": 10, "total_tokens": 10}
    mock.get_embedding_and_usage.return_value = (mock_embedding, mock_usage)

    return mock

@pytest.fixture(scope="module")
def qdrant_instance(mock_embedder):
    """Real Qdrant-backed instance with a temporary collection."""
    collection_name = f"test_qdrant_filters_{uuid4().hex}"

    q = Qdrant(
        collection=collection_name,
        embedder=mock_embedder,
        search_type=SearchType.vector,
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    # Create collection in real Qdrant
    q.create()

    try:
        yield q
    finally:
        # Clean up collection after tests
        try:
            q.drop()
        except Exception:
            # Best-effort cleanup
            pass


def wait_for_count(q: Qdrant, expected: int, timeout: float = 10.0) -> None:
    """Wait until Qdrant collection has at least `expected` points."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if q.get_count() >= expected:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for collection to reach count {expected}")

@pytest.fixture(scope="module")
def seeded_documents(qdrant_instance: Qdrant):
    """Seed some real documents into Qdrant for filter tests."""
    docs = [
        Document(
            name="doc1",
            content="same content for all docs",
            meta_data={"status": "published", "views": 50, "category": "tech"},
        ),
        Document(
            name="doc2",
            content="same content for all docs",
            meta_data={
                "status": "published",
                "views": 150,
                "category": "science",
                "type": "article",
                "word_count": 600,
            },
        ),
        Document(
            name="doc3",
            content="same content for all docs",
            meta_data={
                "status": "draft",
                "views": 200,
                "category": "tech",
                "type": "tutorial",
                "word_count": 400,
                "difficulty": "beginner",
            },
        ),
        Document(
            name="doc4",
            content="same content for all docs",
            meta_data={
                "status": "archived",
                "views": 80,
                "category": "science",
                "type": "tutorial",
                "word_count": 700,
                "difficulty": "advanced",
            },
        ),
    ]

    qdrant_instance.insert(content_hash="seed_hash", documents=docs, filters=None)
    wait_for_count(qdrant_instance, expected=len(docs))

    return docs


class TestSearchIntegration:
    def test_eq_filter_returns_only_matching_docs(self, qdrant_instance: Qdrant, seeded_documents):
        q = qdrant_instance

        expr = EQ("status", "published")
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        assert names == {"doc1", "doc2"}

    def test_in_filter(self, qdrant_instance: Qdrant, seeded_documents):
        q = qdrant_instance

        expr = IN("category", ["science"])
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        assert names == {"doc2", "doc4"}

    def test_gt_filter(self, qdrant_instance: Qdrant, seeded_documents):
        q = qdrant_instance

        expr = GT("views", 100)
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        # views > 100 → doc2 (150), doc3 (200), doc4 (80 excluded)
        assert names == {"doc2", "doc3"}

    def test_lt_filter(self, qdrant_instance: Qdrant, seeded_documents):
        q = qdrant_instance

        expr = LT("views", 100)
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        # views < 100 → doc1 (50), doc4 (80)
        assert names == {"doc1", "doc4"}

    def test_and_filter(self, qdrant_instance: Qdrant, seeded_documents):
        q = qdrant_instance

        expr = AND(EQ("status", "published"), GT("views", 100))
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        # published AND views > 100 → only doc2
        assert names == {"doc2"}

    def test_or_filter(self, qdrant_instance: Qdrant, seeded_documents):
        q = qdrant_instance

        expr = OR(EQ("status", "draft"), EQ("status", "archived"))
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        assert names == {"doc3", "doc4"}

    def test_not_filter(self, qdrant_instance: Qdrant, seeded_documents):
        q = qdrant_instance

        expr = NOT(EQ("status", "draft"))
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        # all docs except the draft one
        assert names == {"doc1", "doc2", "doc4"}

    def test_nested_filter(self, qdrant_instance: Qdrant, seeded_documents):
        q = qdrant_instance

        expr = OR(
            AND(EQ("type", "article"), GT("word_count", 500)),
            AND(EQ("type", "tutorial"), NOT(EQ("difficulty", "beginner"))),
        )

        results = q.search("whatever", limit=10, filters=[expr])
        names = {d.name for d in results}

        # doc2: article & word_count 600 → matches
        # doc4: tutorial & difficulty != beginner → matches
        # doc3: tutorial & difficulty == beginner → excluded
        assert names == {"doc2", "doc4"}

    # ---------- New, more extensive coverage below ----------

    def test_filter_list_implicit_and(self, qdrant_instance: Qdrant, seeded_documents):
        """Passing a list of FilterExpr should behave like AND of all."""
        q = qdrant_instance

        filters = [EQ("status", "published"), GT("views", 100)]
        results = q.search("whatever", limit=10, filters=filters)

        names = {d.name for d in results}
        # published AND views > 100 → only doc2
        assert names == {"doc2"}

    def test_operator_overloads_expression(self, qdrant_instance: Qdrant, seeded_documents):
        """Use &, |, ~ operators to build complex expressions."""
        q = qdrant_instance

        # (published & views > 100) | archived
        expr = (EQ("status", "published") & GT("views", 100)) | EQ("status", "archived")
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        # (doc2) OR (doc4) → doc2, doc4
        assert names == {"doc2", "doc4"}

    def test_in_and_numeric_combination(self, qdrant_instance: Qdrant, seeded_documents):
        """Combine IN + GT/LT to ensure multiple conditions work together."""
        q = qdrant_instance

        # category in {tech, science} AND views > 100
        expr = AND(
            IN("category", ["tech", "science"]),
            GT("views", 100),
        )
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        # views > 100: doc2 (science), doc3 (tech) → both match category filter
        assert names == {"doc2", "doc3"}

    def test_no_match_filter_returns_empty_list(self, qdrant_instance: Qdrant, seeded_documents):
        """Filter that matches nothing should return empty results."""
        q = qdrant_instance

        expr = EQ("status", "nonexistent-status")
        results = q.search("whatever", limit=10, filters=[expr])

        assert results == []

    def test_dict_filters_integration(self, qdrant_instance: Qdrant, seeded_documents):
        """Dict-style filters should work end-to-end via _format_filters."""
        q = qdrant_instance

        # Equivalent to EQ("status", "published") on meta_data.status
        filters_dict = {"status": "published"}
        results = q.search("whatever", limit=10, filters=filters_dict)

        names = {d.name for d in results}
        assert names == {"doc1", "doc2"}

    def test_mixed_or_not_expression(self, qdrant_instance: Qdrant, seeded_documents):
        """More complex expression with OR and NOT at the same level."""
        q = qdrant_instance

        # (status == published OR status == draft) AND NOT(category == science)
        expr = AND(
            OR(EQ("status", "published"), EQ("status", "draft")),
            NOT(IN("category", ["science"])),
        )

        results = q.search("whatever", limit=10, filters=[expr])
        names = {d.name for d in results}

        # published: doc1 (tech), doc2 (science)
        # draft: doc3 (tech)
        # category != science → doc1, doc3
        assert names == {"doc1", "doc3"}