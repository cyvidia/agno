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
from qdrant_client.http.models import PayloadSchemaType



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

    client = q._client
    assert client is not None

    # create indexes

    for field, schema in [
        ("status", PayloadSchemaType.KEYWORD),
        ("views", PayloadSchemaType.INTEGER),
        ("category", PayloadSchemaType.KEYWORD),
        ("type", PayloadSchemaType.KEYWORD),
        ("word_count", PayloadSchemaType.INTEGER),
        ("difficulty", PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(collection_name, field_name=f"meta_data.{field}", field_schema=schema)
    try:
        yield q
    finally:
        # Clean up collection after tests
        try:
            q.drop()
        except Exception:
            # Best-effort cleanup
            pass



@pytest.fixture(scope="module")
def seeded_documents(qdrant_instance: Qdrant):
    """Seed some real documents into Qdrant for filter tests."""
    docs = [
        Document(
            name="doc1",
            content="doc 1 content",
            meta_data={"status": "published", "views": 50, "category": "tech"},
        ),
        Document(
            name="doc2",
            content="doc 2 content",
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
            content="doc 3 content",
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
            content="doc 4 content",
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

    return docs


class TestSearchIntegration:
    # -------------------- basic semantics --------------------

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

    def test_or_with_in_on_multiple_fields(self, qdrant_instance: Qdrant, seeded_documents):
        """OR(IN(...), IN(...)) across different fields should match union of both sets."""
        q = qdrant_instance

        # status ∈ {draft}  OR  views ∈ {50, 80}
        expr = OR(
            IN("status", ["draft"]),
            IN("views", [50, 80]),
        )

        results = q.search("whatever", limit=10, filters=[expr])
        names = {d.name for d in results}

        # From seeded docs:
        # doc1: status=published, views=50         → matches via views
        # doc2: status=published, views=150        → no match
        # doc3: status=draft,     views=200        → matches via status
        # doc4: status=archived,  views=80         → matches via views
        assert names == {"doc1", "doc3", "doc4"}

    # -------------------- additional coverage --------------------

    def test_no_filters_returns_all_docs(self, qdrant_instance: Qdrant, seeded_documents):
        """Calling search with no filters should return all seeded docs (up to limit)."""
        q = qdrant_instance

        results = q.search("whatever", limit=10)
        names = {d.name for d in results}

        assert names == {d.name for d in seeded_documents}

    def test_empty_filter_list_behaves_like_no_filters(self, qdrant_instance: Qdrant, seeded_documents):
        """filters=[] should behave like no filters at all."""
        q = qdrant_instance

        results = q.search("whatever", limit=10, filters=[])
        names = {d.name for d in results}

        assert names == {d.name for d in seeded_documents}

    def test_limit_parameter_is_respected(self, qdrant_instance: Qdrant, seeded_documents):
        """Ensure the limit parameter is respected by the underlying search."""
        q = qdrant_instance

        results = q.search("whatever", limit=2)
        assert len(results) == 2

    def test_eq_on_numeric_field(self, qdrant_instance: Qdrant, seeded_documents):
        """EQ should also work on numeric fields like views."""
        q = qdrant_instance

        expr = EQ("views", 80)
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        assert names == {"doc4"}

    def test_numeric_range_with_and(self, qdrant_instance: Qdrant, seeded_documents):
        """Combine GT and LT to form an open numeric range."""
        q = qdrant_instance

        # 400 < word_count < 700
        expr = AND(
            GT("word_count", 400),
            LT("word_count", 700),
        )

        results = q.search("whatever", limit=10, filters=[expr])
        names = {d.name for d in results}

        # doc2: 600 → in range
        # doc3: 400 → excluded (not > 400)
        # doc4: 700 → excluded (not < 700)
        assert names == {"doc2"}

    def test_nested_not_over_or(self, qdrant_instance: Qdrant, seeded_documents):
        """NOT(OR(...)) should correctly exclude both branches."""
        q = qdrant_instance

        # NOT(status in {draft, archived})
        expr = NOT(OR(EQ("status", "draft"), EQ("status", "archived")))
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        # Only published docs remain
        assert names == {"doc1", "doc2"}

    def test_bitwise_not_operator_shortcut(self, qdrant_instance: Qdrant, seeded_documents):
        """~expr should behave like NOT(expr)."""
        q = qdrant_instance

        expr = ~EQ("status", "archived")
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        # All except archived
        assert names == {"doc1", "doc2", "doc3"}

    def test_bitwise_not_in_complex_expression(self, qdrant_instance: Qdrant, seeded_documents):
        """Mix ~ with AND/IN for more complex expressions."""
        q = qdrant_instance

        # NOT(status == archived) AND category in {science, tech}
        expr = AND(
            ~EQ("status", "archived"),
            IN("category", ["science", "tech"]),
        )

        results = q.search("whatever", limit=10, filters=[expr])
        names = {d.name for d in results}

        # All docs except doc4 (archived), but only matching category {science, tech}
        # doc1 (tech), doc2 (science), doc3 (tech)
        assert names == {"doc1", "doc2", "doc3"}

    def test_in_on_numeric_field(self, qdrant_instance: Qdrant, seeded_documents):
        """IN should work on numeric payload fields as well."""
        q = qdrant_instance

        # views ∈ {80, 200}
        expr = IN("views", [80, 200])
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        assert names == {"doc3", "doc4"}

    def test_in_on_word_count_field(self, qdrant_instance: Qdrant, seeded_documents):
        """IN on integer word_count metadata."""
        q = qdrant_instance

        expr = IN("word_count", [400, 700])
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        assert names == {"doc3", "doc4"}

    def test_multi_key_dict_filters_act_as_and(self, qdrant_instance: Qdrant, seeded_documents):
        """Dict filters with multiple keys should AND all equality conditions."""
        q = qdrant_instance

        filters_dict = {
            "status": "published",
            "category": "science",
        }
        results = q.search("whatever", limit=10, filters=filters_dict)

        names = {d.name for d in results}
        # Only doc2 is published AND science
        assert names == {"doc2"}

    def test_nested_and_or_three_way_logic(self, qdrant_instance: Qdrant, seeded_documents):
        """Extra coverage for mixing AND/OR with multiple branches."""
        q = qdrant_instance

        # (status == published OR status == draft) AND views > 100
        expr = AND(
            OR(EQ("status", "published"), EQ("status", "draft")),
            GT("views", 100),
        )

        results = q.search("whatever", limit=10, filters=[expr])
        names = {d.name for d in results}

        # published with views > 100: doc2
        # draft with views > 100: doc3
        assert names == {"doc2", "doc3"}

    def test_single_expression_in_list_same_as_direct_expr(self, qdrant_instance: Qdrant, seeded_documents):
        """filters=[expr] should behave the same as filters=expr (if supported)."""
        q = qdrant_instance

        expr = EQ("category", "tech")
        results_from_list = q.search("whatever", limit=10, filters=[expr])
        names_from_list = {d.name for d in results_from_list}

        # sanity: doc1, doc3 are tech
        assert names_from_list == {"doc1", "doc3"}

    def test_missing_field_in_filter_excludes_doc(self, qdrant_instance: Qdrant, seeded_documents):
        """
        Filtering on a field that some docs don't have should exclude those docs,
        but still match ones that do have it.
        """
        q = qdrant_instance

        # difficulty == advanced
        expr = EQ("difficulty", "advanced")
        results = q.search("whatever", limit=10, filters=[expr])

        names = {d.name for d in results}
        # Only doc4 has difficulty=advanced
        assert names == {"doc4"}

    def test_three_simple_filter_expressions_implicit_and(
        self, qdrant_instance: Qdrant, seeded_documents
        ):
            """
            Multiple simple expressions in the filters list should all be AND-ed together.
            """
            q = qdrant_instance

            filters = [
                EQ("status", "published"),
                IN("category", ["science", "tech"]),
                LT("views", 100),
            ]
            results = q.search("whatever", limit=10, filters=filters)

            names = {d.name for d in results}
            # Only doc1: published, category=tech, views=50 (< 100)
            assert names == {"doc1"}

    def test_multiple_filter_expressions_with_or_and_gt_implicit_and(
        self, qdrant_instance: Qdrant, seeded_documents
    ):
        """
        A list containing a complex OR expression plus another expression
        should still be treated as implicit AND for the list items.
        """
        q = qdrant_instance

        # (status == draft OR status == archived) AND views > 100
        filters = [
            OR(EQ("status", "draft"), EQ("status", "archived")),
            GT("views", 100),
        ]
        results = q.search("whatever", limit=10, filters=filters)

        names = {d.name for d in results}
        # draft + views>100 → doc3; archived has views=80 → excluded
        assert names == {"doc3"}

    def test_multiple_filter_expressions_with_not_and_gt_implicit_and(
        self, qdrant_instance: Qdrant, seeded_documents
    ):
        """
        List of NOT(...) and numeric comparison should also AND together.
        """
        q = qdrant_instance

        # NOT(category in {science}) AND views > 50
        filters = [
            NOT(IN("category", ["science"])),
            GT("views", 50),
        ]
        results = q.search("whatever", limit=10, filters=filters)

        names = {d.name for d in results}
        # doc3: tech, views=200 → matches
        # doc1: tech, views=50 → fails GT(views, 50)
        # science docs excluded by NOT(IN(...))
        assert names == {"doc3"}

    def test_multiple_filter_expressions_for_tutorial_range(
        self, qdrant_instance: Qdrant, seeded_documents
    ):
        """
        Multiple filters targeting tutorials with word_count constraint.
        """
        q = qdrant_instance

        # type == tutorial AND word_count > 500 AND views < 1000
        filters = [
            EQ("type", "tutorial"),
            GT("word_count", 500),
            LT("views", 1000),
        ]
        results = q.search("whatever", limit=10, filters=filters)

        names = {d.name for d in results}
        # doc3: tutorial, word_count=400 → excluded
        # doc4: tutorial, word_count=700, views=80 → matches
        assert names == {"doc4"}

