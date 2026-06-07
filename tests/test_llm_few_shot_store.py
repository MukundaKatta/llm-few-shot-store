"""Tests for llm-few-shot-store."""

from __future__ import annotations

import pytest

from llm_few_shot_store import ExampleNotFoundError, ExampleStore, FewShotExample

# ---------------------------------------------------------------------------
# FewShotExample — construction
# ---------------------------------------------------------------------------


def test_example_minimal():
    ex = FewShotExample(id=1, user_input="q", assistant_output="a")
    assert ex.id == 1
    assert ex.user_input == "q"
    assert ex.assistant_output == "a"
    assert ex.tags == frozenset()
    assert ex.metadata == {}
    assert ex.created_at == pytest.approx(0.0)


def test_example_with_tags():
    ex = FewShotExample(
        id=1, user_input="q", assistant_output="a", tags=frozenset({"math", "easy"})
    )
    assert "math" in ex.tags
    assert "easy" in ex.tags


def test_example_repr_short():
    ex = FewShotExample(id=1, user_input="short", assistant_output="a")
    r = repr(ex)
    assert "1" in r
    assert "short" in r


def test_example_repr_long_truncated():
    ex = FewShotExample(id=2, user_input="x" * 40, assistant_output="a")
    assert "..." in repr(ex)


# ---------------------------------------------------------------------------
# FewShotExample — tag_overlap
# ---------------------------------------------------------------------------


def test_tag_overlap_full():
    ex = FewShotExample(
        id=1, user_input="q", assistant_output="a", tags=frozenset({"a", "b", "c"})
    )
    assert ex.tag_overlap(frozenset({"a", "b", "c"})) == 3


def test_tag_overlap_partial():
    ex = FewShotExample(
        id=1, user_input="q", assistant_output="a", tags=frozenset({"a", "b"})
    )
    assert ex.tag_overlap(frozenset({"b", "c"})) == 1


def test_tag_overlap_none():
    ex = FewShotExample(id=1, user_input="q", assistant_output="a", tags=frozenset())
    assert ex.tag_overlap(frozenset({"x"})) == 0


# ---------------------------------------------------------------------------
# FewShotExample — serialisation
# ---------------------------------------------------------------------------


def test_example_to_dict():
    ex = FewShotExample(
        id=3,
        user_input="hello",
        assistant_output="hi",
        tags=frozenset({"greet"}),
        metadata={"k": "v"},
        created_at=5.0,
    )
    d = ex.to_dict()
    assert d["id"] == 3
    assert d["user_input"] == "hello"
    assert d["assistant_output"] == "hi"
    assert d["tags"] == ["greet"]  # sorted list
    assert d["metadata"] == {"k": "v"}
    assert d["created_at"] == pytest.approx(5.0)


def test_example_from_dict_round_trip():
    original = FewShotExample(
        id=7,
        user_input="q",
        assistant_output="a",
        tags=frozenset({"x", "y"}),
        metadata={"n": 1},
        created_at=10.0,
    )
    restored = FewShotExample.from_dict(original.to_dict())
    assert restored.id == original.id
    assert restored.user_input == original.user_input
    assert restored.assistant_output == original.assistant_output
    assert restored.tags == original.tags
    assert restored.metadata == original.metadata
    assert restored.created_at == pytest.approx(original.created_at)


def test_example_from_dict_defaults():
    ex = FewShotExample.from_dict({"id": 1, "user_input": "q", "assistant_output": "a"})
    assert ex.tags == frozenset()
    assert ex.metadata == {}


# ---------------------------------------------------------------------------
# ExampleStore — add
# ---------------------------------------------------------------------------


def test_store_add_assigns_id():
    store = ExampleStore(clock=lambda: 0.0)
    ex = store.add("q", "a")
    assert ex.id == 1


def test_store_add_increments_id():
    store = ExampleStore(clock=lambda: 0.0)
    ex1 = store.add("q1", "a1")
    ex2 = store.add("q2", "a2")
    assert ex1.id == 1
    assert ex2.id == 2


def test_store_add_records_timestamp():
    store = ExampleStore(clock=lambda: 42.0)
    ex = store.add("q", "a")
    assert ex.created_at == pytest.approx(42.0)


def test_store_add_with_tags():
    store = ExampleStore(clock=lambda: 0.0)
    ex = store.add("q", "a", tags={"math"})
    assert "math" in ex.tags


def test_store_add_with_metadata():
    store = ExampleStore(clock=lambda: 0.0)
    ex = store.add("q", "a", metadata={"source": "manual"})
    assert ex.metadata == {"source": "manual"}


# ---------------------------------------------------------------------------
# ExampleStore — get
# ---------------------------------------------------------------------------


def test_store_get():
    store = ExampleStore(clock=lambda: 0.0)
    added = store.add("question", "answer")
    retrieved = store.get(added.id)
    assert retrieved is added


def test_store_get_missing_raises():
    store = ExampleStore()
    with pytest.raises(ExampleNotFoundError) as exc_info:
        store.get(99)
    assert exc_info.value.example_id == 99


# ---------------------------------------------------------------------------
# ExampleStore — all / count / len
# ---------------------------------------------------------------------------


def test_store_all_empty():
    store = ExampleStore()
    assert store.all() == []


def test_store_all_returns_copy():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q", "a")
    copy = store.all()
    copy.clear()
    assert store.count() == 1


def test_store_count():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q1", "a1")
    store.add("q2", "a2")
    assert store.count() == 2
    assert len(store) == 2


# ---------------------------------------------------------------------------
# ExampleStore — by_tag
# ---------------------------------------------------------------------------


def test_by_tag_match():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q1", "a1", tags={"math"})
    store.add("q2", "a2", tags={"science"})
    store.add("q3", "a3", tags={"math"})
    results = store.by_tag("math")
    assert len(results) == 2
    assert all("math" in ex.tags for ex in results)


def test_by_tag_no_match():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q", "a", tags={"math"})
    assert store.by_tag("science") == []


# ---------------------------------------------------------------------------
# ExampleStore — query (no tags)
# ---------------------------------------------------------------------------


def test_query_no_tags_returns_first_n():
    store = ExampleStore(clock=lambda: 0.0)
    for i in range(5):
        store.add(f"q{i}", f"a{i}")
    results = store.query(3)
    assert len(results) == 3
    assert results[0].id == 1


def test_query_n_zero_returns_empty():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q", "a")
    assert store.query(0) == []


def test_query_n_exceeds_count():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q1", "a1")
    store.add("q2", "a2")
    results = store.query(10)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# ExampleStore — query (with tags)
# ---------------------------------------------------------------------------


def test_query_with_tags_ranks_by_overlap():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q1", "a1", tags={"math"})
    store.add("q2", "a2", tags={"math", "easy"})
    store.add("q3", "a3", tags={"science"})
    results = store.query(2, tags={"math", "easy"})
    # q2 has 2 overlapping tags; q1 has 1; q3 has 0
    assert results[0].user_input == "q2"
    assert results[1].user_input == "q1"


def test_query_with_tags_respects_n():
    store = ExampleStore(clock=lambda: 0.0)
    for i in range(5):
        store.add(f"q{i}", f"a{i}", tags={"tag"})
    results = store.query(2, tags={"tag"})
    assert len(results) == 2


def test_query_require_all_tags_filters():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q1", "a1", tags={"math"})
    store.add("q2", "a2", tags={"math", "easy"})
    store.add("q3", "a3", tags={"easy"})
    results = store.query(10, tags={"math", "easy"}, require_all_tags=True)
    assert len(results) == 1
    assert results[0].user_input == "q2"


def test_query_require_all_tags_empty_result():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q", "a", tags={"math"})
    results = store.query(10, tags={"math", "hard"}, require_all_tags=True)
    assert results == []


# ---------------------------------------------------------------------------
# ExampleStore — remove
# ---------------------------------------------------------------------------


def test_remove_existing():
    store = ExampleStore(clock=lambda: 0.0)
    ex = store.add("q", "a")
    store.remove(ex.id)
    assert store.count() == 0


def test_remove_missing_raises():
    store = ExampleStore()
    with pytest.raises(ExampleNotFoundError):
        store.remove(42)


def test_remove_middle():
    store = ExampleStore(clock=lambda: 0.0)
    e1 = store.add("q1", "a1")
    e2 = store.add("q2", "a2")
    e3 = store.add("q3", "a3")
    store.remove(e2.id)
    ids = [ex.id for ex in store.all()]
    assert e1.id in ids
    assert e2.id not in ids
    assert e3.id in ids


# ---------------------------------------------------------------------------
# ExampleStore — clear
# ---------------------------------------------------------------------------


def test_clear_removes_all():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q1", "a1")
    store.add("q2", "a2")
    store.clear()
    assert store.count() == 0


def test_clear_resets_id_counter():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q", "a")
    store.clear()
    ex = store.add("new", "new_a")
    assert ex.id == 1


# ---------------------------------------------------------------------------
# ExampleStore — serialisation
# ---------------------------------------------------------------------------


def test_store_to_dict_round_trip():
    store = ExampleStore(clock=lambda: 1.0)
    store.add("q1", "a1", tags={"math"})
    store.add("q2", "a2", tags={"science"})

    restored = ExampleStore.from_dict(store.to_dict(), clock=lambda: 0.0)
    assert restored.count() == 2
    assert restored.get(1).user_input == "q1"
    assert restored.get(2).user_input == "q2"


def test_store_to_dict_preserves_next_id():
    store = ExampleStore(clock=lambda: 0.0)
    store.add("q", "a")  # id=1
    d = store.to_dict()
    assert d["next_id"] == 2


def test_from_dict_missing_next_id_derives_from_max_id():
    # Data without an explicit "next_id" (e.g. hand-written) must still
    # produce a counter past the highest existing ID, so that subsequent
    # adds cannot collide with restored examples.
    data = {
        "examples": [
            {"id": 1, "user_input": "q1", "assistant_output": "a1"},
            {"id": 5, "user_input": "q5", "assistant_output": "a5"},
        ]
    }
    store = ExampleStore.from_dict(data, clock=lambda: 0.0)
    new = store.add("new", "new_a")
    assert new.id == 6
    ids = [ex.id for ex in store.all()]
    assert len(ids) == len(set(ids))  # no duplicate IDs


def test_from_dict_missing_next_id_empty_store():
    store = ExampleStore.from_dict({"examples": []}, clock=lambda: 0.0)
    assert store.add("q", "a").id == 1


def test_store_repr():
    store = ExampleStore()
    assert "ExampleStore" in repr(store)
