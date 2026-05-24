# llm-few-shot-store

Store and retrieve few-shot examples for LLM prompts.

Add input/output pairs with tags, then query for the best-matching examples using tag overlap. No embeddings, no external dependencies.

## Install

```bash
pip install llm-few-shot-store
```

## Quick start

```python
from llm_few_shot_store import ExampleStore

store = ExampleStore()
store.add("What is 2+2?",           "4",      tags={"math", "arithmetic"})
store.add("Solve x² = 9.",          "x=±3",   tags={"math", "algebra"})
store.add("What is the capital of France?", "Paris", tags={"geography"})

# top-2 examples most relevant to a math question
examples = store.query(2, tags={"math"})
for ex in examples:
    print(f"User: {ex.user_input}")
    print(f"Assistant: {ex.assistant_output}")
```

## API

### `ExampleStore`

| Method | Description |
|--------|-------------|
| `add(user_input, assistant_output, *, tags, metadata)` | Add example; returns `FewShotExample` |
| `get(id)` | Return example by integer ID |
| `all()` | All examples in insertion order |
| `by_tag(tag)` | Examples containing a specific tag |
| `query(n, *, tags, require_all_tags)` | Top-*n* by tag overlap |
| `remove(id)` | Delete by ID |
| `clear()` | Remove all; resets ID counter |
| `to_dict()` / `ExampleStore.from_dict(d)` | Serialise / restore |
| `count()` | Number of stored examples |

### `query()` ranking

Examples are sorted by the number of tags they share with the requested tag set. Ties are broken by insertion order (oldest first). Pass `require_all_tags=True` to filter to examples that contain *every* requested tag.

### `FewShotExample`

| Attribute | Type |
|-----------|------|
| `id` | `int` (auto-assigned) |
| `user_input` | `str` |
| `assistant_output` | `str` |
| `tags` | `frozenset[str]` |
| `metadata` | `dict` |
| `created_at` | `float` (Unix time) |

Method: `tag_overlap(query_tags)` → `int`

Serialisation: `to_dict()` / `FewShotExample.from_dict(d)`

## License

MIT
