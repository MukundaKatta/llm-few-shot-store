"""Store and retrieve few-shot examples for LLM prompts.

:class:`FewShotExample` holds a single input/output pair together with tags
and arbitrary metadata.  :class:`ExampleStore` manages a collection of
examples and provides tag-based retrieval.

When you call :meth:`ExampleStore.query`, examples are ranked by the number
of tags they share with the requested tag set — the most-overlapping examples
come first.  Ties are broken by insertion order (oldest first).  This gives a
lightweight, zero-dependency heuristic for relevance without embeddings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


class ExampleNotFoundError(KeyError):
    """Raised when an example ID is not found in the store."""

    def __init__(self, example_id: int) -> None:
        self.example_id = example_id
        super().__init__(f"Example {example_id!r} not found.")


@dataclass
class FewShotExample:
    """A single few-shot example.

    Attributes:
        id: Auto-assigned integer ID (set by :class:`ExampleStore`).
        user_input: The user turn to show in the prompt.
        assistant_output: The expected assistant response.
        tags: Labels for filtering and ranking.
        metadata: Arbitrary extra data.
        created_at: Unix timestamp of insertion.
    """

    id: int
    user_input: str
    assistant_output: str
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def tag_overlap(self, query_tags: frozenset[str]) -> int:
        """Number of tags shared with *query_tags*."""
        return len(self.tags & query_tags)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "id": self.id,
            "user_input": self.user_input,
            "assistant_output": self.assistant_output,
            "tags": sorted(self.tags),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FewShotExample:
        """Reconstruct a :class:`FewShotExample` from a plain dict."""
        return cls(
            id=int(data["id"]),
            user_input=data["user_input"],
            assistant_output=data["assistant_output"],
            tags=frozenset(data.get("tags") or []),
            metadata=dict(data.get("metadata") or {}),
            created_at=float(data.get("created_at", 0.0)),
        )

    def __repr__(self) -> str:
        preview = (
            self.user_input[:35] + "..."
            if len(self.user_input) > 35
            else self.user_input
        )
        return (
            f"FewShotExample(id={self.id}, tags={set(self.tags)!r}, input={preview!r})"
        )


class ExampleStore:
    """An ordered collection of :class:`FewShotExample` objects.

    IDs are auto-assigned integers starting at 1.

    Args:
        clock: Callable returning current Unix time.  Defaults to
            :func:`time.time`.

    Example::

        store = ExampleStore()
        store.add("What is 2+2?", "4", tags={"math", "arithmetic"})
        store.add("What is the capital of France?", "Paris", tags={"geography"})

        # top-2 examples that overlap with math tag
        examples = store.query(n=2, tags={"math"})
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._examples: list[FewShotExample] = []
        self._next_id: int = 1
        self._clock: Callable[[], float] = clock if clock is not None else time.time

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(
        self,
        user_input: str,
        assistant_output: str,
        *,
        tags: set[str] | frozenset[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FewShotExample:
        """Add a new example and return it.

        Args:
            user_input: The user turn.
            assistant_output: The expected response.
            tags: Optional set of string labels.
            metadata: Optional extra data.

        Returns:
            The new :class:`FewShotExample` with its assigned ID.
        """
        example = FewShotExample(
            id=self._next_id,
            user_input=user_input,
            assistant_output=assistant_output,
            tags=frozenset(tags or []),
            metadata=dict(metadata or {}),
            created_at=self._clock(),
        )
        self._examples.append(example)
        self._next_id += 1
        return example

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, example_id: int) -> FewShotExample:
        """Return the example with *example_id*.

        Raises:
            ExampleNotFoundError: If not found.
        """
        for ex in self._examples:
            if ex.id == example_id:
                return ex
        raise ExampleNotFoundError(example_id)

    def all(self) -> list[FewShotExample]:
        """All examples in insertion order."""
        return list(self._examples)

    def by_tag(self, tag: str) -> list[FewShotExample]:
        """Return all examples that include *tag*, in insertion order."""
        return [ex for ex in self._examples if tag in ex.tags]

    def query(
        self,
        n: int,
        *,
        tags: set[str] | frozenset[str] | None = None,
        require_all_tags: bool = False,
    ) -> list[FewShotExample]:
        """Return up to *n* examples, ranked by tag overlap with *tags*.

        Args:
            n: Maximum number of examples to return.
            tags: Tag set to rank by.  If ``None``, returns the first *n*
                examples in insertion order.
            require_all_tags: If ``True``, only return examples that contain
                every tag in *tags*.

        Returns:
            Up to *n* :class:`FewShotExample` objects, best match first.
        """
        if n <= 0:
            return []
        candidates = self._examples
        if tags:
            query_tags = frozenset(tags)
            if require_all_tags:
                candidates = [ex for ex in candidates if query_tags <= ex.tags]
            # rank by overlap descending; stable sort preserves insertion order
            candidates = sorted(
                candidates,
                key=lambda ex: ex.tag_overlap(query_tags),
                reverse=True,
            )
        return candidates[:n]

    def count(self) -> int:
        """Total number of stored examples."""
        return len(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, example_id: int) -> None:
        """Remove the example with *example_id*.

        Raises:
            ExampleNotFoundError: If not found.
        """
        for i, ex in enumerate(self._examples):
            if ex.id == example_id:
                self._examples.pop(i)
                return
        raise ExampleNotFoundError(example_id)

    def clear(self) -> None:
        """Remove all examples.  Resets the ID counter to 1."""
        self._examples.clear()
        self._next_id = 1

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the store to a plain dict."""
        return {
            "next_id": self._next_id,
            "examples": [ex.to_dict() for ex in self._examples],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        clock: Callable[[], float] | None = None,
    ) -> ExampleStore:
        """Reconstruct an :class:`ExampleStore` from a plain dict."""
        store = cls(clock=clock)
        for ed in data.get("examples", []):
            store._examples.append(FewShotExample.from_dict(ed))
        if "next_id" in data:
            store._next_id = int(data["next_id"])
        else:
            # No explicit counter: derive it from the existing IDs so that
            # freshly added examples never collide with restored ones.
            highest = max((ex.id for ex in store._examples), default=0)
            store._next_id = highest + 1
        return store

    def __repr__(self) -> str:
        return f"ExampleStore(count={len(self._examples)})"
