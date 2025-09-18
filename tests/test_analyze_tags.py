from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from analyze_tags import (  # type: ignore[attr-defined]
    build_card_family_projection,
    build_collapsed_counts,
    build_collapsed_tag_map,
    build_family_counts,
    build_family_summary,
    compute_cooccurrence,
    compute_tag_counts,
)


SAMPLE_CARDS = [
    {
        "id": "1",
        "name": "First Card",
        "tags": ["mechanic:scry", "theme:spell_matter", "role:card_draw"],
    },
    {
        "id": "2",
        "name": "Second Card",
        "tags": ["mechanic:scry", "role:card_draw", "theme:control"],
    },
    {
        "id": "3",
        "name": "Third Card",
        "tags": ["mechanic:venture", "theme:dungeon"],
    },
]


def test_compute_tag_counts() -> None:
    counts = compute_tag_counts(SAMPLE_CARDS)
    assert counts["mechanic:scry"] == 2
    assert counts["mechanic:venture"] == 1
    assert counts["theme:spell_matter"] == 1


def test_compute_cooccurrence_filters_threshold() -> None:
    pairs = compute_cooccurrence(SAMPLE_CARDS, min_count=2)
    assert pairs == [("mechanic:scry", "role:card_draw", 2)]


def test_family_helpers() -> None:
    counts = compute_tag_counts(SAMPLE_CARDS)
    collapsed_map = build_collapsed_tag_map(counts, collapse_threshold=1)

    # mechanic:venture should collapse to mechanic:_other due to threshold
    assert collapsed_map["mechanic:venture"] == "mechanic:_other"

    summary = build_family_summary(counts, collapsed_map)
    assert summary["mechanic:scry"]["family"] == "mechanic"
    assert summary["mechanic:venture"]["collapsed"] == "mechanic:_other"

    family_counts = build_family_counts(counts)
    assert family_counts["mechanic"] == 3

    collapsed_counts = build_collapsed_counts(SAMPLE_CARDS, collapsed_map)
    assert collapsed_counts["mechanic:_other"] == 1


def test_card_family_projection() -> None:
    counts = compute_tag_counts(SAMPLE_CARDS)
    collapsed_map = build_collapsed_tag_map(counts, collapse_threshold=1)
    projected = build_card_family_projection(SAMPLE_CARDS, collapsed_map)

    first = next(card for card in projected if card["id"] == "1")
    assert first["tag_families"]["mechanic"] == ["mechanic:scry"]
    third = next(card for card in projected if card["id"] == "3")
    assert third["tag_families"]["mechanic"] == ["mechanic:_other"]
