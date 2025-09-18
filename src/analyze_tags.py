#!/usr/bin/env python3
"""Utilities for analyzing generated Scryfull tag data."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


CardRecord = Mapping[str, object]


def load_tagged_cards(path: str) -> List[CardRecord]:
    """Load the tagged card JSON array produced by ``tag_cards``."""

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def iter_card_tags(cards: Iterable[CardRecord]) -> Iterator[List[str]]:
    """Yield a sorted list of unique tags for each card."""

    for card in cards:
        raw_tags = card.get("tags") or []
        # ``tags`` is already sorted/unique when produced by ``tag_cards``, but we
        # normalise here to make the function robust to hand-crafted input.
        unique_tags = sorted({tag for tag in raw_tags if isinstance(tag, str) and tag})
        yield unique_tags


def compute_tag_counts(cards: Iterable[CardRecord]) -> Counter:
    """Count how many times each tag appears across the card set."""

    counter: Counter = Counter()
    for tags in iter_card_tags(cards):
        counter.update(tags)
    return counter


def compute_cooccurrence(
    cards: Iterable[CardRecord], min_count: int = 1
) -> List[Tuple[str, str, int]]:
    """Compute co-occurrence counts for tag pairs.

    Args:
        cards: Iterable of card records.
        min_count: Minimum number of shared appearances before keeping a pair.

    Returns:
        A list of ``(tag_a, tag_b, count)`` tuples sorted by decreasing count and
        lexicographical order.
    """

    pair_counts: MutableMapping[Tuple[str, str], int] = defaultdict(int)

    for tags in iter_card_tags(cards):
        for tag_a, tag_b in itertools.combinations(tags, 2):
            pair_counts[(tag_a, tag_b)] += 1

    filtered_pairs = [
        (tag_a, tag_b, count)
        for (tag_a, tag_b), count in pair_counts.items()
        if count >= min_count
    ]

    filtered_pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return filtered_pairs


def split_tag_family(tag: str) -> Tuple[str, str]:
    """Return ``(family, suffix)`` for a tag.

    Tags follow a ``family:value`` structure. If the delimiter is missing, the
    family defaults to ``"misc"``.
    """

    if ":" in tag:
        family, suffix = tag.split(":", 1)
        return family, suffix
    return "misc", tag


def build_collapsed_tag_map(
    tag_counts: Mapping[str, int],
    collapse_threshold: int,
    misc_family: str = "misc",
) -> Dict[str, str]:
    """Return a mapping from each original tag to its collapsed representation."""

    collapsed: Dict[str, str] = {}
    for tag, count in tag_counts.items():
        family, _ = split_tag_family(tag)
        if count <= collapse_threshold:
            collapsed_family = family or misc_family
            collapsed[tag] = f"{collapsed_family}:_other"
        else:
            collapsed[tag] = tag

    return collapsed


def build_family_summary(
    tag_counts: Mapping[str, int],
    collapsed_map: Mapping[str, str],
) -> Dict[str, Dict[str, object]]:
    """Build a JSON-serialisable summary for each tag."""

    summary: Dict[str, Dict[str, object]] = {}
    for tag, count in tag_counts.items():
        family, suffix = split_tag_family(tag)
        summary[tag] = {
            "family": family,
            "suffix": suffix,
            "count": count,
            "collapsed": collapsed_map.get(tag, tag),
        }

    return summary


def build_family_counts(tag_counts: Mapping[str, int]) -> Counter:
    """Aggregate tag counts by family prefix."""

    family_counts: Counter = Counter()
    for tag, count in tag_counts.items():
        family, _ = split_tag_family(tag)
        family_counts[family] += count
    return family_counts


def build_collapsed_counts(
    cards: Iterable[CardRecord], collapsed_map: Mapping[str, str]
) -> Counter:
    """Compute counts for collapsed tags using the provided mapping."""

    counter: Counter = Counter()
    for tags in iter_card_tags(cards):
        for tag in tags:
            counter[collapsed_map.get(tag, tag)] += 1
    return counter


def build_card_family_projection(
    cards: Iterable[CardRecord], collapsed_map: Mapping[str, str]
) -> List[Dict[str, object]]:
    """Add a condensed family projection for each card record."""

    projected_cards: List[Dict[str, object]] = []
    for card in cards:
        tags = [tag for tag in card.get("tags", []) if isinstance(tag, str) and tag]
        family_groups: MutableMapping[str, set] = defaultdict(set)

        for tag in tags:
            collapsed_tag = collapsed_map.get(tag, tag)
            family, _ = split_tag_family(collapsed_tag)
            family_groups[family].add(collapsed_tag)

        projected_cards.append(
            {
                "id": card.get("id"),
                "name": card.get("name"),
                "tags": tags,
                "tag_families": {family: sorted(values) for family, values in family_groups.items()},
                "flattened_families": sorted(
                    {collapsed for values in family_groups.values() for collapsed in values}
                ),
            }
        )

    return projected_cards


def dump_json(data: object, path: str, indent: Optional[int] = 2) -> None:
    """Serialise *data* to *path* (or stdout when ``path == '-'``)."""

    target = Path(path)
    if path == "-":
        json.dump(data, sys.stdout, ensure_ascii=False, indent=indent)  # pragma: no cover
        sys.stdout.write("\n")  # pragma: no cover
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse Scryfull tag output.")
    parser.add_argument("--input", required=True, help="Tagged cards JSON file.")
    parser.add_argument(
        "--cooccurrence-output",
        help="Destination for tag co-occurrence data (omit to skip).",
    )
    parser.add_argument(
        "--tag-summary-output",
        help="Destination for per-tag summary data (omit to skip).",
    )
    parser.add_argument(
        "--card-families-output",
        help="Destination for card family projection data (omit to skip).",
    )
    parser.add_argument(
        "--family-threshold",
        type=int,
        default=5,
        help="Collapse tags whose frequency is <= threshold (default: 5).",
    )
    parser.add_argument(
        "--min-cooccurrence",
        type=int,
        default=5,
        help="Minimum co-occurrence count before a pair is included (default: 5).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for JSON outputs (negative for compact).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - CLI glue
    args = parse_args(argv)
    indent = None if args.indent is not None and args.indent < 0 else args.indent

    cards = load_tagged_cards(args.input)

    tag_counts = compute_tag_counts(cards)
    collapsed_map = build_collapsed_tag_map(tag_counts, args.family_threshold)

    if args.cooccurrence_output:
        cooccurrence = compute_cooccurrence(cards, args.min_cooccurrence)
        dump_json(cooccurrence, args.cooccurrence_output, indent=indent)

    if args.tag_summary_output:
        summary = {
            "threshold": args.family_threshold,
            "tag_counts": dict(tag_counts),
            "family_counts": dict(build_family_counts(tag_counts)),
            "collapsed_counts": dict(build_collapsed_counts(cards, collapsed_map)),
            "tag_summary": build_family_summary(tag_counts, collapsed_map),
        }
        dump_json(summary, args.tag_summary_output, indent=indent)

    if args.card_families_output:
        projection = build_card_family_projection(cards, collapsed_map)
        dump_json(projection, args.card_families_output, indent=indent)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
