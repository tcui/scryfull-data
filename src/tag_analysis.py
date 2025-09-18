#!/usr/bin/env python3
"""Utilities for analyzing generated Magic card tags."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Set


@dataclass(frozen=True)
class CooccurrenceNeighbor:
    """Statistics describing how frequently a pair of tags appear together."""

    tag: str
    count: int
    support: float
    confidence: float
    lift: float
    jaccard: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "count": self.count,
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
            "jaccard": self.jaccard,
        }


def load_tagged_cards(path: str) -> List[Dict[str, Any]]:
    """Load the tagged card array produced by :mod:`tag_cards`."""

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _unique_tags(record: MutableMapping[str, Any]) -> List[str]:
    """Return a sorted list of unique, truthy tag strings from ``record``."""

    tags = record.get("tags") or []
    cleaned: Set[str] = set()
    for tag in tags:
        if isinstance(tag, str):
            normalized = tag.strip()
            if normalized:
                cleaned.add(normalized)
    return sorted(cleaned)


def compute_tag_counts(records: Iterable[MutableMapping[str, Any]]) -> Counter:
    """Count how many cards each tag appears on."""

    counts: Counter = Counter()
    for record in records:
        unique_tags = _unique_tags(record)
        counts.update(unique_tags)
    return counts


def derive_tag_cooccurrence(
    records: Iterable[MutableMapping[str, Any]],
    *,
    min_pair_count: int = 2,
    top_n: Optional[int] = 20,
) -> Dict[str, Any]:
    """Compute co-occurrence statistics for card tags.

    Parameters
    ----------
    records:
        Tagged card mappings containing a ``tags`` list.
    min_pair_count:
        Discard pairs that appear together fewer than this many times.
    top_n:
        Limit the number of neighbors returned per tag. ``None`` keeps all.
    """

    pair_counts: Dict[str, Counter] = defaultdict(Counter)
    tag_counts: Counter = Counter()
    card_count = 0

    for record in records:
        unique_tags = _unique_tags(record)
        if not unique_tags:
            continue
        card_count += 1
        tag_counts.update(unique_tags)
        for index, left in enumerate(unique_tags):
            for right in unique_tags[index + 1 :]:
                pair_counts[left][right] += 1
                pair_counts[right][left] += 1

    results: Dict[str, List[CooccurrenceNeighbor]] = {}
    for tag, neighbors in pair_counts.items():
        formatted: List[CooccurrenceNeighbor] = []
        for other, count in neighbors.items():
            if count < min_pair_count:
                continue
            jaccard_denominator = tag_counts[tag] + tag_counts[other] - count
            jaccard = count / jaccard_denominator if jaccard_denominator else 0.0
            support = count / card_count if card_count else 0.0
            confidence = count / tag_counts[tag] if tag_counts[tag] else 0.0
            other_frequency = tag_counts[other] / card_count if card_count else 0.0
            lift = confidence / other_frequency if other_frequency else 0.0
            formatted.append(
                CooccurrenceNeighbor(
                    tag=other,
                    count=count,
                    support=support,
                    confidence=confidence,
                    lift=lift,
                    jaccard=jaccard,
                )
            )

        formatted.sort(key=lambda item: (-item.count, -item.jaccard, item.tag))
        if top_n is not None:
            formatted = formatted[:top_n]
        results[tag] = formatted

    for tag in tag_counts:
        results.setdefault(tag, [])

    sorted_counts = {tag: tag_counts[tag] for tag in sorted(tag_counts)}
    serialized_neighbors = {
        tag: [neighbor.to_dict() for neighbor in neighbors]
        for tag, neighbors in sorted(results.items())
    }

    return {
        "total_cards": card_count,
        "tag_counts": sorted_counts,
        "cooccurrences": serialized_neighbors,
    }


def _family_tag(tag: str, *, tag_counts: Counter, mechanic_rare_threshold: int) -> str:
    """Map a tag string to a condensed family representation."""

    prefix, sep, _ = tag.partition(":")
    if not sep:
        return prefix or tag
    if prefix == "mechanic":
        if tag_counts[tag] <= mechanic_rare_threshold:
            return "mechanic:other"
        return tag
    return prefix


def build_tag_family_export(
    records: Iterable[MutableMapping[str, Any]],
    *,
    mechanic_rare_threshold: int = 5,
) -> List[Dict[str, Any]]:
    """Produce downstream-friendly records with original and family-level tags."""

    materialized = list(records)
    tag_counts = compute_tag_counts(materialized)

    exports: List[Dict[str, Any]] = []
    for record in materialized:
        tags = _unique_tags(record)
        family_tags = sorted(
            {
                _family_tag(tag, tag_counts=tag_counts, mechanic_rare_threshold=mechanic_rare_threshold)
                for tag in tags
            }
        )
        exports.append(
            {
                "id": record.get("id"),
                "oracle_id": record.get("oracle_id"),
                "name": record.get("name"),
                "tags": tags,
                "tag_families": family_tags,
            }
        )

    return exports


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze generated Magic card tags.")
    parser.add_argument("--input", required=True, help="Path to the tagged card JSON array.")
    parser.add_argument(
        "--cooccurrence-output",
        help="Destination for co-occurrence statistics (omit to skip).",
    )
    parser.add_argument(
        "--families-output",
        help="Destination for tag family export (omit to skip).",
    )
    parser.add_argument("--min-pair-count", type=int, default=2, help="Ignore rare tag pairings.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Limit co-occurrence neighbors per tag (use -1 to keep all).",
    )
    parser.add_argument(
        "--mechanic-rare-threshold",
        type=int,
        default=5,
        help="Collapse mechanic:* tags appearing ≤ this many times into mechanic:other.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Pretty-print JSON with this indent (use <0 for compact output).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    indent = None if args.indent is not None and args.indent < 0 else args.indent

    records = load_tagged_cards(args.input)

    if not args.cooccurrence_output and not args.families_output:
        raise SystemExit("Specify at least one output flag to perform analysis.")

    if args.cooccurrence_output:
        cooccurrence = derive_tag_cooccurrence(
            records,
            min_pair_count=args.min_pair_count,
            top_n=None if args.top_n is not None and args.top_n < 0 else args.top_n,
        )
        _dump_json(cooccurrence, args.cooccurrence_output, indent)

    if args.families_output:
        families = build_tag_family_export(
            records,
            mechanic_rare_threshold=args.mechanic_rare_threshold,
        )
        _dump_json(families, args.families_output, indent)


def _dump_json(data: Any, path: str, indent: Optional[int]) -> None:
    stream = sys.stdout if path == "-" else open(path, "w", encoding="utf-8")
    try:
        json.dump(data, stream, ensure_ascii=False, indent=indent)
        if indent is not None or path != "-":
            stream.write("\n")
    finally:
        if stream is not sys.stdout:
            stream.close()


if __name__ == "__main__":
    main()
