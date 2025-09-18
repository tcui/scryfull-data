#!/usr/bin/env python3
"""Helpers for preparing embedding training corpora from tagged cards."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

from .tag_analysis import build_tag_family_export, load_tagged_cards

# Default fields that are stitched together to form the embedding input text.
DEFAULT_TEXT_FIELDS: Sequence[str] = ("name", "type_line", "oracle_text")

# Metadata fields preserved on each example in addition to tag information.
DEFAULT_METADATA_FIELDS: Sequence[str] = (
    "oracle_id",
    "set",
    "set_name",
    "collector_number",
    "rarity",
    "released_at",
    "mana_cost",
    "cmc",
    "colors",
    "color_identity",
    "produced_mana",
    "keywords",
    "layout",
    "games",
    "edhrec_rank",
    "scryfall_uri",
)


def _record_key(record: MutableMapping[str, Any]) -> Optional[str]:
    """Return the identifier used to join tag family metadata."""

    identifier = record.get("id") or record.get("oracle_id")
    if isinstance(identifier, str) and identifier.strip():
        return identifier
    return None


def _normalize_tags(values: Iterable[Any]) -> List[str]:
    """Return a sorted list of unique, non-empty string tags."""

    cleaned = {str(value).strip() for value in values if isinstance(value, str) and value.strip()}
    return sorted(cleaned)


def _coerce_text(value: Any) -> Optional[str]:
    """Convert assorted field values into a display string."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Iterable):
        parts: List[str] = []
        for item in value:
            if item is None:
                continue
            rendered = _coerce_text(item)
            if rendered:
                parts.append(rendered)
        if parts:
            return ", ".join(parts)
    return None


def build_embedding_text(record: MutableMapping[str, Any], text_fields: Sequence[str]) -> str:
    """Join the requested fields from ``record`` into a text blob."""

    pieces: List[str] = []
    for field in text_fields:
        value = record.get(field)
        rendered = _coerce_text(value)
        if rendered:
            pieces.append(rendered)
    return "\n\n".join(pieces)


def prepare_embedding_examples(
    records: Iterable[MutableMapping[str, Any]],
    *,
    mechanic_rare_threshold: int = 5,
    text_fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
    metadata_fields: Sequence[str] = DEFAULT_METADATA_FIELDS,
    drop_empty_text: bool = True,
) -> List[Dict[str, Any]]:
    """Project tagged card records into embedding-friendly JSON objects."""

    materialized = list(records)
    families = build_tag_family_export(
        materialized,
        mechanic_rare_threshold=mechanic_rare_threshold,
    )
    family_map = {
        _record_key(entry): entry.get("tag_families", [])
        for entry in families
    }

    examples: List[Dict[str, Any]] = []
    for record in materialized:
        key = _record_key(record)
        tags = _normalize_tags(record.get("tags") or [])
        tag_families = family_map.get(key, [])
        text = build_embedding_text(record, text_fields)
        if drop_empty_text and not text:
            continue

        metadata: Dict[str, Any] = {}
        for field in metadata_fields:
            if field in ("tags", "tag_families"):
                # Explicitly exclude tag information since it lives top-level.
                continue
            if field not in record:
                continue
            value = record.get(field)
            if value in (None, "", [], {}):
                continue
            metadata[field] = value

        example = {
            "id": key,
            "name": record.get("name"),
            "input": text,
            "tags": tags,
            "tag_families": tag_families,
        }
        if metadata:
            example["metadata"] = metadata
        examples.append(example)

    return examples


def dump_examples(
    examples: Sequence[MutableMapping[str, Any]],
    destination: str,
    *,
    output_format: str = "jsonl",
    indent: Optional[int] = 2,
) -> None:
    """Serialize prepared examples to disk."""

    stream = sys.stdout if destination == "-" else open(destination, "w", encoding="utf-8")
    try:
        if output_format == "jsonl":
            for example in examples:
                json.dump(example, stream, ensure_ascii=False)
                stream.write("\n")
        elif output_format == "json":
            effective_indent = None if indent is not None and indent < 0 else indent
            json.dump(list(examples), stream, ensure_ascii=False, indent=effective_indent)
            if effective_indent is not None or destination != "-":
                stream.write("\n")
        else:
            raise ValueError(f"Unsupported output format: {output_format!r}")
    finally:
        if stream is not sys.stdout:
            stream.close()


def _parse_fields(value: Optional[str], *, fallback: Sequence[str]) -> Sequence[str]:
    if value is None:
        return fallback
    fields = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(fields) if fields else fallback


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare JSON or JSONL training data for card tag embeddings.",
    )
    parser.add_argument("--input", required=True, help="Path to the tagged card JSON array.")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination file or '-' for stdout.",
    )
    parser.add_argument(
        "--mechanic-rare-threshold",
        type=int,
        default=5,
        help="Collapse mechanic:* tags appearing ≤ this many times into mechanic:other.",
    )
    parser.add_argument(
        "--text-fields",
        help="Comma-separated fields to stitch together for the embedding input text.",
    )
    parser.add_argument(
        "--metadata-fields",
        help="Comma-separated fields preserved under the metadata key.",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "json"),
        default="jsonl",
        help="Output format for prepared examples.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Pretty-print JSON output when using --format json (use <0 for compact output).",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep records even when the generated text would be empty.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    text_fields = _parse_fields(args.text_fields, fallback=DEFAULT_TEXT_FIELDS)
    metadata_fields = _parse_fields(args.metadata_fields, fallback=DEFAULT_METADATA_FIELDS)

    records = load_tagged_cards(args.input)
    examples = prepare_embedding_examples(
        records,
        mechanic_rare_threshold=args.mechanic_rare_threshold,
        text_fields=text_fields,
        metadata_fields=metadata_fields,
        drop_empty_text=not args.keep_empty,
    )

    dump_examples(
        examples,
        args.output,
        output_format=args.format,
        indent=args.indent,
    )


if __name__ == "__main__":
    main()
