import json
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_prep import (
    DEFAULT_METADATA_FIELDS,
    DEFAULT_TEXT_FIELDS,
    build_embedding_text,
    dump_examples,
    main,
    prepare_embedding_examples,
)


@pytest.fixture
def sample_cards():
    return [
        {
            "id": "card-1",
            "oracle_id": "oracle-1",
            "name": "Insightful Scholar",
            "type_line": "Creature — Human Wizard",
            "oracle_text": "When Insightful Scholar enters the battlefield, draw a card. Whenever you scry, gain 1 life.",
            "mana_cost": "{2}{U}",
            "cmc": 3,
            "colors": ["U"],
            "color_identity": ["U"],
            "tags": ["mechanic:scry", "role:card_draw", "theme:lifegain"],
        },
        {
            "id": "card-2",
            "oracle_id": "oracle-2",
            "name": "Knowledge Burst",
            "type_line": "Instant",
            "oracle_text": "Scry 2, then draw two cards.",
            "mana_cost": "{1}{U}{U}",
            "cmc": 3,
            "colors": ["U"],
            "color_identity": ["U"],
            "tags": ["mechanic:scry", "role:card_draw"],
        },
        {
            "id": "card-3",
            "oracle_id": "oracle-3",
            "name": "Reckless Gambit",
            "type_line": "Sorcery",
            "oracle_text": "Destroy target creature. Venture into the dungeon.",
            "mana_cost": "{3}{R}",
            "cmc": 4,
            "colors": ["R"],
            "color_identity": ["R"],
            "tags": ["mechanic:venture", "role:removal"],
        },
    ]


def test_build_embedding_text_handles_varied_fields(sample_cards):
    record = sample_cards[0]
    text = build_embedding_text(record, DEFAULT_TEXT_FIELDS)

    assert "Insightful Scholar" in text
    assert "Creature — Human Wizard" in text
    assert "draw a card" in text


def test_prepare_embedding_examples_creates_expected_structure(sample_cards):
    examples = prepare_embedding_examples(
        sample_cards,
        mechanic_rare_threshold=1,
    )

    assert len(examples) == 3

    scholar = next(item for item in examples if item["id"] == "card-1")
    assert scholar["tags"] == ["mechanic:scry", "role:card_draw", "theme:lifegain"]
    assert "mechanic:scry" in scholar["tag_families"]
    assert "role" in scholar["tag_families"]
    assert "theme" in scholar["tag_families"]

    venture = next(item for item in examples if item["id"] == "card-3")
    assert "mechanic:other" in venture["tag_families"]

    for example in examples:
        assert example["input"]
        assert example["name"]
        if "metadata" in example:
            assert "tags" not in example["metadata"]


def test_prepare_embedding_examples_respects_custom_fields(sample_cards):
    examples = prepare_embedding_examples(
        sample_cards,
        mechanic_rare_threshold=1,
        text_fields=("name",),
        metadata_fields=("mana_cost",),
    )

    assert all(example["input"] == example["name"] for example in examples)
    assert all(example.get("metadata", {}).get("mana_cost") for example in examples)


def test_dump_examples_jsonl_round_trip(tmp_path, sample_cards):
    examples = prepare_embedding_examples(sample_cards, mechanic_rare_threshold=1)
    destination = tmp_path / "output.jsonl"

    dump_examples(examples, str(destination), output_format="jsonl")

    lines = destination.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(examples)
    payload = json.loads(lines[0])
    assert set(payload) >= {"id", "input", "tags", "tag_families"}


def test_cli_end_to_end(tmp_path, sample_cards, monkeypatch):
    input_path = tmp_path / "cards.json"
    input_path.write_text(json.dumps(sample_cards), encoding="utf-8")

    output_path = tmp_path / "examples.jsonl"

    argv = [
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--mechanic-rare-threshold",
        "1",
        "--text-fields",
        "name,type_line",
        "--metadata-fields",
        "mana_cost,cmc",
    ]

    main(argv)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(sample_cards)

    first = json.loads(lines[0])
    assert first["input"].count("\n\n") == 1  # two fields stitched together
    assert set(first.get("metadata", {})) == {"mana_cost", "cmc"}

