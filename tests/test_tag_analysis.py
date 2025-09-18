import math
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tag_analysis import build_tag_family_export, derive_tag_cooccurrence


@pytest.fixture
def sample_records():
    return [
        {
            "id": "card-1",
            "name": "Insightful Scholar",
            "tags": ["mechanic:scry", "role:card_draw", "theme:lifegain"],
        },
        {
            "id": "card-2",
            "name": "Knowledge Burst",
            "tags": ["mechanic:scry", "role:card_draw"],
        },
        {
            "id": "card-3",
            "name": "Reckless Gambit",
            "tags": ["mechanic:venture", "role:removal"],
        },
    ]


def test_cooccurrence_metrics(sample_records):
    report = derive_tag_cooccurrence(sample_records, min_pair_count=1, top_n=None)

    assert report["total_cards"] == 3
    assert report["tag_counts"]["mechanic:scry"] == 2
    assert report["tag_counts"]["role:card_draw"] == 2

    scry_neighbors = report["cooccurrences"]["mechanic:scry"]
    assert scry_neighbors[0]["tag"] == "role:card_draw"
    assert scry_neighbors[0]["count"] == 2
    assert math.isclose(scry_neighbors[0]["support"], 2 / 3)
    assert math.isclose(scry_neighbors[0]["confidence"], 1.0)
    assert math.isclose(scry_neighbors[0]["lift"], 1.5)
    assert math.isclose(scry_neighbors[0]["jaccard"], 1.0)

    venture_neighbors = report["cooccurrences"]["mechanic:venture"]
    assert venture_neighbors[0]["tag"] == "role:removal"
    assert venture_neighbors[0]["count"] == 1
    assert math.isclose(venture_neighbors[0]["support"], 1 / 3)
    assert math.isclose(venture_neighbors[0]["confidence"], 1.0)
    assert math.isclose(venture_neighbors[0]["lift"], 3.0)
    assert math.isclose(venture_neighbors[0]["jaccard"], 1.0)


def test_family_export(sample_records):
    export = build_tag_family_export(sample_records, mechanic_rare_threshold=1)

    first = next(item for item in export if item["id"] == "card-1")
    assert first["tags"] == ["mechanic:scry", "role:card_draw", "theme:lifegain"]
    assert first["tag_families"] == ["mechanic:scry", "role", "theme"]

    third = next(item for item in export if item["id"] == "card-3")
    assert third["tag_families"] == ["mechanic:other", "role"]
