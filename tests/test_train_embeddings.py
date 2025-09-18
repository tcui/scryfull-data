from pathlib import Path
import random

import pytest

from src import train_embeddings as te


SAMPLE_PATH = Path("data/tag_output/sample_tag_embeddings.jsonl")


def load_sample_records():
    return te.load_embedding_records(SAMPLE_PATH)


def test_group_resolution_and_index():
    records = load_sample_records()
    groups_per_record = te.resolve_record_groups(
        records, use_tags=True, use_tag_families=True
    )

    assert len(groups_per_record) == len(records)
    assert all(isinstance(groups, set) for groups in groups_per_record)

    index = te.build_group_index(groups_per_record, min_group_size=2)
    # The sample data contains at least the 'role' family shared by all records.
    assert "role" in index
    assert all(len(members) >= 2 for members in index.values())


def test_positive_pair_generation_and_conversion():
    records = load_sample_records()
    groups_per_record = te.resolve_record_groups(
        records, use_tags=True, use_tag_families=True
    )
    index = te.build_group_index(groups_per_record, min_group_size=2)

    rng = random.Random(13)
    pairs = list(te.iter_positive_pairs(index, max_pairs_per_group=None, rng=rng))
    assert pairs, "Expected at least one positive pair"

    filtered_pairs = te.filter_pairs_by_records(pairs, groups_per_record)
    assert filtered_pairs == pairs

    examples = te.convert_pairs_to_examples(filtered_pairs, records)
    assert len(examples) == len(filtered_pairs)
    assert all(len(example.texts) == 2 for example in examples)


def test_negative_pair_sampling():
    records = load_sample_records()
    groups_per_record = te.resolve_record_groups(
        records, use_tags=True, use_tag_families=True
    )
    index = te.build_group_index(groups_per_record, min_group_size=2)
    rng = random.Random(5)
    pairs = list(te.iter_positive_pairs(index, max_pairs_per_group=1, rng=rng))

    negatives = te.build_negative_pairs(pairs[:1], records, groups_per_record, seed=19)
    sentences1, sentences2, labels = negatives
    assert len(sentences1) == len(sentences2) == len(labels)
    # One positive + one negative for the sampled pair.
    assert labels.count(1) == 1
    assert labels.count(0) == 1


def test_filter_pairs_requires_intersection():
    records = load_sample_records()
    groups_per_record = te.resolve_record_groups(
        records, use_tags=True, use_tag_families=True
    )
    index = te.build_group_index(groups_per_record, min_group_size=2)
    rng = random.Random(11)
    pairs = list(te.iter_positive_pairs(index, max_pairs_per_group=1, rng=rng))
    assert pairs

    # Break the shared groups so the pair should be dropped.
    adjusted_groups = [set(groups) for groups in groups_per_record]
    pair = pairs[0]
    adjusted_groups[pair.anchor_idx] = {"mechanic:scry"}
    adjusted_groups[pair.positive_idx] = {"role"}

    filtered = te.filter_pairs_by_records(pairs, adjusted_groups)
    assert not filtered


@pytest.mark.parametrize(
    "argument",
    [
        ["--train-batch-size", "0"],
        ["--num-epochs", "0"],
        ["--learning-rate", "0"],
        ["--warmup-ratio", "1.5"],
        ["--eval-split", "1"],
        ["--min-group-size", "1"],
        ["--max-pairs-per-group", "0"],
        ["--limit", "0"],
        ["--max-seq-length", "0"],
    ],
)
def test_argument_validation(argument):
    parser = te.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "input.jsonl", "--output-dir", "models", *argument])
