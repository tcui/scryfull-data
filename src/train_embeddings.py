"""Train a sentence embedding model on prepared Scryfall card data.

This module provides a command line interface that reads the JSON/JSONL records
produced by :mod:`src.embedding_prep`, groups cards by shared tags, and
fine-tunes a transformer encoder using the
``MultipleNegativesRankingLoss`` objective from the
`sentence-transformers <https://www.sbert.net/>`_ library.

Example usage::

    python -m src.train_embeddings \
        --input data/tag_output/tag_embeddings.jsonl \
        --output-dir models/card-embeddings \
        --model-name sentence-transformers/all-MiniLM-L12-v2 \
        --num-epochs 2 --train-batch-size 64

The trainer automatically builds positive text pairs for cards that share at
least one tag (or tag family) and samples contrasting negatives from cards that
do not share any labels. An optional evaluation split uses a
``BinaryClassificationEvaluator`` to monitor how well the model separates
positive/negative pairs over time.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:  # pragma: no cover - exercised indirectly in tests
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.evaluation import BinaryClassificationEvaluator
except ImportError as exc:  # pragma: no cover - handled gracefully for environments without the dep
    class _FallbackInputExample:  # pragma: no cover - simple placeholder for tests
        def __init__(self, texts: Sequence[str], label: Optional[float] = None) -> None:
            self.texts = list(texts)
            self.label = label

    InputExample = _FallbackInputExample  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment]
    losses = None  # type: ignore[assignment]
    BinaryClassificationEvaluator = None  # type: ignore[assignment]
    _IMPORT_ERROR = ImportError(
        "sentence-transformers is required for training; install it to enable model fine-tuning"
    )
    _IMPORT_ERROR.__cause__ = exc
else:  # pragma: no cover - ensures attribute exists when dependency is present
    _IMPORT_ERROR = None

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from torch.utils.data import DataLoader


LOGGER = logging.getLogger(__name__)


Record = Dict[str, object]


def _int_min(min_value: int) -> Callable[[str], int]:
    """Return a parser that accepts integers greater than or equal to ``min_value``."""

    def parser(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:  # pragma: no cover - argparse reports the error
            raise argparse.ArgumentTypeError(f"Expected an integer, received {value!r}.") from exc

        if parsed < min_value:
            raise argparse.ArgumentTypeError(
                f"Value must be >= {min_value}, received {parsed}."
            )
        return parsed

    return parser


def _float_min(min_value: float, *, inclusive: bool = True) -> Callable[[str], float]:
    """Return a parser that enforces a minimum floating point value."""

    def parser(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:  # pragma: no cover - argparse reports the error
            raise argparse.ArgumentTypeError(f"Expected a floating point value, received {value!r}.") from exc

        if inclusive:
            if parsed < min_value:
                raise argparse.ArgumentTypeError(
                    f"Value must be >= {min_value}, received {parsed}."
                )
        else:
            if parsed <= min_value:
                raise argparse.ArgumentTypeError(
                    f"Value must be > {min_value}, received {parsed}."
                )
        return parsed

    return parser


def _float_range(
    min_value: float,
    max_value: float,
    *,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> Callable[[str], float]:
    """Return a parser enforcing that ``min_value <= x <= max_value`` (respecting inclusivity)."""

    def parser(value: str) -> float:
        parsed = _float_min(min_value, inclusive=inclusive_min)(value)
        max_threshold = max_value if inclusive_max else math.nextafter(max_value, float("-inf"))
        if parsed > max_threshold:
            comparator = "<=" if inclusive_max else "<"
            raise argparse.ArgumentTypeError(
                f"Value must be {comparator} {max_value}, received {parsed}."
            )
        return parsed

    return parser


@dataclass(frozen=True)
class PairedExample:
    """Stores the indices for a positive training pair."""

    anchor_idx: int
    positive_idx: int
    group: str


def load_embedding_records(path: Path) -> List[Record]:
    """Load embedding-ready records from ``path``.

    Parameters
    ----------
    path:
        Location of a JSON or JSONL export created by
        :mod:`src.embedding_prep`.
    """

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
    else:
        records = json.loads(path.read_text())

    if not isinstance(records, list):
        raise ValueError("Expected the input file to contain a list of records.")

    LOGGER.info("Loaded %d records from %s", len(records), path)
    return records


def resolve_record_groups(
    records: Sequence[Record],
    *,
    use_tags: bool,
    use_tag_families: bool,
) -> List[Set[str]]:
    """Return the label groups associated with each record."""

    if not use_tags and not use_tag_families:
        raise ValueError("At least one of use_tags/use_tag_families must be True.")

    groups_per_record: List[Set[str]] = []
    for record in records:
        record_groups: Set[str] = set()
        if use_tags:
            record_groups.update(record.get("tags", []) or [])
        if use_tag_families:
            record_groups.update(record.get("tag_families", []) or [])

        if not record_groups:
            LOGGER.debug("Record %s has no groups and will be ignored.", record.get("id"))

        groups_per_record.append(record_groups)

    return groups_per_record


def build_group_index(
    groups_per_record: Sequence[Set[str]],
    *,
    min_group_size: int = 2,
) -> Dict[str, List[int]]:
    """Create a mapping from group label to the list of record indices."""

    index: Dict[str, List[int]] = {}
    for idx, groups in enumerate(groups_per_record):
        for group in groups:
            index.setdefault(group, []).append(idx)

    filtered_index = {group: idxs for group, idxs in index.items() if len(idxs) >= min_group_size}
    LOGGER.info(
        "Tracking %d groups (minimum size %d).", len(filtered_index), min_group_size
    )
    return filtered_index


def iter_positive_pairs(
    group_index: Dict[str, List[int]],
    *,
    max_pairs_per_group: Optional[int],
    rng: random.Random,
) -> Iterator[PairedExample]:
    """Yield positive pairs sampled from each group."""

    for group, members in group_index.items():
        if len(members) < 2:
            continue

        candidate_pairs: List[Tuple[int, int]] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidate_pairs.append((members[i], members[j]))

        rng.shuffle(candidate_pairs)

        if max_pairs_per_group is not None:
            candidate_pairs = candidate_pairs[:max_pairs_per_group]

        for anchor_idx, positive_idx in candidate_pairs:
            yield PairedExample(anchor_idx=anchor_idx, positive_idx=positive_idx, group=group)


def filter_pairs_by_records(
    pairs: Iterable[PairedExample],
    groups_per_record: Sequence[Set[str]],
) -> List[PairedExample]:
    """Drop pairs whose records no longer share any groups."""

    filtered: List[PairedExample] = []
    for pair in pairs:
        anchor_groups = groups_per_record[pair.anchor_idx]
        positive_groups = groups_per_record[pair.positive_idx]
        if anchor_groups and positive_groups and anchor_groups & positive_groups:
            filtered.append(pair)
    return filtered


def convert_pairs_to_examples(
    pairs: Sequence[PairedExample],
    records: Sequence[Record],
) -> List[InputExample]:
    """Convert ``PairedExample`` entries into ``InputExample`` objects."""

    examples: List[InputExample] = []
    for pair in pairs:
        text_a = str(records[pair.anchor_idx]["input"])
        text_b = str(records[pair.positive_idx]["input"])
        examples.append(InputExample(texts=[text_a, text_b]))
    return examples


def build_negative_pairs(
    pairs: Sequence[PairedExample],
    records: Sequence[Record],
    groups_per_record: Sequence[Set[str]],
    *,
    seed: int,
) -> Tuple[List[str], List[str], List[int]]:
    """Create negative examples for evaluation."""

    rng = random.Random(seed)
    sentences1: List[str] = []
    sentences2: List[str] = []
    labels: List[int] = []

    all_indices = list(range(len(records)))

    for pair in pairs:
        text_a = str(records[pair.anchor_idx]["input"])
        text_b = str(records[pair.positive_idx]["input"])
        sentences1.append(text_a)
        sentences2.append(text_b)
        labels.append(1)

        # Sample a negative counterpart for the anchor.
        attempts = 0
        while True:
            attempts += 1
            negative_idx = rng.choice(all_indices)
            if negative_idx == pair.anchor_idx:
                continue
            if groups_per_record[negative_idx].isdisjoint(groups_per_record[pair.anchor_idx]):
                break
            if attempts > 50:
                # Fall back to any random card if we can't find a disjoint one.
                break

        sentences1.append(text_a)
        sentences2.append(str(records[negative_idx]["input"]))
        labels.append(0)

    return sentences1, sentences2, labels


def create_evaluator(
    eval_pairs: Sequence[PairedExample],
    records: Sequence[Record],
    groups_per_record: Sequence[Set[str]],
    *,
    seed: int,
) -> Optional[BinaryClassificationEvaluator]:
    """Construct a ``BinaryClassificationEvaluator`` if examples are available."""

    if not eval_pairs:
        return None

    if BinaryClassificationEvaluator is None:  # pragma: no cover - executed only without dependency
        raise _IMPORT_ERROR  # type: ignore[misc]

    sentences1, sentences2, labels = build_negative_pairs(
        eval_pairs, records, groups_per_record, seed=seed
    )
    return BinaryClassificationEvaluator(sentences1, sentences2, labels)


def train_model(args: argparse.Namespace) -> None:
    if SentenceTransformer is None or losses is None:  # pragma: no cover
        raise _IMPORT_ERROR  # type: ignore[misc]

    records = load_embedding_records(Path(args.input))
    if args.limit is not None:
        records = records[: args.limit]
        LOGGER.info("Limiting to %d records for training.", len(records))

    groups_per_record = resolve_record_groups(
        records, use_tags=args.use_tags, use_tag_families=args.use_tag_families
    )
    group_index = build_group_index(groups_per_record, min_group_size=args.min_group_size)

    rng = random.Random(args.seed)
    all_pairs = list(
        iter_positive_pairs(
            group_index, max_pairs_per_group=args.max_pairs_per_group, rng=rng
        )
    )

    all_pairs = filter_pairs_by_records(all_pairs, groups_per_record)
    if not all_pairs:
        raise RuntimeError("No positive pairs were generated. Check the input data and filters.")

    rng.shuffle(all_pairs)
    eval_size = int(len(all_pairs) * args.eval_split)
    eval_pairs = all_pairs[:eval_size]
    train_pairs = all_pairs[eval_size:]

    train_examples = convert_pairs_to_examples(train_pairs, records)
    LOGGER.info("Prepared %d training pairs and %d evaluation pairs.", len(train_pairs), len(eval_pairs))

    drop_last = args.train_batch_drop_last and len(train_examples) >= args.train_batch_size
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_examples,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=drop_last,
    )

    evaluator = create_evaluator(eval_pairs, records, groups_per_record, seed=args.seed)

    model = SentenceTransformer(args.model_name)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * args.warmup_ratio)
    LOGGER.info("Warmup steps: %d", warmup_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        scheduler=args.scheduler,
        optimizer_params={"lr": args.learning_rate},
        evaluation_steps=args.evaluation_steps if evaluator else 0,
        output_path=str(output_dir),
        evaluator=evaluator,
        save_best_model=bool(evaluator),
    )

    LOGGER.info("Training completed. Model saved to %s", output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a card text embedder")
    parser.add_argument("--input", required=True, help="Path to JSON/JSONL embedding records")
    parser.add_argument("--output-dir", required=True, help="Directory to store the trained model")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L12-v2",
        help="Base SentenceTransformer model to fine-tune",
    )
    parser.add_argument("--train-batch-size", type=_int_min(1), default=32)
    parser.add_argument("--num-epochs", type=_int_min(1), default=1)
    parser.add_argument("--learning-rate", type=_float_min(0.0, inclusive=False), default=2e-5)
    parser.add_argument(
        "--warmup-ratio",
        type=_float_range(0.0, 1.0, inclusive_min=True, inclusive_max=True),
        default=0.1,
    )
    parser.add_argument("--scheduler", default="warmuplinear")
    parser.add_argument("--evaluation-steps", type=_int_min(0), default=250)
    parser.add_argument(
        "--eval-split",
        type=_float_range(0.0, 1.0, inclusive_min=True, inclusive_max=False),
        default=0.1,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--limit",
        type=_int_min(1),
        help="Optionally limit the number of records",
    )
    parser.add_argument("--max-seq-length", type=_int_min(1))

    group = parser.add_argument_group("pair sampling")
    group.add_argument("--use-tags", action="store_true", help="Include full tags when grouping")
    group.add_argument(
        "--use-tag-families",
        action="store_true",
        help="Include condensed tag families when grouping",
    )
    group.add_argument(
        "--min-group-size",
        type=_int_min(2),
        default=2,
        help="Minimum number of cards required for a tag to form training pairs",
    )
    group.add_argument(
        "--max-pairs-per-group",
        type=_int_min(1),
        help="Cap the number of positive pairs sampled from each group",
    )

    parser.add_argument(
        "--train-batch-drop-last",
        action="store_true",
        help="Drop the last batch even if it is incomplete",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Default to using both tags and tag families unless explicitly disabled.
    if not args.use_tags and not args.use_tag_families:
        args.use_tags = True
        args.use_tag_families = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    train_model(args)


if __name__ == "__main__":  # pragma: no cover
    main()
