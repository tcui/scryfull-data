# scryfull-data
processing Scryfull data

Source data is from https://scryfall.com/docs/api/bulk-data

## Environment setup

Create an isolated Python environment so the tooling does not interfere with
other projects on your machine:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Generate tagged cards

Produce a tagged oracle card export by running `tag_cards.py` against a
Scryfall oracle dump:

```
python -m src.tag_cards \
  --input data/oracle-cards-*.json \
  --output data/tag_output/tagged-cards.json
```

The helper keeps only English cards by default. Use `--language any` to include
every language, `--include-tokens` to keep token layouts, or `--limit` to sample
the first _n_ results while iterating on heuristics.


## Tag analytics

Run the `tag_analysis.py` helper to derive co-occurrence statistics and
generate a condensed tag family export from `tagged-cards.json`:

```
python -m src.tag_analysis \
  --input data/tag_output/tagged-cards.json \
  --cooccurrence-output data/tag_output/tag_cooccurrence.json \
  --families-output data/tag_output/tag_families.json
```

Use `--min-pair-count`, `--top-n`, and `--mechanic-rare-threshold` to tweak the
report granularity for experimentation.

## Embedding preparation

The `embedding_prep.py` helper turns the tagged card export into JSON or JSONL
records that are ready for embedding training or evaluation experiments. The
examples include the raw tags as well as the condensed *tag family* view so you
can test different granularity levels when clustering or searching.

```bash
python -m src.embedding_prep \
  --input data/tag_output/tagged-cards.json \
  --output data/tag_output/tag_embeddings.jsonl \
  --mechanic-rare-threshold 5 \
  --text-fields name,type_line,oracle_text \
  --metadata-fields "oracle_id,set,set_name,rarity,mana_cost,cmc,colors,color_identity"
```

Use `--format json` if you prefer a standard JSON array, and `--keep-empty` to
retain records where the requested text fields are all blank. The script relies
on the same mechanic rarity threshold logic as the analytics helper to collapse
rare `mechanic:*` tags into a catch-all `mechanic:other` bucket.

For quick experimentation, see `data/tag_output/sample_tagged_cards.json` and
the matching `sample_tag_embeddings.jsonl` output that demonstrate the expected
record structure.

## Embedding fine-tuning

After preparing the embedding-ready export, fine-tune a transformer model using
the provided training helper. The trainer builds multi-positive batches based on
shared tags and tag families and optimises a contrastive
`MultipleNegativesRankingLoss` objective:

```bash
python -m src.train_embeddings \
  --input data/tag_output/tag_embeddings.jsonl \
  --output-dir models/card-embeddings \
  --model-name sentence-transformers/all-MiniLM-L12-v2 \
  --num-epochs 2 \
  --train-batch-size 64 \
  --min-group-size 2 \
  --max-group-ratio 0.05
```

Use `--use-tags` and `--use-tag-families` to control which label granularities
form positive pairs, or `--max-pairs-per-group` to subsample very common tags.
Provide `--eval-split` to reserve a hold-out set for binary classification
evaluation during training. The script requires the `sentence-transformers` and
`torch` packages when fine-tuning a model.
