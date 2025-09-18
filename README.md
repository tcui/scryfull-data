# scryfull-data
processing Scryfull data

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
