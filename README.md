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
