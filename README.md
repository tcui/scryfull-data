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
