# scryfull-data
processing Scryfull data

## Tag analysis utilities

Use `src/analyze_tags.py` to derive aggregate statistics from the generated
`tagged-cards.json` file:

```
python src/analyze_tags.py \
  --input data/tag_output/tagged-cards.json \
  --cooccurrence-output data/tag_output/tag_cooccurrence.json \
  --tag-summary-output data/tag_output/tag_summary.json \
  --card-families-output data/tag_output/tagged-card-families.json
```

The script emits:

* co-occurrence counts for frequently paired tags (default min count = 5)
* per-tag and per-family frequency summaries with collapsed `_other` buckets for
  sparse mechanics (threshold configurable via `--family-threshold`)
* a projection of each card's tags into their condensed family representation
