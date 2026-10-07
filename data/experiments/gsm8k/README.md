# GSM8K Experiment Data

- Source: https://huggingface.co/datasets/openai/gsm8k
- Dataset id: `openai/gsm8k`
- Config: `main`
- Split: `test`
- Imported samples: `500`
- Video subset: first `200` rows by `selection_rank`
- Selection seed: `20260615`
- Selection method: `seeded_shuffle_stable_prefix`
- License: MIT, recorded from the Hugging Face dataset card.

Expansion rule: rerun the importer with the same split and seed but a larger
`--limit`. The previous pilot set remains a stable prefix of the larger set.
