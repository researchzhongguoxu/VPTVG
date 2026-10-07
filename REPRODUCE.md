# MathExplainAgent Anonymous Code Release

This package is a curated anonymous code release for reproducing and auditing the MathExplainAgent experiments. It is intended to accompany the manuscript, supplementary material, supplementary artifact package, and reproducibility guide.

## Package Contents

- `src/mathexplain/`: core MathExplainAgent implementation.
- `scripts/`: selected experiment entry points used by the paper experiments.
- `tests/`: regression and schema tests for core modules.
- `configs/`: lightweight configuration files.
- `data/experiments/gsm8k/manifest.jsonl`: fixed GSM8K rendered-image manifest used by the reported experiments.
- `data/experiments/gsm8k/images/`: rendered problem images for the 500-problem answer-mode set.
- `data/experiments/gsm8k/sources/`: source JSONL files used to construct the rendered GSM8K subsets.
- `data/experiments/gsm8k/video_ablation_runs/*/pair_manifest.jsonl`: fixed pair manifests for selected video ablations.
- `.env.example`: placeholder environment variables for model and TTS providers.

This package does not include raw API keys, `.env`, virtual environments, caches, full experiment outputs, full video collections, private absolute paths, or uncurated logs.

## Installation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
copy .env.example .env
```

Edit `.env` with provider credentials before running experiments. Required providers depend on which experiment is run. The main answer and video experiments use Qwen-compatible vision/planner access, optional GLM/Kimi judge or baseline access, and TTS credentials for voiced video rendering.

## Basic Checks

```powershell
python -m pytest tests -q
```

For a low-cost smoke test, run with a small `--limit` before reproducing the full reported settings.

## Reproducing Main Answer Results

```powershell
python scripts/run_gsm8k_qwen_answer_experiment.py ^
  --manifest data/experiments/gsm8k/manifest.jsonl ^
  --output-dir data/experiments/gsm8k/answer_runs/formal_full_vs_qwen_e2e_500 ^
  --limit 500 ^
  --selection-rank-start 0
```

The GLM and Kimi direct image-to-answer baselines are run with:

```powershell
python scripts/run_gsm8k_glm_kimi_e2e_extend_200_to_500.py
```

Expected headline results from the saved formal run are:

- Full System: `460/500`, 92.0%.
- Kimi-k2.6 E2E: `415/500`, 83.0%.
- Qwen3.6-plus E2E: `393/500`, 78.6%.
- GLM-4.6V E2E: `380/500`, 76.0%.

## Reproducing No Verifier Ablation

```powershell
python scripts/run_gsm8k_no_verifier_ablation.py ^
  --manifest data/experiments/gsm8k/manifest.jsonl ^
  --output-dir data/experiments/gsm8k/ablation_runs/formal_no_verifier_qwen_planner_200 ^
  --limit 200 ^
  --selection-rank-start 0
```

Expected formal comparison: Full System `186/200` versus No Verifier `166/200`, exact McNemar `p=1.80e-04`.

## Reproducing Video Generation

```powershell
python scripts/run_gsm8k_video_comparison_experiment.py ^
  --manifest data/experiments/gsm8k/manifest.jsonl ^
  --output-dir data/experiments/gsm8k/video_runs/full_vs_simple_pilot_200 ^
  --limit 200 ^
  --selection-rank-start 0 ^
  --tts ^
  --render-video
```

Expected formal video-chain answer correctness: Full System `188/200` and Plain Simple Pipeline `153/200`.

## Reproducing Video Ablations

No TeachingPlan:

```powershell
python scripts/run_gsm8k_no_teaching_plan_video_ablation.py ^
  --manifest data/experiments/gsm8k/manifest.jsonl ^
  --output-dir data/experiments/gsm8k/video_ablation_runs/no_teaching_plan ^
  --limit 50 ^
  --ab-seed 20260711
```

Weak EDS:

```powershell
python scripts/run_gsm8k_weak_eds_video_ablation.py ^
  --manifest data/experiments/gsm8k/manifest.jsonl ^
  --output-dir data/experiments/gsm8k/video_ablation_runs/weak_eds ^
  --limit 50

python scripts/run_gsm8k_weak_eds_video_judge_ablation.py ^
  --limit 50 ^
  --ab-seed 20260714
```

## Notes on Exact Reproduction

The experiments call external MLLM and TTS providers. Provider-side model updates, network conditions, and rate limits may lead to small deviations in rerun outputs, latency, and token usage. The fixed manifests, saved summaries, supplementary artifacts, and representative structured traces are therefore part of the reproducibility evidence.

This package is intended for anonymous review. If the authors later prepare a public repository, they should add a final license and update venue-specific anonymity labels separately.
