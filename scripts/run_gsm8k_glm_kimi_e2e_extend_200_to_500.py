"""Extend the locked GLM/Kimi 200-question baseline to 500 questions.

This wrapper protects the formal 200-question result directory. It copies the
existing 200-question per-sample cache into a separate 500-question directory,
then runs the GLM/Kimi E2E runner with limit=500. Existing valid cached results
are reused. Failed/unparsable/None-prediction samples are rerun so interrupted
or rate-limited Kimi runs can be completed without overwriting valid results.

PyCharm usage:
1. Keep `scripts/run_gsm8k_glm_kimi_e2e.py` at its default 200-question setup.
2. Run this file directly.
3. Do not add `--overwrite`; this script intentionally does not expose it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_200_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/formal_glm_kimi_e2e_200"
TARGET_500_DIR = REPO_ROOT / "data/experiments/gsm8k/answer_runs/formal_glm_kimi_e2e_calibrated_500"
RUNNER = REPO_ROOT / "scripts/run_gsm8k_glm_kimi_e2e.py"

LIMIT = 500
EXPECTED_SOURCE_COUNT = 200
PROVIDERS = "glm,kimi"
RERUN_FAILED = True


def main() -> int:
    if not SOURCE_200_DIR.exists():
        raise FileNotFoundError(f"Missing locked 200-question source directory: {SOURCE_200_DIR}")

    seed_provider_cache("glm")
    seed_provider_cache("kimi")
    print_cache_status("before")

    command = [
        sys.executable,
        str(RUNNER),
        "--limit",
        str(LIMIT),
        "--output-dir",
        str(TARGET_500_DIR),
        "--providers",
        PROVIDERS,
    ]
    if RERUN_FAILED:
        command.append("--rerun-failed")
    print("[extend] Source 200 cache:", SOURCE_200_DIR, flush=True)
    print("[extend] Target 500 directory:", TARGET_500_DIR, flush=True)
    print(
        "[extend] Valid cached samples will be reused; failed/unparsable/None samples will be rerun.",
        flush=True,
    )
    print("[extend] Command:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    print_cache_status("after")
    return int(completed.returncode)


def seed_provider_cache(provider: str) -> None:
    source_provider_dir = SOURCE_200_DIR / provider
    target_provider_dir = TARGET_500_DIR / provider
    if not source_provider_dir.exists():
        raise FileNotFoundError(f"Missing {provider} source cache: {source_provider_dir}")

    source_count = count_result_files(source_provider_dir)
    if source_count < EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            f"{provider} source cache has only {source_count} result.json files; "
            f"expected at least {EXPECTED_SOURCE_COUNT}."
        )

    target_provider_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for sample_dir in source_provider_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        source_result = sample_dir / "result.json"
        target_sample_dir = target_provider_dir / sample_dir.name
        target_result = target_sample_dir / "result.json"
        if target_result.exists():
            skipped += 1
            continue
        shutil.copytree(sample_dir, target_sample_dir, dirs_exist_ok=True)
        copied += 1

    print(
        f"[extend] {provider}: source_result_count={source_count}, copied={copied}, already_present={skipped}",
        flush=True,
    )


def count_result_files(provider_dir: Path) -> int:
    return sum(1 for path in provider_dir.glob("*/result.json") if path.is_file())


def print_cache_status(label: str) -> None:
    """Print a compact cache diagnostic before/after running the base runner."""

    print(f"[extend] Cache status {label}:", flush=True)
    for provider in PROVIDERS.split(","):
        provider = provider.strip()
        if not provider:
            continue
        stats = cache_stats(TARGET_500_DIR / provider)
        print(
            "[extend]"
            f" {provider}: result_files={stats['result_files']},"
            f" valid={stats['valid']},"
            f" failed_or_unparsed={stats['failed_or_unparsed']},"
            f" none_prediction={stats['none_prediction']}",
            flush=True,
        )


def cache_stats(provider_dir: Path) -> dict[str, int]:
    stats = {
        "result_files": 0,
        "valid": 0,
        "failed_or_unparsed": 0,
        "none_prediction": 0,
    }
    if not provider_dir.exists():
        return stats
    for result_path in provider_dir.glob("*/result.json"):
        stats["result_files"] += 1
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            stats["failed_or_unparsed"] += 1
            stats["none_prediction"] += 1
            continue
        prediction = result.get("prediction")
        prediction_empty = prediction is None or str(prediction).strip() == ""
        valid = (
            result.get("status") == "passed"
            and result.get("parse_success") is True
            and not prediction_empty
        )
        if valid:
            stats["valid"] += 1
        else:
            stats["failed_or_unparsed"] += 1
        if prediction_empty:
            stats["none_prediction"] += 1
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
