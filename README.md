# MathExplainAgent

MathExplainAgent is a modular research prototype for generating math explanation videos from problem images. The system decomposes problem-image-to-video generation into structured problem parsing, problem classification, CAS-backed solving, verifier-guided repair, teaching-script generation, executable video direction, TTS synthesis, and deterministic rendering.

This anonymous code package is prepared to support reproducibility and artifact inspection for the accompanying manuscript. For detailed experiment commands and expected outputs, see `REPRODUCE.md`.

## Core Components

1. Vision Parser
2. Problem Classifier
3. Solver
4. Verifier
5. Explainer
6. Script Director
7. TTS and Renderer

## Main Intermediate Representations

- SPR: Structured Problem Representation.
- EMR: Executable Mathematical Representation.
- SCS: Solution Chain Structure.
- EDS: Executable Director Script.

## Important Paths

- `src/mathexplain/`: implementation of agents, schemas, services, rendering, and configuration.
- `scripts/`: selected experiment entry points used by the reported results.
- `tests/`: regression and schema tests.
- `configs/`: lightweight runtime configuration files.
- `data/experiments/gsm8k/`: rendered GSM8K-style images, fixed manifest, source JSONL files, and pair manifests used by reported experiments.

## Privacy and Anonymization Notes

The package excludes API keys, `.env`, caches, virtual environments, full historical experiment outputs, uncurated logs, private absolute paths, and redundant video collections. It is intended as an anonymous supplementary code package for review.
