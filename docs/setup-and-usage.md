# Autoresearch Chess PGN — Setup & Usage

## Overview

This repo trains a tiny LLM on chess PGN game data from Lichess. An AI agent autonomously experiments with architecture and hyperparameters in `train.py`, keeping changes that lower val_bpb (bits per byte) and reverting those that don't.

Public-facing branch summary:

- Baseline logged result: `1.345913`
- Best logged result: `0.539555`
- Improvement vs baseline: about `59.9%`
- Current evaluation: `val_bpb` plus qualitative PGN sampling
- Not yet included: legality checks, FEN-aware evaluation, or engine-strength benchmarks

See also:

- `../chess-pgn-progress.md`
- `../progress.png`
- `../progress-waterfall.png`
- `../progress-categories.png`

## Prerequisites

- NVIDIA GPU (tested on RTX 3070 Laptop, 8GB VRAM)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Initial Setup

### 1. Install uv (if not already installed)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Download data and train tokenizer

```bash
export AUTORESEARCH_CACHE_DIR="h:/autoresearch/.cache"
uv run prepare.py --dataset chesspgn
```

This downloads 6 months of Lichess PGN games (Jan–Jun 2013, ~136 MB compressed) and trains a BPE tokenizer with 8192 vocab size.

### 4. Verify with smoke test

```bash
AUTORESEARCH_CACHE_DIR=h:/autoresearch/.cache uv run train.py --dataset chesspgn --smoke-test
```

Expected: completes in ~80s, val_bpb ~2.83, peak VRAM ~5.4 GB.

## Running a Training Experiment

Single 5-minute training run:

```bash
AUTORESEARCH_CACHE_DIR=h:/autoresearch/.cache uv run train.py --dataset chesspgn
```

## Running the Chess Benchmark

Board-aware continuation benchmark:

```bash
AUTORESEARCH_CACHE_DIR=h:/autoresearch/.cache uv run eval_chess.py --dataset chesspgn --checkpoint checkpoint_pre_eval.pt
```

Optional knobs:

- `--device auto|cpu|cuda`
- `--max-games 100`
- `--plies 8 16`
- `--greedy-samples 3`

Reported metrics:

- `legal_move_rate`
- `next_move_top1_accuracy`
- `next_move_top3_accuracy`
- `avg_legal_candidates`
- deterministic printed examples showing prompt, generated move, legality, ground truth, and top-ranked legal candidates

## Interpreting Results Correctly

The current branch should be described as a chess-PGN modeling project, not a verified chess-playing model.

What the current results support:

- the model learns PGN formatting and metadata structure
- the autonomous loop finds repeatable metric improvements
- the checkpoint can generate plausible-looking chess headers and opening lines

What the current results do not support yet:

- reliable legal move generation
- continuation accuracy from a known board state
- engine-measured move quality
- claims of playing strength

## Running the Autonomous Research Agent

1. Create a branch for the run:
   ```bash
   git checkout -b autoresearch/mar21
   ```

2. Point your AI agent (Claude, Codex, etc.) at this repo and prompt:
   ```
   Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
   ```

3. The agent will loop autonomously: modify `train.py` -> train 5 min -> keep/revert -> repeat. ~12 experiments/hour, ~100 overnight.

4. Results are logged to `results.tsv`.

## Environment Variables

| Variable | Purpose | Value |
|----------|---------|-------|
| `AUTORESEARCH_CACHE_DIR` | Data storage location | `h:/autoresearch/.cache` |
| `AUTORESEARCH_DATASET` | Override active dataset | `chesspgn` |

## Key Files

| File | Role | Editable? |
|------|------|-----------|
| `train.py` | Model, optimizer, training loop | Yes (agent edits this) |
| `prepare.py` | Data pipeline, tokenizer, evaluation | No |
| `program.md` | Agent instructions | Yes (human edits this) |
| `results.tsv` | Experiment log | Auto-generated |

## Data Location

- Compressed PGN files: `h:\autoresearch\.cache\datasets\chesspgn\data\`
- Tokenizer: `h:\autoresearch\.cache\datasets\chesspgn\tokenizer\`

## GPU Notes (RTX 3070 Laptop 8GB)

- Runs in "compatibility" mode (laptop GPU not in official desktop matrix)
- `torch.compile` is disabled
- Activation checkpointing is enabled by default
- Batch size 16 works, peak VRAM ~5.4 GB
- Keep experiments under ~6 GB VRAM to avoid OOM

## Recommended LLM for the Agent

**Claude Code with Sonnet** is the recommended model for running the autonomous loop.

| Model | Code Quality | Cost per ~100 runs | Verdict |
|-------|-------------|---------------------|---------|
| **Claude Sonnet** | Great | ~$5-10 | Best choice — fast, reliable, cost-effective |
| Claude Opus | Excellent | ~$50-100 | Overkill — changes are simple, not worth 10x cost |
| Claude Haiku | Decent | ~$1-2 | Too weak for architecture experiments |
| Local LLM (LM Studio) | Variable | Free | Not recommended — unreliable for long autonomous runs |

**Why Sonnet:**
- The task is repetitive (tweak hyperparameter, run, check, repeat)
- Code changes are simple (single file, mostly numeric knobs)
- ~10x cheaper than Opus for nearly the same quality on this task
- Fast enough to not waste time between experiments

**How to launch:**
Open a new Claude Code session (Sonnet) in this repo, set permissions to auto-approve, and prompt:
```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

**Local LLMs (e.g. Qwen 2.5 Coder 32B, DeepSeek Coder V2 via LM Studio):** Zero API cost but reliability drops significantly for autonomous multi-step loops. Local models tend to get confused after many iterations and make sloppy edits. Not recommended for overnight unattended runs.

## Repo Origin

- Forked from: [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows RTX fork)
- Original: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Private copy: [brainit-consulting/autoresearch](https://github.com/brainit-consulting/autoresearch)
- Adapted to train on chess PGN data instead of TinyStories
