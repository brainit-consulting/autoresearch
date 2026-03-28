# Chess PGN Progress

## Summary

This branch adapts `autoresearch` from its default TinyStories-oriented setup into a short-run autonomous research loop for chess PGN modeling. Instead of optimizing for general story text, the agent iterates on `train.py` to reduce validation bits per byte (`val_bpb`) on a Lichess PGN dataset under a fixed 5-minute training budget.

The core result so far is clear: the repo is learning the structure of chess game records much better than the starting baseline. At the same time, the current evidence is still about PGN modeling quality, not chess-playing strength.

## What Changed

The branch switched the training target from TinyStories-style text to compressed Lichess PGN files and trained a tokenizer over that domain. The autonomous loop stayed intentionally narrow: the agent edits only `train.py`, runs a timed experiment, records the result in `results.tsv`, keeps changes that improve `val_bpb`, and discards the rest.

This was a good fit for autonomous experimentation because chess PGN is compact, repetitive, and highly structured. That makes it easier to test whether short research loops can discover better architectures and hyperparameters quickly, without needing long training runs.

## Why Chess PGN

Chess PGN is a useful target for this kind of experiment because it combines rigid formatting with meaningful long-range structure. Headers, move numbering, openings, castling notation, captures, checks, and results all follow recurring patterns. That gives the training loop a domain where compression-style improvements are easier to measure than in noisier natural-language text.

It also creates a cleaner test of whether the autonomous loop can discover practical improvements on its own. A 5-minute experiment budget is restrictive, so a structured domain is more likely to show signal quickly.

## How The Loop Works

The workflow is simple:

1. Start from the current best branch state.
2. Let the agent modify `train.py`.
3. Run a fixed-budget training experiment on the chess PGN dataset.
4. Evaluate with `val_bpb`.
5. Record the run in `results.tsv`.
6. Keep the change only if the metric improves.

This means the experiment history is not a set of hand-picked anecdotes. It is a log of accepted and rejected changes under a consistent metric and time budget.

## Quantitative Results

The baseline logged in [results.tsv](./results.tsv) is `val_bpb = 1.345913` for the initial chess setup. The current best logged result is `0.539555`, also in [results.tsv](./results.tsv). That is an absolute improvement of `0.806358`, or about `59.9%` relative to the starting baseline.

A few trends stand out from the kept experiments:

- Reducing effective batch size improved step count sharply under the fixed wall-clock budget.
- A smaller, faster model outperformed the deeper starting baseline for this domain.
- Several later gains came from optimizer and initialization tuning rather than large architectural changes.
- The best current result comes from reducing the token embedding initialization standard deviation from `1.0` to `0.5`.

In other words, the branch has moved well beyond “it runs” and into a regime with repeatable measured improvement.

## Qualitative Results

Qualitative spot checks of the current checkpoint show that it can generate plausible PGN-looking text, including game headers, metadata fields, opening names, results, and opening move sequences. That is meaningful progress compared with a model that only memorizes fragments or produces malformed notation.

However, the current checkpoint is still weak at continuation fidelity. In direct sampling tests, it can restart a fresh game record or drift away from the prompted line instead of reliably continuing a partial game. That means the model has learned a lot about the distribution of chess PGN text, but it is not yet dependable as a move-by-move continuation model.

## What This Does Not Yet Prove

This branch does **not** yet demonstrate meaningful chess-playing ability.

Current limitations:

- There is no legality checker for generated moves.
- There is no FEN-aware evaluation.
- There is no Stockfish or other engine-based strength measurement.
- There is no robust benchmark for continuation accuracy from a partial game state.
- The main ground-truth metric is still `val_bpb`, which measures predictive compression quality rather than chess strength directly.

Because of that, the correct claim today is narrow: this repo shows meaningful progress on modeling chess PGN structure and formatting, not evidence of strong chess reasoning or playing strength.

## Why The Results Still Matter

Even with those limits, the branch is already useful as a proof of process. It shows that an autonomous loop can explore a structured domain, reject many bad ideas, keep a smaller number of better ones, and drive a large quantitative gain without changing the evaluation harness.

That makes this repo a credible experiment in autonomous training research, even before it becomes a credible chess model.

## Next Steps

The next upgrades should focus on evaluation quality rather than only pushing `val_bpb` lower:

- Add legality checks for generated continuations.
- Add continuation benchmarks from real partial PGNs.
- Add engine-based evaluation for move quality.
- Make the public README and repo metadata match the actual chess branch state.
- Add a tracked `LICENSE` file so the repo can be published cleanly.

## Public Release Status

This branch is close to being explainable in public, but it is not quite publication-ready yet. The results are real, but the repo still needs small cleanup work around licensing, public-facing messaging, and clearer statements about what the current evaluation does and does not measure.
