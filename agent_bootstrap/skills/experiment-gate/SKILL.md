---
name: experiment-gate
description: MANDATORY checklist before designing/launching ANY new experiment in a competition campaign — 5 checks derived from measured failure meta-analysis (precondition, own-history, instrument-family, research-gate, matched-control) + trap signatures + working patterns
---

# experiment-gate — walk this before ANY new experiment

Full operational version: `rogii/notes/EXPERIMENT_GATE.md` (repo) and memory
`experiment_selection_lessons.md`. Summary:

## The 5 checks
1. **Precondition**: does the input provably contain the signal the method needs?
   Run the cheapest direct probe FIRST (landscape scan / KS audit / cache inspection).
2. **Own-history**: grep scripts/, notes/, the dead-end ledger, and memory for the axis —
   the answer may already be on disk (exp140 lesson).
3. **Instrument-family**: classify the candidate by information source. Instruments
   (G*, confirm law) only validate within families they have publicly scored.
   New family ⇒ probe-grade prereg, wide band, small weight (driver36 lesson).
4. **Research-gate**: papers cited + math written (headroom, null vs hypothesis)
   + dead-list checked (with named implementation difference if re-testing a kill).
5. **Matched-control**: every eval carries the control isolating exactly the tested
   change (r187 refit-freedom lesson).

## Auto-reject trap signatures
- pooled+conf improve, G* flat/worse → G*-bait, never cashed publicly
- tune-gain + confirm-loss → non-transfer overfit
- weights collapse to global under leave-pool-out CV → pool noise
- gains correlate with sibling distance → label-transfer mirage

## Patterns that work — copy them
Oracle-headroom before build · twin/duplicate-label noise-floor instruments ·
per-band/per-bin decomposition · best/worst control cohorts · pre-registration ·
detached long processes (nohup+disown + file-marker watcher).
