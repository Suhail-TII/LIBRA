# Responsible disclosure

## Status

The model providers whose systems are measured here (OpenAI, Meta, Alibaba and
Mistral) were **not** notified individually before this work was submitted or
this repository was published.

That is a deliberate position, stated plainly so that readers can weigh it
rather than assume one way or the other. The reasoning follows.

## What this work contains, and what it does not

This work introduces no new jailbreak primitive. Its contribution is
measurement: it fixes a finite space of 103 configurations assembled from
already-published technique families, measures every one of them exhaustively,
and evaluates a budgeted search procedure against the optimum that exhaustive
measurement reveals.

The components are public before this paper:

- The harmful instructions are AdvBench, a public benchmark released with
  Zou et al. (2023).
- The semantic framings instantiate technique families documented in the
  published literature and in the L1B3RT4S corpus of jailbreak prompts
  circulated in practice, which is itself a public repository.
- The structural templates are the UTOS structures published with
  StructuralSleight (Li et al., 2025).

What is new is which combinations work on which model, how much choosing badly
costs, and how cheaply a good choice can be found.

## What is released

- Implementations and full prompt text of all eleven framings and four
  templates. Reproducing the measurements requires them.
- Per-response classifier labels for every scored response of the probe and
  screening cohorts, 128,414 responses. The held-out cohort's per-response
  labels are not included; see below.
- Replay caches, selection results, and the scripts that produce the paper's
  tables and figures. Two of those scripts read inputs that are not released,
  so a clean clone regenerates most tables but not the held-out results of
  Sections IV-D and V-D. `NOTES.md` names which, and the derived outputs for
  those two are published instead.

## What is not released

- **No generated completions.** Not for the probe grids, not for the held-out
  grid, not for the screening runs. The per-cell outcomes behind the held-out
  comparison are withheld for the same reason; only the four summary rows in
  `results/eval_half/heldout_103.csv` are published.
- No API request or response bodies.

## The honest caveat

Releasing the configuration text means this repository contains attack
configurations that reached over 90% attack success under the paper's
evaluator on two of the measured models, at the model versions and generation
settings recorded in the release manifest. Withholding that text would make
the measurements unreproducible, which is the trade this release makes. It is
a trade, not a free choice, and a reader who disagrees with it has a
reasonable position.

What limits the uplift is that the configurations are compositions of
components that were already public, applied to a public benchmark, and
measured under a stated evaluator whose own error is characterised in the
paper. A reader who wanted these attacks could have assembled them from the
cited sources. What they could not have known without this work is which one
to pick for a given model, which is also what makes the result useful to a
defender.

## For model providers

Everything needed to reproduce these results against your own model is in this
repository, including the search procedure, so you can measure your own
exposure under a query budget you choose rather than relying on the numbers
reported here. Section V of the paper describes the procedure and
`heldout_replay.py` and `bootstrap_search.py` show how it was evaluated.

If you want the withheld per-response data for a model you operate, or want
the configuration text removed or gated, open an issue or contact the
corresponding author. We will respond.

## Version

This statement describes the repository at the commit in which it appears. The
model versions, snapshot identifiers where preserved, and generation settings
are recorded in the release manifest; API-served results are not reproducible
against a moving endpoint and the paper says so.
