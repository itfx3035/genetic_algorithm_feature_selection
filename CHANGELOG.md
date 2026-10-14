# Changelog

All notable changes to this project are documented here.

## 0.4.0

**Subset-size constraints and a built-in sparsity penalty**

- New `min_features` / `max_features` parameters: **hard** bounds on the
  subset size the GA can propose, enforce and return. Enforced by
  *repairing* chromosomes back into bounds after every operator (and on
  initialisation), so the entire search — every elite, offspring and
  history `best_subset`, not just the final answer — stays inside the
  range. `max_features=K` is the built-in "find me the best K features".
- New `penalty_per_feature` parameter: **soft** Occam-razor. When set, the
  GA optimizes `error + penalty_per_feature × len(subset)` (selection,
  elitism, early-stopping and best-tracking all act on this objective),
  while the evaluator's own number is kept and reported separately:
  `best["raw_error"]`, `run()`'s error is the penalized objective and every
  `top_subsets()` entry gains a `raw_error` field. The two dials
  (bounds = hard, penalty = soft) are independent and compose.
- **Backwards compatible**: with `min_features=None, max_features=None,
  penalty_per_feature=None` (all defaults) the RNG draw order and search
  are exactly what v0.3.0 did — the bounds repair only consumes RNG when a
  chromosome is actually out of bounds, and the objective equals the raw
  evaluator error when no penalty is set.
- The demo/test suite grew a `TestSizeConstraints` + `TestPenalty` group
  (bounds active and respected in the *whole* history, per-feature
  penalty bookkeeping, raw-vs-objective separation) and `picklable_metric`
  gained `flat_error` / `shrink_error`.
- `__repr__` now shows the constraint knobs when they are set.

## 0.3.0

**Parallel evaluation (`n_jobs`)**

- New `n_jobs` parameter on `GeneticFeatureSelector` and `select_features_with_ga`
  (`n_jobs=None` or `int >= 2`). When set, each generation's *fresh* feature
  subsets (the ones missing from the evaluation cache) are scored by
  `n_jobs` worker processes; already-scored subsets are always reused from
  the cache. The GA's own search (RNG draw order, elite/tournament/crossover
  decisions) never touches the pool, so a seeded run with a deterministic
  evaluator is **bit-identical to the `n_jobs=None` run** for any `n_jobs` —
  `n_jobs` is a wall-time optimization, not a search change.

- Evaluation-function pickling is layered: `cloudpickle` (by value; closures
  and lambdas) when installed, else plain `pickle` (by reference; top-level
  functions in importable modules). A failed pickling or a worker-pool
  failure *falls back to serial evaluation* rather than crashing, and the
  fallback is reported explicitly (see observability below).

- **Observability** (so "does it parallelize here?" is answerable, not
  assumed):
  - `parallel_subsets_scored` — how many subset evaluations the worker pool
    actually performed this run (`0` if the pool never ran),
  - `parallel_unavailable_reason` — why the pool was not used (e.g.
    unpicklable evaluator, sandbox without working process spawning), or
    `None` when the pool is/ran fine,
  - the verbose output now ends with an explicit
    `Evaluation: N subset(s) scored by K worker process(es)` line, or
    `Evaluation: SERIAL only (...)` when the pool never ran.
  - On the *first* pool failure the library disables parallel evaluation for
    the remainder of the run (one `WARNING`, not one per generation).

- Worker processes receive the library's directory *and* the evaluator
  module's directory on their `sys.path`, so pickled-by-reference evaluators
  imported from your project's files work.

**Testing**

- Real pytest suite under `tests/` (34 tests): constructor validation,
  seed reproducibility, cache accounting, history invariants,
  `top_subsets`/`feature_prevalence` contracts, invalid-value/exception
  handling, `fail_on_error`, early stopping, metric helpers, wrapper
  equivalence, and `n_jobs` ↔ serial parity with the observability
  assertions above. Run: `python -m pytest tests/ -q`.
- The in-file self-test (`python ga_feature_selection_lib.py`) gained an
  `n_jobs` parity section that also reports whether the pool truly ran.

- `example_usage.py`: the GA run now lives under `if __name__ == "__main__":`
  (required so forkserver-started workers do not re-import and re-run the
  script) and prints the parallel-evaluation accounting line.

## 0.2.0

- **Evaluator-contract defense.** `None`, `NaN`, `±inf`, non-numeric results,
  *and raised exceptions* mark a subset *invalid*: it is counted, excluded
  from selection, and can never be returned. New `fail_on_error=True` makes
  raised exceptions propagate instead. A run ends with a `UserWarning`
  summarizing invalid/raised-subset counts.
- **Post-run state** on the instance: per-generation `history`, `best`,
  `unique_evaluations`, `invalid_evaluation_count`, `last_evaluation_error`
  (all reset on each `run()`, so a selector can be reused).
- **Post-run analysis**: `top_subsets(k)` (best scored subsets) and
  `feature_prevalence(top_k)` (per-feature inclusion frequency among the
  top-`k` scored subsets — a robust importance signal).
- **Early stopping**: `stall_generations=K` stops the run after `K`
  consecutive generations without a strict best-error improvement (`None` =
  full budget, the default).
- **Logging**: progress events are always logged on the
  `ga_feature_selection` logger (`INFO` at start/finish, `DEBUG` per
  generation); `verbose=True` additionally prints them to stdout.

## 0.1.0

- Initial released version: `GeneticFeatureSelector`, the
  `select_features_with_ga` wrapper, `build_binary_error` /
  `build_regression_error` helpers, `random_state` seeding, evaluation cache
  (each distinct subset scored at most once per run), tournament selection
  over valid individuals, uniform crossover, bit-flip mutation, elitism,
  in-file self-test, pip installability (`pyproject.toml`), and the
  Wine Quality demo.
