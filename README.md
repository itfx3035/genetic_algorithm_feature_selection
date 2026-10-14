# GA Feature Selection

A small Python library that finds feature subsets with a **genetic algorithm (GA)**. You supply an evaluation function that scores a feature subset; the GA evolves subsets (tournament selection, uniform crossover, bit-flip mutation, elitism) to minimize it.

> **Experimental / Alpha.** Single-file library, no stability guarantees. The API and defaults may change. The "results" section below is a demonstration, not a benchmark.

---

## How it works

Each candidate subset is a binary chromosome (gene = 1 → feature included). Every generation:

1. **Evaluate** all chromosomes with your `evaluation_function` (lower is better). Results are **cached per distinct subset**, and each distinct subset is scored by your function at most once per run. With `n_jobs=K`, the subsets each generation has not seen before are scored by `K` worker **processes**; already-scored subsets are always reused from the cache. A subset is *invalid* if the function returns `None`/non-finite, or **raises** — invalid subsets are excluded from selection and (by default) never crash the run; the counter is reported at the end.
2. **Track the best** valid subset seen so far (`selector.best`), by the GA's objective — your evaluator's number plus, if `penalty_per_feature` is set, `penalty_per_feature × #features`. The evaluator's own number is kept separately as `best["raw_error"]`.
3. **Build the next generation**: the top `elite_size` valid individuals (best by the GA's objective) are copied unchanged; the rest are produced by **tournament selection** (among *valid* individuals only) → **uniform crossover** (two children, second is the complement draw) → **per-gene bit-flip mutation** with probability `mutation_rate`. If `min_features`/`max_features` are set, every child (and the initial population) is **repaired into the size bounds** afterwards — bounds are hard constraints on the search, not a filter on results.
4. Repeat up to `max_generations` generations. With `stall_generations=K`, the run stops early once the best error has not *strictly* improved for K consecutive generations.

After `run()`, the instance holds per-generation `history`, the `best` solution, `unique_evaluations`, `invalid_evaluation_count`, `last_evaluation_error`, plus `top_subsets(k)` and `feature_prevalence(top_k)` for post-hoc analysis.

If *every* individual in a generation is invalid, the population is re-initialized and logged — with a pathological evaluator this can loop forever, which is intentional (it surfaces the problem rather than returning garbage).

```
                  +---------------------------+
                  |   initial random          |
                  |   population              |
                  +-------------+-------------+
                                |
            +-------------------+-------------------+
            |                   v                   |
            |      +--------------------------+     |
            +-----+| evaluate all chromosomes |     |
            |      | (user fn; cached;        |     |
            |      |  None/raise = invalid    |     |
            |      +------------+-------------+     |
            |                 |                     |
            |      +----------v-----------+         |
            |      | update best-so-far   |         |
            |      +----------+-----------+         |
            |                 |                     |
            |      +----------v-----------+         |
            |      | next generation:     |         |
            |      |  elites (unchanged)  |         |
            |      |  + tournament select |         |
            |      |  + uniform crossover |         |
            |      |  + bit-flip mutation |         |
            |      +----------+-----------+         |
            |                 |                     |
            +-----------------+---------------------+
                             |
              more generations (and not stalled)?
                             |
                  report best subset + its score
```

## Installation

The library itself has **no third-party dependencies** (standard library only).

```bash
pip install .              # or: pip install -e .  for an editable install
pip install ".[dev]"       # scikit-learn / pandas / numpy for the demo
pip install ".[parallel]"  # cloudpickle: n_jobs with closure/lambda evaluators
pip install ".[test]"      # pytest for the test suite
```


The demo dataset (`wine-quality-red.csv`, UCI Wine Quality, red wine, `CC0`) is included in this directory.

**Multiprocessing caveat.** `n_jobs` uses Python's `multiprocessing` to spawn worker *processes*. If your environment cannot do that (some sandboxes, or certain startup settings), the library detects it, **falls back to serial evaluation for that run**, and says so explicitly in the verbose output (`Evaluation: SERIAL only (...)`) and in `selector.parallel_unavailable_reason` — results are always correct, just unaccelerated.

## Quick start

No ML library required — just an evaluation function:

```python
from ga_feature_selection_lib import select_features_with_ga

def dummy_eval(features):
    """Lower is better. Return None for invalid subsets."""
    n = len(features)
    return max(0.1, 0.2 * abs(n - 8)) / 2     # fake landscape, min at 8

features = [f"feature_{i}" for i in range(20)]

selected, error = select_features_with_ga(
    features=features,
    evaluation_function=dummy_eval,
    population_size=30,
    mutation_rate=0.05,
    max_generations=20,
    random_state=42,          # make runs reproducible
    # n_jobs=2,                # optional: evaluate on 2 worker processes
    verbose=True,
)
print(len(selected), "selected:", selected)
```

The GA is stochastic. Pass `random_state=<int>` and a *deterministic* evaluation function to get identical results across runs.

## Real-world example: Wine Quality + RandomForest

`example_usage.py` runs the full workflow on the included UCI Wine Quality (red) dataset (1,599 rows × 11 features, multi-class `quality`):

```python
from ga_feature_selection_lib import GeneticFeatureSelector, build_binary_error

# ... df = pd.read_csv("wine-quality-red.csv", sep=";") ...
# ... 80/20 stratified train/test split (random_state=42) ...

def cv_error(selected):                       # GA objective (train split only)
    if not selected:
        return None                           # empty subset: cannot score
    X = X_train[selected]
    err = 0.0
    for tr, te in folds:                      # 5-fold StratifiedKFold on train
        model = RandomForestClassifier(n_estimators=60, random_state=42)
        model.fit(X.iloc[tr], y_train.iloc[tr])
        err += build_binary_error(list(model.predict(X.iloc[te])),
                                  list(y_train.iloc[te]))
    return err / len(folds)

selector = GeneticFeatureSelector(
    features,                     # the 11 feature names
    cv_error,
    population_size=30, mutation_rate=0.03, elite_size=5, tournament_size=3,
    max_generations=40,
    stall_generations=3,          # stop once 3 generations go without progress
    random_state=42,
    n_jobs=2,                     # optional: score fresh subsets in processes
)
selected, ga_error = selector.run(verbose=True)
selector.history                          # per-generation record
selector.feature_prevalence(top_k=30)    # robust importance ranking
selector.top_subsets(5)                  # the best subsets scored
```


Run it:

```bash
pip install ".[dev]"
python example_usage.py
```

**Measured on this machine (for calibration, not a benchmark):** 40-generation budget with a 3-generation stall → **early stop at 7 generations, 96 unique subsets scored** (cache), **60-tree RandomForest × 5-fold CV on 1,279 training rows**. Wall time of the GA loop itself: **4.3 min serial (`n_jobs=None`), 3.0 min with `n_jobs=2`** (the `Evaluation:` line in the output is the honest source; see [Parallel evaluation](#parallel-evaluation) for the caveats). Outcome on the untouched test split, comparing the **same model the GA optimized** (RandomForest, fit once on the full train set): **test accuracy 0.6750 with the GA's 7-feature subset, vs 0.6750 with all 11 features** — i.e. the GA found a 7-of-11-feature subset that matches the full-feature baseline on this single split. That is a legitimate possible outcome but **not a benchmark**: one seed, one split, one model; a real selection experiment would validate with repeated or nested CV. (The GA's in-sample CV error, ≈ 0.303, is an optimistic estimate — the test comparison is the honest one.)

`example_usage.py` prints all of this: prevalence, top-5 scored subsets, the test-split comparison, and feature importances, plus the full `history`-aware progress output.

## API reference

### `GeneticFeatureSelector`

```python
from ga_feature_selection_lib import GeneticFeatureSelector

selector = GeneticFeatureSelector(
    features,               # required
    evaluation_function,    # required
    population_size=50,
    mutation_rate=0.01,     # per-gene flip probability, 0..1
    elite_size=5,
    tournament_size=3,
    max_generations=100,
    random_state=None,      # int seed for reproducibility
    stall_generations=None, # early stop: N generations without best improving
    fail_on_error=False,    # True = let evaluator exceptions propagate
    min_features=None,      # hard lower bound on subset size (0..n features)
    max_features=None,      # hard upper bound ("find me the best K")
    penalty_per_feature=None,  # soft Occam-razor: error + lambda*size
    n_jobs=None,            # see "Parallel evaluation" below
)
best_features, best_error = selector.run(verbose=True)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `features` | `list[str]` | — | Feature names. Non-empty, no duplicates. |
| `evaluation_function` | `Callable[[list[str]], float \| None]` | — | Scores a subset; **lower is better**. May be called with *any* subset, **including the empty list** — return `None` for subsets your model can't use. `None`/`NaN`/`±inf`/non-numeric and **raised exceptions** make the subset invalid (counted, never returned). Called at most once per distinct subset (cached). |
| `population_size` | `int` | `50` | Individuals per generation (≥ 2). |
| `mutation_rate` | `float` | `0.01` | Per-gene flip probability. Expected flips per chromosome = `mutation_rate × n_features`. |
| `elite_size` | `int` | `5` | Best valid individuals copied unchanged. `0 ≤ elite_size ≤ population_size`. If `elite_size == population_size`, no new offspring are created and the run stagnates. |
| `tournament_size` | `int` | `3` | Candidates per tournament (≥ 2). Larger = stronger selection pressure. |
| `max_generations` | `int` | `100` | Generation budget (≥ 1). Always *at most* this many; may stop earlier via `stall_generations`. |
| `random_state` | `int \| None` | `None` | Seed. Same seed + deterministic evaluator ⇒ identical run. |
| `stall_generations` | `int \| None` | `None` | Early stop after this many consecutive generations without a strict best-error improvement (first valid result always counts as an improvement). `None` = never (uses the full budget). |
| `fail_on_error` | `bool` | `False` | Re-raise evaluator exceptions instead of marking that subset invalid. |
| `min_features` | `int \| None` | `None` | Hard **lower** bound on subset size: the GA only ever proposes (and can only return) subsets with ≥ `min_features` features. Enforced by repairing chromosomes back into bounds after every operator. |
| `max_features` | `int \| None` | `None` | Hard **upper** bound, same mechanism. The built-in way to ask "find me the best **K** features": `max_features=K`. |
| `penalty_per_feature` | `float \| None` | `None` | **Soft** Occam-razor: the GA optimizes `error + penalty_per_feature × #features` instead of `error` (the raw number is still reported as `best["raw_error"]`). Trades a little score against subset size; pick it by feel (e.g. `0.005` on a 0-1 error) and check `top_subsets` afterwards. |
| `n_jobs` | `int \| None` | `None` | Evaluate each generation's *new* subsets on `n_jobs` worker processes (cache misses only). `None` = serial. Same-seed runs are identical for any setting (see [Parallel evaluation](#parallel-evaluation)). |

**Returns** `(selected_features: list[str], best_error: float)` — the best *valid* subset found (in `features` order) and its error: the GA's objective, i.e. your evaluator's return value plus `penalty_per_feature × len(subset)` (identical to the evaluator's value when no penalty is set; the evaluator's own number is `selector.best["raw_error"]`). If no subset was ever valid, returns `([], 0.0)` (with a warning if the evaluator raised).

**Instance state after `run()`** (all reset on each call, so a selector can be reused):

| Attribute | Meaning |
|---|---|
| `history` | One dict per completed generation: `generation`, `improved` (bool), `best_error`, `best_subset`, `new_evaluations`, `unique_evaluations`, `invalid_evaluations`. |
| `best` | `{"features", "error", "raw_error", "generation"}` of the best valid subset, or `None` if none. `error` is what the GA minimized (with `penalty_per_feature` set, that *includes* the penalty); `raw_error` is the evaluator's own number for the same subset. |
| `unique_evaluations` (property) | Distinct subsets scored by your function (cache size). |
| `invalid_evaluation_count` | Distinct subsets that scored invalid (including ones that raised). |
| `last_evaluation_error` | Most recent exception raised by the evaluator, or `None`. |
| `parallel_subsets_scored` | How many evaluations the worker pool actually performed this run (`0` when `n_jobs` was not set or the pool never ran). |
| `parallel_unavailable_reason` | Why the pool was not used (`None` = pool ran fine / not requested) — e.g. unpicklable evaluator or no working process spawning. The verbose output prints the same, on the final `Evaluation:` line. |

**Methods (post-run analysis):**

| Method | Returns |
|---|---|
| `run(verbose=True)` | `(best_features, best_error)`; populates the state above. |
| `top_subsets(k=10)` | The `k` best scored subsets by GA objective: `[{"features": list, "error": float, "raw_error": float}]`, best first. |
| `feature_prevalence(top_k=None)` | Per-feature inclusion frequency among the `top_k` best-scored subsets (all of them if `None`), best-ranked by GA objective: `[{"feature", "count", "of", "prevalence"}]`, sorted descending. |

Parameter validation raises `ValueError` with a message for any rule above.

**Logging.** `verbose=True` additionally prints to stdout. Progress is **always** logged on the `ga_feature_selection` logger (start/finish at `INFO`, per-generation lines at `DEBUG`); under the default last-resort configuration only `WARNING`+ reaches stderr, so to see per-generation lines set the level, e.g. `logging.getLogger("ga_feature_selection").setLevel(logging.DEBUG)`. A `UserWarning` is emitted once at the end of the run if the evaluator ever raised.

### `select_features_with_ga(...)`

Convenience wrapper with the exact same parameters (plus `verbose: bool = True`) and return value:

```python
selected, error = select_features_with_ga(
    features, evaluation_function,   # same keyword args as the class...
)
```

It does *not* expose the instance state — if you want `history` / `top_subsets` / `feature_prevalence`, build a `GeneticFeatureSelector` directly (as in `example_usage.py`).

### Metric helpers

```python
from ga_feature_selection_lib import build_binary_error, build_regression_error

build_binary_error(y_pred, y_true)        # 1 - accuracy
build_regression_error(y_true, y_pred)    # MSE, normalized by range of y_true
```

For classifiers with more than two classes, compute whatever "error" you want (e.g. `1 - accuracy` from `cross_val_score`) yourself — the helpers are just convenience implementations of the same contract: numeric, lower is better.

### Evaluator checklist

- Returns a **number, lower is better** (or `None` for invalid).
- Accepts **any subset in any order**, including empty.
- **Deterministic** if you set `random_state` (fix CV `random_state`s, model seeds, …).
- **Cheap-ish** — see cost below.

## Parallel evaluation (`n_jobs`)

- **What changes / what doesn't.** Only *evaluation* is parallelized: each
  generation's subsets that are **missing from the cache** are picked (with
  `cloudpickle` if installed, else plain `pickle`) and scored on `n_jobs`
  worker processes. The GA itself — the RNG draw order, the tournament /
  crossover / mutation decisions, the cache — lives in the main process and
  is never touched by the pool. Therefore, for a **deterministic** evaluator
  with a fixed seed, the result of an `n_jobs=K` run is **bit-identical** to
  the `n_jobs=None` run (selected subset, error, and per-generation history).
  `n_jobs` buys wall time, not a different answer.
- **Your evaluator must be picklable.** Top-level functions and classes in
  importable modules work with plain `pickle` (the worker is given the
  library's and your module's directories on its path). Closures and lambdas
  — the common case for "evaluate a model on these columns" — need
  `pip install cloudpickle` (`pip install ".[parallel]"`).
- **Keep the GA call under `if __name__ == "__main__":`** in scripts.
  Worker processes re-import your script's main module; code that runs there
  at import time would run again in every worker.
- **Honest fallback.** If the evaluator can't be pickled, or the environment
  can't spawn processes at all, the run falls back to **serial** evaluation
  (one `WARNING`, then serial for the rest of the run) and reports it
  unambiguously: the verbose output's final line says
  `Evaluation: SERIAL only (reason)`, and
  `selector.parallel_unavailable_reason` holds the same text. On success it
  says `Evaluation: N subset(s) scored by K worker process(es)`; both figures
  are also in `selector.parallel_subsets_scored`.
- **When it helps (and how much).** `n_jobs` helps when evaluation is the
  bottleneck. Measured on this laptop, the wine demo's GA loop over the same
  96 unique subsets: **257 s serial → 183 s with `n_jobs=2`** (~ 1.4× on two
  workers — under 50% of the ideal 2×, because per-evaluation cost here is
  only five 60-tree fits, so scheduling/round-trip is a non-trivial share).
  Both runs scored the same 96 subsets and produced the identical GA result.
  On heavier evaluators (bigger models, more folds, larger data) the
  gain approaches `n_jobs`×. The GA's *result* is unchanged either way; the
  demo prints the `Evaluation:` line so you can see exactly which path your
  machine took and how many evaluations the pool performed.

## Choosing hyperparameters

| Parameter | Start here | Notes |
|---|---|---|
| `population_size` | 30–50 | Larger = more diversity, linearly more evaluations. 100+ rarely helps on small feature spaces. |
| `mutation_rate` | 0.01–0.03 | Expected flips = `rate × n_features`. If the best score freezes for many generations, raise toward 0.05–0.1; if it wanders, lower it. |
| `elite_size` | ~10% of population | Preserves progress; too high stalls diversity. `== population_size` = guaranteed stagnation. |
| `tournament_size` | 3 | 2 = soft/explore, 4–5 = strong pressure/converge fast. |
| `max_generations` | 25–100 | Generation *budget*; with `stall_generations=3` a typical run stops at the plateau in a fraction of it. Watch `history` or the verbose output for where it plateaus. |

For **small feature spaces** (n ≤ ~15), a random/brute-force sweep of a few thousand subsets is often *more* reliable than a GA — there are fewer unique subsets than the GA can explore. The GA's value shows up with ~20–30+ features where the search space is large.

**"Give me fewer features / only K."** Two built-in dials, use them together if you like:

- **`max_features=K`** — hard upper bound. The search is *repaired* into `≤ K` after every operator, so the GA can only ever return a subset of at most `K` features. The answer you get is the best ≤ K subset it found.
- **`min_features=m`** — hard lower bound, same mechanism (useful when small subsets are unscoreable or when you want at least a minimum size).
- **`penalty_per_feature=λ`** — soft preference *within* the bounds: the GA optimizes `error + λ×#features`, so two subsets at (near-)equal score are broken toward the smaller one, without forbidding anything. `best["raw_error"]` and every `top_subsets` entry keep the evaluator's own number so you can see how much you gave up.

With neither set the GA behaves as before and will happily grow subsets to squeeze out small score gains. (You can still hand-roll a penalty inside your evaluator — but the built-in one is what keeps the raw score visible.)

For quick sanity checks, `selector.history` (per-generation best error), `selector.top_subsets()` and `selector.feature_prevalence()` answer "did it plateau?", "what's actually good in this landscape?", and "which features matter, robustly?" without re-running.

## Limitations

- **Search cost, measured**: worst case *without* the cache, one generation re-scores ≈ `population + 2 × (population − elite_size) × tournament_size` subsets (≈ 320 with the defaults — mostly repeats). **With** the cache, your function is called **once per distinct subset**: at most `population_size` new subsets in generation 1, then at most `2 × (population_size − elite_size)` new per generation (two offspring per tournament pair); the demo above scored only **96** distinct subsets in 7 generations. Budget your wall time as `#unique subsets × cost-per-evaluation ÷ n_jobs` (wine demo: 257 s serial for its 96 subsets at 60-tree RF × 5-fold CV; `n_jobs` divides the evaluation portion by the parallelism you actually get).
- **The returned score is optimistic.** It was chosen *by* the GA from the same scores it was scored on, so it overstates real generalization. Validate the returned subset on data the GA never saw (the example does exactly this).
- **Selection overfitting.** If the evaluator scores on in-sample or train-fold accuracy, the GA is free to overfit that metric with large, lucky subsets. Keep the evaluator's split fixed and out of the GA's direct reach.
- **Non-determinism without a seed.** Every unseeded run is different. Two seeded runs with a stochastic evaluator (e.g. CV without fixed folds) may also differ.
- **Single-objective.** Subset-size bounds (`min_features`/`max_features`) and a per-feature penalty are built in, but there are no *other* constraints (must-include/must-exclude features) and no multi-objective trade-offs — handle those inside your evaluator.
- **No guarantee.** GA = heuristic. For n ≤ ~15 features, compare against a brute-force/random sweep before trusting a particular subset.
- **Thread-safety**: a selector instance is not safe to share across threads; `run()` also resets its state, so call it from one thread at a time.

## Testing

```bash
pip install ".[test]"          # pytest
python -m pytest tests/ -q    # 39 tests, runs in ~1 s
python ga_feature_selection_lib.py   # in-file self-test (no dependencies)
```

The suite covers: parameter validation, seed reproducibility, the evaluation
cache, per-generation `history` invariants, `top_subsets` / `feature_prevalence`,
invalid-value and exception handling, `fail_on_error`, early stopping, the
metric helpers, the `select_features_with_ga` wrapper, the subset-size
constraints (`min_features` / `max_features`) and `penalty_per_feature`, and
`n_jobs` ↔ serial
parity (with explicit observability that the worker pool really ran). Each
test's evaluator is tiny and deterministic, so the whole suite is fast and
CI-friendly.

## Project layout

| File | Purpose |
|---|---|
| `ga_feature_selection_lib.py` | The library (self-test via `python ga_feature_selection_lib.py`). |
| `example_usage.py` | End-to-end Wine Quality + RandomForest demo. |
| `tests/` | Pytest suite (`python -m pytest tests/ -q`). |
| `wine-quality-red.csv` | UCI Wine Quality (red), 1,599 rows, `;`-delimited, CC0. |
| `pyproject.toml` | Installable as `pip install .` / `.[dev]` / `.[parallel]` / `.[test]`. |
| `CHANGELOG.md` | Release history. |

## License

MIT — see [LICENSE](LICENSE).
