"""
Genetic Algorithm Feature Selection
===================================

A genetic-algorithm (GA) approach to black-box feature subset selection.

Feature subsets are encoded as binary chromosomes (gene = 1 means the
feature is selected) and evolved until a user-supplied evaluation
function is minimized. The evaluation function defines a *search
landscape*, not a true model score: because no fixed train/test split is
exposed to it, the GA can overfit the selection process to the
evaluation protocol. See the README for guidance.
"""

import logging
import math
import os
import pickle
import random
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import multiprocessing as _mp
except ImportError:
    _mp = None

_PARALLEL: Dict[str, Any] = {"pool": None, "n": 0}
_ERROR_PREFIX = "__ga_eval_error__::"

__all__ = [
    "GeneticFeatureSelector",
    "select_features_with_ga",
    "build_binary_error",
    "build_regression_error",
]

__version__ = "0.4.0"

logger = logging.getLogger("ga_feature_selection")


def _ga_worker_initializer(search_dirs) -> None:
    """Inside each worker: make the GA module + the evaluator module's
    package importable (needed by pickle-by-reference workers)."""
    for d in search_dirs:
        if d is not None and d not in __import__("sys").path:
            __import__("sys").path.insert(0, d)


def _ga_worker_job(payload: Tuple[bytes, Tuple[str, ...]]) -> Tuple[Tuple[str, ...], Any]:
    """(Re)evaluate one subset inside a worker process.

    ``payload`` = (pickled evaluator, feature-name tuple); the tuple keys
    the parent's result map so worker output stays order-independent.
    Exceptions are encoded as ``_ERROR_PREFIX + repr`` so they can never
    crash the pool and are indistinguishable (in structure) from a normal
    return value to the dispatcher.
    """
    evaluator_bytes, feature_names = payload
    try:
        import cloudpickle  # type: ignore[import-not-found]
        evaluator = cloudpickle.loads(evaluator_bytes)
    except ModuleNotFoundError:
        evaluator = pickle.loads(evaluator_bytes)
    try:
        return feature_names, evaluator(list(feature_names))
    except BaseException as exc:  # noqa: BLE001 - contract boundary
        return feature_names, _ERROR_PREFIX + repr(exc)


def _ga_parallel_evaluate(
    evaluation_function: Callable[[List[str]], Optional[float]],
    missing_keys: List[Tuple[str, ...]],
    n_jobs: int,
) -> Dict[Tuple[str, ...], Any]:
    """Evaluate ``missing_keys`` on a reusable worker pool.

    The evaluator is pickled with ``cloudpickle`` when installed (closures
    by value - needed for demo scripts), else with :mod:`pickle` (top-level
    importable functions only). The pool is created once per ``n_jobs``
    value in this process and reused across generations (warm workers).

    Returns a dict keyed by feature-name tuple; a value that is a string
    starting with ``_ERROR_PREFIX`` means the evaluator raised in the
    worker. Order of results is irrelevant (keys carry the identity).
    """
    if _mp is None:
        raise RuntimeError("multiprocessing is unavailable in this interpreter")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [base_dir]
    # Also expose the directory containing the evaluator's module so a
    # pickle-by-reference pickled evaluator can be imported in workers.
    mod_name = getattr(evaluation_function, "__module__", None)
    mod_obj = sys.modules.get(mod_name) if mod_name else None
    mod_file = getattr(mod_obj, "__file__", None)
    if mod_file and os.path.isdir(os.path.dirname(os.path.abspath(mod_file))):
        search_dirs.append(os.path.dirname(os.path.abspath(mod_file)))
    try:
        import cloudpickle  # type: ignore[import-not-found]
        payload = cloudpickle.dumps(evaluation_function)
    except ModuleNotFoundError:
        payload = pickle.dumps(evaluation_function)  # may raise PicklingError
    pool = _PARALLEL.get("pool")
    n = _PARALLEL.get("n", 0)
    if pool is None or n != n_jobs:
        if pool is not None:
            try:
                pool.terminate()
            except Exception:  # noqa: BLE001
                pass
        pool = _mp.Pool(processes=n_jobs,
                        initializer=_ga_worker_initializer,
                        initargs=(search_dirs,))
        _PARALLEL["pool"] = pool
        _PARALLEL["n"] = n_jobs
    work = [(payload, k) for k in missing_keys]
    results = pool.map(_ga_worker_job, work, chunksize=8)
    return dict(results)



class GeneticFeatureSelector:

    """
    Select a subset of features by evolutionary search.

    Parameters
    ----------
    features:
        Feature names to consider. Must be non-empty with no duplicates.
    evaluation_function:
        ``fn(selected_features: List[str]) -> float``. Lower return value
        is better (e.g. ``1 - accuracy``). ``None`` and non-finite results
        (``NaN``/``inf``) mark the subset as *invalid*: it can never be
        the returned result and its individuals are excluded from parent
        selection. An exception raised by the evaluator also marks the
        subset as invalid (and increments ``invalid_evaluation_count``),
        unless ``fail_on_error=True``.

        NOTE: your function may be called with *any* subset the GA
        proposes (any length, any order matching ``features``), possibly
        the empty list - return ``None`` for subsets you cannot score. This
        is intentional so the GA can be used as a pure black-box search
        over the space of subsets.
    population_size:
        Number of individuals per generation (int, >= 2).
    mutation_rate:
        Probability of flipping each gene (0..1, inclusive).
    elite_size:
        Number of best valid individuals copied unchanged into the next
        generation (0 <= elite_size <= population_size).
    tournament_size:
        Number of valid individuals competing in each tournament
        (int, >= 2). If fewer individuals are valid, the tournament
        shrinks to the number of valid ones.
    max_generations:
        Generation budget (int, >= 1). The GA runs at most this many
        generations (fewer, if ``stall_generations`` early-stops it).
    random_state:
        Optional int seed. With the same seed and deterministic evaluation
        the run is fully reproducible (a stochastic evaluation function,
        e.g. cross-validation without a fixed split, will not be).
    stall_generations:
        Optional int. When set, the run stops early once the best error
        has not *strictly* improved for this many consecutive generations
        ("best" means best-so-far; the very first valid result always
        counts as an improvement). ``None`` (default) disables early
        stopping.
    fail_on_error:
        When True, exceptions raised by ``evaluation_function`` propagate
        out of :meth:`run` instead of marking that subset invalid.
    n_jobs:
        Optional int >= 2. When set, feature subsets not yet in the
        evaluation cache (the ones each generation actually needs) are
        scored by worker processes; all others are reused from the cache.
        The GA itself (RNG draw order, evolution) is unaffected, so a
        seeded run with a deterministic evaluator gives the same result
        for any ``n_jobs``. The evaluation function is pickled -
        top-level functions and classes always work; closures and lambdas
        work when ``cloudpickle`` is installed (``pip install
        cloudpickle``, used by the demo). If pickling fails, the run
        *falls back to serial* with a warning instead of crashing.
    min_features:
        Optional int >= 0. The GA will only ever *propose* (and therefore
        score/return) subsets of at least this many features. Enforced by
        repairing chromosomes back into bounds after each operator, so the
        entire search stays inside the constraint. ``None`` (default) = no
        lower bound.
    max_features:
        Optional int >= 1. Upper bound on subset size, same enforcement as
        ``min_features``. With ``max_features=K`` the GA can return at most
        ``K`` features - the built-in way to ask "find me the best K
        features". ``None`` (default) = no upper bound.
    penalty_per_feature:
        Optional non-negative float. When set, the GA optimizes
        ``error + penalty_per_feature * len(subset)`` (a simple Occam-razor
        term that trades a little score against subset size), while the
        evaluator's own number is kept separately and reported as
        ``best["raw_error"]`` (and in every ``top_subsets`` entry). This is
        a *soft* preference; use ``max_features`` for a *hard* limit.
        ``None`` (default) = no penalty.

    Operators
    ---------
    - Selection: tournament over *valid* individuals only.
    - Crossover: uniform, producing two children (the second child is the
      complement draw of the first).
    - Mutation: independent bit flip on each gene.
    - Elitism: top ``elite_size`` valid individuals carried over unchanged.

    State after :meth:`run`
    -----------------------
    - ``history``  : one dict per completed generation.
    - ``best``     : ``{"features", "error", "raw_error", "generation"}``
      or ``None`` (``error`` is what the GA minimized - with
      ``penalty_per_feature`` set, that includes the penalty;
      ``raw_error`` is the evaluator's own number for the same subset).
    - ``unique_evaluations`` (property): number of distinct subsets scored.
    - ``invalid_evaluation_count``: distinct subsets that scored invalid
      (returned None/non-finite or raised).
    - ``last_evaluation_error``: the most recent exception raised by the
      evaluator, if any.
    """

    def __init__(
        self,
        features: List[str],
        evaluation_function: Callable[[List[str]], Optional[float]],
        population_size: int = 50,
        mutation_rate: float = 0.01,
        elite_size: int = 5,
        tournament_size: int = 3,
        max_generations: int = 100,
        random_state: Optional[int] = None,
        stall_generations: Optional[int] = None,
        fail_on_error: bool = False,
        min_features: Optional[int] = None,
        max_features: Optional[int] = None,
        penalty_per_feature: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ):
        if not features:
            raise ValueError("features must be a non-empty list")
        if len(set(features)) != len(features):
            raise ValueError("features must be unique")
        if not isinstance(population_size, int) or population_size < 2:
            raise ValueError("population_size must be an int >= 2")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if elite_size < 0 or elite_size > population_size:
            raise ValueError(
                "elite_size must satisfy 0 <= elite_size <= population_size"
            )
        if not isinstance(tournament_size, int) or tournament_size < 2:
            raise ValueError("tournament_size must be an int >= 2")
        if not isinstance(max_generations, int) or max_generations < 1:
            raise ValueError("max_generations must be an int >= 1")
        if stall_generations is not None and (
            not isinstance(stall_generations, int) or stall_generations < 1
        ):
            raise ValueError("stall_generations must be a positive int or None")
        if not isinstance(fail_on_error, bool):
            raise ValueError("fail_on_error must be a bool")
        if n_jobs is not None and (
            not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 2
        ):
            raise ValueError("n_jobs must be None or an int >= 2")
        if min_features is not None and (
            not isinstance(min_features, int) or isinstance(min_features, bool)
            or min_features < 0 or min_features > len(features)
        ):
            raise ValueError(
                "min_features must be None or an int in [0, "
                f"{len(features)}]"
            )
        if max_features is not None and (
            not isinstance(max_features, int) or isinstance(max_features, bool)
            or max_features < 1
        ):
            raise ValueError("max_features must be None or an int >= 1")
        if min_features is not None and max_features is not None and \
                min_features > max_features:
            raise ValueError("min_features must be <= max_features")
        if penalty_per_feature is not None and (
            not isinstance(penalty_per_feature, (int, float))
            or isinstance(penalty_per_feature, bool)
            or penalty_per_feature < 0
        ):
            raise ValueError(
                "penalty_per_feature must be None or a number >= 0"
            )

        self.features = list(features)
        self.evaluation_function = evaluation_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.random_state = random_state
        self.stall_generations = stall_generations
        self.fail_on_error = fail_on_error
        self.min_features: Optional[int] = min_features
        self.max_features: Optional[int] = max_features
        self.penalty_per_feature: Optional[float] = (
            float(penalty_per_feature)
            if penalty_per_feature is not None else None
        )
        self.n_jobs: Optional[int] = n_jobs
        self._rng = random.Random(random_state)

        # Per-run state, (re)initialised by run().
        self._eval_cache: Dict[Tuple[str, ...], Optional[float]] = {}
        self.history: List[Dict[str, Any]] = []
        self.best: Optional[Dict[str, Any]] = None
        self.invalid_evaluation_count = 0
        self.last_evaluation_error: Optional[BaseException] = None
        self._first_eval_error: Optional[BaseException] = None
        # Parallel-evaluation observability (populated by run()):
        self.parallel_subsets_scored = 0        # scored by worker processes
        self.parallel_unavailable_reason = None # why (if any) it wasn't used
        self._verbose = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # compact, mirrors the constructor
        parts = [
            f"GeneticFeatureSelector(features={len(self.features)}, "
            f"population_size={self.population_size}, "
            f"mutation_rate={self.mutation_rate}, elite_size={self.elite_size}, "
            f"tournament_size={self.tournament_size}, "
            f"max_generations={self.max_generations}",
        ]
        if self.min_features is not None:
            parts.append(f"min_features={self.min_features}")
        if self.max_features is not None:
            parts.append(f"max_features={self.max_features}")
        if self.penalty_per_feature is not None:
            parts.append(f"penalty_per_feature={self.penalty_per_feature!r}")
        parts.append(f"random_state={self.random_state}, n_jobs={self.n_jobs}")
        return "(" + ", ".join(parts) + ")"

    @property
    def unique_evaluations(self) -> int:
        """Number of distinct subsets scored by the evaluator so far."""
        return len(self._eval_cache)

    def _say(self, line: str) -> None:
        """Emit a progress line: to stdout when verbose, always at DEBUG."""
        logger.debug("%s", line)
        if self._verbose:
            print(line)

    def _create_individual(self) -> List[int]:
        """Create a random chromosome (always within the size bounds)."""
        return self._enforce_bounds(
            [self._rng.randint(0, 1) for _ in self.features]
        )

    def _create_population(self) -> List[List[int]]:
        """Create the initial population."""
        return [self._create_individual() for _ in range(self.population_size)]

    @staticmethod
    def _coerce_error(raw: Any) -> Optional[float]:
        """Coerce an evaluator result to a finite float, or ``None`` if it
        does not represent a valid error (None, non-numeric, NaN, ±inf).
        """
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    def _evaluate(self, key: Tuple[str, ...], selected: List[str]) -> Optional[float]:
        """Evaluate a subset via the user function, caching the result.

        Invalid results (see class docstring) are cached as ``None``.
        """
        if key in self._eval_cache:
            return self._eval_cache[key]
        try:
            raw = self.evaluation_function(selected)
        except Exception as exc:  # noqa: BLE001 - contract boundary
            self.invalid_evaluation_count += 1
            self.last_evaluation_error = exc
            if self._first_eval_error is None:
                self._first_eval_error = exc
            logger.debug(
                "Evaluating subset %s raised %s: %s",
                selected, type(exc).__name__, exc,
            )
            if self.fail_on_error:
                raise
            self._eval_cache[key] = None
            return None

        result = self._coerce_error(raw)
        if result is None:
            self.invalid_evaluation_count += 1
        self._eval_cache[key] = result
        return result

    def _tournament(self, valid_idx: List[int],
                    evals: List[Optional[float]]) -> int:
        """Return the population index of the tournament winner."""
        pool = self._rng.sample(valid_idx, min(self.tournament_size, len(valid_idx)))
        return min(pool, key=lambda i: evals[i])

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Uniform crossover producing two children. The second child is the
        complement draw of the first (wherever child1 took a gene from
        parent1, child2 takes it from parent2)."""
        n = len(parent1)
        flips = [self._rng.random() < 0.5 for _ in range(n)]
        child1 = [parent1[i] if flips[i] else parent2[i] for i in range(n)]
        child2 = [parent2[i] if flips[i] else parent1[i] for i in range(n)]
        return child1, child2

    def _mutate(self, individual: List[int]) -> List[int]:
        """Bit-flip mutation on a copy of the individual."""
        if self.mutation_rate <= 0.0:
            return list(individual)
        rng = self._rng
        return [
            (1 - g) if (rng.random() < self.mutation_rate) else g
            for g in individual
        ]

    def _enforce_bounds(self, individual: List[int]) -> List[int]:
        """Repair a chromosome so its number of 1s lies within
        ``[min_features, max_features]`` (each bound optional).

        Flips as few random genes as needed, chosen with ``self._rng``, and
        only draws from the RNG when the chromosome is actually outside the
        bounds - so with both bounds unset the RNG sequence (and hence the
        whole run) is exactly what earlier versions did.
        """
        n_on = sum(individual)
        if self.min_features is not None and n_on < self.min_features:
            zeros = [i for i, g in enumerate(individual) if g == 0]
            self._rng.shuffle(zeros)
            for i in zeros[: self.min_features - n_on]:
                individual[i] = 1
        elif self.max_features is not None and n_on > self.max_features:
            ones = [i for i, g in enumerate(individual) if g == 1]
            self._rng.shuffle(ones)
            for i in ones[: n_on - self.max_features]:
                individual[i] = 0
        return individual

    def _effective_error(self, key: Tuple[str, ...],
                         raw: Optional[float]) -> Optional[float]:
        """The GA's objective for a subset: the raw evaluator error plus
        ``penalty_per_feature`` x size (``None`` when the raw was invalid).
        """
        if raw is None or self.penalty_per_feature is None:
            return raw
        return raw + self.penalty_per_feature * len(key)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> Tuple[List[str], float]:
        """Run the GA.

        Parameters
        ----------
        verbose:
            Print progress to stdout. Either way, progress is also logged
            at DEBUG and key events at INFO/WARNING on the
            ``ga_feature_selection`` logger, and a final :class:`UserWarning`
            is emitted if the evaluator raised exceptions.

        Returns
        -------
        (best_features, best_error)
            The best valid feature subset found (feature order, as given
            in ``features``) and its error - the GA's objective, i.e. your
            evaluator's return value plus ``penalty_per_feature * len``
            (with no penalty set, exactly the evaluator's value). The
            evaluator's own number for that subset is
            ``self.best["raw_error"]``. If no subset is ever valid,
            returns ``([], 0.0)``.

        With ``n_jobs >= 2``, evaluations not present in the cache are
        dispatched to ``n_jobs`` worker processes in their population order
        (the pool's results are applied back in dispatch order, so the GA's
        RNG - and therefore the whole run - stays reproducible).
        The evaluation function is pickled, so it must be picklable
        (top-level functions or classes; not lambdas or closures).

        Populates ``self.history``, ``self.best``, ``self.invalid_evaluation_count``
        and the ``unique_evaluations`` cache.
        """
        population = self._create_population()
        n_features = len(self.features)
        self._verbose = verbose

        # Reset per-run state so run() can be called again.
        self._eval_cache = {}
        self.history = []
        self.best = None
        self.invalid_evaluation_count = 0
        self.last_evaluation_error = None
        self._first_eval_error = None
        self.parallel_subsets_scored = 0
        self.parallel_unavailable_reason = None
        self._parallel_disabled = False

        def fmt(key: Tuple[str, ...]) -> str:
            names = list(key)
            return (f"({len(names)} features) " +
                    ", ".join(names[:6]) + ("..." if len(names) > 6 else ""))

        self._say("Starting GA feature selection")
        self._say(f"  Features:          {n_features}")
        self._say(f"  Population size:   {self.population_size}")
        self._say(f"  Elite size:        {self.elite_size}")
        self._say(f"  Mutation rate:     {self.mutation_rate}")
        self._say(f"  Tournament size:   {self.tournament_size}")
        self._say(f"  Max generations:   {self.max_generations}")
        if self.min_features is not None and self.min_features > 0:
            self._say(f"  Min features:    {self.min_features}")
        if self.max_features is not None:
            self._say(f"  Max features:    {self.max_features}")
        if self.penalty_per_feature is not None:
            self._say(f"  Penalty/feature: {self.penalty_per_feature!r}")
        if self.stall_generations is not None:
            self._say(f"  Stop on stall:     {self.stall_generations} generations "
                      f"without improvement")
        if self.random_state is not None:
            self._say(f"  Random state:      {self.random_state}")
        if self.n_jobs is not None:
            self._say(f"  n_jobs:            {self.n_jobs}")
        self._say("-" * 50)
        logger.info("GA run started: %r", self)

        best_key: Optional[Tuple[str, ...]] = None
        best_error = math.inf
        best_generation = 0
        stall = 0

        for g in range(self.max_generations):
            generation = g + 1
            unique_before = len(self._eval_cache)
            invalid_before = self.invalid_evaluation_count

            # --- Evaluate this generation (cached) ------------------------
            keys: List[Tuple[str, ...]] = []
            evals: List[Optional[float]] = []
            for ind in population:
                keys.append(tuple(
                    self.features[i] for i in range(n_features) if ind[i]
                ))

            # Fill the cache: with n_jobs the *missing* keys go to the
            # worker pool (worker exceptions arrive as _ERROR_PREFIX
            # strings and are decoded here; the GA's RNG never sees the
            # pool), otherwise everything is evaluated serially. Cached
            # keys are reused from _eval_cache either way.
            if self.n_jobs is not None and not self._parallel_disabled:
                missing = [k for k in keys if k not in self._eval_cache]
                if missing:
                    try:
                        remote = _ga_parallel_evaluate(
                            self.evaluation_function, missing, self.n_jobs)
                    except Exception as exc:  # infra/pickling failure
                        # An *infrastructure* failure (pickling the
                        # evaluator, or the worker pool) - NOT an exception
                        # your evaluator raised, so it is not recorded in
                        # last_evaluation_error. Try again next generation
                        # only if the evaluator became picklable (impossible
                        # mid-run), so disable permanently + serial.
                        reason = f"{type(exc).__name__}: {exc}".replace(
                            "\n", " ")[:200]
                        self._parallel_disabled = True
                        self.parallel_unavailable_reason = \
                            f"n_jobs={self.n_jobs} failed: {reason}"
                        self._say(f"WARNING: {self.parallel_unavailable_reason}\n"
                                  f"         falling back to serial evaluation "
                                  f"for the rest of this run")
                        logger.warning("n_jobs unavailable; serial fallback: %s",
                                       self.parallel_unavailable_reason)
                        for key in missing:
                            self._evaluate(key, list(key))
                    else:
                        self.parallel_subsets_scored += len(missing)
                        for key in missing:
                            raw = remote.get(key,
                                             _ERROR_PREFIX + "key lost (bug)")
                            if isinstance(raw, str) and \
                                    raw.startswith(_ERROR_PREFIX):
                                self.last_evaluation_error = RuntimeError(
                                    raw[len(_ERROR_PREFIX):].strip())
                                if self._first_eval_error is None:
                                    self._first_eval_error = (
                                        self.last_evaluation_error)
                                if self.fail_on_error:
                                    raise RuntimeError(
                                        "evaluator raised in worker; "
                                        + raw[len(_ERROR_PREFIX):].strip()
                                    ) from None
                                self._eval_cache[key] = None
                                self.invalid_evaluation_count += 1
                            else:
                                self._eval_cache[key] = self._coerce_error(raw)
                                if self._eval_cache[key] is None:
                                    self.invalid_evaluation_count += 1
            else:
                for key in keys:
                    self._evaluate(key, list(key))
            evals = [self._eval_cache[k] for k in keys]

            # The objective the GA optimizes: raw error + penalty*size.
            # Identical to `evals` when no penalty is set. Selection, elitism
            # and the best-tracking all act on this, not on the raw number.
            effs = [self._effective_error(k, e) for k, e in zip(keys, evals)]

            # --- Track best so far ------------------------------------------
            improved = False
            for key, error in zip(keys, effs):
                if error is not None:
                    if error < best_error:
                        best_key, best_error, best_generation = \
                            key, error, generation
                        improved = True
            if improved:
                stall = 0
                self.best = {
                    "features": list(best_key),
                    "error": best_error,
                    "raw_error": self._eval_cache[best_key],
                    "generation": best_generation,
                }
            else:
                stall += 1
            if verbose and (
                (generation % 10 == 0)
                or generation == self.max_generations
                or self.stall_generations is not None
            ):
                if best_key is not None:
                    self._say(f"Generation {generation:>4} | "
                              f"best error: {best_error:.4f} | "
                              f"best: {fmt(best_key)}")
                else:
                    self._say(f"Generation {generation:>4} | no valid subset yet")

            # --- Evolve ------------------------------------------------------
            valid_idx = [i for i, e in enumerate(effs) if e is not None]
            if valid_idx:
                new_population: List[List[int]] = [
                    list(population[i])
                    for i in sorted(valid_idx, key=lambda i: effs[i])
                        [:self.elite_size]
                ]
                while len(new_population) < self.population_size:
                    p1 = self._tournament(valid_idx, effs)
                    p2 = self._tournament(valid_idx, effs)
                    c1, c2 = self._crossover(population[p1], population[p2])
                    new_population.append(
                        self._enforce_bounds(self._mutate(c1)))
                    if len(new_population) < self.population_size:
                        new_population.append(
                            self._enforce_bounds(self._mutate(c2)))
            else:
                # Nothing valid this generation: restart fresh.
                self._say(f"Generation {generation}: all individuals invalid, "
                          f"reinitializing population")
                new_population = self._create_population()

            population = new_population

            self.history.append({
                "generation": generation,
                "improved": improved,
                "best_error": best_error if best_key is not None else None,
                "best_subset": list(best_key) if best_key is not None else None,
                "new_evaluations": len(self._eval_cache) - unique_before,
                "unique_evaluations": len(self._eval_cache),
                "invalid_evaluations":
                    self.invalid_evaluation_count - invalid_before,
            })

            if self.stall_generations is not None and \
                    stall >= self.stall_generations and \
                    generation < self.max_generations:
                self._say(f"Early stop: no improvement for {stall} consecutive "
                          f"generations; stopping at generation {generation} "
                          f"of {self.max_generations}")
                break

        # --- Report ----------------------------------------------------------
        self._say("-" * 50)
        self._say(f"Completed after {len(self.history)} generation(s); "
                  f"{self.unique_evaluations} unique subsets evaluated, "
                  f"{self.invalid_evaluation_count} invalid")
        if self.n_jobs is not None:
            if self.parallel_unavailable_reason is not None:
                self._say("Evaluation: SERIAL only (worker pool unused - "
                          f"{self.parallel_unavailable_reason})")
            else:
                self._say(f"Evaluation: {self.parallel_subsets_scored} "
                          f"subset(s) scored by {self.n_jobs} worker process(es)")
        logger.info("GA run finished: %d unique subsets, %d invalid, best=%r",
                    self.unique_evaluations, self.invalid_evaluation_count,
                    self.best)
        if self.best is not None:
            self._say(f"Best solution (found at generation {self.best['generation']}):")
            self._say(f"  ({len(self.best['features'])} features) "
                      f"error: {self.best['error']:.6f}")
            if self.penalty_per_feature is not None:
                self._say(
                    f"  (GA objective = raw error + {self.penalty_per_feature}"
                    f" per feature; raw error: {self.best['raw_error']:.6f})")
            if len(self.best["features"]) <= 12:
                self._say(f"  Selected: {self.best['features']}")
        else:
            self._say("No valid feature subset was found "
                      "(evaluation returned None/non-finite, or raised, for "
                      "every subset).")

        if self._first_eval_error is not None:
            warnings.warn(
                f"{self.invalid_evaluation_count} feature subset(s) scored "
                f"invalid; the evaluator raised at least once (first example: "
                f"{type(self._first_eval_error).__name__}: "
                f"{self._first_eval_error}). Check its signature and edge "
                f"cases; set fail_on_error=True to make such errors fatal.",
                UserWarning,
            )

        # Release the worker pool if we used one; a later run() recreates it
        # (so a selector instance can be reused across serial+parallel runs).
        if _PARALLEL.get("pool") is not None:
            try:
                _PARALLEL["pool"].terminate()
            except Exception:  # noqa: BLE001
                pass
            _PARALLEL["pool"] = None
            _PARALLEL["n"] = 0

        if self.best is None:
            return [], 0.0
        return list(self.best["features"]), self.best["error"]

    # ------------------------------------------------------------------
    # Post-run analysis
    # ------------------------------------------------------------------

    def _valid_cached(self) -> List[Tuple[float, float, Tuple[str, ...]]]:
        """(effective error, raw error, feature-tuple) for all valid cached
        subsets, sorted by the GA's objective (raw error plus any penalty),
        then feature names (deterministic)."""
        valid = [
            (self._effective_error(feats, score), score, feats)
            for feats, score in self._eval_cache.items()
            if score is not None
        ]
        valid.sort(key=lambda t: (t[0], t[2]))
        return valid

    def top_subsets(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return the ``k`` best-valid subsets from the evaluation cache.

        Only meaningful after :meth:`run`. Each entry is
        ``{"features": list, "error": float, "raw_error": float}``, best
        first. ``error`` is what the GA minimized (evaluator error plus
        ``penalty_per_feature`` x size); ``raw_error`` is the evaluator's
        own number for that subset. If fewer than ``k`` valid subsets were
        scored, all of them are returned.
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive int")
        return [
            {"features": list(feats), "error": eff, "raw_error": raw}
            for eff, raw, feats in self._valid_cached()[:k]
        ]

    def feature_prevalence(self, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Per-feature inclusion frequency among the best cached subsets.

        Considered set: the ``top_k`` lowest-error valid subsets in the
        evaluation cache (all valid subsets when ``top_k`` is ``None`` or
        larger than the number of valid subsets). Only meaningful after
        :meth:`run`.

        Returns a list of dicts, one per feature, sorted by prevalence
        descending: ``{"feature", "count", "of", "prevalence"}`` where
        ``count`` is how many of the ``of`` considered subsets contain it.

        A single GA "best" is one lucky chromosome; which features keep
        appearing in good solutions is a much sturdier importance signal,
        e.g. for deciding how many features a model needs.
        """
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k must be a positive int or None")
        valid = self._valid_cached()
        if not valid:
            return []
        k = len(valid) if top_k is None else min(top_k, len(valid))
        counts = {f: 0 for f in self.features}
        for _eff, _raw, feats in valid[:k]:
            for f in feats:
                counts[f] += 1
        return [
            {
                "feature": f,
                "count": counts[f],
                "of": k,
                "prevalence": counts[f] / k,
            }
            for f in sorted(counts, key=lambda f: (-counts[f], f))
        ]


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------

def build_binary_error(y_pred: List, y_true: List) -> float:
    """Binary classification error: ``1 - accuracy``. Lower is better."""
    if not y_pred or not y_true or len(y_pred) != len(y_true):
        return 1.0
    correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    return 1.0 - correct / len(y_pred)


def build_regression_error(y_true: List, y_pred: List,
                           normalize_by_range: bool = True) -> float:
    """Regression error: MSE, optionally divided by the range of ``y_true``
    so the score is scale-invariant and comparable across datasets.
    Lower is better.
    """
    if not y_pred or not y_true or len(y_pred) != len(y_true):
        return float("inf")
    n = len(y_pred)
    mse = sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / n
    if normalize_by_range:
        span = max(y_true) - min(y_true)
        if span > 0:
            mse = mse / span
    return mse


# ----------------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------------

def select_features_with_ga(
    features: List[str],
    evaluation_function: Callable[[List[str]], Optional[float]],
    population_size: int = 50,
    mutation_rate: float = 0.01,
    elite_size: int = 5,
    tournament_size: int = 3,
    max_generations: int = 100,
    random_state: Optional[int] = None,
    stall_generations: Optional[int] = None,
    fail_on_error: bool = False,
    min_features: Optional[int] = None,
    max_features: Optional[int] = None,
    penalty_per_feature: Optional[float] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[str], float]:
    """One-call wrapper around :class:`GeneticFeatureSelector`.

    Same parameters as the class (plus ``verbose``). Use the class itself
    to access ``history``/``best``/``feature_prevalence()`` after the run.

    Returns ``(best_feature_subset, best_error)``; ``( [], 0.0)`` if the
    evaluation function never produces a valid score.
    """
    selector = GeneticFeatureSelector(
        features=features,
        evaluation_function=evaluation_function,
        population_size=population_size,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        tournament_size=tournament_size,
        max_generations=max_generations,
        random_state=random_state,
        stall_generations=stall_generations,
        fail_on_error=fail_on_error,
        min_features=min_features,
        max_features=max_features,
        penalty_per_feature=penalty_per_feature,
        n_jobs=n_jobs,
    )
    return selector.run(verbose=verbose)


# ----------------------------------------------------------------------
# Self-test (no external dependencies)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("GA Feature Selection - self-test")
    print("=" * 60)

    def dummy_metric(feats):
        """Deterministic fake landscape: minimal error at 8 features,
        degrading as the count drifts from 8."""
        return max(0.1, 0.2 * abs(len(feats) - 8)) / 2

    features = [f"f{i:02d}" for i in range(20)]
    kwargs = dict(population_size=30, max_generations=20)

    # 1. Reproducibility -------------------------------------------------
    r1 = select_features_with_ga(features, dummy_metric,
                                 random_state=42, verbose=False, **kwargs)
    r2 = select_features_with_ga(features, dummy_metric,
                                 random_state=42, verbose=False, **kwargs)
    print(f"repro: seed=42 run1 n={len(r1[0])} err={r1[1]:.4f} | "
          f"run2 n={len(r2[0])} err={r2[1]:.4f}")
    assert r1 == r2, "same seed must give identical results"

    # 2. Cache + history + best ------------------------------------------
    sel = GeneticFeatureSelector(features, dummy_metric,
                                population_size=30, max_generations=20,
                                random_state=99)
    r3, _ = sel.run(verbose=False)
    print(f"cache: full 20-gen run scored {sel.unique_evaluations} unique "
          f"subsets (would be ~{30*20} evals without the cache)")
    assert sel.unique_evaluations < 30 * 20

    assert len(sel.history) == 20
    assert sel.best["features"] == r3
    assert sel.best["error"] == max(0.1, 0.2 * abs(len(r3) - 8)) / 2
    assert all(set(h) >= {"generation", "improved", "best_error",
                          "best_subset", "new_evaluations",
                          "unique_evaluations", "invalid_evaluations"}
               for h in sel.history)
    assert sel.history[-1]["unique_evaluations"] == sel.unique_evaluations
    print(f"history: {len(sel.history)} entries; "
          f"best at generation {sel.best['generation']} "
          f"error {sel.best['error']:.4f}")

    # 3. top_subsets / feature_prevalence ---------------------------------
    tops = sel.top_subsets(5)
    assert len(tops) == 5 and tops[0]["error"] <= tops[-1]["error"]
    prev = sel.feature_prevalence(top_k=len(tops))
    assert prev and all(0.0 <= row["prevalence"] <= 1.0 for row in prev)
    assert all(row["of"] == len(tops) for row in prev)
    order = [row["prevalence"] for row in prev]
    assert order == sorted(order, reverse=True)
    print(f"prevalence (top {len(tops)} subsets): "
          f"{[(r['feature'], r['prevalence']) for r in prev[:3]]}")

    # 4. Invalid handling: None / NaN / exceptions -------------------------
    def bad_sometimes(feats):
        if not feats:
            return None
        if len(feats) % 2:
            return float("nan")
        if len(feats) == 4:
            raise ValueError("model cannot fit exactly 4 features")
        return 0.5 * len(feats)

    import warnings

    with warnings.catch_warnings():
        sel2 = GeneticFeatureSelector(
            [f"g{i}" for i in range(6)], bad_sometimes,
            population_size=20, max_generations=15, random_state=7)
        sub2, err2 = sel2.run(verbose=False)
    print(f"invalid: returned n={len(sub2)} err={err2:.2f} | "
          f"invalid_evaluation_count={sel2.invalid_evaluation_count} "
          f"last_error={sel2.last_evaluation_error!r}")
    assert len(sub2) >= 2
    assert isinstance(sel2.last_evaluation_error, ValueError)
    assert sel2.invalid_evaluation_count > 0
    assert len(sel2.history) == 15   # history exists even with invalids

    # 5. fail_on_error propagates ------------------------------------------
    sel3 = GeneticFeatureSelector(
        [f"g{i}" for i in range(6)], bad_sometimes,
        population_size=12, max_generations=60,
        random_state=7, fail_on_error=True)
    try:
        sel3.run(verbose=False)
    except ValueError as e:
        print(f"fail_on_error: raised as expected: {e!r}")
    else:
        raise AssertionError("fail_on_error=True should re-raise")

    # 6. Early stopping ------------------------------------------------------
    def flat_metric(feats):
        return 0.1  # no subset beats any other

    sel4 = GeneticFeatureSelector(
        [f"h{i}" for i in range(8)], flat_metric,
        population_size=10, max_generations=50, random_state=1,
        stall_generations=3)
    _, err4 = sel4.run(verbose=False)
    print(f"early stop: {len(sel4.history)} of 50 generations used "
          f"(stall_generations=3)")
    assert len(sel4.history) == 4          # gen 1 improved, then 3 stall gens
    assert sel4.history[-1]["generation"] == 4
    assert err4 == 0.1

    # 7. All-invalid landscape ---------------------------------------------
    sel5 = GeneticFeatureSelector(
        [f"z{i}" for i in range(4)], lambda feats: None,
        population_size=6, max_generations=3, random_state=2,
        stall_generations=3)
    sub5, err5 = sel5.run(verbose=False)
    assert sub5 == [] and err5 == 0.0
    print("all-invalid: returned ([], 0.0) as documented")

    # 8. n_jobs parity: when the worker pool actually runs, results must
    # match the serial run for a deterministic evaluator. (The pool may be
    # unusable in some sandboxes - detect that explicitly, never silently.)
    if _mp is None:
        print("n_jobs: skipped (multiprocessing unavailable)")
    else:
        sel_p = GeneticFeatureSelector(
            features, dummy_metric,
            population_size=30, max_generations=10,
            random_state=42, n_jobs=2)
        sub_p, err_p = sel_p.run(verbose=False)
        sel_s = GeneticFeatureSelector(
            features, dummy_metric,
            population_size=30, max_generations=10,
            random_state=42)
        sub_s, err_s = sel_s.run(verbose=False)
        print(f"n_jobs parity: serial n={len(sub_s)} err={err_s:.4f} | "
              f"parallel n={len(sub_p)} err={err_p:.4f}")
        assert sub_p == sub_s and err_p == err_s, \
            "n_jobs=2 must match the serial run (deterministic evaluator)"
        assert sel_p.unique_evaluations == sel_s.unique_evaluations
        if sel_p.parallel_unavailable_reason is not None:
            print(f"  ... but the pool never ran "
                  f"({sel_p.parallel_unavailable_reason[:120]}).")
            print("  PARITY CHECK IS DEGENERATE (serial vs serial); "
                  "skipped in this environment.")
        else:
            assert sel_p.parallel_subsets_scored > 0, \
                "pool ran but scored zero subsets?"
            print(f"  parallel pool genuinely scored "
                  f"{sel_p.parallel_subsets_scored} subset(s)")

    # 9. Subset-size constraints: max_features / min_features -------------
    # Hard bound: the GA must never *return* a subset outside [min, max],
    # and the bound should actually be *active* (the unconstrained optimum
    # for greedy_metric is the full 20-feature set, so max_features=6 must
    # cap it).
    def greedy_metric(feats):
        # monotonically decreasing in size: best = all 20 features
        return -0.01 * len(feats)

    sel6 = GeneticFeatureSelector(
        features, greedy_metric,
        population_size=30, mutation_rate=0.05, elite_size=5,
        tournament_size=3, max_generations=15, random_state=11,
        max_features=6)
    sub6, err6 = sel6.run(verbose=False)
    assert len(sub6) <= 6, f"max_features violated: {len(sub6)} > 6"
    assert len(sub6) > 0
    print(f"max_features=6: returned {len(sub6)}/20 (unconstrained greedy "
          f"optimum is 20) err={err6:.4f}")
    assert len(sub6) == 6

    sel7 = GeneticFeatureSelector(
        features, greedy_metric,
        population_size=30, mutation_rate=0.05, max_generations=15,
        random_state=11, min_features=12, max_features=14)
    sub7, _ = sel7.run(verbose=False)
    assert 12 <= len(sub7) <= 14, \
        f"min/max violated: {len(sub7)} not in [12,14]"
    print(f"min_features=12, max_features=14: returned {len(sub7)}/20")

    # Validation: bad bounds are rejected at construction time.
    for bad_kwargs in (dict(min_features=-1), dict(max_features=0),
                       dict(min_features=20, max_features=5)):
        try:
            GeneticFeatureSelector(features, dummy_metric, **bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"expected ValueError for {bad_kwargs}")
    print("constraint validation: bad min/max rejected correctly")

    # 10. penalty_per_feature -----------------------------------------
    # The GA optimizes error + penalty*size, while reporting the raw number
    # separately. greedy_metric wants MORE features (-0.01 per feature, so
    # unconstrained optimum = all 20). penalty=0.02 makes objective(n) =
    # -0.01n + 0.02n = +0.01n, i.e. FEWER features win; min_features=1
    # rules out the empty set, so the unique optimum is exactly 1 feature:
    # objective 0.01, raw error -0.01.
    sel8 = GeneticFeatureSelector(
        features, greedy_metric,
        population_size=30, mutation_rate=0.05, max_generations=15,
        random_state=17, min_features=1, penalty_per_feature=0.02)
    sub8, err8 = sel8.run(verbose=False)
    assert sel8.best is not None
    assert len(sub8) == 1, f"expected the 1-feature optimum, got {len(sub8)}"
    assert abs(err8 - 0.01) < 1e-12          # -0.01 + 0.02
    assert abs(sel8.best["raw_error"] + 0.01) < 1e-12
    print(f"penalty=0.02 on a greedy landscape: returned 1 feature "
          f"(objective {err8:+.4f}, raw error {sel8.best['raw_error']:+.4f})")
    tops8 = sel8.top_subsets(3)
    # every entry's objective is its raw error plus 0.02 per feature
    assert all(abs((t["error"] - t["raw_error"]) - 0.02 * len(t["features"]))
               < 1e-12 for t in tops8)
    assert tops8[0]["error"] <= tops8[-1]["error"]
    print(f"top_subsets carry raw_error: e.g. "
          f"{[round(t['error'], 4) for t in tops8[:2]]}")

    print("\nAll self-tests passed.")
