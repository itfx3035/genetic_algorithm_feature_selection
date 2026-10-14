"""
Wine-quality demo: GA feature subset selection with a held-out test split.

Pipeline:
- one 80/20 stratified train/test split (fixed seed)
- GA minimizes 5-fold cross-validation error on the TRAIN split only
- final comparison fits on the full train split and scores the untouched
  TEST split (so the GA never sees it)
- n_jobs scores each generation's fresh subsets on worker processes.
  The GA's own search is identical to the serial run for the same seed;
  only the wall time changes. The evaluator here is a *closure*, so the
  worker pool needs ``cloudpickle`` (``pip install cloudpickle``); without
  it, the library falls back to serial with a warning and reports exactly
  that on the final "Evaluation:" line.

NOTE: the GA call sits inside ``if __name__ == "__main__":`` - required so
that worker processes (forkserver start method) do not re-import and re-run
this script.
"""

import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from ga_feature_selection_lib import GeneticFeatureSelector, build_binary_error

RANDOM_STATE = 42


def main():
    # ------------------------------------------------------------------
    # 1. Load + inspect
    # ------------------------------------------------------------------
    df = pd.read_csv("wine-quality-red.csv", sep=";")
    df.columns = [c.strip() for c in df.columns]
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Class balance:\n{df.quality.value_counts(normalize=True).sort_index()}")
    print()

    # ------------------------------------------------------------------
    # 2. One train/test split; the GA only ever sees the train split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("quality", axis=1),
        df["quality"],
        test_size=0.2,
        stratify=df["quality"],
        random_state=RANDOM_STATE,
    )
    features = list(X_train.columns)
    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows, "
          f"features: {len(features)}")

    # ------------------------------------------------------------------
    # 3. GA objective: 5-fold CV error on the TRAIN split
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_train, y_train))

    def cv_error(selected):
        """Training-CV error for a feature subset (lower is better).

        Demonstrates the evaluator contract:
        - any subset, including the empty one, may be passed in
        - an empty subset cannot be scored: return None (marked invalid,
          counted, and can never be the returned result)
        - return a plain Python float (not np.float64)
        """
        if not selected:
            return None
        X = X_train[selected]
        err = 0.0
        for tr, te in folds:
            model = RandomForestClassifier(n_estimators=60,
                                          random_state=RANDOM_STATE)
            model.fit(X.iloc[tr], y_train.iloc[tr])
            preds = model.predict(X.iloc[te])
            err += build_binary_error(list(preds), list(y_train.iloc[te]))
        return err / len(folds)

    GA_KWARGS = dict(
        population_size=30,
        mutation_rate=0.03,
        elite_size=5,
        tournament_size=3,
        max_generations=40,
        stall_generations=3,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # 4. The GA run
    # ------------------------------------------------------------------
    import os
    n_jobs = int(os.environ.get("GA_NJOBS", "2"))  # 0 = serial

    start = time.perf_counter()
    selector = GeneticFeatureSelector(
        features, cv_error,
        n_jobs=n_jobs if n_jobs >= 2 else None,
        **GA_KWARGS,
    )
    best_subset, best_train_err = selector.run()
    elapsed = time.perf_counter() - start

    mode = (f"SERIAL fallback - {selector.parallel_unavailable_reason}"
            if (n_jobs >= 2 and selector.parallel_unavailable_reason)
            else f"n_jobs={n_jobs} (worker pool)" if n_jobs >= 2
            else "serial")
    print(f"\nGA wall time (60-tree RF, 5-fold CV, {mode}): {elapsed:.1f}s")
    print(f"Unique subsets scored: {selector.unique_evaluations} (no re-eval)")
    print(f"Invalid evaluations:   {selector.invalid_evaluation_count}")
    print(f"Best training-CV error: {best_train_err:.4f}")
    print(f"Selected features ({len(best_subset)}/{len(features)}): "
          f"{best_subset}\n")

    print("Per-generation history:")
    for row in selector.history:
        print(f"  g{row['generation']:>2} | best so far "
              f"{row['best_error']:.4f} | new subsets "
              f"{row['new_evaluations']:>2} | total {row['unique_evaluations']:>3}")

    print("\nFeature prevalence (share of top-30 subsets containing the feature):")
    for row in selector.feature_prevalence(top_k=30):
        print(f"  {row['feature']:<22s} {row['prevalence']:.2f}"
              f"   ({row['count']}/{row['of']})")

    print("\nTop 5 lowest-CV-error subsets scored:")
    for row in selector.top_subsets(5):
        print(f"  n={len(row['features'])}  err={row['error']:.4f}  "
              f"{row['features']}")

    # ------------------------------------------------------------------
    # 5. GA subset vs. all features, on the untouched TEST split
    #    (same model the GA optimized - apples to apples, fit once on
    #    the full train set)
    # ------------------------------------------------------------------
    def evaluate_on_test(cols):
        model = RandomForestClassifier(n_estimators=60,
                                      random_state=RANDOM_STATE)
        model.fit(X_train[cols], y_train)
        return model.score(X_test[cols], y_test), model

    acc_ga, _ = evaluate_on_test(best_subset)
    acc_all, model_all = evaluate_on_test(features)

    print(f"\nTest accuracy (RandomForest) | all {len(features)} features: "
          f"{acc_all:.4f} |"
          f" GA {len(best_subset)}-feature subset: {acc_ga:.4f}")
    print("Test-set accuracy is a single demonstration on one held-out "
          "split, not an independent benchmark.")

    print("\nFeature importances (RandomForest on the full train set):")
    for name, imp in sorted(zip(model_all.feature_names_in_,
                                model_all.feature_importances_),
                            key=lambda t: -t[1]):
        print(f"  {name:<22s} {imp:.4f}")


if __name__ == "__main__":
    main()
