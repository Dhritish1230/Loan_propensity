import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(r"C:\Users\m\Desktop\PROJECT INTERNSHIP")
MT_DIR = ROOT / "multi_month_training"
OUT_DIR = MT_DIR / "experiment_outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MT_DIR) not in sys.path:
    sys.path.insert(0, str(MT_DIR))

from evaluate_model_slice import FEATURE_SETS, MONTHS, RAW_CALL_FEATURES, load_month  # noqa: E402
import train_t0_call_targeting_model as t0_stage  # noqa: E402
from aggregate_call_sequence_features import SEQUENCE_FEATURES  # noqa: E402


T0_TARGETS = {
    "answer_any": lambda df: (df["answered_calls"] > 0).astype(int),
    "engaged_5s": lambda df: ((df["answered_calls"] > 0) & (df["avg_call_duration"] >= 5)).astype(int),
    "engaged_10s": lambda df: ((df["answered_calls"] > 0) & (df["avg_call_duration"] >= 10)).astype(int),
    "engaged_15s": lambda df: ((df["answered_calls"] > 0) & (df["avg_call_duration"] >= 15)).astype(int),
}

T1_BASE_CAT = FEATURE_SETS["t1"]["categorical"]
T1_BASE_NUM = FEATURE_SETS["t1"]["numeric"]
T1_EXTRA_NUM = [
    "total_calls_log1p",
    "answered_calls_log1p",
    "avg_call_duration_log1p",
    "raw_total_calls_log1p",
    "raw_total_duration_log1p",
    "unanswered_calls",
    "unanswered_rate",
    "duration_per_answered_call",
    "raw_duration_per_answered_call",
    "has_any_call",
    "has_answered_call",
    "has_10s_avg_call",
    "has_30s_avg_call",
    "has_60s_avg_call",
    "raw_answered_x_duration",
    "raw_outbound_x_answered_share",
] + SEQUENCE_FEATURES

SEQUENCE_CACHE_DIR = MT_DIR / "sequence_call_cache"


def metric_block(y_true: pd.Series, pred_prob: np.ndarray) -> dict:
    y = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    scored = pd.DataFrame({"y": y.to_numpy(), "p": pred_prob})
    scored["decile"] = pd.qcut(scored["p"].rank(method="first"), 10, labels=False)
    top_rate = float(scored.groupby("decile")["y"].mean().sort_index(ascending=False).iloc[0])
    baseline = float(scored["y"].mean())
    ranked = scored.sort_values("p", ascending=False)
    return {
        "rows": int(len(scored)),
        "positives": int(scored["y"].sum()),
        "baseline_rate": baseline,
        "auc": float(roc_auc_score(scored["y"], scored["p"])) if scored["y"].nunique() == 2 else None,
        "precision_at_100": float(ranked.head(100)["y"].mean()),
        "precision_at_500": float(ranked.head(500)["y"].mean()),
        "precision_at_1000": float(ranked.head(1000)["y"].mean()),
        "top_decile_rate": top_rate,
        "top_decile_lift": top_rate / baseline if baseline else 0.0,
    }


def make_linear_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "ohe",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            min_frequency=500,
                                            max_categories=80,
                                        ),
                                    ),
                                ]
                            ),
                            cat_cols,
                        ),
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler()),
                                ]
                            ),
                            num_cols,
                        ),
                    ]
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=160,
                    tol=1e-2,
                    random_state=42,
                    average=True,
                ),
            ),
        ]
    )


def make_tree_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "ohe",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            min_frequency=500,
                                            max_categories=80,
                                        ),
                                    ),
                                ]
                            ),
                            cat_cols,
                        ),
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="median")),
                                ]
                            ),
                            num_cols,
                        ),
                    ]
                ),
            ),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=60,
                    max_depth=14,
                    min_samples_leaf=60,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )


def make_model(model_type: str, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    if model_type == "linear":
        return make_linear_pipeline(cat_cols, num_cols)
    if model_type == "extra_trees":
        return make_tree_pipeline(cat_cols, num_cols)
    raise ValueError(f"Unknown model_type: {model_type}")


def maybe_sample(df: pd.DataFrame, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df
    y = df[label_col].astype(int)
    # Keep class balance roughly stable in quick experiments.
    sampled_parts = []
    rng = np.random.default_rng(seed)
    for value in sorted(y.unique()):
        idx = y.index[y == value]
        take = max(1, int(max_rows * len(idx) / len(df)))
        take = min(take, len(idx))
        sampled_idx = rng.choice(idx.to_numpy(), size=take, replace=False)
        sampled_parts.append(df.loc[sampled_idx])
    return pd.concat(sampled_parts, ignore_index=True).sample(frac=1, random_state=seed)


def add_t0_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for target, builder in T0_TARGETS.items():
        out[target] = builder(out)
    return out


def load_t0_frames() -> dict[str, pd.DataFrame]:
    return {month: add_t0_targets(t0_stage.load_training_month(month)) for month in MONTHS}


def add_t1_enhancements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in set(T1_BASE_NUM + RAW_CALL_FEATURES + SEQUENCE_FEATURES):
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["total_calls_log1p"] = np.log1p(out["total_calls"].clip(lower=0))
    out["answered_calls_log1p"] = np.log1p(out["answered_calls"].clip(lower=0))
    out["avg_call_duration_log1p"] = np.log1p(out["avg_call_duration"].clip(lower=0))
    out["raw_total_calls_log1p"] = np.log1p(out["raw_total_calls"].clip(lower=0))
    out["raw_total_duration_log1p"] = np.log1p(out["raw_total_duration"].clip(lower=0))
    out["unanswered_calls"] = (out["total_calls"] - out["answered_calls"]).clip(lower=0)
    out["unanswered_rate"] = np.where(out["total_calls"] > 0, out["unanswered_calls"] / out["total_calls"], 0)
    out["duration_per_answered_call"] = np.where(
        out["answered_calls"] > 0,
        out["avg_call_duration"] * out["total_calls"] / out["answered_calls"],
        0,
    )
    out["raw_duration_per_answered_call"] = np.where(
        out["raw_status_answered_calls"] > 0,
        out["raw_total_duration"] / out["raw_status_answered_calls"],
        0,
    )
    out["has_any_call"] = (out["total_calls"] > 0).astype(int)
    out["has_answered_call"] = (out["answered_calls"] > 0).astype(int)
    out["has_10s_avg_call"] = (out["avg_call_duration"] >= 10).astype(int)
    out["has_30s_avg_call"] = (out["avg_call_duration"] >= 30).astype(int)
    out["has_60s_avg_call"] = (out["avg_call_duration"] >= 60).astype(int)
    out["raw_answered_x_duration"] = out["raw_status_answered_share"] * out["raw_avg_duration"]
    out["raw_outbound_x_answered_share"] = out["raw_type_outbound_share"] * out["raw_status_answered_share"]
    return out


def load_t1_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for month in MONTHS:
        df = load_month(month)
        seq_path = SEQUENCE_CACHE_DIR / f"{month}_sequence_call_features.csv"
        if seq_path.exists() and "user_id" in df.columns:
            seq = pd.read_csv(seq_path, low_memory=False)
            seq["user_id"] = seq["user_id"].astype(str).str.strip()
            df["user_id"] = df["user_id"].astype(str).str.strip()
            df = df.merge(seq, on="user_id", how="left")
        frames[month] = add_t1_enhancements(df)
    return frames


def run_lomo(
    frames: dict[str, pd.DataFrame],
    features: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    label_col: str,
    model_type: str,
    experiment: str,
    max_train_rows: int,
) -> list[dict]:
    rows = []
    for holdout in MONTHS:
        train_df = pd.concat([df for month, df in frames.items() if month != holdout], ignore_index=True)
        train_df = maybe_sample(train_df, label_col, max_train_rows, seed=42)
        holdout_df = frames[holdout]

        model = make_model(model_type, cat_cols, num_cols)
        model.fit(train_df[features], train_df[label_col].astype(int))
        pred = model.predict_proba(holdout_df[features])[:, 1]
        metrics = metric_block(holdout_df[label_col], pred)

        rows.append(
            {
                "experiment": experiment,
                "model_type": model_type,
                "label_col": label_col,
                "holdout_month": holdout,
                **metrics,
            }
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "auc",
        "precision_at_100",
        "precision_at_500",
        "precision_at_1000",
        "top_decile_lift",
        "top_decile_rate",
        "baseline_rate",
    ]
    return (
        results.groupby(["experiment", "model_type", "label_col"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["experiment", "precision_at_500", "top_decile_lift"], ascending=[True, False, False])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["all", "t0", "t1"],
        default="all",
        help="Run all experiments or only one stage.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    if args.mode in {"all", "t0"}:
        t0_frames = load_t0_frames()
        for target in T0_TARGETS:
            all_rows.extend(
                run_lomo(
                    frames=t0_frames,
                    features=t0_stage.FEATURES,
                    cat_cols=t0_stage.CATEGORICAL_FEATURES,
                    num_cols=t0_stage.NUMERIC_FEATURES,
                    label_col=target,
                    model_type="linear",
                    experiment="t0_target_search",
                    max_train_rows=args.max_train_rows,
                )
            )
        for model_type in ["linear", "extra_trees"]:
            all_rows.extend(
                run_lomo(
                    frames=t0_frames,
                    features=t0_stage.FEATURES,
                    cat_cols=t0_stage.CATEGORICAL_FEATURES,
                    num_cols=t0_stage.NUMERIC_FEATURES,
                    label_col="engaged_10s",
                    model_type=model_type,
                    experiment="t0_model_search",
                    max_train_rows=args.max_train_rows,
                )
            )

    if args.mode in {"all", "t1"}:
        t1_frames = load_t1_frames()
        base_features = T1_BASE_CAT + T1_BASE_NUM
        enhanced_num = T1_BASE_NUM + [col for col in T1_EXTRA_NUM if col not in T1_BASE_NUM]
        enhanced_features = T1_BASE_CAT + enhanced_num
        for experiment, features, num_cols in [
            ("t1_base_features", base_features, T1_BASE_NUM),
            ("t1_enhanced_features", enhanced_features, enhanced_num),
        ]:
            for model_type in ["linear", "extra_trees"]:
                all_rows.extend(
                    run_lomo(
                        frames=t1_frames,
                        features=features,
                        cat_cols=T1_BASE_CAT,
                        num_cols=num_cols,
                        label_col="converted",
                        model_type=model_type,
                        experiment=experiment,
                        max_train_rows=args.max_train_rows,
                    )
                )

    results = pd.DataFrame(all_rows)
    summary = summarize(results)

    suffix = args.mode
    results_path = OUT_DIR / f"stage_model_lomo_results_{suffix}.csv"
    summary_path = OUT_DIR / f"stage_model_lomo_summary_{suffix}.csv"
    json_path = OUT_DIR / f"stage_model_lomo_summary_{suffix}.json"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "max_train_rows": args.max_train_rows,
                "results_csv": str(results_path),
                "summary_csv": str(summary_path),
                "best_by_experiment": summary.groupby("experiment").head(3).to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved results:", results_path)
    print("Saved summary:", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
