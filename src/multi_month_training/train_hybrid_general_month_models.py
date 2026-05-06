import json
import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from evaluate_model_slice import FEATURE_SETS, MONTHS, load_month, make_pipeline


ROOT = Path(os.getenv("LOAN_PROPENSITY_WORKSPACE", Path(__file__).resolve().parents[2])).resolve()
SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT / "multi_month_training"
LOAN_API_DIR = Path(os.getenv("LOAN_PROPENSITY_MODEL_DIR", ROOT / "models")).resolve()


def precision_at_k(y_true: pd.Series, pred_prob, k: int) -> float:
    ranked = pd.DataFrame({"y": y_true.values, "p": pred_prob}).sort_values("p", ascending=False)
    return float(ranked.head(k)["y"].mean())


def top_decile_lift(y_true: pd.Series, pred_prob) -> float:
    scored = pd.DataFrame({"y": y_true.values, "p": pred_prob})
    scored["decile"] = pd.qcut(scored["p"], 10, labels=False, duplicates="drop")
    table = scored.groupby("decile")["y"].mean().sort_index(ascending=False)
    baseline = float(scored["y"].mean())
    if baseline == 0 or table.empty:
        return 0.0
    return float(table.iloc[0] / baseline)


def metrics(y_true: pd.Series, pred_prob) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, pred_prob)),
        "precision_at_100": precision_at_k(y_true, pred_prob, 100),
        "precision_at_500": precision_at_k(y_true, pred_prob, 500),
        "top_decile_lift": top_decile_lift(y_true, pred_prob),
    }


def fit_model(df: pd.DataFrame, model_family: str):
    config = FEATURE_SETS[model_family]
    features = config["categorical"] + config["numeric"]
    model = make_pipeline(config["categorical"], config["numeric"])
    model.fit(df[features], df["converted"])
    return model, features


def evaluate_hybrid(month_frames: dict[str, pd.DataFrame], model_family: str) -> pd.DataFrame:
    rows = []
    blend_weights = [0.25, 0.50, 0.75]

    for holdout_month, holdout_df in month_frames.items():
        train_months_df = pd.concat(
            [df for month, df in month_frames.items() if month != holdout_month],
            ignore_index=True,
        )
        month_train_df, month_test_df = train_test_split(
            holdout_df,
            test_size=0.3,
            random_state=42,
            stratify=holdout_df["converted"],
        )

        general_model, features = fit_model(train_months_df, model_family)
        month_model, _ = fit_model(month_train_df, model_family)

        y_test = month_test_df["converted"].astype(int)
        general_pred = general_model.predict_proba(month_test_df[features])[:, 1]
        month_pred = month_model.predict_proba(month_test_df[features])[:, 1]

        candidate_scores = {
            "general_only": general_pred,
            "month_specific_only": month_pred,
        }
        for weight in blend_weights:
            candidate_scores[f"hybrid_month_weight_{weight:.2f}"] = (
                (1 - weight) * general_pred + weight * month_pred
            )

        for strategy, pred in candidate_scores.items():
            result = metrics(y_test, pred)
            rows.append(
                {
                    "model_family": model_family,
                    "month": holdout_month,
                    "strategy": strategy,
                    "rows": int(len(month_test_df)),
                    "conversions": int(y_test.sum()),
                    "baseline_rate": float(y_test.mean()),
                    **result,
                }
            )

    return pd.DataFrame(rows)


def train_final_hybrid_bundle(month_frames: dict[str, pd.DataFrame], model_family: str) -> dict:
    all_df = pd.concat(month_frames.values(), ignore_index=True)
    general_model, features = fit_model(all_df, model_family)
    month_models = {}
    for month, df in month_frames.items():
        month_model, _ = fit_model(df, model_family)
        month_models[month] = month_model

    return {
        "model_family": model_family,
        "features": features,
        "general_model": general_model,
        "month_models": month_models,
        "default_month_weight": 0.50,
        "known_months": list(month_models.keys()),
        "description": (
            "Hybrid bundle: blends a global multi-month model with a same-month model "
            "when the month is known; use general_model alone for unseen future months."
        ),
    }


def main():
    month_frames = {month: load_month(month) for month in MONTHS}
    report_parts = []
    artifact_paths = {}

    for model_family in ["t0", "t1"]:
        family_report = evaluate_hybrid(month_frames, model_family)
        report_parts.append(family_report)

        bundle = train_final_hybrid_bundle(month_frames, model_family)
        artifact_path = LOAN_API_DIR / f"loan_model_{model_family}_hybrid_general_month.pkl"
        feature_path = LOAN_API_DIR / f"feature_columns_{model_family}_hybrid_general_month.pkl"
        joblib.dump(bundle, artifact_path)
        joblib.dump(bundle["features"], feature_path)
        artifact_paths[f"{model_family}_bundle"] = str(artifact_path)
        artifact_paths[f"{model_family}_features"] = str(feature_path)
        print(f"Saved {model_family.upper()} hybrid bundle:", artifact_path)

    report = pd.concat(report_parts, ignore_index=True)
    report_path = OUT_DIR / "hybrid_general_month_comparison.csv"
    json_path = OUT_DIR / "hybrid_general_month_summary.json"
    report.to_csv(report_path, index=False)

    best_rows = (
        report.sort_values(
            ["model_family", "month", "precision_at_100", "auc"],
            ascending=[True, True, False, False],
        )
        .groupby(["model_family", "month"], as_index=False)
        .first()
    )
    summary = {
        "artifacts": artifact_paths,
        "best_strategy_by_month": best_rows.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved report:", report_path)
    print("Saved summary:", json_path)
    print("Best strategies:")
    print(best_rows[["model_family", "month", "strategy", "auc", "precision_at_100"]].to_string(index=False))


if __name__ == "__main__":
    main()
