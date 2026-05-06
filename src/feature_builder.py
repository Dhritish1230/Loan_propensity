import numpy as np
import pandas as pd


BASE_CATEGORICAL_COLS = [
    "language",
    "state",
    "campaign_id",
    "scheme",
    "flow_phase",
    "zone",
    "branch",
    "base_type",
    "flow_type",
]

BASE_NUMERIC_COLS = [
    "age",
    "decile",
    "minimum_loan_amount",
    "maximum_loan_amount",
    "loan_amount_span",
    "created_dayofweek",
    "created_hour",
]

CALL_NUMERIC_COLS = [
    "total_calls",
    "answered_calls",
    "outbound_calls",
    "avg_call_duration",
    "answered_rate",
    "unanswered_calls",
    "has_any_call",
    "has_answered_call",
    "total_calls_log1p",
    "answered_calls_log1p",
    "avg_call_duration_log1p",
    "outbound_share",
]


def get_call_feature_cols():
    return CALL_NUMERIC_COLS.copy()


def validate_t0_feature_columns(feature_cols):
    call_features = set(get_call_feature_cols())
    leaked = sorted(set(feature_cols) & call_features)
    if leaked:
        raise ValueError(
            f"T0 must not use call/post-contact features, but found: {', '.join(leaked)}"
        )


def validate_t1_feature_columns(feature_cols):
    missing = [col for col in CALL_NUMERIC_COLS if col not in feature_cols]
    if missing:
        raise ValueError(
            f"T1 should include call/post-contact features, but is missing: {', '.join(missing)}"
        )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def get_series(name, default=np.nan):
        if name in out.columns:
            return out[name]
        return pd.Series([default] * len(out), index=out.index)

    created_ts = pd.to_datetime(get_series("created_at"), errors="coerce")
    expiry_ts = pd.to_datetime(get_series("expiry_date"), errors="coerce")

    out["created_dayofweek"] = created_ts.dt.dayofweek.fillna(-1)
    out["created_hour"] = created_ts.dt.hour.fillna(-1)
    days_to_expiry = (expiry_ts - created_ts).dt.days
    out["days_to_expiry"] = days_to_expiry.clip(lower=0, upper=365).fillna(-1)

    min_amt = pd.to_numeric(get_series("minimum_loan_amount", 0), errors="coerce").fillna(0)
    max_amt = pd.to_numeric(get_series("maximum_loan_amount", 0), errors="coerce").fillna(0)
    out["minimum_loan_amount"] = min_amt
    out["maximum_loan_amount"] = max_amt
    out["loan_amount_span"] = (max_amt - min_amt).clip(lower=0)

    out["has_agreement_id"] = get_series("agreement_id").notna().astype(int)
    out["min_loan_amount_log1p"] = np.log1p(min_amt.clip(lower=0))
    out["max_loan_amount_log1p"] = np.log1p(max_amt.clip(lower=0))

    total_calls = pd.to_numeric(get_series("total_calls", 0), errors="coerce").fillna(0)
    answered_calls = pd.to_numeric(get_series("answered_calls", 0), errors="coerce").fillna(0)
    outbound_calls = pd.to_numeric(get_series("outbound_calls", 0), errors="coerce").fillna(0)
    avg_call_duration = pd.to_numeric(get_series("avg_call_duration", 0), errors="coerce").fillna(0)
    answered_rate = pd.to_numeric(get_series("answered_rate", 0), errors="coerce").fillna(0)

    out["total_calls"] = total_calls
    out["answered_calls"] = answered_calls
    out["outbound_calls"] = outbound_calls
    out["avg_call_duration"] = avg_call_duration
    out["answered_rate"] = answered_rate
    out["unanswered_calls"] = (total_calls - answered_calls).clip(lower=0)
    out["has_any_call"] = (total_calls > 0).astype(int)
    out["has_answered_call"] = (answered_calls > 0).astype(int)
    out["total_calls_log1p"] = np.log1p(total_calls)
    out["answered_calls_log1p"] = np.log1p(answered_calls)
    out["avg_call_duration_log1p"] = np.log1p(avg_call_duration.clip(lower=0))
    out["outbound_share"] = np.where(total_calls > 0, outbound_calls / total_calls, 0)

    for col in BASE_CATEGORICAL_COLS:
        if col not in out.columns:
            out[col] = "unknown"

    return out


def get_t0_feature_lists():
    feature_cols = BASE_CATEGORICAL_COLS.copy() + BASE_NUMERIC_COLS.copy()
    validate_t0_feature_columns(feature_cols)
    return BASE_CATEGORICAL_COLS.copy(), BASE_NUMERIC_COLS.copy()


def get_t1_feature_lists():
    feature_cols = BASE_CATEGORICAL_COLS.copy() + BASE_NUMERIC_COLS.copy() + CALL_NUMERIC_COLS.copy()
    validate_t1_feature_columns(feature_cols)
    return BASE_CATEGORICAL_COLS.copy(), BASE_NUMERIC_COLS.copy() + CALL_NUMERIC_COLS.copy()


def get_t0_feature_lists_focused():
    categorical_cols = [
        "language",
        "state",
        "campaign_id",
        "scheme",
        "base_type",
        "flow_phase",
    ]
    numeric_cols = [
        "age",
        "decile",
        "minimum_loan_amount",
        "maximum_loan_amount",
        "loan_amount_span",
        "created_dayofweek",
        "created_hour",
    ]
    validate_t0_feature_columns(categorical_cols + numeric_cols)
    return categorical_cols, numeric_cols


def get_t1_feature_lists_focused():
    categorical_cols, numeric_cols = get_t0_feature_lists_focused()
    feature_cols = categorical_cols + numeric_cols + CALL_NUMERIC_COLS.copy()
    validate_t1_feature_columns(feature_cols)
    return categorical_cols, numeric_cols + CALL_NUMERIC_COLS.copy()


def get_t0_feature_lists_ranking():
    categorical_cols = [
        "language",
        "state",
        "campaign_id",
        "scheme",
        "flow_phase",
        "zone",
        "branch",
        "base_type",
    ]
    numeric_cols = [
        "age",
        "decile",
        "minimum_loan_amount",
        "maximum_loan_amount",
        "loan_amount_span",
        "min_loan_amount_log1p",
        "max_loan_amount_log1p",
        "created_dayofweek",
        "created_hour",
        "days_to_expiry",
        "has_agreement_id",
    ]
    validate_t0_feature_columns(categorical_cols + numeric_cols)
    return categorical_cols, numeric_cols


def get_t0_feature_lists_generalized():
    categorical_cols = [
        "language",
        "state",
        "campaign_id",
        "scheme",
        "flow_phase",
        "base_type",
    ]
    numeric_cols = [
        "age",
        "decile",
        "minimum_loan_amount",
        "maximum_loan_amount",
        "loan_amount_span",
        "min_loan_amount_log1p",
        "max_loan_amount_log1p",
        "created_dayofweek",
        "created_hour",
        "days_to_expiry",
        "has_agreement_id",
    ]
    validate_t0_feature_columns(categorical_cols + numeric_cols)
    return categorical_cols, numeric_cols
