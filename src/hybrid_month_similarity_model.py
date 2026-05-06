import numpy as np
import pandas as pd


class SingleHybridMonthModel:
    """One deployable model object that blends global and month-specific models."""

    def __init__(
        self,
        t0_bundle,
        t1_bundle,
        month_profiles,
        profile_features,
        default_month_weight=0.35,
        call_signal_columns=None,
        t0_profile_features=None,
        t1_profile_features=None,
    ):
        self.t0_bundle = t0_bundle
        self.t1_bundle = t1_bundle
        self.month_profiles = month_profiles
        self.profile_features = profile_features
        self.t0_profile_features = t0_profile_features or profile_features
        self.t1_profile_features = t1_profile_features or profile_features
        self.default_month_weight = default_month_weight
        self.call_signal_columns = call_signal_columns or [
            "total_calls",
            "answered_calls",
            "answered_rate",
        ]

    @property
    def features(self):
        return sorted(set(self.t0_bundle["features"] + self.t1_bundle["features"]))

    def _prepare(self, df):
        out = df.copy()
        for col in self.features + self.profile_features + self.call_signal_columns:
            if col not in out.columns:
                out[col] = np.nan

        for col in ["language", "state", "flow_phase"]:
            out[col] = out[col].fillna("unknown").astype(str).str.strip()
        out["language"] = out["language"].str.lower()
        out["state"] = out["state"].str.upper()

        numeric_cols = sorted(
            set(self.features + self.profile_features + self.call_signal_columns)
            - {"language", "state", "flow_phase"}
        )
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        return out

    def _has_call_signal(self, df):
        signal = pd.Series(False, index=df.index)
        for col in self.call_signal_columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            signal = signal | (vals > 0)
        return signal

    def _batch_profile(self, df, profile_features=None):
        profile_features = profile_features or self.profile_features
        return {
            feature: float(pd.to_numeric(df[feature], errors="coerce").fillna(0).mean())
            for feature in profile_features
        }

    def _month_weights(self, df, profile_features=None):
        profile_features = profile_features or self.profile_features
        profile = self._batch_profile(df, profile_features)
        distances = {}
        for month, month_profile in self.month_profiles.items():
            distance = 0.0
            for feature in profile_features:
                scale = abs(month_profile.get(f"{feature}_std", 0.0)) + 1.0
                distance += abs(profile[feature] - month_profile[feature]) / scale
            distances[month] = distance

        similarities = {
            month: 1.0 / (1.0 + distance)
            for month, distance in distances.items()
        }
        total = sum(similarities.values())
        if total <= 0:
            return {month: 1.0 / len(similarities) for month in similarities}
        return {month: value / total for month, value in similarities.items()}

    def _score_bundle(self, df, bundle, profile_features=None):
        features = bundle["features"]
        global_pred = bundle["global_model"].predict_proba(df[features])[:, 1]
        weights = self._month_weights(df, profile_features)
        month_pred = np.zeros(len(df), dtype=float)
        for month, weight in weights.items():
            model = bundle["month_models"][month]
            month_pred += weight * model.predict_proba(df[features])[:, 1]
        return (1.0 - self.default_month_weight) * global_pred + self.default_month_weight * month_pred

    def predict_proba(self, X):
        df = self._prepare(pd.DataFrame(X).copy())
        t0_pred = self._score_bundle(df, self.t0_bundle, self.t0_profile_features)
        t1_pred = self._score_bundle(df, self.t1_bundle, self.t1_profile_features)
        use_t1 = self._has_call_signal(df).to_numpy()
        pred = np.where(use_t1, t1_pred, t0_pred)
        pred = np.clip(pred, 0.0, 1.0)
        return np.column_stack([1.0 - pred, pred])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def explain_month_similarity(self, X):
        df = self._prepare(pd.DataFrame(X).copy())
        return {
            "t0": self._month_weights(df, self.t0_profile_features),
            "t1": self._month_weights(df, self.t1_profile_features),
        }


class SingleFamilyHybridMonthModel:
    """One deployable T0-or-T1 model that blends global and month-specific models."""

    def __init__(
        self,
        model_family,
        bundle,
        month_profiles,
        profile_features,
        default_month_weight=0.35,
        categorical_columns=None,
    ):
        self.model_family = model_family
        self.bundle = bundle
        self.month_profiles = month_profiles
        self.profile_features = profile_features
        self.default_month_weight = default_month_weight
        self.categorical_columns = categorical_columns or ["language", "state", "flow_phase"]

    @property
    def features(self):
        return self.bundle["features"]

    def _prepare(self, df):
        out = df.copy()
        for col in self.features + self.profile_features:
            if col not in out.columns:
                out[col] = np.nan

        categorical_columns = getattr(self, "categorical_columns", ["language", "state", "flow_phase"])
        for col in categorical_columns:
            if col in out.columns:
                out[col] = out[col].fillna("unknown").astype(str).str.strip()
        if "language" in out.columns:
            out["language"] = out["language"].str.lower()
        if "state" in out.columns:
            out["state"] = out["state"].str.upper()

        numeric_cols = sorted(
            set(self.features + self.profile_features)
            - set(categorical_columns)
        )
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        return out

    def _batch_profile(self, df):
        return {
            feature: float(pd.to_numeric(df[feature], errors="coerce").fillna(0).mean())
            for feature in self.profile_features
        }

    def _month_weights(self, df):
        profile = self._batch_profile(df)
        distances = {}
        for month, month_profile in self.month_profiles.items():
            distance = 0.0
            for feature in self.profile_features:
                scale = abs(month_profile.get(f"{feature}_std", 0.0)) + 1.0
                distance += abs(profile[feature] - month_profile[feature]) / scale
            distances[month] = distance

        similarities = {
            month: 1.0 / (1.0 + distance)
            for month, distance in distances.items()
        }
        total = sum(similarities.values())
        if total <= 0:
            return {month: 1.0 / len(similarities) for month in similarities}
        return {month: value / total for month, value in similarities.items()}

    def predict_proba(self, X):
        df = self._prepare(pd.DataFrame(X).copy())
        features = self.bundle["features"]
        global_pred = self.bundle["global_model"].predict_proba(df[features])[:, 1]
        weights = self._month_weights(df)
        month_pred = np.zeros(len(df), dtype=float)
        for month, weight in weights.items():
            model = self.bundle["month_models"][month]
            month_pred += weight * model.predict_proba(df[features])[:, 1]
        pred = (1.0 - self.default_month_weight) * global_pred + self.default_month_weight * month_pred
        pred = np.clip(pred, 0.0, 1.0)
        return np.column_stack([1.0 - pred, pred])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def explain_month_similarity(self, X):
        df = self._prepare(pd.DataFrame(X).copy())
        return self._month_weights(df)
