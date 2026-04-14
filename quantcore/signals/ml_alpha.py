"""
Machine learning alpha signal combination.

Inspired by Two Sigma and D.E. Shaw's approach of using ML to combine
multiple weak signals into a stronger composite signal.

References:
  - Lopez de Prado (2018) "Advances in Financial Machine Learning" (published book)
  - Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning"
    - Review of Financial Studies (publicly available)
  - Two Sigma public blog posts on ML in finance

Key idea: instead of equal-weighting signals, use ML to learn
the optimal weighting across signals and market regimes.

Original implementation. Uses only sklearn — no proprietary code.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional, Tuple


class MLAlphaModel:
    """
    ML-based signal combination model.

    Takes multiple raw alpha signals and learns optimal weights
    using walk-forward cross-validation — critical for avoiding
    look-ahead bias (the #1 mistake in quant ML).

    Parameters
    ----------
    model_type : str
        'ridge'   — regularised linear (fast, interpretable)
        'elastic' — Elastic Net (sparsity + L2)
        'gbm'     — Gradient Boosting (captures non-linearity)
        'rf'      — Random Forest
    lookback : int
        Training window in days (walk-forward).
    """

    def __init__(
        self,
        model_type: str = "ridge",
        lookback:   int = 252,
        n_splits:   int = 5,
    ):
        self.model_type = model_type
        self.lookback   = lookback
        self.n_splits   = n_splits
        self.scaler     = StandardScaler()
        self._model     = None

    def _make_model(self):
        if self.model_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.model_type == "elastic":
            return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)
        elif self.model_type == "gbm":
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, subsample=0.8,
            )
        elif self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=100, max_depth=5,
                min_samples_leaf=10, n_jobs=-1,
            )
        return Ridge(alpha=1.0)

    # ------------------------------------------------------------------
    # Feature engineering (Lopez de Prado style)
    # ------------------------------------------------------------------

    @staticmethod
    def build_features(
        signals: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        lookback: int = 21,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Build a feature matrix from multiple signal DataFrames.

        For each signal and each asset, compute:
          - Current value
          - 5-day mean (short-term average)
          - 21-day mean (medium-term average)
          - Z-score vs 63-day window
          - Signal momentum (current - 21d ago)

        Returns feature matrix and feature names.
        """
        feature_rows = []
        tickers = None

        for name, df in signals.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < lookback:
                continue

            window = df.iloc[max(0, idx - 63): idx + 1]
            current = df.loc[date]
            mean5   = window.tail(5).mean()
            mean21  = window.tail(21).mean()
            std63   = window.std()
            zscore  = (current - window.mean()) / (std63 + 1e-8)
            mom     = current - window.iloc[-min(21, len(window))]

            if tickers is None:
                tickers = list(current.index)

            for feat, vals in [
                (f"{name}_cur", current),
                (f"{name}_m5",  mean5),
                (f"{name}_m21", mean21),
                (f"{name}_z",   zscore),
                (f"{name}_mom", mom),
            ]:
                feature_rows.append(vals.reindex(tickers).fillna(0).values)

        if not feature_rows or tickers is None:
            return None

        # Shape: [n_features, n_assets] → transpose to [n_assets, n_features]
        X = np.array(feature_rows).T
        return X, tickers

    # ------------------------------------------------------------------
    # Walk-forward training and prediction
    # ------------------------------------------------------------------

    def walk_forward_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        rebal_dates: pd.DatetimeIndex,
        horizon: int = 21,
    ) -> pd.DataFrame:
        """
        Walk-forward ML alpha prediction.

        At each rebalancing date:
          1. Train on the past `lookback` days of signal/return data
          2. Predict next-period returns
          3. Use predictions as weights

        This is the correct way to use ML in finance — never train
        on future data (Lopez de Prado, 2018).

        Returns
        -------
        pd.DataFrame of ML-predicted alpha scores.
        """
        all_predictions = []

        for i, date in enumerate(rebal_dates):
            if i < self.lookback // 21:
                continue

            # Build training data from past dates
            train_dates = rebal_dates[max(0, i - self.lookback // 21): i]
            X_train_list, y_train_list = [], []

            for td in train_dates:
                feat = self.build_features(signals, td)
                if feat is None:
                    continue
                X_t, tickers_t = feat

                fwd_date_idx = forward_returns.index.searchsorted(td) + horizon
                if fwd_date_idx >= len(forward_returns):
                    continue

                fwd_ret = forward_returns.iloc[fwd_date_idx].reindex(tickers_t).fillna(0).values

                X_train_list.append(X_t)
                y_train_list.append(fwd_ret)

            if len(X_train_list) < 10:
                continue

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)

            # Fit model
            try:
                model = self._make_model()
                X_scaled = self.scaler.fit_transform(X_train)
                model.fit(X_scaled, y_train)
            except Exception:
                continue

            # Predict for current date
            feat_cur = self.build_features(signals, date)
            if feat_cur is None:
                continue
            X_cur, tickers = feat_cur

            try:
                X_cur_scaled = self.scaler.transform(X_cur)
                if hasattr(model, "predict"):
                    preds = model.predict(X_cur_scaled)
                else:
                    continue
            except Exception:
                continue

            pred_series = pd.Series(preds, index=tickers, name=date)
            all_predictions.append(pred_series)

        if not all_predictions:
            return pd.DataFrame()

        result = pd.DataFrame(all_predictions)
        # Cross-sectional z-score
        mu  = result.mean(axis=1)
        std = result.std(axis=1)
        return result.sub(mu, axis=0).div(std + 1e-8, axis=0)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        signals: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        sample_date: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """
        Train a Random Forest on recent data and return feature importances.
        Useful for understanding which signals are driving the alpha.
        """
        rebal_dates = forward_returns.resample("ME").last().index
        idx = rebal_dates.searchsorted(sample_date)
        train_dates = rebal_dates[max(0, idx - 24): idx]

        X_list, y_list = [], []
        feat_names = None

        for td in train_dates:
            feat = self.build_features(signals, td)
            if feat is None:
                continue
            X_t, tickers = feat
            fwd_idx = forward_returns.index.searchsorted(td) + 21
            if fwd_idx >= len(forward_returns):
                continue
            y_t = forward_returns.iloc[fwd_idx].reindex(tickers).fillna(0).values
            X_list.append(X_t)
            y_list.append(y_t)

        if not X_list:
            return None

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        model = RandomForestRegressor(n_estimators=50, max_depth=4, n_jobs=-1)
        model.fit(X, y)

        signal_names = list(signals.keys())
        feat_labels  = []
        for name in signal_names:
            for suffix in ["_cur", "_m5", "_m21", "_z", "_mom"]:
                feat_labels.append(name + suffix)

        importance = pd.Series(
            model.feature_importances_[:len(feat_labels)],
            index=feat_labels[:len(model.feature_importances_)],
        ).sort_values(ascending=False)

        return importance
