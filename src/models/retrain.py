from __future__ import annotations
import os
import pandas as pd
from typing import Optional, Dict, List

from .isolation_forest import IsolationForestModel
from .autoencoder import AutoencoderModel


class ModelRetrainer:
    def __init__(self):
        pass

    @staticmethod
    def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
        return df.select_dtypes(include=["number"]).copy()

    @staticmethod
    def _aligned_features(df: pd.DataFrame, prev_features: List[str]) -> List[str]:
        numeric_cols = set(df.select_dtypes(include=["number"]).columns)
        inter = [c for c in prev_features if c in numeric_cols]
        if inter:
            return inter
        return list(df.select_dtypes(include=["number"]).columns)

    def retrain_isolation_forest(
        self, df: pd.DataFrame, prev_path: str, contamination: float = 0.01
    ) -> str:
        X_all = self._numeric_frame(df)

        prev = IsolationForestModel.load(prev_path)
        features = self._aligned_features(X_all, prev.features)
        X = X_all[features].copy()

        model = IsolationForestModel(
            contamination=float(contamination),
            n_estimators=int(prev._cfg.get("n_estimators", 500)),
            max_samples=int(prev._cfg.get("max_samples", 256)),
            max_features=float(prev._cfg.get("max_features", 1.0)),
            bootstrap=bool(prev._cfg.get("bootstrap", False)),
            random_state=prev.random_state,
            use_robust_scaler=(prev.scaler_type == "robust"),
        )
        model.fit(X, feature_names=features)
        out_path = prev_path.replace(".pkl", "_retrained.pkl")
        model.save(out_path)
        return out_path

    def retrain_autoencoder(
        self, df: pd.DataFrame, prev_path: str, contamination: float = 0.01
    ) -> str:
        X_all = self._numeric_frame(df)

        prev = AutoencoderModel.load(prev_path)
        features = self._aligned_features(X_all, prev.features)
        X = X_all[features].copy()

        model = AutoencoderModel(
            encoding_dim=int(prev.encoding_dim),
            hidden_layers=list(prev.hidden_layers),
            activation=str(prev.activation),
            contamination=float(contamination),
            epochs=int(prev.metadata.get("epochs", 100)) if "epochs" in prev.metadata else 100,
            batch_size=int(prev.metadata.get("batch_size", 128)) if "batch_size" in prev.metadata else 128,
            random_state=prev.metadata.get("random_state", prev.random_state),
            loss=str(prev.loss),
            l2_reg=float(prev.l2_reg),
            noise_std=float(prev.noise_std),
            use_robust_scaler=(prev.scaler_type == "robust"),
        )
        model.fit(X, feature_names=features, validation_split=0.1)
        out_path = prev_path.replace(".pkl", "_retrained.pkl")
        model.save(out_path)
        return out_path
