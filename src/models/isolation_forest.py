from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any, cast
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler


class IsolationForestModel:
    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 500,
        max_samples: int = 256,
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: Optional[int] = 42,
        use_robust_scaler: bool = True,
    ) -> None:
        self.contamination: float = float(contamination)
        self.random_state: Optional[int] = int(random_state) if random_state is not None else None

        self._cfg: Dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "bootstrap": bootstrap,
        }
        self._coerce_cfg_types()

        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.scaler_type: str = "robust" if use_robust_scaler else "standard"

        self.model: IsolationForest = IsolationForest(
            n_estimators=int(self._cfg["n_estimators"]),
            max_samples=int(self._cfg["max_samples"]),
            max_features=float(self._cfg["max_features"]),
            bootstrap=bool(self._cfg["bootstrap"]),
            contamination=float(self.contamination),
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.features: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.threshold: Optional[float] = None

  
    def _coerce_cfg_types(self) -> None:
        self._cfg["n_estimators"] = int(self._cfg.get("n_estimators", 500))
        self._cfg["max_samples"] = int(self._cfg.get("max_samples", 256))
        self._cfg["max_features"] = float(self._cfg.get("max_features", 1.0))
        self._cfg["bootstrap"] = bool(self._cfg.get("bootstrap", False))
        self.contamination = float(self.contamination)

    @staticmethod
    def _ensure_array(X: pd.DataFrame | np.ndarray, features: List[str] | None) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if features:
                miss = [c for c in features if c not in X.columns]
                if miss:
                    X = X.copy()
                    for c in miss:
                        X[c] = 0.0
                X_arr = X[features].to_numpy(copy=False)
            else:
                X_arr = X.to_numpy(copy=False)
        else:
            X_arr = X
        X_arr = np.asarray(X_arr, dtype=np.float64)
        return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: pd.DataFrame | np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        if isinstance(X, pd.DataFrame):
            self.features = feature_names or X.columns.tolist()
            X_arr = X[self.features].to_numpy(copy=False) if feature_names else X.to_numpy(copy=False)
        else:
            self.features = feature_names or []
            X_arr = X
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled)

        s_native = -self.model.score_samples(X_scaled)  
        q = 1.0 - float(self.contamination)
        self.threshold = float(np.quantile(s_native, q))

        self.metadata = {
            "model_type": "IsolationForest",
            "contamination": float(self.contamination),
            "n_estimators": int(self._cfg["n_estimators"]),
            "max_samples": int(self._cfg["max_samples"]),
            "max_features": float(self._cfg["max_features"]),
            "bootstrap": bool(self._cfg["bootstrap"]),
            "random_state": self.random_state,
            "scaler": self.scaler_type,
            "threshold": float(self.threshold),
            "n_features": len(self.features),
            "features": list(self.features),
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": int(X_arr.shape[0]),
        }

    def calibrate_threshold(self, X_cal: pd.DataFrame | np.ndarray, desired_rate: Optional[float] = None) -> None:
     
        X_arr = self._ensure_array(X_cal, self.features)
        X_scaled = cast(np.ndarray, self.scaler.transform(X_arr))
        s_native = -self.model.score_samples(X_scaled)
        rate = float(self.contamination) if desired_rate is None else float(desired_rate)
        rate = min(max(rate, 1e-6), 0.5)  # clamp sanely
        self.threshold = float(np.quantile(s_native, 1.0 - rate))
        self.metadata["threshold"] = float(self.threshold)
        self.metadata["threshold_source"] = "calibration"
        self.metadata["threshold_rate"] = float(rate)


    def native_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Model not fitted: scaler missing.")
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("Model not fitted: estimator missing.")
        X_arr = self._ensure_array(X, self.features)
        X_scaled = cast(np.ndarray, self.scaler.transform(X_arr))
        raw = self.model.score_samples(X_scaled)
        return -raw

    def get_aml_scores(self, X: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = self.native_score(X) 
        s_min, s_max = float(np.min(s)), float(np.max(s))
        if (s_max - s_min) > 1e-12:
            ui_scores = (s - s_min) / (s_max - s_min)
        else:
            ui_scores = np.zeros_like(s)

        thr = (
            float(self.threshold)
            if self.threshold is not None
            else float(np.quantile(s, 1.0 - float(self.contamination)))
        )
        labels = (s >= thr).astype(int)
        return ui_scores, labels

   
    def save(self, file_path: str) -> None:
        payload: Dict[str, Any] = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "features": self.features,
            "metadata": self.metadata,
            "threshold": self.threshold,
            "contamination": float(self.contamination),
            "random_state": self.random_state,
            "cfg": {
                "n_estimators": int(self._cfg["n_estimators"]),
                "max_samples": int(self._cfg["max_samples"]),
                "max_features": float(self._cfg["max_features"]),
                "bootstrap": bool(self._cfg["bootstrap"]),
            },
            "model": self.model,  
        }
        joblib.dump(payload, file_path)

    @classmethod
    def load(cls, file_path: str) -> "IsolationForestModel":
        data: Dict[str, Any] = joblib.load(file_path)

        cfg = data.get("cfg", {})
        instance = cls(
            contamination=float(data.get("contamination", 0.01)),
            n_estimators=int(cfg.get("n_estimators", 500)),
            max_samples=int(cfg.get("max_samples", 256)),
            max_features=float(cfg.get("max_features", 1.0)),
            bootstrap=bool(cfg.get("bootstrap", False)),
            random_state=data.get("random_state", 42),
            use_robust_scaler=(data.get("scaler_type", "standard") == "robust"),
        )
        instance._coerce_cfg_types()

        instance.scaler = data["scaler"]
        instance.scaler_type = str(data.get("scaler_type", instance.scaler_type))
        instance.features = list(data.get("features", []))
        instance.metadata = dict(data.get("metadata", {}))
        instance.threshold = data.get("threshold")

        if "model" in data and isinstance(data["model"], IsolationForest):
            instance.model = data["model"]
        else:
            instance.model = IsolationForest(
                n_estimators=int(instance._cfg["n_estimators"]),
                max_samples=int(instance._cfg["max_samples"]),
                max_features=float(instance._cfg["max_features"]),
                bootstrap=bool(instance._cfg["bootstrap"]),
                contamination=float(instance.contamination),
                random_state=instance.random_state,
                n_jobs=-1,
            )

        return instance
