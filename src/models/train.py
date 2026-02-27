from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from .isolation_forest import IsolationForestModel
from .autoencoder import AutoencoderModel


class ModelTrainer:
    def __init__(self):
        pass

    @staticmethod
    def _numeric_frame(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        if features:
            cols = [c for c in features if c in df.columns]
            return df[cols].select_dtypes(include=["number"]).copy()
        return df.select_dtypes(include=["number"]).copy()

    def train_from_dataframe(
        self,
        df: pd.DataFrame,
        model_types: List[str],
        models_dir: str,
        contamination: float = 0.01,
    ) -> Dict[str, str]:
        os.makedirs(models_dir, exist_ok=True)
        X = self._numeric_frame(df)
        features = list(X.columns)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved: Dict[str, str] = {}

        if "isolation_forest" in model_types:
            if_model = IsolationForestModel(
                contamination=float(contamination),
                n_estimators=1000,
                max_samples=256,
                max_features=1.0,
                bootstrap=False,
                random_state=42,
                use_robust_scaler=True,  
            )
            if_model.fit(X, feature_names=features)
            if_path = os.path.join(models_dir, f"model_isolation_forest_{timestamp}.pkl")
            if_model.save(if_path)
            with open(if_path.replace(".pkl", "_metadata.json"), "w") as f:
                json.dump(if_model.metadata, f, indent=2)
            saved["isolation_forest"] = if_path

        if "autoencoder" in model_types:
            ae_model = AutoencoderModel(
                encoding_dim=8,
                hidden_layers=[128, 64, 32],
                activation="relu",
                contamination=float(contamination),
                epochs=100,
                batch_size=128,
                random_state=42,
                loss="mse",
                l2_reg=1e-4,
                noise_std=0.02,
                use_robust_scaler=True, 
            )
            ae_model.fit(X, feature_names=features, validation_split=0.1)
            ae_path = os.path.join(models_dir, f"model_autoencoder_{timestamp}.pkl")
            ae_model.save(ae_path)
            with open(ae_path.replace(".pkl", "_metadata.json"), "w") as f:
                json.dump(ae_model.metadata, f, indent=2)
            saved["autoencoder"] = ae_path

        return saved
