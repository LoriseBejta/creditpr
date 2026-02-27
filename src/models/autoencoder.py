from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, cast

import keras
import tensorflow as tf
from keras import layers, models, callbacks, regularizers
from sklearn.preprocessing import StandardScaler, RobustScaler


class AutoencoderModel:
    def __init__(
        self,
        encoding_dim: int = 8,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        contamination: float = 0.01,
        epochs: int = 100,
        batch_size: int = 128,
        random_state: Optional[int] = 42,
        loss: str = "mse",
        l2_reg: float = 1e-4,
        noise_std: float = 0.02,
        use_robust_scaler: bool = True
    ):
        self.encoding_dim = int(encoding_dim)
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.activation = activation
        self.contamination = float(contamination)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = random_state
        self.loss = loss
        self.l2_reg = float(l2_reg)
        self.noise_std = float(noise_std)

        self.model: Optional[keras.Model] = None
        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.scaler_type = "robust" if use_robust_scaler else "standard"

        self.features: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.threshold: Optional[float] = None

    def _ensure_array(self, X: pd.DataFrame | np.ndarray, use_features: bool = True) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if use_features and self.features:
                miss = [c for c in self.features if c not in X.columns]
                if miss:
                    X = X.copy()
                    for c in miss:
                        X[c] = 0.0
                X_arr = X[self.features].to_numpy(copy=False)
            else:
                X_arr = X.to_numpy(copy=False)
        else:
            X_arr = X
        X_arr = np.asarray(X_arr, dtype=np.float64)
        return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _reconstruction_error_from_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not built. Call fit() first."
        recon = self.model.predict(X_scaled, verbose=cast(Any, 0))
        if self.loss == "mse":
            err = np.mean((X_scaled - recon) ** 2, axis=1)
        else:
            err = np.mean(np.abs(X_scaled - recon), axis=1)
        return err


    def build_model(self, input_dim: int):
        inp = layers.Input(shape=(input_dim,), name="inputs")
        x = layers.GaussianNoise(self.noise_std)(inp)

        for h in self.hidden_layers:
            x = layers.Dense(h, activation=None,
                             kernel_regularizer=regularizers.l2(self.l2_reg))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)

        encoded = layers.Dense(self.encoding_dim, activation=None, name="encoding",
                               kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Activation(self.activation)(encoded)
        y = encoded
        for h in reversed(self.hidden_layers):
            y = layers.Dense(h, activation=None,
                             kernel_regularizer=regularizers.l2(self.l2_reg))(y)
            y = layers.BatchNormalization()(y)
            y = layers.Activation(self.activation)(y)

        out = layers.Dense(input_dim, activation="linear", name="recon")(y)

        self.model = models.Model(inputs=inp, outputs=out)
        self.model.compile(optimizer="adam", loss=self.loss, metrics=["mse"])

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.1
    ):
        if isinstance(X, pd.DataFrame):
            self.features = feature_names or X.columns.tolist()
            X_arr = X[self.features].to_numpy(copy=False) if feature_names else X.to_numpy(copy=False)
        else:
            self.features = feature_names or []
            X_arr = X
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.scaler.fit_transform(X_arr)

        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
            keras.utils.set_random_seed(self.random_state)
        self.build_model(X_scaled.shape[1])
        assert self.model is not None, "Model not built."

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        )
        best_weights_path = "/tmp/ae_best.weights.h5"
        model_ckpt = callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        )

        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, model_ckpt, reduce_lr],
            verbose="auto"
        )
        if os.path.exists(best_weights_path):
            try:
                self.model.load_weights(best_weights_path)
            except Exception:
                pass

        errs = self._reconstruction_error_from_scaled(X_scaled)
        self.threshold = float(np.quantile(errs, 1.0 - float(self.contamination)))

        self.metadata = {
            "model_type": "Autoencoder",
            "encoding_dim": self.encoding_dim,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "loss": self.loss,
            "l2_reg": self.l2_reg,
            "noise_std": self.noise_std,
            "scaler": self.scaler_type,
            "contamination": float(self.contamination),
            "threshold": float(self.threshold),
            "n_features": len(self.features),
            "features": list(self.features),
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": int(X_arr.shape[0]),
            "final_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "threshold_source": "train_quantile",
            "random_state": self.random_state,
        }

    def reconstruction_error(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = self._ensure_array(X, use_features=True)
        X_scaled = cast(np.ndarray, self.scaler.transform(X_arr))
        return self._reconstruction_error_from_scaled(X_scaled)

    def get_aml_scores(self, X: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        errs = self.reconstruction_error(X)  # native (higher = more anomalous)
        # UI score 0..1
        e_min, e_max = float(errs.min()), float(errs.max())
        if (e_max - e_min) > 1e-12:
            scores = (errs - e_min) / (e_max - e_min)
        else:
            scores = np.zeros_like(errs)
        thr = self.threshold if self.threshold is not None else float(np.quantile(errs, 1.0 - float(self.contamination)))
        labels = (errs >= thr).astype(int)
        return scores, labels

    def save(self, file_path: str):
        assert self.model is not None, "Model not trained. Call fit() first."
        keras_path = file_path.replace(".pkl", "_keras.keras").replace(".joblib", "_keras.keras")
        self.model.save(keras_path)

        save_obj = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "features": self.features,
            "metadata": self.metadata,
            "threshold": self.threshold,
            "encoding_dim": self.encoding_dim,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "contamination": self.contamination,
            "loss": self.loss,
            "l2_reg": self.l2_reg,
            "noise_std": self.noise_std,
            "keras_model_path": keras_path,
            "random_state": self.random_state,
        }
        joblib.dump(save_obj, file_path)

    @classmethod
    def load(cls, file_path: str) -> "AutoencoderModel":
        data = joblib.load(file_path)

        instance = cls(
            encoding_dim=int(data.get("encoding_dim", 8)),
            hidden_layers=list(data.get("hidden_layers", [128, 64, 32])),
            activation=str(data.get("activation", "relu")),
            contamination=float(data.get("contamination", 0.01)),
            loss=str(data.get("loss", "mse")),
            l2_reg=float(data.get("l2_reg", 1e-4)),
            noise_std=float(data.get("noise_std", 0.02)),
            use_robust_scaler=(data.get("scaler_type", "standard") == "robust"),
            random_state=data.get("random_state", 42),
        )

        keras_model_path = data.get("keras_model_path")
        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(f"Keras model file not found: {keras_model_path}")

        loaded = keras.models.load_model(keras_model_path, compile=False)
        instance.model = cast(keras.Model, loaded)
        instance.scaler = data["scaler"]
        instance.features = list(data.get("features", []))
        instance.metadata = dict(data.get("metadata", {}))
        instance.threshold = data.get("threshold")
        return instance
