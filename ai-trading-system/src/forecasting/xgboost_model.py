from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostForecaster:
    """
    Simple XGBoost forecaster for direction prediction.
    Can be trained on historical features or use a simple rule-based fallback.
    """
    
    def __init__(self, model_path: str | None = None, retrain: bool = False) -> None:
        self.model_path = model_path
        self.model = None
        self.trained = False
        self.feature_names = ["rsi_14", "volume_surge_pct", "thesis_score", "ltp"]
        
        if XGBOOST_AVAILABLE and model_path and not retrain:
            try:
                self.model = xgb.Booster()
                self.model.load_model(model_path)
                self.trained = True
            except Exception:
                self.model = None
    
    def train(self, df: pd.DataFrame, target_col: str = "direction") -> None:
        """Train XGBoost model on historical data."""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available for training")
            return
        
        if self.model is None:
            self.model = xgb.Booster()
        
        # Prepare features
        X = df[self.feature_names].fillna(0)
        
        # Prepare target
        y = df[target_col].astype(int)
        
        # Simple train/test split
        split_idx = int(0.8 * len(df))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train
        self.model = xgb.Booster()
        self.model.set_params({
            "objective": "multi:softprob",
            "num_class": 3,  # DOWN, NEUTRAL, UP
            "max_depth": 3,
            "eta": 0.1,
        })
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model.train(dtrain, num_boost_round=50)
        
        # Evaluate
        dtest = xgb.DMatrix(X_test, label=y_test)
        preds = self.model.predict(dtest)
        pred_labels = np.argmax(preds, axis=1)
        accuracy = (preds.argmax(axis=1) == y_test).mean()
        
        self.trained = True
        print(f"XGBoost trained. Accuracy: {accuracy:.4f}")
    
    def predict(self, features: Dict[str, float]) -> Dict[str, any]:
        """Predict direction given feature values."""
        if not self.trained or self.model is None:
            # Rule-based fallback
            rsi = features.get("rsi_14", 50.0)
            thesis = features.get("thesis_score", 0.0)
            
            if rsi < 30 and thesis > 0.5:
                direction = "UP"
                confidence = 0.7
                magnitude = 2.0
            elif rsi > 70 and thesis < -0.5:
                direction = "DOWN"
                confidence = 0.7
                magnitude = 2.0
            else:
                direction = "NEUTRAL"
                confidence = 0.5
                magnitude = 0.0
        else:
            # XGBoost prediction
            dmatrix = xgb.DMatrix(pd.DataFrame([features], columns=self.feature_names))
            preds = self.model.predict(dmatrix)
            pred_idx = np.argmax(preds, axis=1)[0]
            
            direction_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
            direction = direction_map.get(pred_idx, "NEUTRAL")
            confidence = abs(preds[0][pred_idx] - 0.5) * 2
            magnitude = abs(preds[0][pred_idx] - 0.5) * 4
        
        return {
            "direction": direction,
            "confidence": round min(confidence, 1.0), 3),
            "magnitude": round(magnitude, 2),
            "model_version": "xgb_fallback" if not self.trained else "xgb_trained",
        }