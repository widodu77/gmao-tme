"""
Supervised model for predicting equipment failure within 30 days
and whether an order will require spare parts.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


# Features to use for prediction (numerical only, exclude targets and IDs)
EXCLUDE_COLS = {
    'id_ordre_travail', 'id_demande', 'id_actif',
    'will_fail_within_30_days', 'will_need_spare_parts',
    'anomaly_type_encoded', 'spare_parts_cost_target',
    'intervention_duration_target', 'days_to_next_failure',
    'iso_anomaly', 'iso_score', 'ocsvm_anomaly', 'ocsvm_score',
    'anomaly_consensus', 'anomaly_confidence',
}


class FailurePredictor:
    """Predict equipment failure within 30 days using Random Forest."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=10,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
        self._fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select numerical features, excluding targets and IDs."""
        numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numerical if c not in EXCLUDE_COLS]
        return features

    def fit(self, df: pd.DataFrame, target: str = 'will_fail_within_30_days'):
        """Train the model. Returns train/test metrics dict."""
        self.feature_cols = self._prepare_features(df)
        X = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df[target].astype(int)

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y,
        )

        self.model.fit(X_train, y_train)
        self._fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='roc_auc')

        metrics = {
            'target': target,
            'n_samples': len(df),
            'n_features': len(self.feature_cols),
            'positive_rate': float(y.mean()),
            'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'test_f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
            'cv_roc_auc_mean': float(cv_scores.mean()),
            'cv_roc_auc_std': float(cv_scores.std()),
            'classification_report': classification_report(
                y_test, y_pred, target_names=['No Failure', 'Failure'], output_dict=True,
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': dict(
                sorted(
                    zip(self.feature_cols, self.model.feature_importances_),
                    key=lambda x: x[1], reverse=True,
                )[:15]
            ),
        }

        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prediction columns to dataframe."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)

        result = df.copy()
        result['failure_prediction'] = self.model.predict(X_scaled)
        result['failure_probability'] = self.model.predict_proba(X_scaled)[:, 1]
        return result

    def save(self, models_dir: Path):
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(models_dir / 'failure_predictor_model.pkl', 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'features': self.feature_cols}, f)

    @classmethod
    def load(cls, models_dir: Path) -> 'FailurePredictor':
        models_dir = Path(models_dir)
        predictor = cls.__new__(cls)
        with open(models_dir / 'failure_predictor_model.pkl', 'rb') as f:
            data = pickle.load(f)
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_cols = data['features']
        predictor._fitted = True
        return predictor


class SparePartsPredictor:
    """Predict whether a maintenance order will require spare parts."""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=150, max_depth=8, learning_rate=0.1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
        self._fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> list[str]:
        numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        extra_exclude = EXCLUDE_COLS | {
            'has_spare_parts', 'spare_parts_cost_ratio',
            'num_spare_parts_requests', 'total_parts_quantity', 'total_parts_cost',
        }
        return [c for c in numerical if c not in extra_exclude]

    def fit(self, df: pd.DataFrame) -> dict:
        """Train spare parts prediction model."""
        self.feature_cols = self._prepare_features(df)
        X = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df['will_need_spare_parts'].astype(int)

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y,
        )

        self.model.fit(X_train, y_train)
        self._fitted = True

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='roc_auc')

        return {
            'target': 'will_need_spare_parts',
            'n_samples': len(df),
            'n_features': len(self.feature_cols),
            'positive_rate': float(y.mean()),
            'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'test_f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
            'cv_roc_auc_mean': float(cv_scores.mean()),
            'cv_roc_auc_std': float(cv_scores.std()),
            'classification_report': classification_report(
                y_test, y_pred, target_names=['No Parts', 'Needs Parts'], output_dict=True,
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': dict(
                sorted(
                    zip(self.feature_cols, self.model.feature_importances_),
                    key=lambda x: x[1], reverse=True,
                )[:15]
            ),
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)
        result = df.copy()
        result['spare_parts_prediction'] = self.model.predict(X_scaled)
        result['spare_parts_probability'] = self.model.predict_proba(X_scaled)[:, 1]
        return result

    def save(self, models_dir: Path):
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(models_dir / 'spare_parts_predictor_model.pkl', 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'features': self.feature_cols}, f)

    @classmethod
    def load(cls, models_dir: Path) -> 'SparePartsPredictor':
        models_dir = Path(models_dir)
        predictor = cls.__new__(cls)
        with open(models_dir / 'spare_parts_predictor_model.pkl', 'rb') as f:
            data = pickle.load(f)
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_cols = data['features']
        predictor._fitted = True
        return predictor
