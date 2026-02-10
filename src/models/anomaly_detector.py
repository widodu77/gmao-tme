"""
Anomaly detection for maintenance orders.
Uses Isolation Forest + One-Class SVM with consensus scoring.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


ANOMALY_FEATURES = [
    'cout_total', 'duree_intervention_par_heure', 'time_to_closure_hours',
    'time_to_start_hours', 'num_spare_parts_requests', 'total_parts_cost',
    'equipment_failure_count', 'equipment_avg_cost', 'days_since_last_maintenance',
    'equipment_age_days', 'site_failure_count', 'zone_failure_count',
    'rolling_7d_failures', 'rolling_30d_failures', 'rolling_7d_avg_cost',
    'rolling_30d_avg_cost',
]


class AnomalyDetector:
    """Detect anomalies in corrective maintenance using ensemble methods."""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            contamination=contamination, random_state=42,
            n_estimators=100, n_jobs=-1,
        )
        self.ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=contamination)
        self.features = None
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit both anomaly detection models."""
        self.features = [f for f in ANOMALY_FEATURES if f in df.columns]
        X = df[self.features].copy()
        X = X.fillna(X.median()).replace([np.inf, -np.inf], np.nan).fillna(X.median())

        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        self.ocsvm.fit(X_scaled)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies and return df with added columns."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        X = df[self.features].copy()
        X = X.fillna(X.median()).replace([np.inf, -np.inf], np.nan).fillna(X.median())
        X_scaled = self.scaler.transform(X)

        result = df.copy()
        iso_pred = self.iso_forest.predict(X_scaled)
        result['iso_anomaly'] = (iso_pred == -1).astype(int)
        result['iso_score'] = self.iso_forest.score_samples(X_scaled)

        ocsvm_pred = self.ocsvm.predict(X_scaled)
        result['ocsvm_anomaly'] = (ocsvm_pred == -1).astype(int)
        result['ocsvm_score'] = self.ocsvm.decision_function(X_scaled)

        result['anomaly_consensus'] = (
            (result['iso_anomaly'] == 1) | (result['ocsvm_anomaly'] == 1)
        ).astype(int)
        result['anomaly_confidence'] = result['iso_anomaly'] + result['ocsvm_anomaly']

        return result

    def evaluate(self, df_with_predictions: pd.DataFrame) -> dict:
        """Return summary statistics of detected anomalies."""
        df = df_with_predictions
        n = len(df)
        iso_n = df['iso_anomaly'].sum()
        ocsvm_n = df['ocsvm_anomaly'].sum()
        consensus_n = df['anomaly_consensus'].sum()
        high_conf = (df['anomaly_confidence'] == 2).sum()
        agreement = (df['iso_anomaly'] == df['ocsvm_anomaly']).mean() * 100

        normal = df[df['anomaly_consensus'] == 0]
        anomalous = df[df['anomaly_consensus'] == 1]

        stats = {}
        for feat in ['cout_total', 'duree_intervention_par_heure', 'time_to_closure_hours']:
            if feat in df.columns:
                n_mean = normal[feat].mean()
                a_mean = anomalous[feat].mean()
                diff_pct = ((a_mean - n_mean) / n_mean * 100) if n_mean != 0 else 0
                stats[feat] = {
                    'normal_mean': round(n_mean, 2),
                    'anomaly_mean': round(a_mean, 2),
                    'difference_pct': round(diff_pct, 1),
                }

        return {
            'total_samples': n,
            'isolation_forest_anomalies': int(iso_n),
            'ocsvm_anomalies': int(ocsvm_n),
            'consensus_anomalies': int(consensus_n),
            'high_confidence_anomalies': int(high_conf),
            'agreement_pct': round(agreement, 2),
            'feature_comparison': stats,
        }

    def save(self, models_dir: Path):
        """Save trained models to disk."""
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        with open(models_dir / 'isolation_forest_model.pkl', 'wb') as f:
            pickle.dump(self.iso_forest, f)
        with open(models_dir / 'ocsvm_model.pkl', 'wb') as f:
            pickle.dump(self.ocsvm, f)
        with open(models_dir / 'anomaly_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(models_dir / 'anomaly_features.txt', 'w') as f:
            f.write('\n'.join(self.features))

    @classmethod
    def load(cls, models_dir: Path) -> 'AnomalyDetector':
        """Load trained models from disk."""
        models_dir = Path(models_dir)
        detector = cls.__new__(cls)
        with open(models_dir / 'isolation_forest_model.pkl', 'rb') as f:
            detector.iso_forest = pickle.load(f)
        with open(models_dir / 'ocsvm_model.pkl', 'rb') as f:
            detector.ocsvm = pickle.load(f)
        with open(models_dir / 'anomaly_scaler.pkl', 'rb') as f:
            detector.scaler = pickle.load(f)
        with open(models_dir / 'anomaly_features.txt', 'r') as f:
            detector.features = [line.strip() for line in f if line.strip()]
        detector._fitted = True
        detector.contamination = 0.1
        return detector
