"""
End-to-end pipeline: raw data -> cleaned data -> features -> models -> reports.
Run this script to reproduce all results from scratch.

Usage:
    python run_pipeline.py           # Full pipeline
    python run_pipeline.py --step 1  # Only preprocessing
    python run_pipeline.py --step 2  # Only feature engineering
    python run_pipeline.py --step 3  # Only model training
    python run_pipeline.py --step 4  # Only effectiveness & spare parts analysis
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def step1_preprocessing():
    """Step 1: Load raw data, clean, integrate, save processed."""
    from src.preprocessing import run_preprocessing_pipeline

    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPROCESSING & INTEGRATION")
    print("=" * 70)

    raw_dir = PROJECT_ROOT / 'data' / 'raw'
    output_dir = PROJECT_ROOT / 'data' / 'processed'

    run_preprocessing_pipeline(raw_dir, output_dir)
    print("Step 1 complete.\n")


def step2_features():
    """Step 2: Feature engineering."""
    from src.feature_engineering import run_feature_pipeline

    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    features_dir = PROJECT_ROOT / 'data' / 'features'

    features_df = run_feature_pipeline(processed_dir, features_dir)
    print(f"Step 2 complete. Features shape: {features_df.shape}\n")


def step3_models():
    """Step 3: Train all ML models."""
    import pandas as pd
    from src.models.anomaly_detector import AnomalyDetector
    from src.models.failure_predictor import FailurePredictor, SparePartsPredictor

    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)

    features_dir = PROJECT_ROOT / 'data' / 'features'
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(features_dir / 'corrective_features.csv')

    # --- Anomaly Detection ---
    print("\n--- Anomaly Detection ---")
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(df)
    df_with_anomalies = detector.predict(df)
    eval_results = detector.evaluate(df_with_anomalies)
    detector.save(models_dir)
    print(f"  Consensus anomalies: {eval_results['consensus_anomalies']} "
          f"({eval_results['consensus_anomalies']/eval_results['total_samples']*100:.1f}%)")
    print(f"  Method agreement: {eval_results['agreement_pct']:.1f}%")

    # Save anomaly predictions
    df_with_anomalies.to_csv(models_dir / 'corrective_with_anomalies.csv', index=False)

    # --- Failure Prediction ---
    print("\n--- Failure Prediction (will_fail_within_30_days) ---")
    failure_pred = FailurePredictor()
    failure_metrics = failure_pred.fit(df)
    failure_pred.save(models_dir)
    print(f"  Precision: {failure_metrics['test_precision']:.3f}")
    print(f"  Recall:    {failure_metrics['test_recall']:.3f}")
    print(f"  F1 Score:  {failure_metrics['test_f1']:.3f}")
    print(f"  ROC AUC:   {failure_metrics['test_roc_auc']:.3f}")
    print(f"  CV AUC:    {failure_metrics['cv_roc_auc_mean']:.3f} (+/- {failure_metrics['cv_roc_auc_std']:.3f})")

    # --- Spare Parts Prediction ---
    print("\n--- Spare Parts Prediction (will_need_spare_parts) ---")
    parts_pred = SparePartsPredictor()
    parts_metrics = parts_pred.fit(df)
    parts_pred.save(models_dir)
    print(f"  Precision: {parts_metrics['test_precision']:.3f}")
    print(f"  Recall:    {parts_metrics['test_recall']:.3f}")
    print(f"  F1 Score:  {parts_metrics['test_f1']:.3f}")
    print(f"  ROC AUC:   {parts_metrics['test_roc_auc']:.3f}")
    print(f"  CV AUC:    {parts_metrics['cv_roc_auc_mean']:.3f} (+/- {parts_metrics['cv_roc_auc_std']:.3f})")

    # Save all metrics
    all_metrics = {
        'anomaly_detection': eval_results,
        'failure_prediction': failure_metrics,
        'spare_parts_prediction': parts_metrics,
    }
    with open(models_dir / 'model_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\nAll models and metrics saved to {models_dir}")
    print("Step 3 complete.\n")


def step4_analysis():
    """Step 4: Effectiveness analysis and spare parts forecasting."""
    from src.models.effectiveness import run_effectiveness_analysis
    from src.models.spare_parts_forecaster import run_spare_parts_analysis

    print("\n" + "=" * 70)
    print("STEP 4: EFFECTIVENESS & SPARE PARTS ANALYSIS")
    print("=" * 70)

    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    reports_dir = PROJECT_ROOT / 'reports'

    print("\n--- Maintenance Effectiveness ---")
    run_effectiveness_analysis(processed_dir, reports_dir)

    print("\n--- Spare Parts Analysis ---")
    run_spare_parts_analysis(processed_dir, reports_dir)

    print("Step 4 complete.\n")


def main():
    parser = argparse.ArgumentParser(description='GMAO AI Pipeline')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4],
                        help='Run a specific step (1-4). Omit to run all.')
    args = parser.parse_args()

    print("=" * 70)
    print("GMAO AI PROJECT - FULL PIPELINE")
    print("=" * 70)

    steps = {
        1: step1_preprocessing,
        2: step2_features,
        3: step3_models,
        4: step4_analysis,
    }

    if args.step:
        steps[args.step]()
    else:
        for step_num, step_func in steps.items():
            step_func()

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
