"""
Feature engineering pipeline for GMAO maintenance data.
Creates 95+ features and 5 target variables from corrective maintenance data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import warnings

warnings.filterwarnings('ignore')


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal components from date_creation_ot."""
    df = df.copy()
    if 'date_creation_ot' not in df.columns:
        return df

    dt = df['date_creation_ot']
    df['creation_hour'] = dt.dt.hour
    df['creation_day_of_week'] = dt.dt.dayofweek
    df['creation_day_of_month'] = dt.dt.day
    df['creation_month'] = dt.dt.month
    df['creation_quarter'] = dt.dt.quarter
    df['creation_is_weekend'] = df['creation_day_of_week'].isin([5, 6]).astype(int)
    df['creation_is_business_hours'] = ((df['creation_hour'] >= 8) & (df['creation_hour'] < 18)).astype(int)

    # Time since last failure per asset
    df = df.sort_values(['id_actif', 'date_creation_ot'])
    df['days_since_last_maintenance'] = (
        df.groupby('id_actif')['date_creation_ot'].diff().dt.total_seconds() / 86400
    )
    median_days = df['days_since_last_maintenance'].median()
    df['days_since_last_maintenance'] = df['days_since_last_maintenance'].fillna(median_days)

    return df


def create_equipment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Equipment-level aggregated features."""
    df = df.copy()

    for col_name, agg_col, agg_func in [
        ('equipment_failure_count', 'id_actif', 'size'),
        ('equipment_avg_cost', 'cout_total', 'mean'),
        ('equipment_avg_duration', 'duree_intervention_par_heure', 'mean'),
    ]:
        if agg_func == 'size':
            agg = df.groupby('id_actif').size().reset_index(name=col_name)
        else:
            agg = df.groupby('id_actif')[agg_col].mean().reset_index(name=col_name)
        df = df.merge(agg, on='id_actif', how='left')

    # Family-level stats
    fam_counts = df.groupby('famille').size().reset_index(name='famille_failure_count')
    df = df.merge(fam_counts, on='famille', how='left')
    fam_cost = df.groupby('famille')['cout_total'].mean().reset_index(name='famille_avg_cost')
    df = df.merge(fam_cost, on='famille', how='left')

    # Equipment age proxy
    first_maint = df.groupby('id_actif')['date_creation_ot'].min().reset_index()
    first_maint.columns = ['id_actif', 'first_maintenance_date']
    df = df.merge(first_maint, on='id_actif', how='left')
    df['equipment_age_days'] = (
        (df['date_creation_ot'] - df['first_maintenance_date']).dt.total_seconds() / 86400
    )
    df['equipment_age_days'] = df['equipment_age_days'].fillna(0)

    return df


def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Site and zone aggregated features."""
    df = df.copy()
    site_counts = df.groupby('site').size().reset_index(name='site_failure_count')
    df = df.merge(site_counts, on='site', how='left')
    site_cost = df.groupby('site')['cout_total'].mean().reset_index(name='site_avg_cost')
    df = df.merge(site_cost, on='site', how='left')
    zone_counts = df.groupby(['site', 'zone']).size().reset_index(name='zone_failure_count')
    df = df.merge(zone_counts, on=['site', 'zone'], how='left')
    return df


def create_spare_parts_features(df: pd.DataFrame) -> pd.DataFrame:
    """Spare parts usage features."""
    df = df.copy()
    df['has_spare_parts'] = (df['num_spare_parts_requests'] > 0).astype(int)
    df['spare_parts_cost_ratio'] = df.apply(
        lambda r: r['total_parts_cost'] / r['cout_total'] if r['cout_total'] > 0 else 0, axis=1
    )
    fam_sp = df.groupby('famille')['has_spare_parts'].mean().reset_index(name='famille_spare_parts_usage_rate')
    df = df.merge(fam_sp, on='famille', how='left')
    return df


def create_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary anomaly type features + recurrence."""
    df = df.copy()
    top_anomalies = df['anomalie'].value_counts().head(20).index
    for anomaly in top_anomalies:
        col_name = f"anomaly_{anomaly.replace(' ', '_').lower()}"[:50]
        df[col_name] = (df['anomalie'] == anomaly).astype(int)

    anom_counts = df.groupby(['id_actif', 'anomalie']).size().reset_index(name='anomaly_recurrence_count')
    df = df.merge(
        anom_counts.groupby('id_actif')['anomaly_recurrence_count'].max().reset_index(),
        on='id_actif', how='left',
    )
    return df


def create_urgency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Urgency one-hot + historical urgency rate."""
    df = df.copy()
    if 'urgence' in df.columns:
        urgency_dummies = pd.get_dummies(df['urgence'], prefix='urgence')
        df = pd.concat([df, urgency_dummies], axis=1)
        fam_urg = df.groupby('famille').apply(
            lambda x: (x['urgence'] == 'Fort').sum() / len(x) if len(x) > 0 else 0
        ).reset_index(name='famille_high_urgency_rate')
        df = df.merge(fam_urg, on='famille', how='left')
    return df


def create_provider_features(df: pd.DataFrame) -> pd.DataFrame:
    """Service provider performance features."""
    df = df.copy()
    prov_time = df.groupby('prestataire')['time_to_closure_hours'].mean().reset_index(name='provider_avg_time_to_closure')
    df = df.merge(prov_time, on='prestataire', how='left')
    prov_cost = df.groupby('prestataire')['cout_total'].mean().reset_index(name='provider_avg_cost')
    df = df.merge(prov_cost, on='prestataire', how='left')
    prov_vol = df.groupby('prestataire').size().reset_index(name='provider_work_volume')
    df = df.merge(prov_vol, on='prestataire', how='left')
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling window aggregations."""
    df = df.copy().sort_values(['id_actif', 'date_creation_ot'])
    df['rolling_7d_failures'] = df.groupby('id_actif')['id_ordre_travail'].transform(
        lambda x: x.rolling(window=7, min_periods=1).count()
    )
    df['rolling_30d_failures'] = df.groupby('id_actif')['id_ordre_travail'].transform(
        lambda x: x.rolling(window=30, min_periods=1).count()
    )
    df['rolling_7d_avg_cost'] = df.groupby('id_actif')['cout_total'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df['rolling_30d_avg_cost'] = df.groupby('id_actif')['cout_total'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    return df


def encode_categoricals(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Label-encode high-cardinality categoricals."""
    df = df.copy()
    cat_cols = ['portfolio', 'site', 'zone', 'famille', 'groupe', 'prestataire', 'etat']
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    return df, encoders


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables for ML models."""
    df = df.copy().sort_values(['id_actif', 'date_creation_ot'])

    # Binary: will equipment fail again within 30 days?
    df['next_failure_date'] = df.groupby('id_actif')['date_creation_ot'].shift(-1)
    df['days_to_next_failure'] = (
        (df['next_failure_date'] - df['date_creation_ot']).dt.total_seconds() / 86400
    )
    df['will_fail_within_30_days'] = (df['days_to_next_failure'] <= 30).fillna(False).astype(int)

    # Multi-class: anomaly type
    df['anomaly_type_encoded'] = LabelEncoder().fit_transform(df['anomalie'].astype(str))

    # Regression targets
    df['spare_parts_cost_target'] = df['total_parts_cost']
    df['intervention_duration_target'] = df['duree_intervention_par_heure']

    # Binary: will order require spare parts?
    df['will_need_spare_parts'] = df['has_spare_parts']

    return df


def save_features(df: pd.DataFrame, output_dir: Path):
    """Save final feature dataset with metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drop columns not needed for modeling
    drop_cols = [
        'date_declaration', 'date_creation_ot', 'date_debut_reparation', 'date_cloture',
        'date_reouverture', 'date_validation', 'date_rejet', 'commentaire_cloture',
        'commentaire_ouverture', 'description', 'motif_rejet', 'motif_standby',
        'responsable_cloture', 'responsable_reouverture', 'responsable_rejet',
        'validateur_travaux', 'createur_ot', 'demandeur', 'utilisateur_prestataire',
        'time_since_last_failure', 'first_maintenance_date', 'next_failure_date',
        'parts_articles', 'actif', 'localisation', 'anomalie', 'plateforme_creation',
        'plateforme_prestataire', 'plateforme_cloture', 'plateforme_rejet',
        'plateforme_reouverture', 'plateforme_validation',
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    features_df = df.drop(columns=existing_drops)

    features_df.to_csv(output_dir / 'corrective_features.csv', index=False, encoding='utf-8-sig')
    features_df.to_pickle(output_dir / 'corrective_features.pkl')

    numerical = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = features_df.select_dtypes(include=['object']).columns.tolist()
    targets = [
        'will_fail_within_30_days', 'anomaly_type_encoded',
        'spare_parts_cost_target', 'intervention_duration_target', 'will_need_spare_parts',
    ]

    metadata = {
        'total_samples': len(features_df),
        'total_features': features_df.shape[1],
        'numerical_features': numerical,
        'categorical_features': categorical,
        'target_variables': targets,
        'created_date': datetime.now().isoformat(),
    }
    with open(output_dir / 'feature_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved features: {features_df.shape} to {output_dir}")
    return features_df


def run_feature_pipeline(processed_dir: Path, features_dir: Path) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    processed_dir = Path(processed_dir)
    features_dir = Path(features_dir)

    print("Loading processed data...")
    try:
        df = pd.read_pickle(processed_dir / 'corrective_integrated.pkl')
    except Exception:
        df = pd.read_csv(processed_dir / 'corrective_integrated.csv')
        for col in ['date_creation_ot', 'date_cloture', 'date_declaration', 'date_debut_reparation']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')

    print("Creating features...")
    df = create_time_features(df)
    df = create_equipment_features(df)
    df = create_location_features(df)
    df = create_spare_parts_features(df)
    df = create_anomaly_features(df)
    df = create_urgency_features(df)
    df = create_provider_features(df)
    df = create_rolling_features(df)
    df, _ = encode_categoricals(df, features_dir)
    df = create_targets(df)

    print("Saving feature dataset...")
    features_df = save_features(df, features_dir)

    return features_df
