"""
Data loading, cleaning, and integration pipeline.
Can be re-run on new raw Excel exports in the same format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

RAW_FILES = {
    'corrective': 'Ordres de travail.xlsx',
    'preventive': 'OT_Preventifs_20-01-2026 21_44_04.xlsx',
    'spare_parts': 'a-w.xlsx',
}


def load_raw_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all three raw Excel files."""
    raw_dir = Path(raw_dir)
    data = {}
    for key, filename in RAW_FILES.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            # Try to find any xlsx file matching the pattern
            candidates = list(raw_dir.glob('*.xlsx'))
            raise FileNotFoundError(
                f"Expected '{filename}' in {raw_dir}. Found: {[c.name for c in candidates]}"
            )
        data[key] = pd.read_excel(filepath)
        print(f"  Loaded {key}: {data[key].shape}")
    return data


def clean_corrective(df: pd.DataFrame) -> pd.DataFrame:
    """Clean corrective maintenance data."""
    df = df.copy()

    # Fix encoding issues
    if 'etat' in df.columns:
        df['etat'] = df['etat'].str.replace('Clotur\ufffd', 'Cloture', regex=False)

    # Parse dates
    date_cols = [
        'date_declaration', 'date_creation_ot', 'date_debut_reparation',
        'date_cloture', 'date_reouverture', 'date_validation', 'date_rejet',
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce')

    # Boolean columns
    bool_cols = ['vandalisme', 'palliative', 'avec_demande_pdr', 'est_reouvert']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'Oui': True, 'oui': True, 'Non': False, 'non': False})

    # Fill missing costs with 0
    for col in ['cout_maindoeuvre', 'cout_pdr', 'cout_bc', 'cout_total']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Derived time features
    if 'date_declaration' in df.columns and 'date_cloture' in df.columns:
        df['time_to_closure_hours'] = (
            (df['date_cloture'] - df['date_declaration']).dt.total_seconds() / 3600
        )
    if 'date_creation_ot' in df.columns and 'date_debut_reparation' in df.columns:
        df['time_to_start_hours'] = (
            (df['date_debut_reparation'] - df['date_creation_ot']).dt.total_seconds() / 3600
        )

    return df


def clean_preventive(df: pd.DataFrame) -> pd.DataFrame:
    """Clean preventive maintenance data."""
    df = df.copy()

    date_cols = [
        'date_debut_planifiee', 'date_fin_planifiee', 'date_debut',
        'date_fin', 'date_cloture', 'date_validation',
    ]
    for col in date_cols:
        if col in df.columns:
            if df[col].dtype != 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'cout' in df.columns:
        df['cout'] = df['cout'].fillna(0)

    if 'date_debut_planifiee' in df.columns and 'date_debut' in df.columns:
        df['days_deviation_from_plan'] = (
            (df['date_debut'] - df['date_debut_planifiee']).dt.total_seconds() / 86400
        )

    if 'date_debut' in df.columns and 'date_fin' in df.columns:
        df['intervention_duration_hours'] = (
            (df['date_fin'] - df['date_debut']).dt.total_seconds() / 3600
        )

    return df


def clean_spare_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Clean spare parts data."""
    df = df.copy()

    date_cols = ['date_demande', 'date_validation_superviseur', 'date_cloture', 'date_rejet']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y, %H:%M:%S', errors='coerce')

    if 'quantite_demande' in df.columns:
        df['quantite_demande'] = df['quantite_demande'].fillna(0)
    if 'cout_total' in df.columns:
        df['cout_total'] = df['cout_total'].fillna(0)

    if 'cout_total' in df.columns and 'quantite_demande' in df.columns:
        df['cout_unitaire'] = df.apply(
            lambda r: r['cout_total'] / r['quantite_demande'] if r['quantite_demande'] > 0 else 0,
            axis=1,
        )

    if 'date_demande' in df.columns and 'date_validation_superviseur' in df.columns:
        df['validation_time_hours'] = (
            (df['date_validation_superviseur'] - df['date_demande']).dt.total_seconds() / 3600
        )

    if 'date_validation_superviseur' in df.columns and 'date_cloture' in df.columns:
        df['fulfillment_time_hours'] = (
            (df['date_cloture'] - df['date_validation_superviseur']).dt.total_seconds() / 3600
        )

    return df


def integrate_datasets(
    corrective: pd.DataFrame,
    preventive: pd.DataFrame,
    spare_parts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Integrate the three cleaned datasets. Returns dict of integrated dataframes."""

    # Aggregate spare parts per work order
    sp_agg = spare_parts.groupby('id_ot').agg({
        'id_demande': 'count',
        'quantite_demande': 'sum',
        'cout_total': 'sum',
        'article': lambda x: ', '.join(x.astype(str).unique()[:5]),
        'famille': lambda x: ', '.join(x.astype(str).unique()),
        'etat': lambda x: x.mode()[0] if len(x) > 0 else None,
    }).reset_index()
    sp_agg.columns = [
        'id_ordre_travail', 'num_spare_parts_requests', 'total_parts_quantity',
        'total_parts_cost', 'parts_articles', 'parts_families', 'parts_request_status',
    ]

    corrective_integrated = corrective.merge(sp_agg, on='id_ordre_travail', how='left')
    corrective_integrated['num_spare_parts_requests'] = corrective_integrated['num_spare_parts_requests'].fillna(0).astype(int)
    corrective_integrated['total_parts_quantity'] = corrective_integrated['total_parts_quantity'].fillna(0)
    corrective_integrated['total_parts_cost'] = corrective_integrated['total_parts_cost'].fillna(0)

    # Asset master
    corr_assets = corrective[['id_actif', 'actif', 'famille', 'groupe', 'site', 'zone', 'localisation']].copy()
    corr_assets['source'] = 'corrective'
    prev_assets = preventive[['id_actif', 'actif', 'famille', 'groupe', 'site', 'zone', 'localisation']].copy()
    prev_assets['source'] = 'preventive'

    all_assets = pd.concat([corr_assets, prev_assets], ignore_index=True)
    asset_master = all_assets.groupby('id_actif').agg({
        'actif': lambda x: x.mode()[0] if len(x) > 0 else None,
        'famille': lambda x: x.mode()[0] if len(x) > 0 else None,
        'groupe': lambda x: x.mode()[0] if len(x) > 0 else None,
        'site': lambda x: x.mode()[0] if len(x) > 0 else None,
        'zone': lambda x: x.mode()[0] if len(x) > 0 else None,
        'localisation': lambda x: x.mode()[0] if len(x) > 0 else None,
        'source': lambda x: ', '.join(x.unique()),
    }).reset_index()

    corr_counts = corrective.groupby('id_actif').size().reset_index(name='corrective_count')
    prev_counts = preventive.groupby('id_actif').size().reset_index(name='preventive_count')
    asset_master = asset_master.merge(corr_counts, on='id_actif', how='left')
    asset_master = asset_master.merge(prev_counts, on='id_actif', how='left')
    asset_master['corrective_count'] = asset_master['corrective_count'].fillna(0).astype(int)
    asset_master['preventive_count'] = asset_master['preventive_count'].fillna(0).astype(int)
    asset_master['total_maintenance_count'] = asset_master['corrective_count'] + asset_master['preventive_count']

    # Unified timeline
    corr_tl = corrective_integrated[[
        'id_ordre_travail', 'date_creation_ot', 'date_cloture', 'etat', 'anomalie',
        'urgence', 'site', 'zone', 'famille', 'actif', 'id_actif',
        'cout_total', 'duree_intervention_par_heure', 'prestataire',
        'num_spare_parts_requests', 'total_parts_cost',
    ]].copy()
    corr_tl['maintenance_type'] = 'corrective'
    corr_tl.rename(columns={
        'id_ordre_travail': 'order_id',
        'date_creation_ot': 'date_created',
        'anomalie': 'description',
    }, inplace=True)

    prev_tl = preventive[[
        'numero_ot', 'date_debut_planifiee', 'date_cloture', 'etat', 'nom_intervention',
        'frequence', 'site', 'zone', 'famille', 'actif', 'id_actif',
        'cout', 'prestataire',
    ]].copy()
    prev_tl['maintenance_type'] = 'preventive'
    prev_tl['urgence'] = None
    prev_tl['num_spare_parts_requests'] = 0
    prev_tl['total_parts_cost'] = 0
    if 'intervention_duration_hours' in preventive.columns:
        prev_tl['duree_intervention_par_heure'] = preventive['intervention_duration_hours']
    else:
        prev_tl['duree_intervention_par_heure'] = np.nan
    prev_tl.rename(columns={
        'numero_ot': 'order_id',
        'date_debut_planifiee': 'date_created',
        'nom_intervention': 'description',
        'cout': 'cout_total',
    }, inplace=True)

    unified = pd.concat([corr_tl, prev_tl], ignore_index=True, sort=False)
    unified = unified.sort_values('date_created')

    # Location hierarchy
    corr_locs = corrective[['site', 'zone', 'localisation']].drop_duplicates()
    prev_locs = preventive[['site', 'zone', 'localisation']].drop_duplicates()
    all_locs = pd.concat([corr_locs, prev_locs], ignore_index=True).drop_duplicates()
    loc_hierarchy = all_locs.groupby(['site', 'zone']).size().reset_index(name='location_count')

    corr_by_loc = corrective.groupby(['site', 'zone']).size().reset_index(name='corrective_count')
    prev_by_loc = preventive.groupby(['site', 'zone']).size().reset_index(name='preventive_count')
    loc_hierarchy = loc_hierarchy.merge(corr_by_loc, on=['site', 'zone'], how='left')
    loc_hierarchy = loc_hierarchy.merge(prev_by_loc, on=['site', 'zone'], how='left')
    loc_hierarchy['corrective_count'] = loc_hierarchy['corrective_count'].fillna(0).astype(int)
    loc_hierarchy['preventive_count'] = loc_hierarchy['preventive_count'].fillna(0).astype(int)
    loc_hierarchy['total_maintenance'] = loc_hierarchy['corrective_count'] + loc_hierarchy['preventive_count']

    return {
        'corrective_integrated': corrective_integrated,
        'asset_master': asset_master,
        'unified_timeline': unified,
        'location_hierarchy': loc_hierarchy,
    }


def save_processed(
    corrective: pd.DataFrame,
    preventive: pd.DataFrame,
    spare_parts: pd.DataFrame,
    integrated: dict[str, pd.DataFrame],
    output_dir: Path,
):
    """Save all cleaned and integrated data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enc = 'utf-8-sig'
    corrective.to_csv(output_dir / 'corrective_cleaned.csv', index=False, encoding=enc)
    corrective.to_pickle(output_dir / 'corrective_cleaned.pkl')
    preventive.to_csv(output_dir / 'preventive_cleaned.csv', index=False, encoding=enc)
    preventive.to_pickle(output_dir / 'preventive_cleaned.pkl')
    spare_parts.to_csv(output_dir / 'spare_parts_cleaned.csv', index=False, encoding=enc)
    spare_parts.to_pickle(output_dir / 'spare_parts_cleaned.pkl')

    for name, df in integrated.items():
        df.to_csv(output_dir / f'{name}.csv', index=False, encoding=enc)
        df.to_pickle(output_dir / f'{name}.pkl')

    print(f"  Saved all processed data to {output_dir}")


def run_preprocessing_pipeline(raw_dir: Path, output_dir: Path):
    """Full preprocessing pipeline: load -> clean -> integrate -> save."""
    print("Step 1: Loading raw data...")
    raw = load_raw_data(raw_dir)

    print("Step 2: Cleaning datasets...")
    corrective = clean_corrective(raw['corrective'])
    preventive = clean_preventive(raw['preventive'])
    spare_parts = clean_spare_parts(raw['spare_parts'])
    print(f"  Corrective: {corrective.shape}, Preventive: {preventive.shape}, Spare parts: {spare_parts.shape}")

    print("Step 3: Integrating datasets...")
    integrated = integrate_datasets(corrective, preventive, spare_parts)
    for name, df in integrated.items():
        print(f"  {name}: {df.shape}")

    print("Step 4: Saving processed data...")
    save_processed(corrective, preventive, spare_parts, integrated, output_dir)

    return corrective, preventive, spare_parts, integrated
