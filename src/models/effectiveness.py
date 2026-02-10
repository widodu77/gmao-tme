"""
Maintenance effectiveness analysis.
Compares corrective vs preventive maintenance, calculates ROI,
and identifies optimal maintenance intervals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def compare_maintenance_types(
    corrective: pd.DataFrame,
    preventive: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side comparison of corrective vs preventive maintenance."""

    # Compute duration for preventive (use date_debut -> date_cloture)
    prev = preventive.copy()
    if 'date_debut' in prev.columns and 'date_cloture' in prev.columns:
        prev['date_debut'] = pd.to_datetime(prev['date_debut'], format='mixed', errors='coerce')
        prev['date_cloture'] = pd.to_datetime(prev['date_cloture'], format='mixed', errors='coerce')
        prev['duration_hours'] = (prev['date_cloture'] - prev['date_debut']).dt.total_seconds() / 3600
    elif 'intervention_duration_hours' in prev.columns:
        prev['duration_hours'] = prev['intervention_duration_hours']
    else:
        prev['duration_hours'] = np.nan

    prev_cost_col = 'cout' if 'cout' in prev.columns else 'cout_total'

    corr = corrective.copy()
    corr_duration = corr['duree_intervention_par_heure'] if 'duree_intervention_par_heure' in corr.columns else corr.get('duration_hours', pd.Series([np.nan]))

    metrics = {
        'total_interventions': [len(corr), len(prev)],
        'avg_duration_hours': [corr_duration.mean(), prev['duration_hours'].mean()],
        'median_duration_hours': [corr_duration.median(), prev['duration_hours'].median()],
        'avg_cost': [corr['cout_total'].mean(), prev[prev_cost_col].mean()],
        'total_cost': [corr['cout_total'].sum(), prev[prev_cost_col].sum()],
        'unique_equipment': [corr['id_actif'].nunique(), prev['id_actif'].nunique()],
    }

    comparison = pd.DataFrame(metrics, index=['Corrective', 'Preventive']).T
    comparison['Ratio (Corr/Prev)'] = comparison['Corrective'] / comparison['Preventive'].replace(0, np.nan)

    return comparison


def effectiveness_by_family(
    corrective: pd.DataFrame,
    preventive: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate effectiveness score per equipment family."""
    corr_fam = corrective.groupby('famille').agg({
        'id_ordre_travail': 'count',
        'cout_total': ['mean', 'sum'],
    })
    corr_fam.columns = ['corr_count', 'corr_avg_cost', 'corr_total_cost']
    corr_fam = corr_fam.reset_index()

    prev_cost_col = 'cout' if 'cout' in preventive.columns else 'cout_total'
    prev_fam = preventive.groupby('famille').agg({
        'numero_ot': 'count',
        prev_cost_col: ['mean', 'sum'],
    })
    prev_fam.columns = ['prev_count', 'prev_avg_cost', 'prev_total_cost']
    prev_fam = prev_fam.reset_index()

    merged = corr_fam.merge(prev_fam, on='famille', how='inner')
    merged['cost_ratio'] = merged['corr_avg_cost'] / merged['prev_avg_cost'].replace(0, np.nan)
    merged['intervention_ratio'] = merged['corr_count'] / merged['prev_count'].replace(0, np.nan)

    # Effectiveness score: lower ratios = better preventive maintenance
    merged['effectiveness_score'] = (
        100 / (1 + merged['cost_ratio'].fillna(1) + merged['intervention_ratio'].fillna(1)) * 2
    )
    merged = merged.sort_values('effectiveness_score', ascending=False)

    return merged


def calculate_roi(
    corrective: pd.DataFrame,
    preventive: pd.DataFrame,
    avoidance_factor: float = 0.6,
) -> dict:
    """Calculate ROI of preventive maintenance."""
    prev_cost_col = 'cout' if 'cout' in preventive.columns else 'cout_total'
    total_prev_cost = preventive[prev_cost_col].sum()

    avg_corr_cost_per_equip = corrective.groupby('id_actif')['cout_total'].sum().mean()
    n_prev_equipment = preventive['id_actif'].nunique()
    estimated_avoided = avg_corr_cost_per_equip * n_prev_equipment * avoidance_factor

    roi_pct = ((estimated_avoided - total_prev_cost) / total_prev_cost * 100) if total_prev_cost > 0 else 0

    return {
        'preventive_investment': float(total_prev_cost),
        'estimated_cost_avoided': float(estimated_avoided),
        'net_benefit': float(estimated_avoided - total_prev_cost),
        'roi_percent': float(roi_pct),
    }


def calculate_optimal_intervals(
    corrective: pd.DataFrame,
    preventive: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate days between preventive maintenance and next corrective failure."""
    corr = corrective.copy()
    prev = preventive.copy()

    # Ensure datetime
    if corr['date_creation_ot'].dtype == 'object':
        corr['date_creation_ot'] = pd.to_datetime(corr['date_creation_ot'], format='mixed', errors='coerce')

    prev_date_col = 'date_debut' if 'date_debut' in prev.columns else 'date_debut_planifiee'
    if prev[prev_date_col].dtype == 'object':
        prev[prev_date_col] = pd.to_datetime(prev[prev_date_col], format='mixed', errors='coerce')

    equipment_both = set(corr['id_actif'].dropna()) & set(prev['id_actif'].dropna())
    intervals = []

    for equip in equipment_both:
        prev_dates = prev.loc[prev['id_actif'] == equip, prev_date_col].dropna().sort_values()
        corr_dates = corr.loc[corr['id_actif'] == equip, 'date_creation_ot'].dropna().sort_values()

        for pdate in prev_dates:
            next_corr = corr_dates[corr_dates > pdate]
            if len(next_corr) > 0:
                days = (next_corr.iloc[0] - pdate).days
                if days >= 0:
                    intervals.append({
                        'id_actif': equip,
                        'preventive_date': pdate,
                        'next_corrective_date': next_corr.iloc[0],
                        'days_until_failure': days,
                    })

    return pd.DataFrame(intervals) if intervals else pd.DataFrame(
        columns=['id_actif', 'preventive_date', 'next_corrective_date', 'days_until_failure']
    )


def preventive_compliance(preventive: pd.DataFrame) -> dict:
    """Calculate preventive maintenance compliance metrics."""
    prev = preventive.copy()
    if 'days_deviation_from_plan' not in prev.columns:
        return {'compliance_rate': None, 'avg_deviation_days': None}

    deviations = prev['days_deviation_from_plan'].dropna()
    on_time = (deviations.abs() <= 1).sum()
    total = len(deviations)
    compliance_rate = (on_time / total * 100) if total > 0 else 0

    return {
        'compliance_rate': float(compliance_rate),
        'avg_deviation_days': float(deviations.mean()),
        'median_deviation_days': float(deviations.median()),
        'on_time_count': int(on_time),
        'total_with_data': int(total),
    }


def run_effectiveness_analysis(processed_dir: Path, reports_dir: Path) -> dict:
    """Run the full maintenance effectiveness analysis."""
    processed_dir = Path(processed_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    try:
        corrective = pd.read_pickle(processed_dir / 'corrective_integrated.pkl')
    except Exception:
        corrective = pd.read_csv(processed_dir / 'corrective_integrated.csv')
        for col in ['date_creation_ot', 'date_cloture', 'date_declaration']:
            if col in corrective.columns:
                corrective[col] = pd.to_datetime(corrective[col], format='mixed', errors='coerce')
    try:
        preventive = pd.read_pickle(processed_dir / 'preventive_cleaned.pkl')
    except Exception:
        preventive = pd.read_csv(processed_dir / 'preventive_cleaned.csv')
        for col in ['date_debut_planifiee', 'date_debut', 'date_fin', 'date_cloture']:
            if col in preventive.columns:
                preventive[col] = pd.to_datetime(preventive[col], format='mixed', errors='coerce')

    print("Comparing maintenance types...")
    comparison = compare_maintenance_types(corrective, preventive)
    print(comparison)

    print("\nEffectiveness by equipment family...")
    family_eff = effectiveness_by_family(corrective, preventive)

    print("\nCalculating ROI...")
    roi = calculate_roi(corrective, preventive)
    print(f"  ROI: {roi['roi_percent']:.1f}%")

    print("\nCalculating optimal intervals...")
    intervals = calculate_optimal_intervals(corrective, preventive)
    if len(intervals) > 0:
        q75 = intervals['days_until_failure'].quantile(0.75)
        print(f"  Recommended interval: {q75:.0f} days (75th percentile)")
    else:
        q75 = None
        print("  Insufficient overlap data for interval calculation.")

    print("\nPreventive compliance...")
    compliance = preventive_compliance(preventive)
    print(f"  Compliance rate: {compliance['compliance_rate']:.1f}%")

    # Save results
    comparison.to_csv(reports_dir / 'maintenance_comparison.csv')
    family_eff.to_csv(reports_dir / 'family_effectiveness.csv', index=False)
    if len(intervals) > 0:
        intervals.to_csv(reports_dir / 'maintenance_intervals.csv', index=False)

    summary = {
        'comparison': comparison.to_dict(),
        'roi': roi,
        'compliance': compliance,
        'optimal_interval_days': float(q75) if q75 is not None else None,
        'n_families_analyzed': len(family_eff),
    }
    with open(reports_dir / 'effectiveness_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {reports_dir}")
    return summary
