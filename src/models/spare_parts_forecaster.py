"""
Spare parts consumption analysis and forecasting.
Given only 89 days of data, we use per-family demand estimation
and reorder point calculations rather than time-series forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def pareto_analysis(spare_parts: pd.DataFrame) -> pd.DataFrame:
    """Pareto analysis: which articles account for 80% of cost."""
    summary = spare_parts.groupby('article').agg({
        'cout_total': 'sum',
        'quantite_demande': 'sum',
        'id_demande': 'count',
    }).sort_values('cout_total', ascending=False)
    summary.columns = ['total_cost', 'total_quantity', 'request_count']
    summary['cumulative_cost'] = summary['total_cost'].cumsum()
    summary['cumulative_pct'] = summary['cumulative_cost'] / summary['total_cost'].sum() * 100
    return summary.reset_index()


def consumption_by_family(spare_parts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consumption metrics by spare parts family."""
    return spare_parts.groupby('famille').agg({
        'cout_total': ['sum', 'mean', 'count'],
        'quantite_demande': 'sum',
    }).sort_values(('cout_total', 'sum'), ascending=False)


def demand_forecast(spare_parts: pd.DataFrame, forecast_days: int = 90) -> pd.DataFrame:
    """
    Simple demand forecast per article based on observed daily rate.
    With 89 days of data, we extrapolate daily consumption rate.
    """
    sp = spare_parts.copy()
    sp['date_demande'] = pd.to_datetime(sp['date_demande'], format='mixed', errors='coerce')
    date_range = (sp['date_demande'].max() - sp['date_demande'].min()).days
    if date_range <= 0:
        date_range = 1

    article_demand = sp.groupby('article').agg({
        'quantite_demande': 'sum',
        'cout_total': 'sum',
        'id_demande': 'count',
        'famille': 'first',
    }).reset_index()
    article_demand.columns = ['article', 'total_qty', 'total_cost', 'n_requests', 'famille']

    article_demand['daily_demand'] = article_demand['total_qty'] / date_range
    article_demand['forecast_qty'] = article_demand['daily_demand'] * forecast_days
    article_demand['forecast_cost'] = (
        article_demand['total_cost'] / article_demand['total_qty'].replace(0, np.nan)
    ) * article_demand['forecast_qty']
    article_demand['forecast_cost'] = article_demand['forecast_cost'].fillna(0)

    return article_demand.sort_values('forecast_cost', ascending=False)


def reorder_analysis(spare_parts: pd.DataFrame, safety_factor: float = 1.5) -> pd.DataFrame:
    """
    Calculate reorder points based on average demand and stock availability.
    """
    sp = spare_parts.copy()
    sp['date_demande'] = pd.to_datetime(sp['date_demande'], format='mixed', errors='coerce')
    date_range = max((sp['date_demande'].max() - sp['date_demande'].min()).days, 1)

    article_stats = sp.groupby('article').agg({
        'quantite_demande': ['sum', 'mean', 'std'],
        'quantite_dispo': 'last',
        'famille': 'first',
        'cout_total': 'sum',
    })
    article_stats.columns = [
        'total_demand', 'avg_demand_per_request', 'demand_std',
        'current_stock', 'famille', 'total_cost',
    ]
    article_stats = article_stats.reset_index()
    article_stats['demand_std'] = article_stats['demand_std'].fillna(0)

    article_stats['daily_demand'] = article_stats['total_demand'] / date_range
    article_stats['reorder_point'] = (
        article_stats['daily_demand'] * 7  # 7-day lead time assumption
        + safety_factor * article_stats['demand_std']
    )
    article_stats['needs_reorder'] = article_stats['current_stock'] < article_stats['reorder_point']
    article_stats['suggested_order_qty'] = np.maximum(
        article_stats['reorder_point'] * 2 - article_stats['current_stock'], 0
    ).round(0)

    return article_stats.sort_values('needs_reorder', ascending=False)


def stock_health_summary(spare_parts: pd.DataFrame) -> dict:
    """Summary statistics on stock health."""
    sp = spare_parts.copy()
    sp['stock_out'] = sp['quantite_dispo'] < sp['quantite_demande']
    stock_out_count = sp['stock_out'].sum()
    stock_out_pct = stock_out_count / len(sp) * 100

    rejected = sp[sp['etat'] == 'rejet']
    rejection_rate = len(rejected) / len(sp) * 100

    return {
        'total_requests': len(sp),
        'unique_articles': sp['article'].nunique(),
        'stock_out_count': int(stock_out_count),
        'stock_out_pct': round(stock_out_pct, 1),
        'rejection_count': len(rejected),
        'rejection_rate_pct': round(rejection_rate, 1),
        'total_cost': float(sp['cout_total'].sum()),
        'top_stockout_families': (
            sp[sp['stock_out']].groupby('famille').size()
            .sort_values(ascending=False).head(5).to_dict()
        ),
    }


def run_spare_parts_analysis(processed_dir: Path, reports_dir: Path) -> dict:
    """Full spare parts analysis pipeline."""
    processed_dir = Path(processed_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading spare parts data...")
    try:
        spare_parts = pd.read_pickle(processed_dir / 'spare_parts_cleaned.pkl')
    except Exception:
        spare_parts = pd.read_csv(processed_dir / 'spare_parts_cleaned.csv')

    print("Running Pareto analysis...")
    pareto = pareto_analysis(spare_parts)
    pareto_80 = pareto[pareto['cumulative_pct'] <= 80]
    print(f"  {len(pareto_80)} articles ({len(pareto_80)/len(pareto)*100:.1f}%) = 80% of costs")

    print("Consumption by family...")
    family_consumption = consumption_by_family(spare_parts)

    print("Demand forecasting (next 90 days)...")
    forecast = demand_forecast(spare_parts, forecast_days=90)
    top_forecast = forecast.head(10)
    print(f"  Top article forecast: {top_forecast.iloc[0]['article']} - {top_forecast.iloc[0]['forecast_qty']:.0f} units")

    print("Reorder analysis...")
    reorder = reorder_analysis(spare_parts)
    needs_reorder = reorder[reorder['needs_reorder']]
    print(f"  {len(needs_reorder)} articles need reorder")

    print("Stock health summary...")
    health = stock_health_summary(spare_parts)

    # Save
    pareto.to_csv(reports_dir / 'pareto_analysis.csv', index=False)
    forecast.to_csv(reports_dir / 'demand_forecast.csv', index=False)
    reorder.to_csv(reports_dir / 'reorder_analysis.csv', index=False)

    summary = {
        'stock_health': health,
        'pareto_80_count': len(pareto_80),
        'pareto_total_articles': len(pareto),
        'articles_needing_reorder': len(needs_reorder),
        'total_forecast_cost_90d': float(forecast['forecast_cost'].sum()),
    }
    with open(reports_dir / 'spare_parts_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {reports_dir}")
    return summary
