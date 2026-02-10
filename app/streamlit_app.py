"""
GMAO AI Project - Interactive Dashboard
Consolidated view: Overview, Exploration, Features, Models, Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import pickle
import sys
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="GMAO AI Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 1rem; }
    .sub-header { font-size: 1.8rem; font-weight: bold; color: #2c3e50; margin-top: 1.5rem; margin-bottom: 0.5rem; }
    .insight-box { background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .warning-box { background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #ffc107; }
    .success-box { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #28a745; }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'processed'
    features_dir = project_root / 'data' / 'features'
    models_dir = project_root / 'models'
    reports_dir = project_root / 'reports'

    data = {
        'corrective': pd.read_csv(data_dir / 'corrective_integrated.csv'),
        'preventive': pd.read_csv(data_dir / 'preventive_cleaned.csv'),
        'spare_parts': pd.read_csv(data_dir / 'spare_parts_cleaned.csv'),
        'timeline': pd.read_csv(data_dir / 'unified_timeline.csv'),
        'asset_master': pd.read_csv(data_dir / 'asset_master.csv'),
        'features': pd.read_csv(features_dir / 'corrective_features.csv'),
    }

    for col in ['date_creation_ot', 'date_cloture', 'date_declaration', 'date_debut_reparation']:
        if col in data['corrective'].columns:
            data['corrective'][col] = pd.to_datetime(data['corrective'][col], format='mixed', errors='coerce')

    with open(data_dir / 'data_quality_report.json', 'r', encoding='utf-8') as f:
        data['quality_report'] = json.load(f)
    with open(data_dir / 'relationship_summary.json', 'r', encoding='utf-8') as f:
        data['relationship_summary'] = json.load(f)
    with open(features_dir / 'feature_metadata.json', 'r', encoding='utf-8') as f:
        data['feature_metadata'] = json.load(f)

    for fname, key in [
        (models_dir / 'model_metrics.json', 'model_metrics'),
        (reports_dir / 'effectiveness_summary.json', 'effectiveness_summary'),
        (reports_dir / 'spare_parts_summary.json', 'spare_parts_summary'),
    ]:
        if fname.exists():
            with open(fname, 'r') as f:
                data[key] = json.load(f)

    if (models_dir / 'corrective_with_anomalies.csv').exists():
        data['anomaly_predictions'] = pd.read_csv(models_dir / 'corrective_with_anomalies.csv')
    if (reports_dir / 'demand_forecast.csv').exists():
        data['demand_forecast'] = pd.read_csv(reports_dir / 'demand_forecast.csv')

    return data


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load trained ML models for interactive predictions."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'
    sys.path.insert(0, str(project_root))

    loaded = {}

    try:
        from src.models.failure_predictor import FailurePredictor
        loaded['failure'] = FailurePredictor.load(models_dir)
    except Exception as e:
        loaded['failure_error'] = str(e)

    try:
        from src.models.failure_predictor import SparePartsPredictor
        loaded['spare_parts'] = SparePartsPredictor.load(models_dir)
    except Exception as e:
        loaded['spare_parts_error'] = str(e)

    try:
        from src.models.anomaly_detector import AnomalyDetector
        loaded['anomaly'] = AnomalyDetector.load(models_dir)
    except Exception as e:
        loaded['anomaly_error'] = str(e)

    return loaded


def _build_feature_vector(params: dict, features_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build a full feature vector from user inputs + historical medians."""
    medians = features_df[feature_cols].median().to_dict()
    row = {col: medians.get(col, 0) for col in feature_cols}
    row.update({k: v for k, v in params.items() if k in feature_cols})
    return pd.DataFrame([row])[feature_cols]


# ─── PAGE: PREDICTIONS ──────────────────────────────────────────────────────

def show_predictions(data):
    st.markdown('<p class="sub-header">Prediction Interactive</p>', unsafe_allow_html=True)
    st.markdown("Saisissez les parametres d'un ordre de travail pour obtenir les predictions des modeles.")

    models = load_models()
    features_df = data['features']
    corr = data['corrective']

    # Collect unique values for select boxes
    familles = sorted(corr['famille'].dropna().unique().tolist())
    sites = sorted(corr['site'].dropna().unique().tolist())
    urgences = sorted(corr['urgence'].dropna().unique().tolist()) if 'urgence' in corr.columns else ['Moyen', 'Fort', 'Faible']
    portfolios = sorted(corr['portfolio'].dropna().unique().tolist())
    anomalies = corr['anomalie'].value_counts().head(25).index.tolist()

    # Build encoding lookup tables from the features dataset
    encodings = {}
    for col in ['site', 'famille', 'portfolio', 'groupe', 'prestataire', 'etat', 'zone']:
        enc_col = f'{col}_encoded'
        if col in corr.columns and enc_col in features_df.columns:
            mapping_df = pd.DataFrame({col: corr[col], enc_col: features_df[enc_col]})
            encodings[col] = mapping_df.drop_duplicates().dropna().set_index(col)[enc_col].to_dict()

    # --- Input form ---
    st.markdown("### Parametres de l'ordre de travail")

    col1, col2, col3 = st.columns(3)

    with col1:
        site = st.selectbox("Site", sites, index=0)
        famille = st.selectbox("Famille d'equipement", familles, index=0)
        portfolio = st.selectbox("Portfolio", portfolios, index=0)
        urgence = st.selectbox("Urgence", urgences, index=urgences.index('Moyen') if 'Moyen' in urgences else 0)

    with col2:
        anomalie = st.selectbox("Type d'anomalie", anomalies, index=0)
        cout_total = st.number_input("Cout total estime (EUR)", min_value=0.0, value=500.0, step=50.0)
        duree_intervention = st.number_input("Duree intervention (heures)", min_value=0.0, value=2.0, step=0.5)
        nombre_main_doeuvre = st.number_input("Nombre main d'oeuvre", min_value=1, value=2, step=1)

    with col3:
        creation_hour = st.slider("Heure de creation", 0, 23, 10)
        creation_dow = st.selectbox("Jour de la semaine", ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'],
                                    index=1)
        dow_map = {'Lundi': 0, 'Mardi': 1, 'Mercredi': 2, 'Jeudi': 3, 'Vendredi': 4, 'Samedi': 5, 'Dimanche': 6}
        equipment_failures = st.number_input("Pannes precedentes (cet equip.)", min_value=0, value=1, step=1)
        days_since_last = st.number_input("Jours depuis derniere maintenance", min_value=0.0, value=15.0, step=1.0)

    # Build params dict from user inputs
    # Get historical stats for the selected famille
    fam_data = corr[corr['famille'] == famille]
    fam_avg_cost = fam_data['cout_total'].mean() if len(fam_data) > 0 else features_df['famille_avg_cost'].median()
    fam_failure_count = len(fam_data)
    fam_sp_rate = features_df.loc[features_df.get('famille_encoded', pd.Series()) == encodings.get('famille', {}).get(famille), 'famille_spare_parts_usage_rate'].median() if 'famille' in encodings else features_df['famille_spare_parts_usage_rate'].median()
    if pd.isna(fam_sp_rate):
        fam_sp_rate = features_df['famille_spare_parts_usage_rate'].median()

    site_data = corr[corr['site'] == site]
    site_fc = len(site_data)

    # Anomaly binary flags
    anomaly_flags = {}
    for col_name in features_df.columns:
        if col_name.startswith('anomaly_') and col_name not in ('anomaly_recurrence_count', 'anomaly_type_encoded'):
            anomaly_flags[col_name] = 0
    # Set the matching anomaly flag to 1
    anomaly_col_name = f"anomaly_{anomalie.replace(' ', '_').lower()}"[:50]
    if anomaly_col_name in anomaly_flags:
        anomaly_flags[anomaly_col_name] = 1

    # Urgency flags
    urgency_flags = {}
    for col_name in features_df.columns:
        if col_name.startswith('urgence_'):
            urgency_flags[col_name] = 0
    urg_col = f'urgence_{urgence}'
    if urg_col in urgency_flags:
        urgency_flags[urg_col] = 1

    params = {
        'cout_total': cout_total,
        'duree_intervention_par_heure': duree_intervention,
        'nombre_main_doeuvre': nombre_main_doeuvre,
        'creation_hour': creation_hour,
        'creation_day_of_week': dow_map[creation_dow],
        'creation_day_of_month': 15,
        'creation_month': datetime.now().month,
        'creation_quarter': (datetime.now().month - 1) // 3 + 1,
        'creation_is_weekend': 1 if dow_map[creation_dow] >= 5 else 0,
        'creation_is_business_hours': 1 if 8 <= creation_hour < 18 else 0,
        'equipment_failure_count': equipment_failures,
        'days_since_last_maintenance': days_since_last,
        'equipment_avg_cost': cout_total,
        'equipment_avg_duration': duree_intervention,
        'equipment_age_days': days_since_last * equipment_failures,
        'famille_avg_cost': fam_avg_cost,
        'famille_failure_count': fam_failure_count,
        'famille_spare_parts_usage_rate': fam_sp_rate,
        'site_failure_count': site_fc,
        'anomaly_recurrence_count': max(1, equipment_failures),
        'cout_pdr': 0.0,
        'cout_maindoeuvre': cout_total * 0.3,
        'time_to_closure_hours': duree_intervention * 5,
        'time_to_start_hours': duree_intervention * 0.5,
        'rolling_7d_failures': min(equipment_failures, 3),
        'rolling_30d_failures': equipment_failures,
        'rolling_7d_avg_cost': cout_total,
        'rolling_30d_avg_cost': cout_total * 0.8,
        **anomaly_flags,
        **urgency_flags,
    }

    # Add encoded categoricals
    for col, mapping in encodings.items():
        enc_col = f'{col}_encoded'
        if col == 'site':
            params[enc_col] = mapping.get(site, 0)
        elif col == 'famille':
            params[enc_col] = mapping.get(famille, 0)
        elif col == 'portfolio':
            params[enc_col] = mapping.get(portfolio, 0)
        else:
            # Use median for fields not in form
            params[enc_col] = features_df[enc_col].median() if enc_col in features_df.columns else 0

    # Compute urgency rate for famille
    if 'famille_high_urgency_rate' in features_df.columns:
        params['famille_high_urgency_rate'] = features_df['famille_high_urgency_rate'].median()

    # Provider features - use medians
    for feat in ['provider_avg_time_to_closure', 'provider_avg_cost', 'provider_work_volume']:
        if feat in features_df.columns:
            params[feat] = features_df[feat].median()

    # Zone feature
    if 'zone_failure_count' in features_df.columns:
        params['zone_failure_count'] = features_df['zone_failure_count'].median()

    # Spare parts features for failure model
    params['has_spare_parts'] = 0
    params['spare_parts_cost_ratio'] = 0.0
    params['num_spare_parts_requests'] = 0
    params['total_parts_cost'] = 0.0
    params['total_parts_quantity'] = 0.0

    # --- Run predictions ---
    if st.button("Lancer les predictions", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("### Resultats")

        result_cols = st.columns(3)

        # Failure prediction
        with result_cols[0]:
            if 'failure' in models:
                fp_model = models['failure']
                try:
                    fp_df = _build_feature_vector(params, features_df, fp_model.feature_cols)
                    fp_scaled = fp_model.scaler.transform(fp_df)
                    failure_prob = fp_model.model.predict_proba(fp_scaled)[0, 1]
                    failure_pred = fp_model.model.predict(fp_scaled)[0]

                    st.markdown("#### Risque de Panne")
                    color = "#e74c3c" if failure_prob > 0.5 else "#f39c12" if failure_prob > 0.3 else "#27ae60"
                    st.markdown(f'<div style="text-align:center;padding:1rem;background:{color}20;border-radius:0.5rem;border-left:4px solid {color}">'
                                f'<h2 style="color:{color};margin:0">{failure_prob*100:.1f}%</h2>'
                                f'<p style="margin:0">{"RISQUE ELEVE" if failure_prob > 0.5 else "RISQUE MOYEN" if failure_prob > 0.3 else "RISQUE FAIBLE"}</p>'
                                f'</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilite de panne dans les 30 prochains jours")

                    if failure_prob > 0.5:
                        st.markdown('<div class="warning-box">Planifier une inspection preventive rapidement.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erreur prediction panne: {e}")
            else:
                st.warning(f"Modele non charge: {models.get('failure_error', 'inconnu')}")

        # Spare parts prediction
        with result_cols[1]:
            if 'spare_parts' in models:
                sp_model = models['spare_parts']
                try:
                    sp_df = _build_feature_vector(params, features_df, sp_model.feature_cols)
                    sp_scaled = sp_model.scaler.transform(sp_df)
                    sp_prob = sp_model.model.predict_proba(sp_scaled)[0, 1]

                    st.markdown("#### Besoin Pieces de Rechange")
                    color = "#e74c3c" if sp_prob > 0.5 else "#f39c12" if sp_prob > 0.3 else "#27ae60"
                    st.markdown(f'<div style="text-align:center;padding:1rem;background:{color}20;border-radius:0.5rem;border-left:4px solid {color}">'
                                f'<h2 style="color:{color};margin:0">{sp_prob*100:.1f}%</h2>'
                                f'<p style="margin:0">{"PROBABLE" if sp_prob > 0.5 else "POSSIBLE" if sp_prob > 0.3 else "PEU PROBABLE"}</p>'
                                f'</div>', unsafe_allow_html=True)
                    st.caption("Probabilite de besoin en pieces de rechange")

                    if sp_prob > 0.5:
                        st.markdown('<div class="warning-box">Verifier la disponibilite des pieces avant intervention.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erreur prediction pieces: {e}")
            else:
                st.warning(f"Modele non charge: {models.get('spare_parts_error', 'inconnu')}")

        # Anomaly detection
        with result_cols[2]:
            if 'anomaly' in models:
                ad_model = models['anomaly']
                try:
                    ad_df = _build_feature_vector(params, features_df, ad_model.features)
                    ad_scaled = ad_model.scaler.transform(ad_df)

                    iso_score = ad_model.iso_forest.score_samples(ad_scaled)[0]
                    iso_pred = ad_model.iso_forest.predict(ad_scaled)[0]
                    ocsvm_pred = ad_model.ocsvm.predict(ad_scaled)[0]

                    is_anomaly = (iso_pred == -1) or (ocsvm_pred == -1)
                    confidence = int(iso_pred == -1) + int(ocsvm_pred == -1)

                    st.markdown("#### Score d'Anomalie")
                    # Normalize iso_score: more negative = more anomalous
                    norm_score = max(0, min(1, (0.0 - iso_score) / 0.3))
                    color = "#e74c3c" if is_anomaly else "#27ae60"
                    label = "ANOMALIE DETECTEE" if is_anomaly else "NORMAL"
                    if confidence == 2:
                        label = "ANOMALIE (haute confiance)"

                    st.markdown(f'<div style="text-align:center;padding:1rem;background:{color}20;border-radius:0.5rem;border-left:4px solid {color}">'
                                f'<h2 style="color:{color};margin:0">{label}</h2>'
                                f'<p style="margin:0">Score IF: {iso_score:.3f} | Accord: {confidence}/2</p>'
                                f'</div>', unsafe_allow_html=True)
                    st.caption("Detection par Isolation Forest + One-Class SVM")

                    if is_anomaly:
                        st.markdown('<div class="warning-box">Cet ordre presente des caracteristiques inhabituelles. Verifier les donnees.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erreur detection anomalie: {e}")
            else:
                st.warning(f"Modele non charge: {models.get('anomaly_error', 'inconnu')}")

        # Feature contribution insight
        st.markdown("---")
        st.markdown("### Facteurs les plus influents")
        if 'failure' in models:
            fp_model = models['failure']
            importances = sorted(
                zip(fp_model.feature_cols, fp_model.model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:10]
            fi_df = pd.DataFrame(importances, columns=['Feature', 'Importance'])
            fi_df['Valeur utilisee'] = [f"{params.get(f, 'N/A')}" for f, _ in importances]
            fig = px.bar(fi_df, y='Feature', x='Importance', orientation='h',
                         title='Top 10 features (modele panne)', hover_data=['Valeur utilisee'],
                         color='Importance', color_continuous_scale='Blues')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# ─── PAGE: OVERVIEW ──────────────────────────────────────────────────────────

def show_overview(data):
    st.markdown('<p class="main-header">GMAO AI - Tableau de Bord</p>', unsafe_allow_html=True)
    st.markdown("### Analyse de la maintenance industrielle (Jan - Mar 2025)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", f"{len(data['timeline']):,}", "89 jours")
    c2.metric("Ordres Correctifs", f"{len(data['corrective']):,}")
    c3.metric("Ordres Preventifs", f"{len(data['preventive']):,}")
    c4.metric("Equipements", f"{len(data['asset_master']):,}", "5 sites")

    st.markdown("---")

    # Model performance summary if available
    if 'model_metrics' in data:
        st.markdown("### Performance des Modeles ML")
        m = data['model_metrics']
        c1, c2, c3 = st.columns(3)

        fp = m.get('failure_prediction', {})
        c1.markdown("**Prediction de Panne**")
        c1.metric("F1", f"{fp.get('test_f1', 0):.3f}")
        c1.metric("AUC", f"{fp.get('test_roc_auc', 0):.3f}")

        sp = m.get('spare_parts_prediction', {})
        c2.markdown("**Prediction Pieces de Rechange**")
        c2.metric("F1", f"{sp.get('test_f1', 0):.3f}")
        c2.metric("AUC", f"{sp.get('test_roc_auc', 0):.3f}")

        ad = m.get('anomaly_detection', {})
        c3.markdown("**Detection d'Anomalies**")
        c3.metric("Anomalies", ad.get('consensus_anomalies', 0))
        c3.metric("Accord", f"{ad.get('agreement_pct', 0)}%")

    st.markdown("---")

    # Key findings
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="warning-box">', unsafe_allow_html=True)
    c1.markdown("**Conformite preventive:** seulement 26.5%")
    c1.markdown('</div>', unsafe_allow_html=True)

    c2.markdown('<div class="warning-box">', unsafe_allow_html=True)
    c2.markdown("**Taux de rupture stock:** 49%")
    c2.markdown('</div>', unsafe_allow_html=True)

    c3.markdown('<div class="success-box">', unsafe_allow_html=True)
    c3.markdown("**11% des articles = 80% des couts**")
    c3.markdown('</div>', unsafe_allow_html=True)


# ─── PAGE: DATA & EXPLORATION ────────────────────────────────────────────────

def show_exploration(data):
    st.markdown('<p class="sub-header">Exploration des Donnees</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Qualite & Integration", "Tendances Temporelles",
        "Pieces de Rechange", "Anomalies & Pannes", "Efficacite"
    ])

    # ── Tab 1: Data Quality ──
    with tab1:
        quality = data['quality_report']
        rel = data['relationship_summary']

        c1, c2, c3 = st.columns(3)
        for col_widget, name in [(c1, 'Corrective'), (c2, 'Preventive'), (c3, 'Spare Parts')]:
            q = quality[name]
            col_widget.markdown(f"**{name}**")
            col_widget.metric("Lignes", f"{q['total_rows']:,}")
            col_widget.metric("Colonnes", q['total_columns'])
            col_widget.metric("Doublons", q['duplicates'])
            if q['missing_values']:
                col_widget.markdown("Top valeurs manquantes:")
                for c, info in list(q['missing_values'].items())[:3]:
                    col_widget.text(f"  {c}: {info['percentage']:.1f}%")

        st.markdown("---")
        st.markdown("#### Integration")
        c1, c2, c3 = st.columns(3)
        asset = rel['asset_utilization']
        c1.metric("Actifs totaux", f"{asset['total_assets']:,}")
        c2.metric("Avec les 2 types", asset['assets_both'])
        loc = rel['location_distribution']
        c3.metric("Site le plus actif", f"{loc['most_active_site']} ({loc['most_active_site_count']})")

        # Dataset explorer
        st.markdown("#### Explorer les datasets")
        choice = st.selectbox("Dataset", ["Correctif", "Preventif", "Pieces de rechange", "Actifs", "Timeline"])
        df_map = {"Correctif": data['corrective'], "Preventif": data['preventive'],
                  "Pieces de rechange": data['spare_parts'], "Actifs": data['asset_master'],
                  "Timeline": data['timeline']}
        sel = df_map[choice]
        st.caption(f"{sel.shape[0]:,} lignes x {sel.shape[1]} colonnes")
        st.dataframe(sel.head(100), use_container_width=True, height=300)

    # ── Tab 2: Temporal ──
    with tab2:
        timeline = data['timeline'].copy()
        timeline['date_created'] = pd.to_datetime(timeline['date_created'], format='mixed', errors='coerce')
        timeline['date'] = timeline['date_created'].dt.date
        timeline['day_of_week'] = timeline['date_created'].dt.dayofweek
        timeline['hour'] = timeline['date_created'].dt.hour

        daily = timeline.groupby(['date', 'maintenance_type']).size().reset_index(name='count')
        fig = px.line(daily, x='date', y='count', color='maintenance_type',
                      title='Volume journalier', color_discrete_map={'corrective': '#e74c3c', 'preventive': '#3498db'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            dow = timeline.groupby(['day_of_week', 'maintenance_type']).size().reset_index(name='count')
            dow['jour'] = dow['day_of_week'].map({0:'Lun',1:'Mar',2:'Mer',3:'Jeu',4:'Ven',5:'Sam',6:'Dim'})
            fig = px.bar(dow, x='jour', y='count', color='maintenance_type', barmode='group',
                         title='Par jour de la semaine', color_discrete_map={'corrective':'#e74c3c','preventive':'#3498db'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            hour = timeline.groupby(['hour', 'maintenance_type']).size().reset_index(name='count')
            fig = px.line(hour, x='hour', y='count', color='maintenance_type', markers=True,
                          title='Par heure', color_discrete_map={'corrective':'#e74c3c','preventive':'#3498db'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # MTBF
        corr = data['corrective'].copy().sort_values(['id_actif', 'date_creation_ot'])
        corr['days_bf'] = corr.groupby('id_actif')['date_creation_ot'].diff().dt.total_seconds() / 86400
        mtbf = corr.groupby('famille')['days_bf'].agg(['mean','count'])
        mtbf = mtbf[mtbf['count'] >= 5].sort_values('mean').head(15).reset_index()
        fig = px.bar(mtbf, y='famille', x='mean', orientation='h', title='MTBF par famille (top 15)',
                     color='mean', color_continuous_scale='RdYlGn')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Spare Parts ──
    with tab3:
        sp = data['spare_parts'].copy()
        art = sp.groupby('article').agg({'cout_total':'sum','quantite_demande':'sum','id_demande':'count'}).sort_values('cout_total', ascending=False)
        art.columns = ['total_cost','total_qty','requests']
        art['cum_pct'] = art['total_cost'].cumsum() / art['total_cost'].sum() * 100
        art = art.reset_index()

        top = art.head(20)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=list(range(len(top))), y=top['total_cost'], name="Cout"), secondary_y=False)
        fig.add_trace(go.Scatter(x=list(range(len(top))), y=top['cum_pct'], name="Cumul %",
                                 mode='lines+markers', line=dict(color='red', width=3)), secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color="red", secondary_y=True)
        fig.update_yaxes(title_text="Cout total", secondary_y=False)
        fig.update_yaxes(title_text="% cumulatif", secondary_y=True, range=[0, 100])
        fig.update_layout(title="Analyse Pareto", height=400)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fam = sp.groupby('famille')['cout_total'].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(fam, y='famille', x='cout_total', orientation='h', title='Cout par famille', color='cout_total', color_continuous_scale='Blues')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            status = sp['etat'].value_counts().reset_index()
            status.columns = ['etat', 'count']
            fig = px.pie(status, values='count', names='etat', title='Statut des demandes')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        pareto_80 = art[art['cum_pct'] <= 80]
        st.markdown(f"**Pareto:** {len(pareto_80)} articles ({len(pareto_80)/len(art)*100:.1f}%) = 80% des couts")

    # ── Tab 4: Anomalies & Failures ──
    with tab4:
        corr = data['corrective'].copy()
        c1, c2 = st.columns(2)
        with c1:
            anom = corr['anomalie'].value_counts().head(15).reset_index()
            anom.columns = ['anomalie', 'count']
            fig = px.bar(anom, y='anomalie', x='count', orientation='h', title='Top 15 types d\'anomalies',
                         color='count', color_continuous_scale='Reds')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            site_f = corr.groupby('site').size().reset_index(name='count').sort_values('count', ascending=False)
            fig = px.bar(site_f, x='site', y='count', title='Pannes par site', color='count',
                         color_continuous_scale='Oranges', text='count')
            fig.update_traces(textposition='outside')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        # Urgency
        if 'urgence' in corr.columns:
            c1, c2 = st.columns([2, 1])
            with c1:
                urg = corr['urgence'].value_counts().reset_index()
                urg.columns = ['urgence', 'count']
                fig = px.pie(urg, values='count', names='urgence', title='Niveaux d\'urgence',
                             color='urgence', color_discrete_map={'Fort':'#e74c3c','Moyen':'#f39c12','Faible':'#27ae60'})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                equip_f = corr.groupby('id_actif').size().sort_values(ascending=False)
                recurring = equip_f[equip_f > 1]
                reopened = (corr['est_reouvert'] == True).sum()
                st.metric("Equipements recurrents", len(recurring))
                st.metric("Moy. pannes/equip. recurrent", f"{recurring.mean():.1f}")
                st.metric("OT reouverts", f"{reopened} ({reopened/len(corr)*100:.1f}%)")

    # ── Tab 5: Efficiency ──
    with tab5:
        corr = data['corrective'].copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Moy. temps cloture", f"{corr['time_to_closure_hours'].mean():.1f} h")
        c2.metric("Mediane temps cloture", f"{corr['time_to_closure_hours'].median():.1f} h")
        c3.metric("Moy. duree intervention", f"{corr['duree_intervention_par_heure'].mean():.1f} h")

        fig = px.histogram(corr, x='time_to_closure_hours', nbins=50, title='Distribution temps de cloture',
                           color_discrete_sequence=['#3498db'])
        fig.add_vline(x=corr['time_to_closure_hours'].median(), line_dash="dash", line_color="red")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            port = corr.groupby('portfolio')['cout_total'].agg(['sum','mean','count']).reset_index()
            port.columns = ['portfolio','total','moyenne','nb']
            fig = px.bar(port, x='portfolio', y='total', title='Cout total par portfolio', color='total',
                         color_continuous_scale='Greens', text='total')
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            prov = corr.groupby('prestataire').agg({'id_ordre_travail':'count','time_to_closure_hours':'mean',
                                                     'cout_total':'mean'}).sort_values('id_ordre_travail', ascending=False).head(10).reset_index()
            prov.columns = ['prestataire','nb','avg_time','avg_cost']
            fig = px.scatter(prov, x='avg_time', y='avg_cost', size='nb', hover_data=['prestataire'],
                             title='Performance prestataires', color='nb', color_continuous_scale='Viridis')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# ─── PAGE: FEATURES ──────────────────────────────────────────────────────────

def show_features(data):
    st.markdown('<p class="sub-header">Feature Engineering</p>', unsafe_allow_html=True)

    meta = data['feature_metadata']
    features_df = data['features']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Features", meta['total_features'])
    c2.metric("Samples", f"{meta['total_samples']:,}")
    c3.metric("Numeriques", len(meta['numerical_features']))
    c4.metric("Cibles", len(meta['target_variables']))

    tab1, tab2 = st.tabs(["Categories", "Explorateur"])

    with tab1:
        cats = {
            'Temporelles': [f for f in meta['numerical_features'] if any(x in f for x in ['creation_', 'days_since'])],
            'Equipement': [f for f in meta['numerical_features'] if 'equipment_' in f or 'famille_' in f],
            'Localisation': [f for f in meta['numerical_features'] if 'site_' in f or 'zone_' in f],
            'Pieces rechange': [f for f in meta['numerical_features'] if 'spare' in f or 'parts' in f],
            'Prestataire': [f for f in meta['numerical_features'] if 'provider_' in f],
            'Fenetres glissantes': [f for f in meta['numerical_features'] if 'rolling_' in f],
            'Encodees': [f for f in meta['numerical_features'] if '_encoded' in f],
        }
        cat_counts = {k: len(v) for k, v in cats.items()}
        fig = px.bar(x=list(cat_counts.keys()), y=list(cat_counts.values()), title='Features par categorie',
                     color=list(cat_counts.values()), color_continuous_scale='Blues')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Targets
        st.markdown("#### Variables cibles")
        target_desc = {
            'will_fail_within_30_days': ('Classification binaire', "Panne dans les 30 jours?"),
            'will_need_spare_parts': ('Classification binaire', "Besoin de pieces?"),
            'anomaly_type_encoded': ('Classification multi-classe', "Type d'anomalie"),
            'spare_parts_cost_target': ('Regression', "Cout pieces de rechange"),
            'intervention_duration_target': ('Regression', "Duree d'intervention"),
        }
        for t in meta['target_variables']:
            if t in target_desc and t in features_df.columns:
                typ, desc = target_desc[t]
                if 'binaire' in typ.lower():
                    rate = features_df[t].mean() * 100
                    st.markdown(f"- **{t}** ({typ}): {desc} - taux positif {rate:.1f}%")
                else:
                    st.markdown(f"- **{t}** ({typ}): {desc}")

    with tab2:
        feat = st.selectbox("Feature a analyser",
                            [f for f in meta['numerical_features'] if f not in meta['target_variables']])
        if feat:
            c1, c2 = st.columns(2)
            with c1:
                st.write(features_df[feat].describe())
            with c2:
                fig = px.histogram(features_df, x=feat, nbins=50, title=f'Distribution: {feat}')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        sel = st.multiselect("Correlation entre features", meta['numerical_features'][:20],
                             default=meta['numerical_features'][:5])
        if len(sel) >= 2:
            fig = px.imshow(features_df[sel].corr(), title='Matrice de correlation', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)


# ─── PAGE: MODELS & ANALYSIS ─────────────────────────────────────────────────

def show_models(data):
    st.markdown('<p class="sub-header">Modeles ML & Analyses</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Detection d'Anomalies", "Prediction de Pannes",
        "Pieces de Rechange", "Efficacite Maintenance"
    ])

    metrics = data.get('model_metrics', {})

    # ── Tab 1: Anomaly Detection ──
    with tab1:
        ad = metrics.get('anomaly_detection', {})
        if not ad:
            st.warning("Pas de resultats. Lancez `python run_pipeline.py --step 3`")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Samples", f"{ad['total_samples']:,}")
            c2.metric("Anomalies (consensus)", ad['consensus_anomalies'])
            c3.metric("Haute confiance", ad['high_confidence_anomalies'])
            c4.metric("Accord methodes", f"{ad['agreement_pct']}%")

            if 'feature_comparison' in ad:
                st.markdown("#### Comparaison Normal vs Anomalie")
                comp = pd.DataFrame([
                    {'Feature': f, 'Normal': v['normal_mean'], 'Anomalie': v['anomaly_mean'], 'Diff %': v['difference_pct']}
                    for f, v in ad['feature_comparison'].items()
                ])
                st.dataframe(comp, use_container_width=True)

            if 'anomaly_predictions' in data:
                ap = data['anomaly_predictions']
                if 'site' in ap.columns and 'anomaly_consensus' in ap.columns:
                    sa = ap.groupby('site')['anomaly_consensus'].agg(['sum','count'])
                    sa['rate'] = (sa['sum']/sa['count']*100).round(1)
                    sa = sa.reset_index()
                    fig = px.bar(sa, x='site', y='rate', title='Taux d\'anomalies par site (%)',
                                 color='rate', color_continuous_scale='Reds')
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Failure Prediction ──
    with tab2:
        fp = metrics.get('failure_prediction', {})
        if not fp:
            st.warning("Pas de resultats. Lancez `python run_pipeline.py --step 3`")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{fp['test_precision']:.3f}")
            c2.metric("Recall", f"{fp['test_recall']:.3f}")
            c3.metric("F1", f"{fp['test_f1']:.3f}")
            c4.metric("ROC AUC", f"{fp['test_roc_auc']:.3f}")

            st.caption(f"CV AUC: {fp['cv_roc_auc_mean']:.3f} (+/- {fp['cv_roc_auc_std']:.3f}) | "
                       f"Taux positif: {fp['positive_rate']*100:.1f}%")

            c1, c2 = st.columns(2)
            with c1:
                if 'confusion_matrix' in fp:
                    cm = np.array(fp['confusion_matrix'])
                    fig = px.imshow(cm, labels=dict(x="Predit", y="Reel", color="Nb"),
                                    x=['Pas de panne', 'Panne'], y=['Pas de panne', 'Panne'],
                                    text_auto=True, title='Matrice de confusion', color_continuous_scale='Blues')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'feature_importance' in fp:
                    fi = pd.DataFrame(list(fp['feature_importance'].items()), columns=['Feature', 'Importance'])
                    fig = px.bar(fi, y='Feature', x='Importance', orientation='h',
                                 title='Top 15 features')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Spare Parts ──
    with tab3:
        sp_m = metrics.get('spare_parts_prediction', {})

        # Prediction model metrics
        if sp_m:
            st.markdown("#### Prediction: besoin de pieces?")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{sp_m['test_precision']:.3f}")
            c2.metric("Recall", f"{sp_m['test_recall']:.3f}")
            c3.metric("F1", f"{sp_m['test_f1']:.3f}")
            c4.metric("ROC AUC", f"{sp_m['test_roc_auc']:.3f}")

            c1, c2 = st.columns(2)
            with c1:
                if 'confusion_matrix' in sp_m:
                    cm = np.array(sp_m['confusion_matrix'])
                    fig = px.imshow(cm, labels=dict(x="Predit", y="Reel"),
                                    x=['Sans pieces', 'Avec pieces'], y=['Sans pieces', 'Avec pieces'],
                                    text_auto=True, title='Matrice de confusion', color_continuous_scale='Blues')
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'feature_importance' in sp_m:
                    fi = pd.DataFrame(list(sp_m['feature_importance'].items()), columns=['Feature', 'Importance'])
                    fig = px.bar(fi, y='Feature', x='Importance', orientation='h', title='Top 15 features')
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Forecasting & stock health
        sp_summary = data.get('spare_parts_summary', {})
        if sp_summary:
            st.markdown("#### Sante du stock & Previsions")
            health = sp_summary.get('stock_health', {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Demandes totales", f"{health.get('total_requests',0):,}")
            c2.metric("Taux rupture", f"{health.get('stock_out_pct',0):.1f}%")
            c3.metric("Taux rejet", f"{health.get('rejection_rate_pct',0):.1f}%")
            c4.metric("Articles a reapprovisionner", sp_summary.get('articles_needing_reorder', 0))

            st.metric("Cout previsionnel 90j", f"{sp_summary.get('total_forecast_cost_90d',0):,.0f} EUR")

            if 'demand_forecast' in data:
                forecast = data['demand_forecast']
                top = forecast.head(15)[['article','famille','total_qty','forecast_qty','forecast_cost']].copy()
                top.columns = ['Article','Famille','Qty historique','Qty prevue','Cout prevu']
                st.dataframe(top, use_container_width=True, column_config={
                    'Cout prevu': st.column_config.NumberColumn(format="%.0f"),
                    'Qty prevue': st.column_config.NumberColumn(format="%.0f"),
                })

    # ── Tab 4: Effectiveness ──
    with tab4:
        eff = data.get('effectiveness_summary', {})
        if not eff:
            st.warning("Pas de resultats. Lancez `python run_pipeline.py --step 4`")
        else:
            roi = eff.get('roi', {})
            compliance = eff.get('compliance', {})
            interval = eff.get('optimal_interval_days')

            c1, c2, c3 = st.columns(3)
            c1.metric("ROI Preventif", f"{roi.get('roi_percent',0):.1f}%")
            c2.metric("Conformite", f"{compliance.get('compliance_rate',0):.1f}%")
            c3.metric("Intervalle optimal", f"{interval:.0f} j" if interval else "N/A")

            st.markdown("#### Detail ROI")
            roi_rows = []
            for k, v in roi.items():
                label = k.replace('_', ' ').title()
                if abs(v) > 1000:
                    roi_rows.append({'Metrique': label, 'Valeur': f"{v:,.0f} EUR"})
                else:
                    roi_rows.append({'Metrique': label, 'Valeur': f"{v:.1f}%"})
            st.dataframe(pd.DataFrame(roi_rows), use_container_width=True)

            st.markdown("#### Conformite preventive")
            st.markdown(f"- **Taux dans les delais:** {compliance.get('compliance_rate',0):.1f}%")
            st.markdown(f"- **Ecart moyen:** {compliance.get('avg_deviation_days',0):.1f} jours")
            st.markdown(f"- **Ecart median:** {compliance.get('median_deviation_days',0):.1f} jours")


# ─── PAGE: RECOMMENDATIONS ───────────────────────────────────────────────────

def show_recommendations(data):
    st.markdown('<p class="sub-header">Recommandations</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Constats Critiques")

        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**1. Conformite preventive: 26.5%**")
        st.markdown("Ameliorer le suivi des calendriers et la responsabilite des prestataires")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**2. Fiabilite equipements securite: MTBF ~7 jours**")
        st.markdown("Augmenter la frequence d'inspection ou remplacer les equipements")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**3. Performance prestataires: ecart x25**")
        st.markdown("Renegocier avec les prestataires sous-performants")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("### Opportunites d'Optimisation")

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("**1. Stock pieces:** 11% articles = 80% couts")
        st.markdown("Concentrer l'optimisation sur 144 articles critiques")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("**2. Maintenance predictive:** F1=0.79, AUC=0.97")
        st.markdown("Deployer le modele pour reduire les reparations d'urgence de ~30%")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("**3. Pannes recurrentes:** 267 actifs avec pannes multiples")
        st.markdown("Investiguer les causes racines, envisager le remplacement")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ROI Estime des Recommandations")

    roi_data = pd.DataFrame({
        'Initiative': ['Optimisation stock', 'Maintenance predictive', 'Performance prestataires', 'Remplacement equipements'],
        'Economies attendues': [250000, 400000, 300000, 200000],
        'Cout implementation': [50000, 150000, 20000, 500000],
        'Delai': ['3 mois', '6 mois', '1 mois', '12 mois']
    })
    roi_data['ROI %'] = ((roi_data['Economies attendues'] - roi_data['Cout implementation']) / roi_data['Cout implementation'] * 100).round(1)

    fig = px.bar(roi_data, x='Initiative', y=['Economies attendues', 'Cout implementation'],
                 title='Economies vs Cout', barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(roi_data, use_container_width=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Chargement des donnees..."):
        data = load_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Section", [
        "Vue d'ensemble",
        "Exploration des Donnees",
        "Feature Engineering",
        "Modeles & Analyses",
        "Prediction Interactive",
        "Recommandations",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtres")
    sites = ['Tous'] + sorted(data['corrective']['site'].unique().tolist())
    selected_site = st.sidebar.selectbox("Site", sites)
    if selected_site != 'Tous':
        data['corrective'] = data['corrective'][data['corrective']['site'] == selected_site]
        data['timeline'] = data['timeline'][data['timeline']['site'] == selected_site]

    st.sidebar.markdown("---")
    st.sidebar.caption("GMAO AI Project | Jan-Mar 2025")

    if page == "Vue d'ensemble":
        show_overview(data)
    elif page == "Exploration des Donnees":
        show_exploration(data)
    elif page == "Feature Engineering":
        show_features(data)
    elif page == "Modeles & Analyses":
        show_models(data)
    elif page == "Prediction Interactive":
        show_predictions(data)
    elif page == "Recommandations":
        show_recommendations(data)


if __name__ == "__main__":
    main()
