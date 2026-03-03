# GMAO AI - Maintenance Intelligente

Systeme d'analyse et de prediction pour la maintenance industrielle, base sur des donnees GMAO (Jan-Mar 2025, 5 sites, 7090 actifs).

## Resultats

| Modele | Metrique | Score |
|--------|----------|-------|
| Prediction de panne (30j) | F1 / AUC | 0.790 / 0.971 |
| Prediction pieces de rechange | F1 / AUC | 0.948 / 0.992 |
| Detection d'anomalies | Consensus / Accord | 13.2% / 93.7% |

## Structure

```
gmao-ai-project/
├── src/                    # Modules Python reutilisables
│   ├── preprocessing.py        # Chargement, nettoyage, integration
│   ├── feature_engineering.py  # 95 features + 5 cibles
│   └── models/
│       ├── failure_predictor.py    # RandomForest (panne) + GradientBoosting (pieces)
│       ├── anomaly_detector.py     # Isolation Forest + One-Class SVM
│       ├── effectiveness.py        # ROI, conformite, intervalles optimaux
│       └── spare_parts_forecaster.py  # Pareto, prevision demande, reapprovisionnement
├── app/
│   └── streamlit_app.py   # Dashboard interactif (6 pages)
├── data/
│   ├── raw/                # 3 fichiers Excel source
│   ├── processed/          # CSVs nettoyes + integres
│   └── features/           # Dataset ML-ready (95 features, 2035 samples)
├── models/                 # Modeles entraines (.pkl) + metriques
├── reports/                # Sorties d'analyse (CSV, PNG, JSON)
├── notebooks/              # Notebooks d'exploration
├── run_pipeline.py         # Pipeline bout-en-bout (4 etapes)
├── requirements.txt
└── gmao_tme_technical_report_.pdf
```

## Utilisation

### Installation
```bash
pip install -r requirements.txt
```

### Pipeline complet
```bash
python run_pipeline.py          # Tout lancer (preprocessing → features → modeles → analyses)
python run_pipeline.py --step 3 # Lancer une etape specifique (1-4)
```

### Dashboard
```bash
streamlit run app/streamlit_app.py
```

Le dashboard inclut : vue d'ensemble, exploration des donnees, feature engineering, resultats des modeles, **prediction interactive** (saisir les parametres d'un OT et obtenir les predictions en direct), et recommandations.

### Deploiement Streamlit Cloud
1. Pousser le repo sur GitHub (prive)
2. Connecter le repo sur [share.streamlit.io](https://share.streamlit.io)
3. Configurer : fichier principal = `app/streamlit_app.py`

## Donnees sources

- **Ordres de travail.xlsx** — 2,035 ordres correctifs
- **OT_Preventifs.xlsx** — 9,149 ordres preventifs
- **a-w.xlsx** — 12,573 demandes de pieces de rechange
