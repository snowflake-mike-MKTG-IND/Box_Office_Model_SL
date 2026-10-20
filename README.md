# Box Office Opening Weekend Prediction Model

Interactive visualization and prediction tool for theatrical opening weekend box office revenue.

## Features
- **Cascade Classification**: 2-stage tier prediction (SMALL → MID → LARGE+)
- **53 Input Features**: Budget, star power, Google Trends, sentiment, and more
- **Real-time Predictions**: Adjust inputs and see predictions instantly

## Model Architecture
- CatBoost classifiers for tier assignment
- XGBoost regressors for within-tier prediction
- Trained on 400+ theatrical releases (2019-2025)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Maintainer
**Maintained by Brown.** As of October 20, 2026, data-science development of this model has passed from Mike McCarver
to Brown, Mike's data-science agent. From here on Brown runs the work autonomously (studies, challenger models, shadow
forecasts) and keeps this site current. Predictions on these pages are published by Lomis, the production pipeline
agent, after an independent review by Gruney. Promotions of a new model to production are approved by Mike.

- `data/films.json` (the Predictions page) is the predictions of record, committed by Lomis after each reviewed
  publish. It is the same file the public Opening Weekend Intelligence app reads. Never edit it by hand.
- Everything else (pages, model documentation, the generated `data/brown/` model card and research log) is
  committed by Brown.
