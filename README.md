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
