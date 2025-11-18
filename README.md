# NFL Game Predictor

Advanced machine learning system for predicting NFL game outcomes with 83% accuracy.

## 🎯 Performance Summary

| Model | Accuracy | CV Score | Improvement |
|-------|----------|----------|-------------|
| **Ensemble (Final)** | **82.5%** | **87.1% ± 3.6%** | **+44.7%** |
| XGBoost | 83.3% | 82.7% ± 3.1% | +46.2% |
| Logistic Regression | 82.5% | 86.6% ± 4.5% | +44.7% |
| Random Forest | 80.7% | 80.7% ± 3.8% | +41.6% |
| Baseline (Home Field) | 57.0% | - | - |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/nfl_predictor/main.py

# Make predictions
python src/nfl_predictor/predict.py --week 15 --season 2024
```

## 📊 Advanced ML Features

### 1. Ensemble Learning Architecture
Combines three complementary algorithms using soft voting:
- **Logistic Regression**: Captures linear relationships in betting odds
- **Random Forest**: Identifies non-linear patterns in team matchups
- **XGBoost**: Optimizes for complex feature interactions

### 2. Data Leakage Prevention
Automated detection and removal of result-contaminated features:
```python
# Automatic correlation check
for feature in features:
    if correlation_with_target(feature) > 0.95:
        exclude_feature(feature)
```

### 3. Feature Engineering Pipeline
**47 engineered features** from multiple data sources:

#### Betting Market Features
- Moneyline odds → Implied probability conversion
- Point spread with home field adjustment
- Over/under totals for game pace

#### Player Performance Aggregation
- Team-level statistics from individual player projections
- Fantasy points as composite performance metric
- Position group contributions (QB, RB, WR, etc.)

#### Situational Features
- Rest advantage (days between games)
- Division rivalry indicator
- Season progress weighting

### 4. Cross-Validation Strategy
5-fold stratified cross-validation ensuring:
- Consistent home/away win distribution
- No data leakage between folds
- Robust performance estimates

### 5. Feature Importance Analysis

Top predictive features identified through Random Forest:

| Feature | Importance | Description |
|---------|------------|-------------|
| carries_away_players | 0.125 | Projected rushing attempts |
| carries | 0.112 | Total rushing volume |
| home_implied_prob | 0.058 | Betting market consensus |
| fantasy_points_ppr | 0.057 | Composite performance metric |
| rushing_yards | 0.044 | Ground game projection |

## 🔧 Technical Implementation

### Project Structure
```
capstone-nfl-game-predictor/
├── src/nfl_predictor/
│   ├── model_builder.py    # Core ML pipeline
│   ├── data_preparer.py    # Data collection & processing
│   └── main.py             # Training orchestration
├── data/
│   ├── games_with_features.csv
│   └── weekly_stats.csv
├── models/
│   └── nfl_predictor.pkl   # Trained ensemble model
└── notebooks/
    └── analysis.ipynb      # Exploratory analysis
```

### Key Components

#### Data Collection
```python
# Automated data pipeline
preparer = DataPreparer(seasons=[2023, 2024])
games = preparer.collect_game_data()
player_stats = preparer.collect_player_stats()
```

#### Model Training
```python
# Ensemble configuration
predictor = NFLGamePredictor()
predictor.load_data()
X, y = predictor.prepare_features()
results = predictor.train_models(X, y)
```

#### Prediction
```python
# Make predictions on new games
model = joblib.load('models/nfl_predictor.pkl')
prediction = model.predict(game_features)
probability = model.predict_proba(game_features)
```

## 📈 Model Insights

### What Works
1. **Betting markets are efficient** - Implied probabilities from odds are strongest predictors
2. **Player projections matter** - Aggregated team-level stats improve accuracy by ~5%
3. **Rest differential is non-linear** - Extreme rest advantages (>7 days) have outsized impact
4. **Ensemble reduces variance** - Combining models improves stability by 3-4%

### Challenges Overcome
- **Data Leakage**: Initial 100% accuracy from including post-game features
- **Feature Timing**: Ensuring all features are truly pre-game available
- **Class Imbalance**: 55.6% home win rate handled through stratification

### Realistic Expectations
- **57%**: Baseline (always pick home team)
- **65-70%**: Professional betting models
- **75-80%**: Elite models with proprietary data
- **83%**: Our model (strong performance, validate on future games)
- **>90%**: Likely indicates data leakage

## 🔬 Validation Results

### Cross-Validation Performance
```
Fold 1: 84.2%
Fold 2: 89.5%
Fold 3: 86.8%
Fold 4: 91.2%
Fold 5: 83.7%
Mean: 87.1% ± 3.6%
```

### Test Set Confusion Matrix
```
              Predicted
              Away  Home
Actual Away   [38    12]
       Home   [ 8    56]

Precision: 0.82
Recall: 0.88
F1-Score: 0.85
```

## 📚 Data Sources

- **Game Data**: [nfl_data_py](https://github.com/cooperdff/nfl_data_py) - Official NFL data
- **Betting Lines**: Historical odds and spreads
- **Player Stats**: Weekly projections and performance metrics

## 🛠️ Requirements

- Python 3.8+
- scikit-learn >= 1.7.2
- xgboost >= 3.1.1
- pandas >= 1.5.3
- numpy >= 1.26.4
- nfl-data-py >= 0.3.0

## 📝 Future Enhancements

- [ ] Incorporate weather data
- [ ] Add injury report analysis
- [ ] Implement Elo rating system
- [ ] Create real-time prediction API
- [ ] Add neural network model
- [ ] Include coaching tendencies

## 👨‍💻 Author

**Bin Ware**
Northwest Missouri State University
Capstone Project - Fall 2024

