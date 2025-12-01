# NFL Game Predictor

Machine learning system for predicting NFL game outcomes using betting market data and team statistics.

## 🎯 Performance Summary

| Model | Test Accuracy | CV Score | vs Baseline |
|-------|---------------|----------|-------------|
| **Random Forest** | **76.1%** | **67.8% ± 3.1%** | **+18.9%** |
| Logistic Regression | 75.2% | 61.1% ± 2.0% | +14.0% |
| Ensemble | 73.4% | 65.6% ± 2.7% | +11.2% |
| XGBoost | 69.7% | 64.7% ± 1.9% | +5.6% |
| Baseline (Home Field) | 57.0% | - | - |

> **Note**: CV scores are the reliable metric (test set is only 109 games). 67.8% CV matches professional Vegas-level performance.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (rebuilds data with proper feature lagging)
python src/nfl_predictor/main.py --rebuild

# Train with existing data
python src/nfl_predictor/main.py
```

## 📊 Key Features

### 1. Data Leakage Prevention
Critical fix ensuring all features are truly pre-game available:
- **Lagged player statistics**: Uses `.shift(1)` to only include stats from previous weeks
- **TimeSeriesSplit validation**: Trains on earlier games, tests on later (no future leakage)
- **Automated correlation check**: Excludes features with >95% target correlation

```python
# Player stats are lagged to prevent leakage
team_stats[stat] = team_stats.groupby('team')[stat].transform(
    lambda x: x.rolling(window=3).mean().shift(1)  # Only previous weeks
)
```

### 2. Feature Engineering Pipeline
**66 engineered features** from multiple data sources:

| Category | Features | Description |
|----------|----------|-------------|
| Betting Markets | Moneyline, spread, implied probability | Vegas consensus |
| Team Performance | Rolling averages (3-game window) | Lagged team stats |
| Player Aggregates | Fantasy points, yards, TDs | Team-level rollups |
| Situational | Rest advantage, division game, week | Game context |

### 3. Time-Series Validation
Proper temporal validation to prevent training on future data:

```python
# Chronological split (not random shuffle)
Train: Games 1-435 (earlier in season)
Test:  Games 436-544 (later games)

# TimeSeriesSplit for CV (respects game order)
tscv = TimeSeriesSplit(n_splits=5)
```

### 4. Top Predictive Features

| Feature | Importance | Description |
|---------|------------|-------------|
| home_moneyline | 0.067 | Vegas home team odds |
| away_moneyline | 0.058 | Vegas away team odds |
| home_implied_prob | 0.051 | Converted win probability |
| away_implied_prob | 0.040 | Converted win probability |
| fantasy_points_ppr_avg | 0.029 | Lagged team performance |
| spread_line | 0.027 | Point spread |
| pts_avg_diff | 0.022 | Scoring differential |

## 🔧 Project Structure

```
capstone-nfl-game-predictor/
├── src/nfl_predictor/
│   ├── config.py           # Centralized configuration
│   ├── data_preparer.py    # Data collection with lagging
│   ├── model_builder.py    # ML pipeline with TimeSeriesSplit
│   └── main.py             # CLI with --rebuild flag
├── data/
│   ├── games_with_features.csv
│   └── weekly_stats.csv
├── models/
│   ├── nfl_predictor.pkl   # Trained model
│   ├── scaler.pkl          # Feature scaler
│   └── feature_names.pkl   # Feature list
├── notebooks/
│   └── nfl_eda.ipynb       # Exploratory analysis
└── figures/                # Generated visualizations
```

## 📈 Model Insights

### What Works
1. **Betting markets are efficient** — Implied probabilities are the strongest predictors
2. **Lagged team stats add value** — Rolling averages capture momentum
3. **Rest differential matters** — But effect is non-linear
4. **Ensemble reduces variance** — Though Random Forest performs best solo

### Realistic Expectations

| Accuracy | Meaning |
|----------|---------|
| 52.4% | Break-even point for betting (accounting for vig) |
| 57% | Baseline (always pick home team) |
| **60-68%** | **Professional sports betting models** |
| **67.8%** | **Our model (CV score)** |
| 70%+ | Unusual — verify no leakage |
| 85%+ | Almost certainly data leakage |

### Challenges Overcome
- **Data Leakage**: Initial 83%+ accuracy was from including same-week player stats
- **Temporal Leakage**: StratifiedKFold with shuffle trained on future games
- **Feature Timing**: Ensured all features use only pre-game information

## 📊 Validation Results

### Cross-Validation (TimeSeriesSplit)
```
Random Forest:      67.8% ± 3.1%
Ensemble:           65.6% ± 2.7%
XGBoost:            64.7% ± 1.9%
Logistic Regression: 61.1% ± 2.0%
```

### Model Calibration
- **Brier Score**: 0.189 (lower is better)
- Well-calibrated probability estimates

## 🛠️ Configuration

Edit `config.py` to customize:

```python
@dataclass
class Config:
    seasons: List[int] = [2023, 2024]
    test_size: float = 0.2
    rolling_window: int = 3
    lag_periods: int = 1  # Critical: prevents leakage
    use_time_series_split: bool = True
    correlation_threshold: float = 0.95
```

## 📚 Data Sources

- **Game Data**: [nfl_data_py](https://github.com/cooperdff/nfl_data_py)
- **Betting Lines**: Historical odds from NFL schedules API
- **Player Stats**: Weekly performance metrics

## 🛠️ Requirements

- Python 3.8+
- scikit-learn >= 1.0
- xgboost >= 1.5
- pandas >= 1.5
- numpy >= 1.20
- nfl-data-py >= 0.3.0

## 📝 Future Enhancements

- [ ] Incorporate weather data
- [ ] Add injury report analysis
- [ ] Implement Elo rating system
- [ ] Test on 2025 season (true out-of-sample)
- [ ] Add opponent-adjusted statistics
- [ ] Remove betting features to test "football knowledge"

## 👨‍💻 Author

**Bin Ware**
Northwest Missouri State University
Capstone Project — 2025
