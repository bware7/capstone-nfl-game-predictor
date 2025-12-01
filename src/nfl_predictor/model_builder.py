"""
NFL Game Predictor - Model Builder
FIXED: TimeSeriesSplit validation to prevent temporal leakage
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from config import config

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Install with: pip install xgboost")


class NFLGamePredictor:
    """NFL game prediction model with proper temporal validation"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []
        self.results = {}

        # Features that MUST be excluded (post-game information)
        self.leaked_features = {
            # Target and direct derivatives
            'home_won',
            'home_team_won',
            'winner',
            'result',
            # Final scores
            'home_score',
            'away_score',
            'total',
            'final_score',
            'home_points',
            'away_points',
            # Score differentials
            'score_differential',
            'home_score_differential',
            'away_score_differential',
            # EPA features (calculated post-game)
            'passing_epa',
            'rushing_epa',
            'receiving_epa',
            'passing_epa_home_players',
            'passing_epa_away_players',
            'rushing_epa_home_players',
            'rushing_epa_away_players',
            'receiving_epa_home_players',
            'receiving_epa_away_players',
            # Raw same-week player stats (if not properly lagged)
            'completions_home_players',
            'completions_away_players',
            'passing_yards_home_players',
            'passing_yards_away_players',
            'rushing_yards_home_players',
            'rushing_yards_away_players',
        }

        # Identifier columns to exclude
        self.id_columns = {
            'game_id',
            'old_game_id',
            'gsis',
            'nfl_detail_id',
            'pfr',
            'pff',
            'espn',
            'ftn',
            'recent_team',
            'recent_team_home_players',
            'recent_team_away_players',
            'gameday',
            'gametime',
            'home_team',
            'away_team',
            'stadium',
            'stadium_id',
            'roof',
            'surface',
        }

    def load_data(self, game_data_path=None, player_stats_path=None):
        """Load prepared game data"""
        game_data_path = game_data_path or config.game_data_path
        player_stats_path = player_stats_path or config.player_stats_path

        print("Loading data...")
        self.games_df = pd.read_csv(game_data_path)

        # Ensure chronological order
        if 'season' in self.games_df.columns and 'week' in self.games_df.columns:
            self.games_df = self.games_df.sort_values(['season', 'week']).reset_index(drop=True)

        print(f"  Loaded {len(self.games_df)} games")
        return self.games_df

    def prepare_features(self):
        """Prepare features, strictly excluding any leakage sources"""
        print("\nPreparing features...")

        # Get numeric columns
        numeric_cols = self.games_df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude leaked features and identifiers
        exclude = self.leaked_features | self.id_columns | {'season', 'week'}

        feature_cols = [col for col in numeric_cols if col not in exclude]

        # Correlation check for remaining features
        if 'home_won' in self.games_df.columns:
            y = self.games_df['home_won']
            safe_features = []

            for col in feature_cols:
                try:
                    corr = abs(self.games_df[col].fillna(0).corr(y))
                    if corr < config.correlation_threshold:
                        safe_features.append(col)
                    else:
                        print(f"  ⚠️ Excluding '{col}' - correlation {corr:.3f} too high")
                except:
                    safe_features.append(col)

            feature_cols = safe_features

        self.feature_names = feature_cols
        print(f"  Using {len(feature_cols)} features")

        # Prepare X and y
        X = self.games_df[feature_cols].fillna(0)
        y = self.games_df['home_won'].astype(int)

        # Remove rows with missing target
        valid_idx = ~y.isna()
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)

        print(f"  Dataset shape: {X.shape}")
        print(f"  Home win rate: {y.mean():.2%}")

        return X, y

    def train_models(self, X, y):
        """Train models with proper time-series validation"""
        print("\nTraining models...")

        n_samples = len(X)

        # Time-series split: use last 20% as holdout test set
        train_size = int(n_samples * (1 - config.test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        print(f"  Train: {len(X_train)} games (weeks 1-{train_size})")
        print(f"  Test:  {len(X_test)} games (final {len(X_test)} games)")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        models = {
            'logistic_regression': LogisticRegression(
                random_state=config.random_state, max_iter=1000, C=0.5
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=config.random_state,
            ),
        }

        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=config.random_state,
                eval_metric='logloss',
            )

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=config.cv_folds)

        results = {}
        for name, model in models.items():
            print(f"\n  Training {name}...")

            # Use scaled data for logistic regression
            if name == 'logistic_regression':
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train.values, X_test.values

            # Fit model
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

            # Cross-validation with time series split
            cv_scores = cross_val_score(model, X_tr, y_train, cv=tscv, scoring='accuracy')

            acc = accuracy_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_proba) if y_proba is not None else None

            results[name] = {
                'model': model,
                'test_acc': acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'brier': brier,
                'predictions': y_pred,
            }
            self.models[name] = model

            brier_str = f", Brier: {brier:.4f}" if brier else ""
            print(
                f"    Accuracy: {acc:.3f} | CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}{brier_str}"
            )

        # Create ensemble if multiple models
        if len(models) > 1:
            print(f"\n  Training ensemble...")
            ensemble = VotingClassifier(estimators=list(models.items()), voting='soft')
            ensemble.fit(X_train.values, y_train)
            y_pred = ensemble.predict(X_test.values)
            y_proba = ensemble.predict_proba(X_test.values)[:, 1]

            cv_scores = cross_val_score(ensemble, X_train.values, y_train, cv=tscv)

            acc = accuracy_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_proba)

            results['ensemble'] = {
                'model': ensemble,
                'test_acc': acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'brier': brier,
                'predictions': y_pred,
            }
            self.models['ensemble'] = ensemble

            print(
                f"    Accuracy: {acc:.3f} | CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}, Brier: {brier:.4f}"
            )

        # Select best model by CV score
        best_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_name]['model']
        self.results = results

        print(f"\n  Best model: {best_name} (CV: {results[best_name]['cv_mean']:.3f})")

        return results, X_test, y_test

    def get_feature_importance(self, top_n=10):
        """Get feature importance from tree-based models"""
        if 'random_forest' not in self.models:
            return None

        importance = pd.DataFrame(
            {
                'feature': self.feature_names,
                'importance': self.models['random_forest'].feature_importances_,
            }
        ).sort_values('importance', ascending=False)

        print(f"\nTop {top_n} features:")
        for _, row in importance.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return importance

    def evaluate(self, X_test, y_test):
        """Generate evaluation report with realistic expectations"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        baseline_home = 0.57  # Home field advantage
        vegas_benchmark = 0.66  # Professional benchmark

        for name, result in self.results.items():
            acc = result['test_acc']
            vs_baseline = (acc - baseline_home) / baseline_home * 100
            vs_vegas = (acc - vegas_benchmark) / vegas_benchmark * 100

            print(f"\n{name}:")
            print(f"  Test Accuracy: {acc:.3f}")
            print(f"  CV Score:      {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            print(f"  vs Baseline:   {vs_baseline:+.1f}%")
            print(f"  vs Vegas:      {vs_vegas:+.1f}%")

            if result['brier']:
                print(f"  Brier Score:   {result['brier']:.4f}")

        best_acc = max(r['test_acc'] for r in self.results.values())

        # Performance interpretation
        print("\n" + "-" * 50)
        if best_acc > 0.65:
            print("✅ EXCELLENT: Strong predictive performance!")
        elif best_acc > 0.60:
            print("✅ GOOD: Above baseline, shows predictive signal.")
        else:
            print("💡 BASELINE: Consider adding more predictive features.")

        return best_acc

    def predict(self, X):
        """Make predictions on new data"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call train_models first.")

        X_scaled = self.scaler.transform(X)

        # Use appropriate input based on model type
        if isinstance(self.best_model, VotingClassifier):
            return self.best_model.predict(X)
        elif 'logistic' in str(type(self.best_model)).lower():
            return self.best_model.predict(X_scaled)
        else:
            return self.best_model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call train_models first.")

        if isinstance(self.best_model, VotingClassifier):
            return self.best_model.predict_proba(X)
        elif 'logistic' in str(type(self.best_model)).lower():
            X_scaled = self.scaler.transform(X)
            return self.best_model.predict_proba(X_scaled)
        else:
            return self.best_model.predict_proba(X)


if __name__ == "__main__":
    # Quick test
    predictor = NFLGamePredictor()
    predictor.load_data()
    X, y = predictor.prepare_features()
    results, X_test, y_test = predictor.train_models(X, y)
    predictor.get_feature_importance()
    predictor.evaluate(X_test, y_test)
