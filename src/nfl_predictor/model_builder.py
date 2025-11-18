"""
NFL Game Predictor - Main Model Builder
Consolidated version with data leakage prevention
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Install with: pip install xgboost")


class NFLGamePredictor:
    """Main NFL game prediction model with data leakage prevention"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []

        # Features that leak the target (must be excluded)
        self.leaked_features = [
            'home_won',
            'home_team_won',
            'winner',
            'result',
            'home_score',
            'away_score',
            'total',
            'final_score',
            'home_points',
            'away_points',
            'score_differential',
            'home_score_differential',
            'away_score_differential',
            # EPA features are calculated post-game
            'passing_epa',
            'rushing_epa',
            'receiving_epa',
            'passing_epa_home_players',
            'passing_epa_away_players',
            'rushing_epa_home_players',
            'rushing_epa_away_players',
            'receiving_epa_home_players',
            'receiving_epa_away_players',
        ]

    def load_data(
        self,
        game_data_path='data/games_with_features.csv',
        player_stats_path='data/weekly_stats.csv',
    ):
        """Load game and player data"""
        print("Loading data...")

        # Load main game data
        self.games_df = pd.read_csv(game_data_path)
        print(f"  Loaded {len(self.games_df)} games")

        # Try to load and merge player stats
        try:
            self.player_stats = pd.read_csv(player_stats_path)
            self._merge_player_stats()
        except:
            print("  Player stats not available or couldn't be merged")

        return self.games_df

    def _merge_player_stats(self):
        """Aggregate and merge player statistics at team level"""
        # Only use pre-game projectable stats
        safe_stats = [
            'completions',
            'attempts',
            'passing_yards',
            'passing_tds',
            'carries',
            'rushing_yards',
            'rushing_tds',
            'targets',
            'receptions',
            'receiving_yards',
            'receiving_tds',
            'fantasy_points_ppr',
        ]

        available_stats = [col for col in safe_stats if col in self.player_stats.columns]
        if not available_stats:
            return

        # Aggregate by team and week
        team_stats = (
            self.player_stats.groupby(['recent_team', 'season', 'week'])[available_stats]
            .sum()
            .reset_index()
        )

        # Merge for home and away teams
        self.games_df = self.games_df.merge(
            team_stats,
            left_on=['home_team', 'season', 'week'],
            right_on=['recent_team', 'season', 'week'],
            how='left',
            suffixes=('', '_home_players'),
        ).merge(
            team_stats,
            left_on=['away_team', 'season', 'week'],
            right_on=['recent_team', 'season', 'week'],
            how='left',
            suffixes=('', '_away_players'),
        )

        print(f"  Merged {len(available_stats)} player statistics")

    def prepare_features(self):
        """Prepare features, excluding any that leak the target"""
        print("\nPreparing features...")

        # Get numeric columns
        numeric_cols = self.games_df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude leaked features and identifiers
        exclude = self.leaked_features + [
            'game_id',
            'season',
            'week',
            'gameday',
            'gametime',
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
        ]

        # Filter to safe features only
        feature_cols = [col for col in numeric_cols if col not in exclude]

        # Additional safety check - remove high correlation features
        if 'home_won' in self.games_df.columns:
            y = self.games_df['home_won']
            safe_features = []
            for col in feature_cols:
                if col in self.games_df.columns:
                    try:
                        corr = abs(self.games_df[col].fillna(0).corr(y))
                        if corr < 0.95:  # Keep only features with <95% correlation
                            safe_features.append(col)
                        elif corr >= 0.95:
                            print(f"  Excluding '{col}' - too correlated ({corr:.3f})")
                    except:
                        safe_features.append(col)
            feature_cols = safe_features

        # Engineer additional features
        self._engineer_features()

        # Update feature list with engineered features
        engineered = [
            col for col in self.games_df.columns if col not in numeric_cols and col not in exclude
        ]
        feature_cols.extend(
            [col for col in engineered if self.games_df[col].dtype in ['int64', 'float64']]
        )

        self.feature_names = feature_cols
        print(f"  Using {len(feature_cols)} features")

        # Prepare X and y
        X = self.games_df[feature_cols].fillna(0)
        y = self.games_df['home_won'].astype(int)

        # Remove rows with missing target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"  Final dataset: {X.shape}")
        print(f"  Home win rate: {y.mean():.2%}")

        return X, y

    def _engineer_features(self):
        """Engineer additional features from pre-game data"""
        df = self.games_df

        # Betting market features (if available)
        if 'home_moneyline' in df.columns and 'away_moneyline' in df.columns:
            # Convert moneyline to implied probability
            df['home_implied_prob'] = df['home_moneyline'].apply(
                lambda x: abs(x) / (abs(x) + 100) if x < 0 else 100 / (x + 100) if x > 0 else 0.5
            )
            df['away_implied_prob'] = df['away_moneyline'].apply(
                lambda x: abs(x) / (abs(x) + 100) if x < 0 else 100 / (x + 100) if x > 0 else 0.5
            )

        # Rest advantage
        if 'home_rest' in df.columns and 'away_rest' in df.columns:
            df['rest_differential'] = df['home_rest'] - df['away_rest']

        # Division game indicator
        if 'div_game' in df.columns:
            df['is_division'] = df['div_game'].astype(float)

        # Season progress
        if 'week' in df.columns:
            df['season_progress'] = df['week'] / 18.0

        # Player stats differentials (if available)
        for stat in ['passing_yards', 'rushing_yards', 'receiving_yards', 'fantasy_points_ppr']:
            home_col = f'{stat}_home_players'
            away_col = f'{stat}_away_players'
            if home_col in df.columns and away_col in df.columns:
                df[f'{stat}_diff'] = df[home_col].fillna(0) - df[away_col].fillna(0)

    def train_models(self, X, y):
        """Train multiple models and select best performer"""
        print("\nTraining models...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, C=0.5),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            ),
        }

        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                eval_metric='logloss',
            )

        # Train each model
        results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            # Use scaled data for logistic regression
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=skf)

            acc = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'test_acc': acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
            }
            self.models[name] = model

            print(f"  {name}: {acc:.3f} (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")

        # Create ensemble
        if len(models) > 1:
            ensemble = VotingClassifier(estimators=list(models.items()), voting='soft')
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            cv_scores = cross_val_score(ensemble, X_train, y_train, cv=skf)

            results['ensemble'] = {
                'model': ensemble,
                'test_acc': accuracy_score(y_test, y_pred),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
            }
            self.models['ensemble'] = ensemble

            print(
                f"  ensemble: {results['ensemble']['test_acc']:.3f} "
                f"(CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})"
            )

        # Select best model
        best_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_name]['model']
        self.results = results

        print(f"\nBest model: {best_name} ({results[best_name]['cv_mean']:.3f})")

        return results, X_test, y_test

    def get_feature_importance(self, top_n=10):
        """Get feature importance from tree-based models"""
        if 'random_forest' in self.models:
            importance = pd.DataFrame(
                {
                    'feature': self.feature_names,
                    'importance': self.models['random_forest'].feature_importances_,
                }
            ).sort_values('importance', ascending=False)

            print(f"\nTop {top_n} important features:")
            for _, row in importance.head(top_n).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

            return importance
        return None

    def evaluate(self, X_test, y_test):
        """Generate evaluation report"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        baseline = 0.57  # Home field advantage

        for name, result in self.results.items():
            acc = result['test_acc']
            improvement = (acc - baseline) / baseline * 100
            print(f"\n{name}:")
            print(f"  Accuracy: {acc:.3f}")
            print(f"  CV Score: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
            print(f"  vs Baseline: +{improvement:.1f}%")

        best_acc = max(r['test_acc'] for r in self.results.values())

        # Sanity check
        if best_acc > 0.85:
            print("\n⚠️ WARNING: Accuracy >85% is unusually high for NFL prediction")
            print("  Consider checking for data leakage")
        elif best_acc > 0.75:
            print("\n✅ Good performance - near professional level")
        elif best_acc > 0.65:
            print("\n✅ Solid performance - better than baseline")
        else:
            print("\n💡 Room for improvement - consider more features")

        return best_acc
