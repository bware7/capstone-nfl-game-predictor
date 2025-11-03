"""Clean and prepare NFL data for modeling."""

import pandas as pd

from analytics_project.utils_logger import logger


def create_game_features(games_df):
    """Create features for game prediction."""
    df = games_df.copy()

    # Create target: 1 if home team won
    df['home_won'] = (df['home_score'] > df['away_score']).astype(int)

    # Calculate running averages for each team
    for team_type in ['home', 'away']:
        df[f'{team_type}_avg_score'] = df.groupby(f'{team_type}_team')[
            f'{team_type}_score'
        ].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    # Rest advantage
    df['rest_advantage'] = df['home_rest'] - df['away_rest']

    # Convert spread to numeric
    df['spread_numeric'] = pd.to_numeric(df['spread_line'], errors='coerce')

    logger.info(f"Created features. Shape: {df.shape}")
    return df


def prepare_training_data(df):
    """Prepare data for model training."""
    # Features to use
    features = ['home_avg_score', 'away_avg_score', 'rest_advantage', 'spread_numeric', 'div_game']

    # Remove games with missing features
    df_clean = df[features + ['home_won']].dropna()

    X = df_clean[features]
    y = df_clean['home_won']

    logger.info(f"Training data ready: {X.shape[0]} games")
    return X, y
