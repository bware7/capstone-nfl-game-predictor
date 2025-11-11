"""NFL Data Cleaning and Feature Engineering Module.

This module handles data preprocessing and feature creation
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_nfl_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess NFL game data.

    Args:
        df: Raw NFL game data

    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning...")
    df = df.copy()

    # Convert gameday to datetime
    df['gameday'] = pd.to_datetime(df['gameday'])

    # Create target variable
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

    # Create basic features
    df['point_differential'] = df['home_score'] - df['away_score']
    df['total_points'] = df['home_score'] + df['away_score']
    df['is_playoff'] = (df['game_type'] == 'POST').astype(int)
    df['is_division'] = df['div_game'].fillna(False).astype(int)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def calculate_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rest days between games for each team.

    Args:
        df: Game data with gameday column

    Returns:
        DataFrame with rest day features
    """
    logger.info("Calculating rest days...")
    df = df.sort_values('gameday').copy()

    # Initialize rest day columns
    df['home_rest_days'] = 7  # Default
    df['away_rest_days'] = 7

    # Track last game date for each team
    team_last_game = {}

    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        game_date = row['gameday']

        # Calculate rest for home team
        if home_team in team_last_game:
            rest = (game_date - team_last_game[home_team]).days
            df.at[idx, 'home_rest_days'] = rest

        # Calculate rest for away team
        if away_team in team_last_game:
            rest = (game_date - team_last_game[away_team]).days
            df.at[idx, 'away_rest_days'] = rest

        # Update last game dates
        team_last_game[home_team] = game_date
        team_last_game[away_team] = game_date

    # Calculate rest advantage
    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

    logger.info("Rest days calculated successfully")
    return df


def calculate_rolling_averages(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Calculate rolling performance averages for teams.

    Args:
        df: Game data
        window: Number of games for rolling average

    Returns:
        DataFrame with rolling average features
    """
    logger.info(f"Calculating {window}-game rolling averages...")
    df = df.sort_values('gameday').copy()

    # Initialize rolling average columns
    rolling_cols = [
        f'home_rolling_pts_for_{window}',
        f'home_rolling_pts_against_{window}',
        f'home_rolling_win_rate_{window}',
        f'away_rolling_pts_for_{window}',
        f'away_rolling_pts_against_{window}',
        f'away_rolling_win_rate_{window}',
    ]

    for col in rolling_cols:
        df[col] = np.nan

    # Calculate for each team
    teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

    for team in teams:
        # Get all games for this team
        team_mask_home = df['home_team'] == team
        team_mask_away = df['away_team'] == team

        # Create team game history
        team_games = []

        for idx, row in df.iterrows():
            if row['home_team'] == team:
                team_games.append(
                    {
                        'game_idx': idx,
                        'date': row['gameday'],
                        'pts_for': row['home_score'],
                        'pts_against': row['away_score'],
                        'won': row['home_win'],
                        'is_home': True,
                    }
                )
            elif row['away_team'] == team:
                team_games.append(
                    {
                        'game_idx': idx,
                        'date': row['gameday'],
                        'pts_for': row['away_score'],
                        'pts_against': row['home_score'],
                        'won': 1 - row['home_win'],
                        'is_home': False,
                    }
                )

        # Sort by date
        team_games = sorted(team_games, key=lambda x: x['date'])

        # Calculate rolling averages
        for i, game in enumerate(team_games):
            if i >= 1:  # Need at least 1 previous game
                # Get previous games (up to window size)
                prev_games = team_games[max(0, i - window) : i]

                if prev_games:
                    avg_pts_for = np.mean([g['pts_for'] for g in prev_games])
                    avg_pts_against = np.mean([g['pts_against'] for g in prev_games])
                    avg_win_rate = np.mean([g['won'] for g in prev_games])

                    # Assign to appropriate column
                    idx = game['game_idx']
                    if game['is_home']:
                        df.at[idx, f'home_rolling_pts_for_{window}'] = avg_pts_for
                        df.at[idx, f'home_rolling_pts_against_{window}'] = avg_pts_against
                        df.at[idx, f'home_rolling_win_rate_{window}'] = avg_win_rate
                    else:
                        df.at[idx, f'away_rolling_pts_for_{window}'] = avg_pts_for
                        df.at[idx, f'away_rolling_pts_against_{window}'] = avg_pts_against
                        df.at[idx, f'away_rolling_win_rate_{window}'] = avg_win_rate

    # Fill NaN values with league averages
    for col in rolling_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    logger.info("Rolling averages calculated successfully")
    return df


def create_model_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create final feature set for modeling.

    Args:
        df: Cleaned game data with engineered features

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    logger.info("Creating model features...")

    # Select features for model
    feature_cols = [
        'home_rest_days',
        'away_rest_days',
        'rest_advantage',
        'is_playoff',
        'is_division',
        'home_rolling_pts_for_3',
        'home_rolling_pts_against_3',
        'home_rolling_win_rate_3',
        'away_rolling_pts_for_3',
        'away_rolling_pts_against_3',
        'away_rolling_win_rate_3',
    ]

    # Filter to available columns
    available_features = [col for col in feature_cols if col in df.columns]

    # Create feature matrix
    X = df[available_features].copy()
    y = df['home_win'].copy()

    # Remove rows with NaN
    valid_idx = X.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {X.columns.tolist()}")

    return X, y


def prepare_data_for_modeling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Complete data preparation pipeline.

    Args:
        df: Raw NFL game data

    Returns:
        Tuple of (processed DataFrame, features, target)
    """
    # Clean data
    df_clean = clean_nfl_data(df)

    # Add rest days
    df_clean = calculate_rest_days(df_clean)

    # Add rolling averages
    df_clean = calculate_rolling_averages(df_clean, window=3)

    # Create model features
    X, y = create_model_features(df_clean)

    return df_clean, X, y


if __name__ == "__main__":
    # Example usage
    from src.nfl_predictor.data_collector import collect_nfl_data

    # Collect data
    games = collect_nfl_data([2023, 2024])

    # Prepare for modeling
    df_processed, X, y = prepare_data_for_modeling(games)

    print(f"Processed data shape: {df_processed.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
