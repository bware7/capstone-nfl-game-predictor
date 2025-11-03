"""Collect NFL game data from various sources."""

import nfl_data_py as nfl
import pandas as pd

from analytics_project.utils_logger import logger


def get_games_data(years=[2023, 2024]):
    """Get game results and team stats."""
    logger.info(f"Fetching games for years: {years}")

    # Get game results - this has home/away teams and scores
    games = nfl.import_schedules(years)

    # Log what we got
    logger.info(f"Game columns: {games.columns.tolist()}")

    # Show a sample game
    if not games.empty:
        sample = games[['home_team', 'away_team', 'home_score', 'away_score', 'week']].head(1)
        logger.info(f"Sample game: {sample.to_dict('records')[0]}")

    return games


def save_data(df, filename):
    """Save dataframe to data folder."""
    df.to_csv(f"data/{filename}.csv", index=False)
    logger.info(f"Saved {filename}.csv")
