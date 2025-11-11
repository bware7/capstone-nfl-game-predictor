"""NFL Data Collection Module.

This module handles data collection from nfl_data_py API
"""

import logging
from typing import List, Optional

import nfl_data_py as nfl
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_nfl_data(seasons: List[int], game_types: Optional[List[str]] = None) -> pd.DataFrame:
    """Collect NFL game data for specified seasons.

    Args:
        seasons: List of seasons to collect (e.g., [2023, 2024])
        game_types: List of game types to include (default: ['REG', 'POST'])

    Returns:
        DataFrame with NFL game data
    """
    if game_types is None:
        game_types = ['REG', 'POST']

    logger.info(f"Collecting data for seasons: {seasons}")

    # Collect schedule data
    all_games = []
    for season in seasons:
        try:
            games = nfl.import_schedules([season])
            all_games.append(games)
            logger.info(f"Collected {len(games)} games for season {season}")
        except Exception as e:
            logger.error(f"Error collecting season {season}: {e}")

    if not all_games:
        raise ValueError("No data collected")

    # Combine all seasons
    df = pd.concat(all_games, ignore_index=True)

    # Filter for specified game types and completed games
    df = df[
        (df['game_type'].isin(game_types)) & (df['home_score'].notna()) & (df['away_score'].notna())
    ].copy()

    logger.info(f"Total games collected: {len(df)}")

    return df


def collect_team_stats(seasons: List[int]) -> pd.DataFrame:
    """Collect team statistics for specified seasons.

    Args:
        seasons: List of seasons

    Returns:
        DataFrame with team statistics
    """
    logger.info("Collecting team statistics...")

    try:
        team_stats = nfl.import_seasonal_data(seasons)
        logger.info(f"Collected stats for {len(team_stats)} team-seasons")
        return team_stats
    except Exception as e:
        logger.error(f"Error collecting team stats: {e}")
        return pd.DataFrame()


def collect_player_stats(seasons: List[int], stat_type: str = 'passing') -> pd.DataFrame:
    """Collect player statistics.

    Args:
        seasons: List of seasons
        stat_type: Type of stats ('passing', 'rushing', 'receiving')

    Returns:
        DataFrame with player statistics
    """
    logger.info(f"Collecting {stat_type} statistics...")

    try:
        if stat_type == 'passing':
            stats = nfl.import_seasonal_data(seasons, s_type='pass')
        elif stat_type == 'rushing':
            stats = nfl.import_seasonal_data(seasons, s_type='rush')
        elif stat_type == 'receiving':
            stats = nfl.import_seasonal_data(seasons, s_type='rec')
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")

        logger.info(f"Collected {len(stats)} {stat_type} records")
        return stats
    except Exception as e:
        logger.error(f"Error collecting {stat_type} stats: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    games_df = collect_nfl_data([2023, 2024])
    print(games_df.head())
    print(f"\nShape: {games_df.shape}")
    print(f"Columns: {games_df.columns.tolist()}")
