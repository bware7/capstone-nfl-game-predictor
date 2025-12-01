"""Configuration for NFL Predictor"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Data settings
    seasons: List[int] = field(default_factory=lambda: [2023, 2024])

    # Paths
    game_data_path: str = "data/games_with_features.csv"
    player_stats_path: str = "data/weekly_stats.csv"
    model_output_path: str = "models/nfl_predictor.pkl"
    scaler_output_path: str = "models/scaler.pkl"

    # Model settings
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Validation
    use_time_series_split: bool = True  # Critical: prevents future leakage

    # Feature engineering
    rolling_window: int = 3
    lag_periods: int = 1  # Critical: lag player stats to prevent leakage

    # Leakage detection threshold
    correlation_threshold: float = 0.95


# Default config instance
config = Config()
