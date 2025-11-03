"""NFL Game Predictor main execution."""

from analytics_project.utils_logger import init_logger, logger

from .data_cleaner import create_game_features, prepare_training_data
from .data_collector import get_games_data, save_data
from .model_builder import train_model


def main():
    init_logger()
    logger.info("Starting NFL Game Predictor")

    # Get game data
    games = get_games_data([2023, 2024])

    # Create features
    games_with_features = create_game_features(games)
    save_data(games_with_features, "games_with_features")

    # Prepare for training
    X, y = prepare_training_data(games_with_features)

    # Train model
    model = train_model(X, y)

    logger.info("Model training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
