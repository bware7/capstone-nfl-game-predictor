"""
NFL Game Predictor - Main Script
Run this to train and evaluate the model
"""

import argparse
import os
import sys

from config import config
from data_preparer import DataPreparer
import joblib
from model_builder import NFLGamePredictor


def main(force_rebuild=False):
    """Main execution function"""
    print("=" * 60)
    print("NFL GAME PREDICTOR")
    print("=" * 60)

    # Step 1: Prepare data
    if force_rebuild or not os.path.exists(config.game_data_path):
        print("\nPreparing data...")
        preparer = DataPreparer()
        preparer.run_full_pipeline(force_rebuild=True)
    else:
        print(f"\nUsing existing data from {config.game_data_path}")

    # Step 2: Initialize predictor
    predictor = NFLGamePredictor()

    # Step 3: Load data
    predictor.load_data()

    # Step 4: Prepare features
    X, y = predictor.prepare_features()

    # Step 5: Train models with time-series validation
    results, X_test, y_test = predictor.train_models(X, y)

    # Step 6: Feature importance
    importance = predictor.get_feature_importance(top_n=10)

    # Step 7: Evaluate
    best_accuracy = predictor.evaluate(X_test, y_test)

    # Step 8: Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(predictor.best_model, config.model_output_path)
    joblib.dump(predictor.scaler, config.scaler_output_path)
    joblib.dump(predictor.feature_names, 'models/feature_names.pkl')

    print(f"\n✅ Model saved to {config.model_output_path}")

    return best_accuracy, predictor


def predict_game(predictor, home_stats, away_stats):
    """Make a prediction for a single game"""
    import pandas as pd

    # Combine stats into feature vector
    features = {}
    for k, v in home_stats.items():
        features[f'{k}_home'] = v
    for k, v in away_stats.items():
        features[f'{k}_away'] = v

    # Create differentials
    for k in home_stats:
        if k in away_stats:
            features[f'{k}_diff'] = home_stats[k] - away_stats[k]

    # Create DataFrame with correct feature order
    X = pd.DataFrame([features])[predictor.feature_names].fillna(0)

    prob = predictor.predict_proba(X)[0, 1]
    pred = "Home" if prob > 0.5 else "Away"

    return pred, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NFL Game Predictor')
    parser.add_argument(
        '--rebuild', action='store_true', help='Force rebuild data even if it exists'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        type=int,
        default=[2023, 2024],
        help='Seasons to include (default: 2023 2024)',
    )
    args = parser.parse_args()

    # Update config if seasons specified
    if args.seasons:
        config.seasons = args.seasons

    accuracy, predictor = main(force_rebuild=args.rebuild)
    print(f"\nFinal accuracy: {accuracy:.1%}")
