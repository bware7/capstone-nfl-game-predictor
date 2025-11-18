"""
NFL Game Predictor - Main Script
Run this to train and evaluate the model
"""

import os
import sys

from data_preparer import DataPreparer
import joblib
from model_builder import NFLGamePredictor


def main():
    """Main execution function"""
    print("=" * 60)
    print("NFL GAME PREDICTOR")
    print("=" * 60)

    # Check if data exists
    if not os.path.exists('data/games_with_features.csv'):
        print("\nNo data found. Collecting data...")
        preparer = DataPreparer(seasons=[2023, 2024])
        preparer.run_full_pipeline()

    # Initialize predictor
    predictor = NFLGamePredictor()

    # Load data
    predictor.load_data(
        game_data_path='data/games_with_features.csv', player_stats_path='data/weekly_stats.csv'
    )

    # Prepare features
    X, y = predictor.prepare_features()

    # Train models
    results, X_test, y_test = predictor.train_models(X, y)

    # Get feature importance
    importance = predictor.get_feature_importance(top_n=10)

    # Evaluate
    best_accuracy = predictor.evaluate(X_test, y_test)

    # Save best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(predictor.best_model, 'models/nfl_predictor.pkl')
    joblib.dump(predictor.scaler, 'models/scaler.pkl')
    print("\n✅ Model saved to models/nfl_predictor.pkl")

    return best_accuracy


if __name__ == "__main__":
    accuracy = main()
    print(f"\nFinal accuracy: {accuracy:.1%}")
