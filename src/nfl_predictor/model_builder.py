"""NFL Prediction Model Building Module.

This module handles model training, evaluation, and prediction
"""

import logging
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_prediction_model(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> dict[str, Any]:
    """Build and train logistic regression model for NFL game prediction.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with model, metrics, and additional information
    """
    logger.info("Building prediction model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Calculate baseline (always predict home team wins)
    baseline_accuracy = y_test.mean()

    # Detailed metrics
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    # Feature importance
    feature_importance = pd.DataFrame(
        {
            'feature': X.columns,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0]),
        }
    ).sort_values('abs_coefficient', ascending=False)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Prepare results
    results = {
        'model': model,
        'scaler': scaler,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'improvement_over_baseline': test_accuracy - baseline_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'confusion_matrix': conf_matrix,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
    }

    # Print summary
    print_model_summary(results)

    return results


def print_model_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive model summary.

    Args:
        results: Dictionary with model results
    """
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)

    print(f"\nAccuracy Metrics:")
    print(f"  Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Baseline (Home Win): {results['baseline_accuracy']:.4f}")
    print(
        f"  Improvement: {results['improvement_over_baseline']:.4f} "
        f"({results['improvement_over_baseline'] / results['baseline_accuracy'] * 100:.1f}% relative)"
    )

    print(f"\nDetailed Metrics:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")

    print(f"\nCross-Validation:")
    print(f"  Mean CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")

    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])

    print(f"\nTop 5 Most Important Features:")
    for idx, row in results['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']:30} : {row['coefficient']:+.4f}")

    print("\n" + "=" * 60)


def evaluate_model(
    model: Any, scaler: StandardScaler, X: pd.DataFrame, y: pd.Series
) -> Dict[str, float]:
    """Evaluate model on new data.

    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        X: Features
        y: True labels

    Returns:
        Dictionary with evaluation metrics
    """
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
    }


def predict_game(
    model: Any, scaler: StandardScaler, game_features: pd.DataFrame
) -> Tuple[int, float]:
    """Predict single game outcome.

    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        game_features: Features for one game

    Returns:
        Tuple of (prediction, probability)
    """
    features_scaled = scaler.transform(game_features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0, 1]

    return int(prediction), float(probability)


def save_model(results: Dict[str, Any], filepath: str = 'nfl_prediction_model.pkl') -> None:
    """Save model and scaler to disk.

    Args:
        results: Model results dictionary
        filepath: Path to save model
    """
    model_package = {
        'model': results['model'],
        'scaler': results['scaler'],
        'feature_names': list(results['X_train'].columns),
        'metrics': {
            'test_accuracy': results['test_accuracy'],
            'baseline_accuracy': results['baseline_accuracy'],
            'f1_score': results['f1_score'],
        },
    }

    joblib.dump(model_package, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str = 'nfl_prediction_model.pkl') -> Dict[str, Any]:
    """Load saved model from disk.

    Args:
        filepath: Path to model file

    Returns:
        Model package dictionary
    """
    return joblib.load(filepath)


if __name__ == "__main__":
    # Example usage
    from src.nfl_predictor.data_collector import collect_nfl_data
    from src.nfl_predictor.data_cleaner import prepare_data_for_modeling

    # Collect and prepare data
    games = collect_nfl_data([2023, 2024])
    df_processed, X, y = prepare_data_for_modeling(games)

    # Build model
    results = build_prediction_model(X, y)

    # Save model
    save_model(results)

    print("\nModel training completed successfully!")
