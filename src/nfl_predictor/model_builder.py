"""Build and train NFL game prediction models."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from analytics_project.utils_logger import logger


def train_model(X, y):
    """Train a simple logistic regression model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Model Accuracy: {accuracy:.2%}")
    logger.info(f"Baseline (always predict home): {y_test.mean():.2%}")

    return model
