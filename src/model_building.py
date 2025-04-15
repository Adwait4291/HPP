# src/model_building.py
# Refactored Code

import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import ClassifierMixin # Changed from RegressorMixin
from sklearn.linear_model import LogisticRegression # Changed from LinearRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelBuildingStrategy(ABC):
    """ Abstract base class for model building strategies. """
    @abstractmethod
    def build_and_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin: # Changed return type hint
        """ Abstract method to build and train a model. """
        pass

class LogisticRegressionStrategy(ModelBuildingStrategy): # Renamed class
    """ Concrete strategy for Logistic Regression using scikit-learn. """
    def __init__(self, model_params: dict = None):
        """
        Initialize the strategy with optional model parameters.
        Args:
            model_params (dict, optional): Parameters for LogisticRegression.
        """
        self.model_params = model_params if model_params is not None else {'max_iter': 1000, 'random_state': 42} # Added default params

    def build_and_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        """
        Builds and trains a Logistic Regression model.

        Args:
            X_train: Training data features.
            y_train: Training data labels/target.
        Returns:
            Trained Logistic Regression model instance.
        """
        logging.info("Building and training Logistic Regression model.")
        try:
            model = LogisticRegression(**self.model_params) # Changed model
            model.fit(X_train, y_train)
            logging.info("Logistic Regression model trained successfully.")
            return model
        except Exception as e:
            logging.error(f"Error during Logistic Regression model training: {e}")
            raise

# --- Model Builder Context Class ---
# This class might be less necessary if using scikit-learn Pipelines directly.
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy

    def build_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        logging.info(f"Executing model build/train strategy: {self._strategy.__class__.__name__}")
        return self._strategy.build_and_train(X_train, y_train)