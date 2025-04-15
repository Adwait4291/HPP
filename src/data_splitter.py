# src/data_splitter.py
# Refactored Code

import logging
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplittingStrategy(ABC):
    """ Abstract base class for data splitting strategies. """
    @abstractmethod
    def split(self, df: pd.DataFrame, target_column: str, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """ Abstract method to split data. """
        pass

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """ Concrete strategy for a simple train-test split. """
    def split(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the data using train_test_split.

        Args:
            df (pd.DataFrame): DataFrame to split.
            target_column (str): Name of the target variable column.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before splitting.

        Returns:
            Tuple containing training features, testing features, training target, testing target.
        """
        logging.info(f"Splitting data with test_size={test_size}, random_state={random_state}. Target column: {target_column}")
        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logging.info("Data split successfully.")
            return X_train, X_test, y_train, y_test
        except KeyError:
            logging.error(f"Target column '{target_column}' not found in DataFrame.")
            raise
        except Exception as e:
            logging.error(f"An error occurred during data splitting: {e}")
            raise

class DataSplitter:
    """ Context class using a DataSplittingStrategy. """
    def __init__(self, strategy: DataSplittingStrategy):
        self._strategy = strategy

    def split_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data using the configured strategy.
        """
        return self._strategy.split(df, target_column, test_size, random_state)