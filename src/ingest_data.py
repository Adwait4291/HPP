# src/ingest_data.py
# Refactored Code

import logging
import os
import pandas as pd
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataIngestionStrategy(ABC):
    """
    Abstract base class for data ingestion strategies.
    """
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Abstract method to ingest data from a given file path.

        Args:
            file_path (str): Path to the data file.
        Returns:
            pd.DataFrame: DataFrame containing the ingested data.
        """
        pass

class CSVDataIngestionStrategy(DataIngestionStrategy):
    """
    Concrete strategy for ingesting data from a CSV file.
    """
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Ingests data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
        Returns:
            pd.DataFrame: DataFrame containing the ingested data.
        Raises:
            FileNotFoundError: If the CSV file is not found.
        """
        logging.info(f"Ingesting data from CSV file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully ingested data. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logging.error(f"Error: The file was not found at {file_path}")
            raise
        except Exception as e:
            logging.error(f"An error occurred during CSV ingestion: {e}")
            raise

class DataIngestor:
    """
    Context class that uses a DataIngestionStrategy to ingest data.
    """
    def __init__(self, strategy: DataIngestionStrategy):
        self._strategy = strategy

    def ingest_data(self, file_path: str) -> pd.DataFrame:
        """
        Ingests data using the configured strategy.

        Args:
            file_path (str): Path to the data file.
        Returns:
            pd.DataFrame: DataFrame containing the ingested data.
        """
        return self._strategy.ingest(file_path)

# --- Removed the __main__ block as it contained hardcoded local paths ---
# Example usage (if run as a script, requires adjustment for path):
# if __name__ == "__main__":
#     # Define the path relative to this script or use absolute path
#     # Example assuming data is in ../data relative to src
#     # Adjust this path based on your project structure
#     data_file_path = "../data/AmesHousing.csv" # Example path
#
#     try:
#         csv_strategy = CSVDataIngestionStrategy()
#         data_ingestor = DataIngestor(csv_strategy)
#         df = data_ingestor.ingest_data(data_file_path)
#         print("Data ingested successfully:")
#         print(df.head())
#     except FileNotFoundError:
#         print(f"Error: Data file not found at {data_file_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")