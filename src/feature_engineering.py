# src/feature_engineering.py
# Refactored Code

import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Abstract Base Class ---
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Apply transformation. """
        pass

# --- Concrete Strategies ---
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            # Adding 1 to handle potential zero values before log transform
            df_transformed[feature] = np.log1p(df_transformed[feature])
        return df_transformed

class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler() # Scaler needs to be fitted later

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # This method should ideally just return the configuration.
        # Fitting and transforming happens in the main pipeline.
        logging.warning("StandardScaling apply method called directly. Ensure scaler is fitted appropriately in the pipeline.")
        df_scaled = df.copy()
        if self.features:
             # In a real pipeline, fit_transform on train, transform on test/predict
            df_scaled[self.features] = self.scaler.fit_transform(df_scaled[self.features])
        return df_scaled

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Fitting and applying StandardScaler to features: {self.features}")
        df_scaled = df.copy()
        if self.features:
            df_scaled[self.features] = self.scaler.fit_transform(df_scaled[self.features])
        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying fitted StandardScaler to features: {self.features}")
        df_scaled = df.copy()
        if self.features:
            df_scaled[self.features] = self.scaler.transform(df_scaled[self.features])
        return df_scaled


class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        # Handle unknown categories during prediction
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
         # This method should ideally just return the configuration.
         # Fitting and transforming happens in the main pipeline.
        logging.warning("OneHotEncoding apply method called directly. Ensure encoder is fitted appropriately in the pipeline.")
        df_encoded = df.copy()
        if self.features:
            encoded_data = self.encoder.fit_transform(df_encoded[self.features])
            encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(self.features), index=df_encoded.index)
            df_encoded = pd.concat([df_encoded.drop(columns=self.features), encoded_df], axis=1)
        return df_encoded

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Fitting and applying OneHotEncoder to features: {self.features}")
        df_encoded = df.copy()
        if self.features:
            encoded_data = self.encoder.fit_transform(df_encoded[self.features])
            encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(self.features), index=df_encoded.index)
            df_encoded = pd.concat([df_encoded.drop(columns=self.features), encoded_df], axis=1)
        return df_encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying fitted OneHotEncoder to features: {self.features}")
        df_encoded = df.copy()
        if self.features:
            encoded_data = self.encoder.transform(df_encoded[self.features])
            encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(self.features), index=df_encoded.index)
            df_encoded = pd.concat([df_encoded.drop(columns=self.features), encoded_df], axis=1)
        return df_encoded

class PriceBinner(FeatureEngineeringStrategy):
    """
    Bins the 'SalePrice' column into categories ('Low', 'Medium', 'High').
    """
    def __init__(self, target_column='SalePrice', num_bins=3):
        self.target_column = target_column
        self.num_bins = num_bins
        self.bin_edges = None # Store bin edges after fitting
        self.labels = ['Low', 'Medium', 'High'] # Adjust if num_bins changes

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # This method should fit and transform
        logging.info(f"Applying price binning to column: {self.target_column}")
        df_binned = df.copy()
        if self.target_column in df_binned.columns:
            # Use quantiles for binning
            df_binned[self.target_column + '_Category'], self.bin_edges = pd.qcut(
                df_binned[self.target_column],
                q=self.num_bins,
                labels=self.labels,
                retbins=True,
                duplicates='drop' # Handle duplicate bin edges if necessary
            )
            logging.info(f"Created '{self.target_column}_Category' with bins: {self.bin_edges}")
        else:
            logging.error(f"Target column '{self.target_column}' not found for binning.")
        return df_binned

# --- Feature Engineer Context Class ---
# This class is less relevant now as preprocessing steps will be likely
# combined in a scikit-learn Pipeline in train.py. Keeping for potential modular use.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying feature engineering strategy: {self._strategy.__class__.__name__}")
        return self._strategy.apply(df)