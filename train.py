# train.py
# New script to orchestrate training and model saving

import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib # For saving the pipeline

# Import refactored components from src
from src.ingest_data import DataIngestor, CSVDataIngestionStrategy
from src.feature_engineering import PriceBinner # Only need PriceBinner strategy directly here
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from src.model_building import ModelBuilder, LogisticRegressionStrategy
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluationStrategy
# Note: Outlier/Missing Value handling can be added here or integrated into the pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model(data_path: str = "data/AmesHousing.csv", model_output_path: str = "model.joblib"):
    """
    Trains the house price category prediction model and saves the pipeline.

    Args:
        data_path (str): Path to the input CSV data file.
        model_output_path (str): Path to save the trained pipeline (.joblib).
    """
    logging.info("Starting training pipeline...")

    # 1. Ingest Data
    try:
        ingestor = DataIngestor(CSVDataIngestionStrategy())
        df = ingestor.ingest_data(data_path)
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        return

    # Drop unnecessary columns (e.g., Order, PID if not used as features)
    df = df.drop(columns=['Order', 'PID'], errors='ignore')

    # 2. Feature Engineering (Target Binning)
    try:
        price_binner = PriceBinner(target_column='SalePrice', num_bins=3)
        df = price_binner.apply(df)
        target_column = 'SalePrice_Category' # New target column
        # Drop original SalePrice as it's no longer the direct target
        df = df.drop(columns=['SalePrice'])
    except Exception as e:
        logging.error(f"Target variable binning failed: {e}")
        return

    # 3. Data Splitting
    try:
        # Ensure target column exists before splitting
        if target_column not in df.columns:
             raise ValueError(f"Target column '{target_column}' not found after binning.")

        splitter = DataSplitter(SimpleTrainTestSplitStrategy())
        # Explicitly pass the new target column name
        X_train, X_test, y_train, y_test = splitter.split_data(df, target_column=target_column)
    except Exception as e:
        logging.error(f"Data splitting failed: {e}")
        return

    # 4. Preprocessing (within a pipeline for proper handling)
    # Identify numerical and categorical features (excluding the target)
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=object).columns.tolist()

    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Handle missing numerical values
        ('scaler', StandardScaler())                   # Scale numerical features
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing categorical values
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # One-hot encode
    ])

    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='passthrough') # Keep other columns if any (shouldn't be many after selection)


    # 5. Model Training (using the strategy)
    try:
        model_builder = ModelBuilder(LogisticRegressionStrategy())
        # Create the full pipeline: preprocess -> train
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model_builder._strategy.build_and_train(pd.DataFrame(), pd.Series())) # Pass dummy data, fitting happens in pipeline.fit
        ])

        # Fit the entire pipeline (preprocessing + model)
        logging.info("Fitting the full pipeline...")
        full_pipeline.fit(X_train, y_train)
        logging.info("Pipeline fitting completed.")

    except Exception as e:
        logging.error(f"Model training pipeline failed: {e}")
        return

    # 6. Model Evaluation
    try:
        evaluator = ModelEvaluator(ClassificationModelEvaluationStrategy())
        metrics = evaluator.evaluate_model(full_pipeline, X_test, y_test)
        logging.info(f"Model Evaluation Metrics: {metrics}")
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        # Continue to save the model even if evaluation fails? Or return?
        # For now, log error and continue to save.

    # 7. Save the Pipeline (Preprocessor + Model)
    try:
        logging.info(f"Saving pipeline to {model_output_path}")
        joblib.dump(full_pipeline, model_output_path)
        logging.info("Pipeline saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save pipeline: {e}")

    logging.info("Training pipeline finished.")

if __name__ == "__main__":
    # Define path relative to the script location
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, 'data') # Assuming data is in 'data' subdir
    csv_path = os.path.join(data_dir, 'AmesHousing.csv')
    model_path = os.path.join(script_dir, 'model.joblib') # Save model in root

    # Create data directory if it doesn't exist (for local runs)
    # Note: In deployment, data might be handled differently
    if not os.path.exists(data_dir):
         logging.warning(f"Data directory '{data_dir}' not found. Please place AmesHousing.csv there.")
         # Example: os.makedirs(data_dir) # If needed

    if os.path.exists(csv_path):
        train_model(data_path=csv_path, model_output_path=model_path)
    else:
        logging.error(f"Data file not found at {csv_path}. Training aborted.")