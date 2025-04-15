# api/predict.py
# New Flask API endpoint for Vercel

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Determine the absolute path to the model file
# Vercel typically places the project root differently during build vs. runtime.
# Assuming model.joblib is at the project root relative to the 'api' directory.
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.joblib')
logging.info(f"Attempting to load model from: {model_path}")

# Load the trained pipeline (preprocessor + model)
try:
    pipeline = joblib.load(model_path)
    logging.info("Model pipeline loaded successfully.")
    # Log expected feature names if possible (requires inspection of the preprocessor)
    try:
        # Attempt to get feature names expected by the preprocessor
        # This might vary depending on how the ColumnTransformer was saved
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
             # For newer scikit-learn versions with get_feature_names_out
             # This might only give output feature names, not input ones reliably.
             pass # Input features determined by training data columns usually
        elif hasattr(pipeline.named_steps['preprocessor'], 'transformers_'):
             # Inspect transformers for input features
             input_features = []
             for name, _, features in pipeline.named_steps['preprocessor'].transformers_:
                 if isinstance(features, (list, pd.Index)):
                     input_features.extend(features)
             logging.info(f"Preprocessor expects features (order might matter): {input_features[:20]}...") # Log first few
    except Exception as e:
         logging.warning(f"Could not reliably determine expected input features from pipeline: {e}")


except FileNotFoundError:
    logging.error(f"Error: Model file not found at {model_path}")
    pipeline = None
except Exception as e:
    logging.error(f"Error loading model pipeline: {e}")
    pipeline = None

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests.
    Expects JSON input with features matching the training data columns (except target).
    Example: {"Feature1": value1, "Feature2": value2, ...}
    """
    if pipeline is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        # Convert JSON data to DataFrame - handle potential single record vs list
        if isinstance(data, list):
             # If input is a list of records
             input_df = pd.DataFrame(data)
        elif isinstance(data, dict):
             # If input is a single record, wrap it in a list for DataFrame creation
             input_df = pd.DataFrame([data])
        else:
             return jsonify({"error": "Invalid JSON data format. Expecting a JSON object or list of objects."}), 400

        logging.info(f"Received data for prediction: {input_df.head().to_dict()}")

        # Ensure columns match training data (important!)
        # You might need to load expected columns separately or infer from pipeline
        # This is a simplified check; more robust checking might be needed.
        # try:
        #     training_cols = expected_input_features # Load this list somehow
        #     input_df = input_df[training_cols] # Reorder/select columns
        # except KeyError as e:
        #     return jsonify({"error": f"Missing feature in input data: {e}"}), 400
        # except NameError:
        #      logging.warning("Expected input feature list not available for column check.")


        # Make prediction using the loaded pipeline
        # The pipeline handles preprocessing and prediction
        predictions = pipeline.predict(input_df)
        # predict_proba might be useful too: probabilities = pipeline.predict_proba(input_df)

        # Convert numpy types for JSON serialization if necessary
        if isinstance(predictions, np.ndarray):
            predictions_list = predictions.tolist()
        else:
            predictions_list = predictions # Assuming it's already list-like


        logging.info(f"Predictions generated: {predictions_list}")
        return jsonify({"predictions": predictions_list})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        # Consider more specific error handling based on potential issues
        # (e.g., missing columns, incorrect data types)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Health check endpoint (optional but good practice)
@app.route('/api/health', methods=['GET'])
def health_check():
    """ Basic health check endpoint """
    if pipeline:
        return jsonify({"status": "ok", "message": "Model loaded"}), 200
    else:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500


# If running locally using `flask run` (for testing)
if __name__ == '__main__':
     # Vercel uses a WSGI server like gunicorn, this block is for local dev
     app.run(debug=True)