from flask import Flask, request, jsonify
import os
import pandas as pd
from model import train_model, load_model

# Initialize the app
app = Flask(__name__)
UPLOAD_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global Variables
REQUIRED_COLUMNS_FILE = os.path.join(UPLOAD_FOLDER, 'required_columns.txt')


@app.route('/')
def home():
    return "Welcome to the Manufacturing Predictive API! Use /upload, /train, and /predict to interact."


@app.route('/upload', methods=['POST'])
def upload_data():
    """
    Endpoint to upload a dataset.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, 'uploaded_dataset.csv')
    try:
        file.save(filepath)

        # Infer required columns
        df = pd.read_csv(filepath)
        required_columns = list(df.columns)

        # Save required columns for later use
        with open(REQUIRED_COLUMNS_FILE, 'w') as f:
            f.write(','.join(required_columns))

        return jsonify({"message": f"File uploaded successfully to {filepath}", "columns": required_columns}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to upload file: {str(e)}"}), 500


@app.route('/train', methods=['POST'])
def train():
    """
    Endpoint to train the model on the uploaded dataset.
    """
    try:
        filepath = os.path.join(UPLOAD_FOLDER, 'uploaded_dataset.csv')

        # Train the model
        metrics, model_path, required_columns = train_model(filepath)

        # Save required columns for prediction validation
        with open(REQUIRED_COLUMNS_FILE, 'w') as f:
            f.write(','.join(required_columns))

        return jsonify({"message": "Model trained successfully.", "metrics": metrics}), 200
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions using the trained model.
    """
    try:
        # Load the model and required columns
        model, required_columns = load_model()

        # Validate input data
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided."}), 400

        # Ensure input matches the required columns
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=required_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        return jsonify({
            "Downtime": "Yes" if prediction[0] == 1 else "No",
            "Confidence": probabilities[0][1] if prediction[0] == 1 else probabilities[0][0]
        }), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


