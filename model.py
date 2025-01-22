import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# Directory for saving models
MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)


def preprocess_data(df, target_column=None):
    """
    Preprocess the input DataFrame by handling missing values, encoding, and creating the target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column if it exists in the dataset.

    Returns:
        X (pd.DataFrame): Processed features.
        y (pd.Series): Target variable.
        categorical_columns (list): List of detected categorical columns.
    """
    # If no target column is specified, try to infer it
    if target_column is None:
        potential_targets = ['Downtime_Flag', 'target', 'label']
        target_column = next((col for col in potential_targets if col in df.columns), None)
        if not target_column:
            raise ValueError("No target column found. Please specify a target column explicitly.")

    # Create target column if required (specific to downtime logic)
    if 'Down time Hours' in df.columns and target_column == 'Downtime_Flag':
        df[target_column] = (df['Down time Hours'] > 1).astype(int)

    # Drop unnecessary or known irrelevant columns
    drop_columns = ['Production ID', 'Date', 'Defects', 'Down time Hours']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')

    # Detect categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Encode categorical columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y, categorical_cols


def train_model(file_path, target_column=None):
    """
    Train a Random Forest model using the dataset at `file_path`.

    Args:
        file_path (str): Path to the CSV file.
        target_column (str): Name of the target column if it exists in the dataset.

    Returns:
        metrics (dict): Accuracy and F1 score.
        model_path (str): Path to the saved model.
        columns (list): List of feature columns.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Preprocess the data
    X, y, categorical_cols = preprocess_data(df, target_column)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # Save the model along with required columns
    model_info = {"model": model, "required_columns": list(X.columns)}
    model_path = os.path.join(MODEL_FOLDER, 'trained_model.pkl')
    joblib.dump(model_info, model_path)

    return metrics, model_path, list(X.columns)


def load_model():
    """
    Load the trained model and required columns from disk.

    Returns:
        model: The trained model object.
        required_columns (list): List of required columns for the model.
    """
    model_path = os.path.join(MODEL_FOLDER, 'trained_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Train the model first.")

    model_info = joblib.load(model_path)
    return model_info["model"], model_info["required_columns"]
