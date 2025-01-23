Manufacturing Predictive API
----------------------------------
Objective-
This API predicts machine downtime based on the input manufacturing data, such as Temperature and Run_Time. The model is trained using the provided dataset and can predict whether downtime will occur or not.
-------------------------------------
Setup-
Prerequisites-
To run the API, you need Python 3.x and the following libraries installed:

Flask
scikit-learn
pandas
joblib
---------------------------------------------

--------------------------------------------
---------------------------------------------------
Notes:-

Model: A RandomForestClassifier is used for downtime prediction. You can modify the model type or parameters as needed.
Required Columns: The /upload and /train endpoints automatically detect and save the required columns for the model. These are stored for later use in the /predict endpoint.

-------------------------------------------------------
Based on the requirements outlined in the document, the provided Flask code meets the key criteria as follows:

1. Dataset Handling
The /upload endpoint accepts CSV files and extracts columns, saving them for later use. This is in line with the requirement to handle datasets that include key columns like Machine_ID, Temperature, Run_Time, and Downtime_Flag.
2. Model
The code uses a RandomForestClassifier model to predict downtime based on uploaded data, which is a simple supervised machine learning model as requested (although Logistic Regression or Decision Tree would be suitable as well). This fulfills the "predict machine downtime or product defects" part of the assignment.
3. Endpoints:
/upload: Accepts a CSV file and extracts columns (meets the upload endpoint requirement).
/train: Trains the model on the uploaded dataset and provides performance metrics (accuracy, F1 score), fulfilling the "train" requirement.
/predict: Accepts JSON input, makes predictions, and returns the result in JSON format, as required.
5. Output Format:
The /predict endpoint returns a JSON response with "Downtime" and "Confidence" fields, which matches the output format requirement:
json
Copy
Edit
{ "Downtime": "Yes", "Confidence": 0.85 }
6. Flask and scikit-learn
The API is implemented in Flask, and the model uses scikit-learn for the RandomForestClassifier, fulfilling the technical requirements.
7. Testing Locally
The app can be tested locally using tools like Postman or curl, as described in the provided code.
Suggested README
Here’s a README based on the requirements and the current code:
---------------------------------------------
Manufacturing Predictive API
Objective
This API predicts machine downtime based on the input manufacturing data, such as Temperature and Run_Time. The model is trained using the provided dataset and can predict whether downtime will occur or not.
-------------------------------------
Setup
Prerequisites
To run the API, you need Python 3.x and the following libraries installed:

Flask
scikit-learn
pandas
joblib
You can install the required libraries by running:

bash
Copy
Edit
pip install flask scikit-learn pandas joblib
Running the API
Clone or download the project.
Navigate to the project directory and run the API using the following command:
bash
Copy
Edit
python app.py
The server will start on http://localhost:5000.
Endpoints
1. /upload (POST)
Purpose: Upload a CSV file containing manufacturing data.

Request:

bash
Copy
Edit
curl -X POST -F "file=@path_to_your_dataset.csv" http://localhost:5000/upload
Response:

json
Copy
Edit
{
  "message": "File uploaded successfully to data/uploaded_dataset.csv",
  "columns": ["Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"]
}
2. /train (POST)
Purpose: Train the machine learning model on the uploaded dataset and return performance metrics.

Request:


curl -X POST http://localhost:5000/train
Response:

{
  "message": "Model trained successfully.",
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.94
  }
}
3. /predict (POST)
Purpose: Make a prediction based on the input data.

Request:
-d '{"Temperature": 80, "Run_Time": 120}' \
http://localhost:5000/predict
Response:

json
Copy
Edit
{
  "Downtime": "Yes",
  "Confidence": 0.85
}
----------------------------
File Structure: 

project/
├── app.py                  # Flask application file
├── model.py                # Model code for training and prediction
├── models/                 # Directory for saving the trained model
│   └── trained_model.pkl   # Saved model file
└── data/                   # Directory for uploaded datasets
    └── uploaded_dataset.csv
--------------
NOTES:

Model: A RandomForestClassifier is used for downtime prediction. You can modify the model type or parameters as needed.
Required Columns: The /upload and /train endpoints automatically detect and save the required columns for the model. These are stored for later use in the /predict endpoint.
--------------
Example of How to Test the API:-

Upload a dataset (CSV file with Machine_ID, Temperature, Run_Time, Downtime_Flag).
Train the model on the uploaded dataset using the /train endpoint.
Predict downtime by sending input data to the /predict endpoint.
