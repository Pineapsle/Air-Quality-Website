from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'air_data.csv')

# Load and preprocess data
data = pd.read_csv(CSV_PATH)

# Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Encode categorical variables
data = pd.get_dummies(data, columns=['Country', 'Status'], drop_first=True)

# Separate features and target
X = data.drop(columns=['Date', 'AQI Value']).values.astype(np.float32)
y = data['AQI Value'].values.astype(np.float32)

# PyTorch Model
class AQIPredictor(nn.Module):
    def __init__(self, input_size):
        super(AQIPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        return self.fc3(x)

# Initialize model
input_size = X.shape[1]
model = AQIPredictor(input_size)

# Train model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# Save feature columns
feature_columns = [col for col in data.columns if col not in ['Date', 'AQI Value']]

# Visualization function
def create_aqi_visualization(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Date'], data['AQI Value'], alpha=0.5)
    plt.title('AQI Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('AQI Value')
    plt.xticks(rotation=45)
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    
    # Encode the image to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Root route
@app.route('/', methods=['GET'])
def home():
    return "AQI Prediction API is running!", 200

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check if all necessary fields are present
        required_fields = ['Year', 'Month', 'Day', 'Country_United States of America', 'Status_Moderate']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract features from the data
        features = [
            data['Year'],
            data['Month'],
            data['Day'],
            data['Country_United States of America'],
            data['Status_Moderate'],
        ]

        # Convert features to a numpy array and make prediction
        input_array = np.array(features).reshape(1, -1).astype(np.float32)
        input_tensor = torch.tensor(input_array)

        # Prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()  # Convert prediction to a number

        # Prepare the response
        response = {
            "prediction": prediction,
            "feature_details": {
                "total_samples": len(data),
                "mean_aqi": data['AQI Value'].mean(),
                "max_aqi": data['AQI Value'].max(),
                "min_aqi": data['AQI Value'].min()
            },
            "visualization": create_aqi_visualization(data)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
