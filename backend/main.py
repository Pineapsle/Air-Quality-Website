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

# Encode categorical variables (matching front-end encoding)
data['Country_United States of America'] = (data['Country'] == 'United States of America').astype(int)
data['Status_Moderate'] = (data['Status'] == 'Moderate').astype(int)

# Prepare features
feature_columns = ['Year', 'Month', 'Day', 'Country_United States of America', 'Status_Moderate']
X = data[feature_columns].values.astype(np.float32)
y = data['AQI Value'].values.astype(np.float32)

# Normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

# PyTorch Model
class AQIPredictor(nn.Module):
    def __init__(self, input_size):
        super(AQIPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64) 
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Initialize model
input_size = X_normalized.shape[1]
model = AQIPredictor(input_size)

# Train model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# Visualization function
def create_aqi_visualization(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Date'], data['AQI Value'], alpha=0.5)
    plt.title('AQI Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('AQI Value')
    plt.xticks(rotation=45)
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        input_data = request.get_json()

        # Validate input
        required_fields = ['Year', 'Month', 'Day', 'Country_United States of America', 'Status_Moderate']
        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Prepare input features
        input_features = [
            input_data['Year'], 
            input_data['Month'], 
            input_data['Day'], 
            input_data['Country_United States of America'], 
            input_data['Status_Moderate']
        ]

        # Normalize input features
        input_normalized = (np.array(input_features).astype(np.float32) - X_mean) / X_std
        input_tensor = torch.tensor(input_normalized).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()

        # Prepare the response
        response = {
            "prediction": float(prediction),
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

# Root route
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "AQI Prediction API is running",
        "available_endpoints": ["/predict"]
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)