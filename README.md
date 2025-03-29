# Air Quality Index (AQI) Prediction and Visualization

Welcome to the **AQI Prediction and Visualization** project! This application predicts the Air Quality Index (AQI) using machine learning models, visualizes the AQI trends over time, and provides a simple API for making predictions. This is a full-stack solution with a Flask-based backend and a React.js frontend.

---

## Features

### 🌱 **Predict AQI**
- The model predicts AQI values based on features such as **year**, **month**, **day**, **country**, and **status**.
- The trained PyTorch model uses a neural network to make accurate predictions based on historical data.

### 📊 **Visualization**
- Visualize AQI values over time with interactive graphs.
- The scatter plot dynamically updates to show AQI trends, giving users insight into historical data patterns.

### ⚡ **Interactive API**
- A simple Flask API allows users to make POST requests with the required features and get AQI predictions.
- The API also provides important statistical insights such as the **mean**, **max**, and **min** AQI values from the dataset.

---

## Technology Stack

### Backend
- **Flask**: Lightweight web framework for creating the API and serving the application.
- **PyTorch**: Deep learning library for building and training the AQI prediction model.
- **Pandas**: For data preprocessing and manipulation.
- **NumPy**: Numerical computing for handling large datasets.
- **Matplotlib**: For generating and displaying visualizations of AQI data.
- **Flask-CORS**: Cross-Origin Resource Sharing (CORS) support for enabling the frontend to communicate with the backend.

### Frontend
- **React.js**: JavaScript library for building the frontend UI, allowing users to input AQI-related data and view predictions.
- **Axios**: For making HTTP requests from the frontend to the backend API.

---
