from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import requests

app = Flask(__name__)

# ===================
# Load and prepare data
# ===================
data = pd.read_csv("C:/AI/agent/amzn_raw_data.csv")
data = data.dropna()

data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

X = data[['open', 'high', 'low', 'close', 'volume', 'year', 'month', 'day', 'change_percent', 'avg_vol_20d']].values
y = data['adjusted_close'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===================
# Train KNN model
# ===================
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# ===================
# Evaluate KNN model
# ===================
y_pred_knn = knn_model.predict(X_test)

mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print("===== KNN Model Performance =====")
print(f"Mean Squared Error (MSE): {mse_knn:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_knn:.2f}")
print(f"Mean Absolute Error (MAE): {mae_knn:.2f}")
print(f"R^2 Score: {r2_knn:.4f}")

# ===================
# Optional: Train GradientBoostingRegressor model
# ===================
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr_model.fit(X_train, y_train)

# ===================
# Evaluate GradientBoostingRegressor model
# ===================
y_pred_gbr = gbr_model.predict(X_test)

mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

print("\n===== GradientBoostingRegressor Performance =====")
print(f"Mean Squared Error (MSE): {mse_gbr:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_gbr:.2f}")
print(f"Mean Absolute Error (MAE): {mae_gbr:.2f}")
print(f"R^2 Score: {r2_gbr:.4f}")

# ===================
# Save model & scaler
# ===================
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(gbr_model, 'gbr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ===================
# API Endpoints
# ===================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-adjusted-close', methods=['POST'])
def predict_adjusted_close():
    input_features = request.json['features']

    scaler = joblib.load('scaler.pkl')
    model = joblib.load('knn_model.pkl')  # أو gbr_model.pkl لو عايز تجرب التاني

    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)

    return jsonify({"predicted_adjusted_close": float(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
