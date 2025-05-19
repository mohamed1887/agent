from flask import Flask, jsonify, request
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# إنشاء التطبيق باستخدام Flask
app = Flask(__name__)

# دالة لتحميل البيانات
def load_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2024-12-31')
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)
    return scaled_data, data.index

# دالة لإنشاء مجموعة البيانات الزمنية
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# إنشاء النموذج
def create_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# API لتحميل البيانات وتنبؤ الأسعار
@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', 'TSLA')  # استخدام تسلا بشكل افتراضي
    scaled_data, data_index = load_data(ticker)
    
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = create_model()
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    predictions = model.predict(X)
    predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))

    # تجهيز البيانات للعرض
    response_data = {
        "dates": [str(date) for date in data_index[time_step + 1:]],
        "actual_prices": scaled_data[time_step + 1:].flatten().tolist(),
        "predicted_prices": predicted_prices.flatten().tolist()
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
