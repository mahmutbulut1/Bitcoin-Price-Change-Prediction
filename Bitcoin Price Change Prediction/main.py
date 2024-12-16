import requests
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Binance API anahtarı ve gizli anahtar
api_key = "AzkLFaomGGsyvBJxBVIJaScO4htjsKeqDXsn14V5rmUR5IVVzfsuWW2VdLzJEOQt"
api_secret = "zkmzS1kepXyLx0zTwvEL8cqFlfr128oIEoNoq60O2f3x6qQVi0vB8x47xPfn8DA5"

symbol = "BTCUSDT"  # Örneğin, BTC/USDT çifti

url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": symbol,
    "interval": "1d",  # Günlük veriler
    "startTime": int(datetime(2009, 1, 3).timestamp() * 1000),  # Unix zaman damgası
    "endTime": int(datetime.now().timestamp() * 1000),  # Şu anki Unix zaman damgası
    "limit": 100  # Çekilecek maksimum veri sayısı
}

headers = {
    "X-MBX-APIKEY": api_key
}

response = requests.get(url, params=params, headers=headers)
data = response.json()

# Verileri dosyaya kaydetme
with open("binance_price_data.txt", "w") as file:
    for entry in data:
        date = datetime.fromtimestamp(entry[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        close_price = entry[4]
        file.write(f"Date: {date}, Close Price: {close_price}\n")

print("Veriler binance_price_data.txt dosyasına kaydedildi.")

# Verileri bir dosyadan okuma
with open("binance_price_data.txt", "r") as file:
    lines = file.readlines()

# Verileri float'a dönüştürme ve bir listeye kaydetme
data = []
for line in lines:
    parts = line.strip().split(", ")
    date = datetime.strptime(parts[0], "Date: %Y-%m-%d %H:%M:%S")
    close_price = float(parts[1].split(": ")[1])
    data.append((date, close_price))

# Veriyi ekrana yazdırma
for date, close_price in data:
    print(f"Date: {date}, Close Price: {close_price:.2f}")

# Veriyi sadece kapanış fiyatlarına dönüştürme
close_prices = [price for _, price in data]

# Veriyi [0, 1] aralığına ölçeklendirme
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices = np.array(close_prices).reshape(-1, 1)
close_prices = scaler.fit_transform(close_prices)

# Verileri giriş ve çıkış olarak böler
def prepare_data(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back, 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

# 30 günlük veriye dayalı tahmin yapmak için:
look_back = 30
X, y = prepare_data(close_prices, look_back)

# Veriyi eğitim ve test setlerine böler
train_size = int(len(X) * 0.67)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM modelini oluşturur
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Veriyi yeniden şekillendirir
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Modeli eğitir
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# Gelecekteki Bitcoin fiyat tahminlerini yapar
last_30_days = close_prices[-look_back:].reshape(1, look_back, 1)
future_predictions = []

for i in range(30):
    prediction = model.predict(last_30_days)
    future_predictions.append(prediction[0, 0])
    last_30_days = np.append(last_30_days[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

# Tahmin sonuçlarını gerçek değer aralığına döndürür
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Gelecekteki 30 günlük Bitcoin fiyat tahminlerinin yazdırılması
print("Gelecekteki 30 Günlük Bitcoin Fiyat Tahminleri:")
for i, price in enumerate(future_predictions, start=1):
    print(f"Gün {i}: ${price[0]:.2f}")