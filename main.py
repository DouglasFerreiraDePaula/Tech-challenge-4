# Projeto: Previsão de Preços de Ações com LSTM - Apple (AAPL)

# 1. Coleta e Pré-processamento de Dados
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests

# Coleta de dados históricos da AAPL
data = yf.download('AAPL', start='2015-01-01', end='2024-01-01')
data = data[['Close']]
data.dropna(inplace=True)

# Normalização dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Função para criar sequências para LSTM
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Divisão em treino e teste
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size - 60:]

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 2. Construção e Treinamento do Modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

# 3. Avaliação
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(real_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))

# 4. Salvamento do modelo e scaler
model.save("lstm_aapl_model.h5")
joblib.dump(scaler, "scaler.save")

# 5. API com FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "✅ API de Previsão LSTM está rodando! Acesse /docs para testar."}

class PriceData(BaseModel):
    prices: list

@app.post("/predict")
def predict_price(data: PriceData):
    try:
        scaler = joblib.load("scaler.save")
        model = tf.keras.models.load_model("lstm_aapl_model.h5")
        input_data = np.array(data.prices).reshape(-1, 1)
        scaled_input = scaler.transform(input_data)

        X = []
        X.append(scaled_input[-60:])
        X = np.array(X).reshape(1, 60, 1)

        prediction = model.predict(X)
        predicted_price = scaler.inverse_transform(prediction)
        return {"predicted_price": float(predicted_price[0][0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Executar com: uvicorn nome_arquivo:app --reload
#
# Explicação do comando:
# uvicorn         -> servidor ASGI leve usado para rodar APIs como FastAPI
# nome_arquivo    -> nome do seu arquivo Python (sem .py), ex: main
# app             -> nome do objeto FastAPI dentro do seu código (app = FastAPI())
# --reload        -> ativa recarregamento automático sempre que salvar alterações no código
#
# Exemplo real, se seu arquivo se chama main.py:
# uvicorn main:app --reload

# 6. Script de Teste com dados reais

def test_api():
    data = yf.download('AAPL', period='100d')
    prices = data['Close'].dropna().tolist()
    if len(prices) < 60:
        raise ValueError("Número insuficiente de dados para o modelo (precisa de 60).")
    last_60 = prices[-60:]

    payload = {"prices": last_60}
    url = "http://127.0.0.1:8000/predict"
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("✅ Preço previsto para o próximo dia:", response.json()['predicted_price'])
    else:
        print("❌ Erro:", response.status_code, response.text)

# 7. Dockerfile
# Salvar como Dockerfile no projeto
"""
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# 8. requirements.txt
# Salvar como requirements.txt no projeto
"""
fastapi
uvicorn
tensorflow
scikit-learn
joblib
yfinance
matplotlib
requests
pandas
numpy
"""