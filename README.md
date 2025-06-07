# 📈 Previsão de Preço de Ações (AAPL) com LSTM e FastAPI

Este projeto utiliza uma rede neural LSTM para prever o próximo preço de fechamento das ações da Apple (AAPL), com base nos últimos 60 dias de preços históricos. A aplicação é servida via uma API REST desenvolvida com FastAPI.

---

## 🚀 Funcionalidades

- Coleta automática de dados históricos da AAPL via Yahoo Finance
- Pré-processamento e normalização com MinMaxScaler
- Treinamento de modelo LSTM com TensorFlow/Keras
- API REST com FastAPI para fazer previsões com dados reais
- Suporte a Docker para execução em container

---

## 🧩 Tecnologias utilizadas

- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- yfinance
- FastAPI
- Uvicorn
- Joblib
- Docker (opcional)

---

## 🛠️ Como executar localmente

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o script principal

```bash
python main.py
```

> Isso irá:
> - Baixar os dados históricos da AAPL
> - Treinar o modelo LSTM
> - Salvar os arquivos `lstm_aapl_model.h5` e `scaler.save`

---

## ▶️ Executar a API

Após treinar o modelo, inicie a API com:

```bash
uvicorn main:app --reload
```

A API estará disponível em: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📬 Endpoint `/predict`

### ➤ Método: `POST`

### ➤ URL: `/predict`

### ➤ Body (JSON):

```json
{
  "prices": [150.00, 150.30, 149.80, 150.10, ..., 152.30]
}
```

(Deve conter exatamente **60 valores numéricos**)

### ➤ Resposta:

```json
{
  "predicted_price": 152.34
}
```

---

## 🧪 Testando a API com `curl`

```bash
curl -X POST http://127.0.0.1:8000/predict      -H "Content-Type: application/json"      -d '{"prices": [150.00, 150.30, 149.80, 150.10, ..., 152.30]}'
```

Ou use a função `test_api()` incluída no `main.py` para automatizar o teste com dados reais da AAPL.

---

## 🐳 Executando com Docker

### Dockerfile (incluso no projeto)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Comandos:

```bash
docker build -t lstm-aapl .
docker run -p 8000:8000 lstm-aapl
```

---

## 📂 Estrutura do Projeto

```
.
├── main.py              # Script principal com treinamento e API
├── lstm_aapl_model.h5   # Modelo LSTM treinado (gerado automaticamente)
├── scaler.save          # Escalonador salvo com joblib (gerado automaticamente)
├── requirements.txt     # Dependências do projeto
├── Dockerfile           # Imagem Docker
└── README.md            # Este arquivo
```

---

## 👤 Autor

- Douglas Ferreira de Paula
- GitHub: [@DouglasFerreiraDePaula](https://github.com/DouglasFerreiraDePaula)

---

