# 📊 Análise de Sentimento de Reviews de Produtos

Este projeto foi desenvolvido como parte de um teste técnico, com o objetivo de construir um modelo de Machine Learning capaz de classificar reviews de produtos como **positivas** ou **negativas**, além de disponibilizar essa funcionalidade por meio de uma API.

---

## 🚀 Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

## 📂 Estrutura do projeto
├── data/ # Dataset utilizado
├── model/ # Modelo treinado e vectorizer
├── app/
│ └── main.py # API com FastAPI
├── train.py # Script de treinamento do modelo
├── preprocess.py # Funções de limpeza de texto
├── requirements.txt # Dependências do projeto
└── README.md

## 📊 Dataset

Foi utilizado um dataset público de reviews de produtos contendo mais de **170 mil registros**.

O dataset possui as seguintes colunas:
- ProductName
- ProductPrice
- Rate
- Review
- Summary
- Sentiment

Para este projeto:
- Foram utilizadas apenas as classes **positive** e **negative**
- Registros com classe **neutral** foram removidos

---

## 🧹 Pré-processamento

O pré-processamento incluiu:
- remoção de valores nulos
- filtragem das classes
- limpeza de texto (minúsculas, remoção de caracteres especiais)
- transformação de texto em vetores numéricos usando **TF-IDF**

---

## 🤖 Modelo de Machine Learning

Foi utilizado o modelo:

👉 **Logistic Regression**

Motivos da escolha:
- bom desempenho para classificação de texto
- rápido e eficiente
- fácil de interpretar
- adequado para um primeiro modelo em produção

---

## 📈 Métricas de avaliação

Foram utilizadas:
- Acurácia
- Matriz de confusão
- Precision
- Recall
- F1-score

Resultado obtido:

👉 **Acurácia aproximada: 91%**
## 🔌 API

Foi criada uma API utilizando **FastAPI** para disponibilizar o modelo.

### ▶️ Como executar

```bash'''
uvicorn app.main:app --reload

📍 Endpoint
POST /predict

📥 Exemplo de requisição
{
  "review": "This product is amazing and works perfectly"
}

📤 Exemplo de resposta
{
  "review": "This product is amazing and works perfectly",
  "sentimento_previsto": "positive"
}

☁️ Arquitetura (conceitual - AWS)

Em um cenário de produção, a solução poderia ser estruturada com:

Amazon S3 → armazenamento de dados e modelo
AWS Lambda → execução da inferência
API Gateway → exposição da API
processamento batch para múltiplas análises

💡 Considerações finais
O projeto atende ao objetivo de classificar reviews de produtos e disponibilizar essa funcionalidade via API.

Possíveis melhorias:
testar outros modelos
ajuste de hiperparâmetros
deploy em cloud
monitoramento do modelo em produção