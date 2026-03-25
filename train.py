
#Bibliotecas
import pandas as pd
from preprocess import clean_text
import joblib

# Ler o dataset público de reviews de produtos
df = pd.read_csv("data/sentiment.csv", encoding="latin1", low_memory=False)

print("Primeiras linhas do dataset:")
print(df.head())

print("\nInformações gerais do dataset:")
print(df.info())

print("\nFormato original do dataset:")
print(df.shape)

print("\nNomes das colunas:")
print(df.columns)

print("\nQuantidade original de exemplos por classe:")
print(df["Sentiment"].value_counts())

# Manter apenas classes positive e negative
df = df[df["Sentiment"].isin(["positive", "negative"])]

print("\nFormato após filtrar apenas positive e negative:")
print(df.shape)

print("\nQuantidade de exemplos por classe após filtro:")
print(df["Sentiment"].value_counts())

# Remover valores nulos das colunas principais
df = df.dropna(subset=["Review", "Sentiment"])

# Aplicar limpeza no texto
df["Review"] = df["Review"].apply(clean_text)

print("\nPrimeiras reviews após limpeza:")
print(df["Review"].head())

# Separar entrada (X) e saída (y)
X = df["Review"]
y = df["Sentiment"]

print("\nExemplo de entrada (X):")
print(X.head())

print("\nExemplo de saída (y):")
print(y.head())

from sklearn.model_selection import train_test_split

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTamanho do conjunto de treino:")
print(len(X_train))

print("\nTamanho do conjunto de teste:")
print(len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

# Vetorização com TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nFormato do X_train após TF-IDF:")
print(X_train_tfidf.shape)

print("\nFormato do X_test após TF-IDF:")
print(X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression

# Treinar o modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("\nModelo treinado com sucesso!")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fazer previsões
y_pred = model.predict(X_test_tfidf)

print("\nPrimeiras previsões do modelo:")
print(y_pred[:10])

print("\nPrimeiros valores reais:")
print(y_test.iloc[:10].values)

# Avaliação
accuracy = accuracy_score(y_test, y_pred)

print("\nAcurácia do modelo:")
print(accuracy)

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

# Salvar o modelo
joblib.dump(model, "model/model.pkl")

# Salvar o vectorizer
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModelo e vectorizer salvos com sucesso!")

