import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ==========================================
# FEATURES OFICIAIS DO MODELO
# ==========================================

FEATURES = [
    "indice",
    "crescimento",
    "renda",
    "escolaridade",
    "setor",
    "regiao"
]

# ==========================================
# TREINAMENTO
# ==========================================

def treinar_modelo(df):

    # Encoder do alvo
    le = LabelEncoder()

    df["risco_automacao"] = le.fit_transform(
        df["risco_automacao"]
    )

    # Features
    X = df[FEATURES]

    # Target
    y = df["risco_automacao"]

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Modelo
    modelo = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    # Previsões
    y_pred = modelo.predict(X_test)

    print("\nRelatório:")
    print(classification_report(y_test, y_pred))

    # cria pasta automaticamente
    os.makedirs(
        "output/modelos",
        exist_ok=True
    )

    joblib.dump(
        modelo,
        "output/modelos/random_forest.pkl"
    )

    print("\nModelo salvo com sucesso!")

    return modelo, X_test, y_test, y_pred, X