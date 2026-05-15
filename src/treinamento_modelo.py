import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessamento import preparar_dados

# ==========================================
# FEATURES
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

    # ======================================
    # PREPARAR DADOS
    # ======================================

    df = preparar_dados(df)

    # DEBUG
    print("\nColunas disponíveis:")
    print(df.columns.tolist())

    # ======================================
    # X E Y
    # ======================================

    X = df[FEATURES]

    y = df["risco_automacao"]

    # ======================================
    # SPLIT
    # ======================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ======================================
    # MODELO
    # ======================================

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    # ======================================
    # PREVISÃO
    # ======================================

    y_pred = modelo.predict(X_test)

    print("\nRelatório:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # ======================================
    # SALVAR MODELO
    # ======================================

    os.makedirs(
        "output/modelos",
        exist_ok=True
    )

    joblib.dump(
        modelo,
        "output/modelos/random_forest.pkl"
    )

    print("\nModelo salvo com sucesso!")

    return (
        modelo,
        X_test,
        y_test,
        y_pred,
        X
    )