import pandas as pd
import joblib
import os

# ==========================================
# CARREGAR MODELO
# ==========================================

modelo = None

CAMINHO_MODELO = "output/modelos/random_forest.pkl"

if os.path.exists(CAMINHO_MODELO):

    modelo = joblib.load(
        CAMINHO_MODELO
    )

# ==========================================
# FEATURES
# ==========================================

FEATURES_MODELO = [
    "escolaridade",
    "renda",
    "setor",
    "regiao"
]

# ==========================================
# PREVER RISCO
# ==========================================

def prever_risco(dados):

    if modelo is None:

        return "Modelo ainda não treinado."

    df = pd.DataFrame([dados])

    df = df[FEATURES_MODELO]

    previsao = modelo.predict(df)

    mapa = {
        0: "Baixo",
        1: "Médio",
        2: "Alto"
    }

    return mapa.get(
        previsao[0],
        str(previsao[0])
    )

# ==========================================
# PREVISÃO EM LOTE
# ==========================================

def prever_em_lote(df):

    if modelo is None:

        return pd.DataFrame({
            "Erro": [
                "Modelo ainda não treinado."
            ]
        })

    X = df[FEATURES_MODELO]

    previsoes = modelo.predict(X)

    mapa = {
        0: "Baixo",
        1: "Médio",
        2: "Alto"
    }

    df["risco_previsto"] = [
        mapa.get(p, str(p))
        for p in previsoes
    ]

    return df