import pandas as pd
import joblib

modelo = joblib.load(
    "output/modelos/random_forest.pkl"
)

FEATURES_MODELO = [
    "escolaridade",
    "renda",
    "setor",
    "regiao"
]

# ==========================================
# PREVISÃO INDIVIDUAL
# ==========================================

def prever_risco(dados):

    df = pd.DataFrame([dados])

    df = df[FEATURES_MODELO]

    previsao = modelo.predict(df)

    mapa = {
        0: "🟢 Baixo",
        1: "🟡 Médio",
        2: "🔴 Alto"
    }

    return mapa.get(
        previsao[0],
        str(previsao[0])
    )

# ==========================================
# PREVISÃO EM LOTE
# ==========================================

def prever_em_lote(df):

    X = df[FEATURES_MODELO]

    previsoes = modelo.predict(X)

    mapa = {
        0: "Baixo",
        1: "Médio",
        2: "Alto"
    }

    df["risco_previsto"] = [
        mapa.get(p, p)
        for p in previsoes
    ]

    return df