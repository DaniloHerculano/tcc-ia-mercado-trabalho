import pandas as pd
import joblib

modelo = joblib.load(
    "output/modelos/random_forest.pkl"
)

FEATURES = [
    "indice",
    "crescimento",
    "renda",
    "escolaridade",
    "setor",
    "regiao"
]

MAP_RISCO = {
    0: "🟢 Baixo",
    1: "🟡 Médio",
    2: "🔴 Alto"
}

# ==========================================
# PREVISÃO INDIVIDUAL
# ==========================================

def prever_risco(dados):

    df = pd.DataFrame([dados])

    df = df[FEATURES]

    previsao = modelo.predict(df)[0]

    return MAP_RISCO.get(
        previsao,
        str(previsao)
    )

# ==========================================
# PREVISÃO EM LOTE
# ==========================================

def prever_em_lote(df):

    df_modelo = df[FEATURES]

    previsoes = modelo.predict(df_modelo)

    df["risco_previsto"] = [
        MAP_RISCO.get(p, str(p))
        for p in previsoes
    ]

    return df