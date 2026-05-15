import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CARREGAR DADOS
# ==========================================

def carregar_dados(caminho_arquivo):

    try:

        df = pd.read_csv(
            caminho_arquivo,
            encoding="latin1"
        )

        print("\nDados carregados com sucesso!")
        print(df.head())

        return df

    except Exception as e:

        print(f"Erro ao carregar dados: {e}")

        return None


# ==========================================
# LIMPEZA
# ==========================================

def limpar_dados(df):

    print("\nRemovendo valores ausentes...")

    df = df.dropna()

    print("\nQuantidade de registros após limpeza:")
    print(df.shape)

    return df


# ==========================================
# PREPARAÇÃO
# ==========================================

def preparar_dados(df):

    # ======================================
    # REMOVER NULOS
    # ======================================

    df = df.dropna()

    # ======================================
    # AJUSTAR TEXTO
    # ======================================

    colunas_texto = [
        "escolaridade",
        "setor",
        "regiao",
        "risco_automacao"
    ]

    for coluna in colunas_texto:

        df[coluna] = (
            df[coluna]
            .astype(str)
            .str.strip()
        )

    # ======================================
    # CORRIGIR CODIFICAÇÃO
    # ======================================

    df["escolaridade"] = (
        df["escolaridade"]
        .replace({
            "MÃ©dio": "Médio",
            "TÃ©cnico": "Técnico",
            "PÃ³s-graduaÃ§Ã£o": "Pós-graduação"
        })
    )

    df["setor"] = (
        df["setor"]
        .replace({
            "LogÃstica": "Logística",
            "EducaÃ§Ã£o": "Educação",
            "SaÃºde": "Saúde",
            "IndÃºstria": "Indústria"
        })
    )

    df["risco_automacao"] = (
        df["risco_automacao"]
        .replace({
            "MÃ©dio": "Médio"
        })
    )

    # ======================================
    # MAPEAR ESCOLARIDADE
    # ======================================

    escolaridade_map = {
        "Fundamental": 0,
        "Médio": 1,
        "Técnico": 2,
        "Superior": 3,
        "Pós-graduação": 4
    }

    df["escolaridade"] = (
        df["escolaridade"]
        .map(escolaridade_map)
    )

    # ======================================
    # MAPEAR SETOR
    # ======================================

    setor_map = {
        "Administrativo": 0,
        "Financeiro": 1,
        "Logística": 2,
        "Tecnologia": 3,
        "Saúde": 4,
        "Educação": 5,
        "Indústria": 6
    }

    df["setor"] = (
        df["setor"]
        .map(setor_map)
    )

    # ======================================
    # MAPEAR REGIÃO
    # ======================================

    regiao_map = {
        "Norte": 0,
        "Nordeste": 1,
        "Centro-Oeste": 2,
        "Sudeste": 3,
        "Sul": 4
    }

    df["regiao"] = (
        df["regiao"]
        .map(regiao_map)
    )

    # ======================================
    # MAPEAR RISCO
    # ======================================

    risco_map = {
        "Baixo": 0,
        "Médio": 1,
        "Alto": 2
    }

    df["risco_automacao"] = (
        df["risco_automacao"]
        .map(risco_map)
    )

    # ======================================
    # REMOVER NAN APÓS MAPEAMENTO
    # ======================================

    df = df.dropna(
        subset=[
            "escolaridade",
            "setor",
            "regiao",
            "risco_automacao"
        ]
    )

    # ======================================
    # CONVERTER PARA INT
    # ======================================

    df["escolaridade"] = (
        df["escolaridade"]
        .astype(int)
    )

    df["setor"] = (
        df["setor"]
        .astype(int)
    )

    df["regiao"] = (
        df["regiao"]
        .astype(int)
    )

    df["risco_automacao"] = (
        df["risco_automacao"]
        .astype(int)
    )

    # ======================================
    # NOVAS FEATURES
    # ======================================

    df["indice"] = (
        df["risco_automacao"] * 40
    )

    df["crescimento"] = (
        df["renda"] / 1000
    )

    return df