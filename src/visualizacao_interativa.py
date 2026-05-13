import plotly.express as px


def grafico_interativo_risco(df):

    fig = px.histogram(
        df,
        x="risco_automacao",
        title="Distribuição Interativa de Risco",
        color="risco_automacao"
    )

    return fig


def grafico_ranking(df):

    ranking = (
        df["risco_previsto"]
        .value_counts()
        .reset_index()
    )

    ranking.columns = [
        "Risco",
        "Quantidade"
    ]

    fig = px.bar(
        ranking,
        x="Risco",
        y="Quantidade",
        title="Ranking de Risco"
    )

    return fig