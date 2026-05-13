import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def grafico_distribuicao_risco(df):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x='risco_automacao',
        data=df
    )

    plt.title(
        "Distribuição do Risco de Automação"
    )

    plt.savefig(
        "output/graficos/distribuicao_risco.png"
    )

    plt.close()


def gerar_heatmap(df):

    colunas_numericas = df.select_dtypes(
        include=['int64', 'float64']
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        colunas_numericas.corr(),
        annot=True,
        cmap='coolwarm'
    )

    plt.title("Heatmap de Correlação")

    plt.savefig(
        "output/graficos/heatmap_correlacao.png"
    )

    plt.close()


def gerar_feature_importance(modelo, X):

    importancias = pd.DataFrame({
        'Variavel': X.columns,
        'Importancia': modelo.feature_importances_
    })

    importancias = importancias.sort_values(
        by='Importancia',
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importancias,
        x='Importancia',
        y='Variavel'
    )

    plt.title("Feature Importance")

    plt.savefig(
        "output/graficos/feature_importance.png"
    )

    plt.close()

    return importancias