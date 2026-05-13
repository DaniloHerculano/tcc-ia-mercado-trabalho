from explicabilidade import gerar_shap

from preprocessamento import (
    carregar_dados,
    limpar_dados
)

from analise_exploratoria import (
    grafico_distribuicao_risco,
    gerar_heatmap,
    gerar_feature_importance
)

from treinamento_modelo import (
    treinar_modelo
)

from avaliacao_modelo import (
    gerar_matriz_confusao,
    gerar_relatorio_classificacao
)

CAMINHO_ARQUIVO = "data/dados_ocupacoes.csv"


def main():

    df = carregar_dados(CAMINHO_ARQUIVO)

    df = limpar_dados(df)

    grafico_distribuicao_risco(df)

    gerar_heatmap(df)

    modelo, X_test, y_test, y_pred, X = treinar_modelo(df)

    gerar_shap(X)

    gerar_feature_importance(modelo, X)

    gerar_matriz_confusao(y_test, y_pred)

    gerar_relatorio_classificacao(
        y_test,
        y_pred
    )

    print("\nProjeto executado com sucesso!")


if __name__ == "__main__":
    main()