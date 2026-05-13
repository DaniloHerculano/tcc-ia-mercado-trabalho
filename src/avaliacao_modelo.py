import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def gerar_matriz_confusao(y_test, y_pred):
    matriz = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matriz,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title("Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")

    plt.savefig(
        "output/graficos/matriz_confusao.png"
    )

    plt.close()


def gerar_relatorio_classificacao(y_test, y_pred):
    relatorio = classification_report(
        y_test,
        y_pred
    )

    with open(
        "output/relatorios/classification_report.txt",
        "w"
    ) as arquivo:
        arquivo.write(relatorio)

    print(relatorio)