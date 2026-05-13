import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd

modelo = joblib.load("output/modelos/random_forest.pkl")


def gerar_shap(X):

    explainer = shap.TreeExplainer(modelo)

    shap_values = explainer.shap_values(X)

    plt.figure()

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "output/graficos/shap_summary.png",
        bbox_inches="tight"
    )

    print("\nGráfico SHAP salvo com sucesso!")