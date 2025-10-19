#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

#%% Definindo funções do PCA

def padronizar(X: np.ndarray):
    """
    Padroniza os dados: Z = (X - média) / desvio padrão.
    """
    media = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    return (X - media) / std

def covariancias(X: np.ndarray):
    """
    Calcula a matriz de covariância [p x p] dos preditores padronizados.
    """
    n_amostras = X.shape[0]
    return (X.T @ X) / (n_amostras - 1)

def eigens(cov: np.ndarray):
    """
    Retorna autovalores e autovetores ordenados por variância explicada.
    """
    autovalores, autovetores = np.linalg.eigh(cov)
    idx = np.argsort(autovalores)[::-1]
    return autovalores[idx], autovetores[:, idx]

def variancia_explicada(autovalores: np.ndarray):
    """
    Retorna a variância explicada e a acumulada das componentes principais.
    """
    variancia_total = np.sum(autovalores)
    explicada = autovalores / variancia_total
    acumulada = np.cumsum(explicada)
    return explicada, acumulada

def pca(X: np.ndarray):
    """
    Executa o PCA completo e retorna autovalores, autovetores e variâncias.
    """
    autovalores, autovetores = eigens(covariancias(X))
    explicada, acumulada = variancia_explicada(autovalores)
    return autovalores, autovetores, explicada, acumulada

def projetar_dados(X: np.ndarray, autovetores: np.ndarray, k: int):
    """
    Projeta X nas k primeiras componentes principais.
    """
    W = autovetores[:, :k]
    return X @ W


def scree_plot(variancia_explicada):
    """
    Exibe o Scree Plot com a variância explicada por componente.
    """
    pcs = np.arange(1, len(variancia_explicada) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(pcs, variancia_explicada * 100, marker='s', linestyle="--", label="Variância Explicada (%)")
    for i, v in enumerate(variancia_explicada * 100):
        plt.text(i+1, v + 2, f"{v:.1f}%", ha='center')
    plt.title("Scree Plot")
    plt.xlabel("Componentes Principais")
    plt.ylabel("Variância (%)")
    plt.xticks(pcs)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficos/PCA-screeplot.pdf")
    plt.show()



#%% Definindo X
dados = pd.read_csv("data/clean_star_classification.csv")
X = dados.drop(["class","redshift"] , axis=1) # <- Removendo Colunas Alvo
X = np.array(X)
X_std = padronizar(X)

#%% Realizando PCA

autovalores, autovetores, explicada, acumulada = pca(X_std)
idx = ["PC"+str(x+1) for x in range(len(autovalores))]
vars = pd.DataFrame({"explicada": explicada, "acumulada": acumulada}, index=idx)
vars.to_csv("data/PCA-variancia-explicada.csv", index_label="PCs")
vars
#%% Plotando variância explicada

scree_plot(explicada)

#%% Plotando Dataset com 2 PC

import seaborn as sns
dados_2PC = pd.DataFrame(projetar_dados(X_std, autovetores, k=2))
dados_2PC["class"] = dados["class"].values
sns.scatterplot(data=dados_2PC, x=0, y=1, hue="class", palette="rocket", alpha=0.6, s=20)
plt.title("Projeção dos Dados em 2 Componentes Principais")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("graficos/2PC-projection-scatterplot.pdf")
plt.savefig("graficos/2PC-projection-scatterplot.png", dpi=300)
plt.show()