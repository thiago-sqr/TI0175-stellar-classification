#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8")
    
def padronizar(X: np.ndarray):
    media = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    return (X - media) / std


def covariancias(X: np.ndarray):
    n_amostras = X.shape[0]
    return (X.T @ X) / (n_amostras - 1)


def autos(cov: np.ndarray):
    autovalores, autovetores = np.linalg.eigh(cov)
    idx = np.argsort(autovalores)[::-1]
    return autovalores[idx], autovetores[:, idx]


def variancia_explicada(autovalores: np.ndarray):
    variancia_total = np.sum(autovalores)
    explicada = autovalores / variancia_total
    acumulada = np.cumsum(explicada)
    return explicada, acumulada


def pca_variancia(X: np.ndarray):
    X_centralizado = padronizar(X)
    cov = covariancias(X_centralizado)
    autovalores, _ = autos(cov)
    explicada, acumulada = variancia_explicada(autovalores)
    return autovalores, explicada, acumulada


def projetar_dados(X: np.ndarray, autovetores: np.ndarray, k: int):
    W = autovetores[:, :k]  # pega os k primeiros autovetores
    return X @ W


def scree_plot(variancia_explicada):
    pcs = np.arange(1, len(variancia_explicada) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(pcs, variancia_explicada*100, marker='s', linestyle="--", label="Variância Explicada (%)")
    plt.title("Scree Plot")
    plt.xlabel("Componentes Principais")
    plt.ylabel("Variância (%)")
    plt.xticks(pcs)
    plt.legend()
    plt.grid(True)
    plt.show()


#%%    

dados = pd.read_csv("star_classification.csv")
dados = dados.drop(dados['u'].idxmin()) # <-- Limpando aquele outlier chato

X = dados.drop(["rerun_ID", "class", "redshift"], axis=1)
X = np.array(X)
autovalores, autovetores = autos(covariancias(padronizar(X)))
explicada, acumulada = variancia_explicada(autovalores)

vars = pd.DataFrame({"explicada": explicada, "acumulada": acumulada})
vars

#%%
scree_plot(explicada)

#%%
import seaborn as sns
dados_2PC = pd.DataFrame(projetar_dados(padronizar(X), autovetores, k=2))
sns.scatterplot(data=dados_2PC, x=0, y=1)