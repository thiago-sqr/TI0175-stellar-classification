#%%

# SCRIPT PARA CRIAR GRÁFICOS ESPECíFICOS PRO ARTIGO

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8")

dados = pd.read_csv("data/clean_star_classification.csv")
#%%

redshift = dados[["redshift", "class"]]
redshift_no_stars = redshift[redshift["class"] != "STAR"]
redshift_stars = redshift[redshift["class"] == "STAR"]

fig, axes = plt.subplots(1, 2, figsize=(10,5))  # 1 linha, 2 colunas

# --- Gráfico 1: redshift (sem estrelas)
sns.histplot(
    data=redshift_no_stars,
    x="redshift",
    bins=20,
    hue="class",
    palette="rocket",
    ax=axes[0],
    edgecolor="none"
)
axes[0].set_title("Redshift (Galáxias e Quasares)", fontsize=15)
axes[0].set_xlabel("Redshift", fontsize=15)
axes[0].set_ylabel("Frequência", fontsize=15)


# --- Gráfico 2: redshift (somente estrelas)
sns.histplot(
    data=redshift_stars,
    x="redshift",
    bins=20,
    hue="class",
    palette="rocket_r",
    ax=axes[1],
    edgecolor="none"
)
axes[1].set_title("Redshift (Estrelas)", fontsize=15)
axes[1].set_xlabel("Redshift", fontsize=15)

plt.tight_layout()
plt.savefig("graficos/redshift_comparativo.pdf")
plt.show()

#%%

#grafico de boxplot de redshift por classe
plt.figure(figsize=(8,4))
sns.boxplot(
    data=dados,
    x="class",
    y="redshift",
    palette="mako",
    width=0.5,
    fliersize=2
)
plt.title("Boxplot de Redshift por Classe", fontsize=16)
plt.xlabel("Classe", fontsize=14)
plt.ylabel("Redshift", fontsize=14)
plt.tight_layout()
plt.savefig("graficos/boxplot_redshift_por_classe.pdf", dpi=300)
plt.show() 

#%%
cols = ["r", "i", "z", "class"]
dados_subset = dados[cols]

# Pair plot
sns.pairplot(
    dados_subset,
    hue="class",
    palette="rocket",
    diag_kind="hist",
    corner=True,
    plot_kws={"alpha":0.7, "s":20},
    diag_kws={"bins":20, "edgecolor":"none"}
)

plt.suptitle("Scatter Plot das Variáveis Fotométricas Vermelhas", fontsize=18)
plt.tight_layout()
plt.savefig("graficos/pairplot_fotometricas.png", dpi=300)
plt.show()
