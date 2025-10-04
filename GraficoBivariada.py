#%%
import pandas as pd
import numpy as np
import matplotlib as plt

import seaborn as sns
import matplotlib.pyplot as plt

arquivo = "star_classification.csv"
dados = pd.read_csv(arquivo)
dados_clean = dados.drop(columns=["rerun_ID"])
dados_clean = dados_clean.drop(dados["u"].idxmin())

# %%
#calcula a matriz de correlação
corr_matrix = dados_clean.corr(numeric_only=True)

# %%
#vizualizar matriz de correlação
plt.figure(figsize=(16,12))  # pode ajustar (largura, altura)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Matriz de Correlação", fontsize=16)
plt.xticks(rotation=45, ha="right")  # gira os rótulos do eixo X
plt.yticks(rotation=0)               # mantém os rótulos do eixo Y na horizontal
plt.tight_layout()
plt.show()

#%%
# Seleciona só as colunas numéricas
dados_numericos = dados.select_dtypes(include=[np.number])

# ---
# Remove outliers usando IQR
Q1 = dados_numericos.quantile(0.25)
Q3 = dados_numericos.quantile(0.75)
IQR = Q3 - Q1

# Filtro para manter só valores dentro do intervalo permitido
dados_sem_outliers = dados_numericos[~((dados_numericos < (Q1 - 1.5 * IQR)) | 
                                       (dados_numericos > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Antes:", dados_numericos.shape)
print("Depois:", dados_sem_outliers.shape)

# ---
# Gera o pairplot sem outliers
sns.pairplot(dados_sem_outliers, corner=True, plot_kws={"s": 10, "alpha": 0.5})
plt.show()
#%%
# Seleciona só colunas numéricas + a classe
dados_numericos = dados.select_dtypes(include=[np.number])
dados_numericos["class"] = dados["class"]  # mantém a coluna de classe

# ---
# Remove outliers usando IQR (sem perder a classe)
Q1 = dados_numericos.quantile(0.25, numeric_only=True)
Q3 = dados_numericos.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1

# Filtra sem remover a coluna "class"
filtro = ~((dados_numericos.select_dtypes(include=[np.number]) < (Q1 - 1.5 * IQR)) |
           (dados_numericos.select_dtypes(include=[np.number]) > (Q3 + 1.5 * IQR))).any(axis=1)

dados_sem_outliers = dados_numericos[filtro]

print("Antes:", dados.shape)
print("Depois:", dados_sem_outliers.shape)

# ---
# Pairplot com cores por classe
sns.pairplot(
    dados_sem_outliers,
    hue="class",             # diferencia pelas classes
    corner=True,
    plot_kws={"s": 15, "alpha": 0.6},  # tamanho e transparência dos pontos
    palette="Set2"           # paleta de cores
)

plt.show()