#%%
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt

arquivo = "star_classification.csv"
dados = pd.read_csv(arquivo)

#%%
# ESTRELAS
stars = dados[dados["class"] == "STAR"]
stars.describe()

#%%
# Histograma das estrelas
stars.hist(bins=30, figsize=(15,10), edgecolor="black")
plt.tight_layout()
plt.show()



#%%
#GALAXIAS

galaxies = dados[dados["class"] == "GALAXY"]
galaxies.describe()

#%%
# Histograma das galaxias
galaxies.hist(bins=30, figsize=(15,10), edgecolor="black")
plt.tight_layout()
plt.show()



#%%
# QUASARS

quasars = dados[dados["class"] == "QSO"]
quasars.describe()

#%%
# Histograma dos quasars
quasars.hist(bins=30, figsize=(15,10), edgecolor="black")
plt.tight_layout()
plt.show()



#%%
# Histogramas de todas as classes

colunas = dados.select_dtypes(include=["int64","float64"]).columns

n = len(colunas)
fig, axes = plt.subplots(nrows=n//3+1, ncols=3, figsize=(18, 4*(n//3+1)))

axes = axes.flatten()

for i, col in enumerate(colunas):
    sns.histplot(stars[col], bins=30, color="blue", label="STAR", alpha=0.4, ax=axes[i])
    sns.histplot(galaxies[col], bins=30, color="green", label="GALAXY", alpha=0.4, ax=axes[i])
    sns.histplot(quasars[col], bins=30, color="red", label="QSO", alpha=0.4, ax=axes[i])
    
    axes[i].set_title(f"Distribuição de {col}")
    axes[i].legend()

# Remover eixos extras (se existirem)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
