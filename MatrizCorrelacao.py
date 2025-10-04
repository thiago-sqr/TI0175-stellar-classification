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

#vizualizar matriz de correlação
plt.figure(figsize=(16,12))  # pode ajustar (largura, altura)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Matriz de Correlação", fontsize=16)
plt.xticks(rotation=45, ha="right")  # gira os rótulos do eixo X
plt.yticks(rotation=0)               # mantém os rótulos do eixo Y na horizontal
plt.tight_layout()
plt.show()





# %%
