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


#%%
# sem outlier
dados_numericos = dados.select_dtypes(include=[np.number])
dados_numericos["class"] = dados["class"] 


Q1 = dados_numericos.quantile(0.25, numeric_only=True)
Q3 = dados_numericos.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1


filtro = ~((dados_numericos.select_dtypes(include=[np.number]) < (Q1 - 1.5 * IQR)) |
           (dados_numericos.select_dtypes(include=[np.number]) > (Q3 + 1.5 * IQR))).any(axis=1)

dados_sem_outliers = dados_numericos[filtro]

print("Antes:", dados.shape)
print("Depois:", dados_sem_outliers.shape)

# Pairplot
sns.pairplot(
    dados_sem_outliers,
    hue="class",            
    corner=True,
    plot_kws={"s": 15, "alpha": 0.6},  
    palette="Set2"           # paleta de cores
)

plt.show()
# %%
# com outlier
dados_numericos = dados.select_dtypes(include=[np.number])
dados_numericos["class"] = dados["class"] 
sns.pairplot(
    dados_numericos,
    hue="class",            
    corner=True,
    plot_kws={"s": 15, "alpha": 0.6},  
    palette="Set2"
)
plt.show()
# %%
