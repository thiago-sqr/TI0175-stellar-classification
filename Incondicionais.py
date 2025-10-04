#%%
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt

arquivo = "star_classification.csv"
dados = pd.read_csv(arquivo)

# %%
# Médias e desvio padrão incondicionais
dados.describe()
#%%
# Histograma
dados.hist(bins=30, figsize=(15,10), edgecolor="black")
plt.tight_layout()
plt.show()
#%%
dados.skew(numeric_only=True)
