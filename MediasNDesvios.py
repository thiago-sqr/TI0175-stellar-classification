#%%
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt

arquivo = "star_classification.csv"
dados = pd.read_csv(arquivo)
dados_clean = dados.drop(columns=["rerun_ID"])
# dados_clean = dados_clean.drop(dados["u"].idxmin())

# %%
dados_numericos = dados_clean.select_dtypes(include=[np.number])
dados_numericos["class"] = dados_clean["class"]

# %%
estatisticas = dados_numericos.groupby("class").agg(["mean", "std"])
print(estatisticas)