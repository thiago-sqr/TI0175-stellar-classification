#%%
import pandas as pd

dados = pd.read_csv("data/star_classification.csv")
dados.head()
#%%
id_colunas = ["obj_ID","run_ID","rerun_ID","cam_col","field_ID","spec_obj_ID","fiber_ID"]
dados.drop(id_colunas, axis=1, inplace=True)

#%%
# Aqui já percebemos outlier nas colunas (u, g, z)
dados.describe()

#%%
# Perceba um valor de skewness descomunal em (u, g, z)
dados.skew(numeric_only=True)

#%%
# Visualizando dado problemático:
dados["u"].sort_values(ascending=True)
# menor valor de u: -9999
# segundo menos valor: 10.99623 -> discrepante

#%%
# Limpando dado problemático:
dados = dados.drop(dados['u'].idxmin())

#%%
dados.describe()

#%%
# Perceba a melhora no skewness
dados.skew(numeric_only=True)

#%% ==============================================================
# Tendencia central condicional:
galaxy = dados[dados["class"] == "GALAXY"]
quasar = dados[dados["class"] == "QSO"]
stars = dados[dados["class"] == "STAR"]

galaxy.describe()

#%%
quasar.describe()

#%%
stars.describe()

#%%
# Skewness condicional:
galaxy.skew(numeric_only=True)

#%%
quasar.skew(numeric_only=True)

#%%
stars.skew(numeric_only=True)

#%%
# Salvando todos os dados num csv:

dados.to_csv("data/clean_star_classification.csv", index=False) # sem colunas de ID e sem outlier

dados.describe().to_csv("data/tendencia_central.csv", index_label="stats")
galaxy.describe().to_csv("data/tendencia_central_galaxy.csv", index_label="stats")
quasar.describe().to_csv("data/tendencia_central_quasar.csv", index_label="stats")
stars.describe().to_csv("data/tendencia_central_stars.csv", index_label="stats")

dados.skew(numeric_only=True).to_csv("data/skewness.csv", index_label="feature")
galaxy.skew(numeric_only=True).to_csv("data/skewness_galaxy.csv", index_label="feature")
quasar.skew(numeric_only=True).to_csv("data/skewness_quasar.csv", index_label="feature")
stars.skew(numeric_only=True).to_csv("data/skewness_stars.csv", index_label="feature")

