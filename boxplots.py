#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dados = pd.read_csv("star_classification.csv")

# colunas quantitativas do dataset e removendo coluna inutil
colunas = dados.drop(["class", "rerun_ID"], axis=1).columns

#%% Plotando Boxplot de todas as colunas, com outlier
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
axes = axes.flatten()
fig.suptitle("Boxplot Incondicional com Outlier", fontsize=45)

for i, coluna in enumerate(colunas):
    sns.boxplot(
        data=dados, 
        y=coluna, 
        ax=axes[i],
        width=0.6,
        fliersize=2
    )
    axes[i].set_title(f"Boxplot de {coluna}")
    axes[i].tick_params(axis="y", labelsize=10) 

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("imagens/boxplot-incond1.pdf", dpi=300)
plt.show()

#%% Plotando Boxplot de todas as colunas, sem outlier

dados_clean =  dados.drop(dados['u'].idxmin())

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20,20))
axes = axes.flatten()
fig.suptitle("Boxplot Incondicional sem Outlier", fontsize=45)

for i, coluna in enumerate(colunas):
    sns.boxplot(
        data=dados_clean,
        y=coluna,
        ax=axes[i],
        width=0.6,
        fliersize=2
    )
    axes[i].set_title(f"Boxplot de {coluna}")
    axes[i].tick_params(axis="y", labelsize=10) 

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("imagens/boxplot-incond2.pdf", dpi=300)
plt.show()

#%% Boxplot Condicionais, sem outlier

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20,20))
axes = axes.flatten()
fig.suptitle("Boxplot Condicional sem Outlier", fontsize=45)

for i, coluna in enumerate(colunas):
    sns.boxplot(
        data=dados_clean, 
        x="class", 
        y=coluna, 
        ax=axes[i],
        hue="class",
        palette="mako",
        width=0.6,
        fliersize=2
    )
    axes[i].set_title(f"Boxplot de {coluna}")
    axes[i].tick_params(axis="y", labelsize=10) 

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("imagens/boxplot-cond.pdf", dpi=300)
plt.show()

