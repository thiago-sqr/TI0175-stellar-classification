#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")

dados = pd.read_csv("data/clean_star_classification.csv")

#%% HISTOGRAMA INCONDICIONAL DAS COLUNAS
dados.hist(bins=30, figsize=(15,10), edgecolor="black")
plt.suptitle("Distribuição das Variáveis", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("graficos/hist_incond.pdf", dpi=300)
plt.show()

#%% HISTOGRAMA CONDICIONAL DAS COLUNAS + BAR PLOT DAS CLASSES

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 13))

axes = axes.flatten()

for i, col in enumerate(dados.columns):
    sns.histplot(
        data=dados, 
        x=col, 
        hue="class", 
        bins=30, 
        ax=axes[i], 
        palette="mako", 
        alpha=0.6
    )
    axes[i].set_title(f"Distribuição de {col}")

plt.suptitle("Distribuições das variáveis por classe", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("graficos/hist_cond.pdf", dpi=300)
plt.show()

#%% HISTOGRAMA CONDICIONAL DAS COLUNAS + KDE PLOT 

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 13))

axes = axes.flatten()

for i, col in enumerate(dados.columns):
    sns.histplot(
        data=dados, 
        x=col, 
        hue="class", 
        bins=30, 
        kde=True,
        ax=axes[i], 
        palette="mako", 
        alpha=0.7
    )
    axes[i].set_title(f"Distribuição de {col}")

plt.suptitle("Distribuições das variáveis por classe", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("graficos/kde_cond.pdf", dpi=300)
plt.show()

#%% BOX PLOT INCONDICIONAL

colunas = dados.drop(["class"], axis=1).columns
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,20))
axes = axes.flatten()

for i, coluna in enumerate(colunas):
    sns.boxplot(
        data=dados,
        y=coluna,
        ax=axes[i],
        width=0.4,
        fliersize=2
    )
    axes[i].set_title(f"Boxplot de {coluna}")
    axes[i].tick_params(axis="y", labelsize=10) 

fig.suptitle("Boxplot Incondicional", fontsize=18, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("graficos/boxplot-incond.pdf", dpi=300)
plt.show()

#%% BOX PLOT CONDICIONAL

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,20))
axes = axes.flatten()
fig.suptitle("Boxplot Condicional", fontsize=18)

for i, coluna in enumerate(colunas):
    sns.boxplot(
        data=dados, 
        x="class", 
        y=coluna, 
        ax=axes[i],
        hue="class",
        palette="mako",
        width=0.5,
        fliersize=2
    )
    axes[i].set_title(f"Boxplot de {coluna}")
    axes[i].tick_params(axis="y", labelsize=10) 

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("graficos/boxplot-cond.pdf", dpi=300)
plt.show()