#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dados = pd.read_csv("data/clean_star_classification.csv")

#%% MATRIZ DE CORRELAÇÃO

corr_matrix = dados.corr(numeric_only=True)

plt.figure(figsize=(14,12))  
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Matriz de Correlação", fontsize=16)
plt.xticks(rotation=45, ha="right")  
plt.yticks(rotation=0)   
plt.tight_layout()
plt.savefig("graficos/matriz_correlacao.pdf")
plt.show()

#%% SCATTER PLOT + KDE PLOT DAS COLUNAS COLORIDO POR CLASSE

sns.pairplot(
    dados,
    hue="class",            
    corner=True,
    plot_kws={"s": 15, "alpha": 0.6},  
    palette="mako"
)
plt.tight_layout()
plt.savefig("graficos/scatterplot.pdf")
plt.savefig("graficos/scatterplot.png")
plt.show()

