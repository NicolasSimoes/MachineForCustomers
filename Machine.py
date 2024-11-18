import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Gerando dados fictícios de clientes
data = {
    'idade': [25, 34, 45, 23, 36, 52, 46, 44, 32, 33, 40, 37, 50, 35, 26, 49],
    'frequencia_compras': [5, 6, 12, 4, 10, 7, 3, 9, 8, 10, 2, 11, 5, 7, 6, 8],
    'gasto_medio': [200, 500, 700, 150, 300, 450, 600, 520, 410, 630, 550, 400, 680, 540, 330, 290]
}

# Criando DataFrame
df = pd.DataFrame(data)

# Normalizando os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Usando K-means para criar clusters (número de clusters pode variar)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Visualizando os clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='idade', y='gasto_medio', hue='cluster', data=df, palette='viridis', s=100)
plt.title("Segmentação de Clientes com K-means")
plt.xlabel("Idade")
plt.ylabel("Gasto Médio")
plt.legend(title='Cluster')
plt.show()

# Exibindo os perfis
for i in range(3):
    cluster_data = df[df['cluster'] == i]
    print(f"\nPerfil do Cluster {i}:")
    print(f"Idade Média: {cluster_data['idade'].mean():.1f}")
    print(f"Frequência Média de Compras: {cluster_data['frequencia_compras'].mean():.1f}")
    print(f"Gasto Médio: R${cluster_data['gasto_medio'].mean():.2f}")
