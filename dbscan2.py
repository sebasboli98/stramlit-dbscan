import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# --- Título de la app ---
st.title("🔍 Detección de Perfiles Atípicos con DBSCAN")

# --- Generar datos artificiales ---
st.subheader("📊 Datos simulados")
centers = [[2, 2], [8, 8], [5, 1]]
X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.8, random_state=42)

# Añadir algunos puntos atípicos manualmente
outliers = np.random.uniform(low=-2, high=12, size=(20, 2))
X_total = np.vstack([X, outliers])

df = pd.DataFrame(X_total, columns=["X1", "X2"])
st.dataframe(df.head())

# --- Estadísticas básicas ---
st.write("**Resumen de los datos:**")
st.write(df.describe())

# --- Visualización inicial de los datos ---
st.subheader("🔎 Distribución inicial de los datos (antes de clustering)")
fig1, ax1 = plt.subplots()
ax1.scatter(df["X1"], df["X2"], c="gray", edgecolor='k', s=60, alpha=0.6)
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_title("Distribución de puntos (incluye outliers)")
st.pyplot(fig1)

# --- Parámetros del usuario ---
st.sidebar.header("Parámetros de DBSCAN")
eps = st.sidebar.slider("eps (distancia máxima)", 0.1, 2.0, 0.5, 0.1)
min_samples = st.sidebar.slider("min_samples (mínimo vecinos)", 2, 10, 5, 1)

# --- Modelo DBSCAN ---
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_total)
df["Cluster"] = labels

# --- Visualización ---
st.subheader("📍 Resultados del Clustering")
fig2, ax2 = plt.subplots()
colors = { -1: "red", 0: "blue", 1: "green", 2: "orange", 3: "purple", 4: "cyan" }

for cluster_id in np.unique(labels):
    ax2.scatter(
        df[df["Cluster"] == cluster_id]["X1"],
        df[df["Cluster"] == cluster_id]["X2"],
        label=f"Cluster {cluster_id}" if cluster_id != -1 else "Ruido/Atípicos",
        c=colors.get(cluster_id, "gray"),
        s=60,
        alpha=0.6,
        edgecolor='k'
    )

ax2.legend()
ax2.set_xlabel("X1")
ax2.set_ylabel("X2")
ax2.set_title("Clusters detectados por DBSCAN")
st.pyplot(fig2)

# --- Métricas de Desempeño ---
num_outliers = sum(df["Cluster"] == -1)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

st.subheader("📈 Métricas de Desempeño")
st.write("Número de perfiles atípicos detectados:", num_outliers)
st.write("Número total de clusters (sin contar atípicos):", num_clusters)
st.write("Porcentaje de atípicos:", f"{100 * num_outliers / len(df):.2f}%")

# Silhouette Score solo si hay al menos 2 clusters reales
if num_clusters >= 2:
    silhouette = silhouette_score(X_total[labels != -1], labels[labels != -1])
    st.write("Silhouette Score (sin ruido):", f"{silhouette:.3f}")
else:
    st.write("Silhouette Score: No se puede calcular (menos de 2 clusters)")

# --- Nota adicional ---
st.info("""
**Nota sobre los datos:**
Se simulan 2 grupos principales con cierta dispersión y se inyectan 20 puntos aleatorios como posibles atípicos. 
Esto permite probar cómo DBSCAN puede diferenciar entre estructuras densas (clusters reales) y puntos aislados (ruido).
""")
