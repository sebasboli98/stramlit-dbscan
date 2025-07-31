import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

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
fig, ax = plt.subplots()
colors = { -1: "red", 0: "blue", 1: "green", 2: "orange", 3: "purple", 4: "cyan" }

for cluster_id in np.unique(labels):
    ax.scatter(
        df[df["Cluster"] == cluster_id]["X1"],
        df[df["Cluster"] == cluster_id]["X2"],
        label=f"Cluster {cluster_id}" if cluster_id != -1 else "Ruido/Atípicos",
        c=colors.get(cluster_id, "gray"),
        s=60,
        alpha=0.6,
        edgecolor='k'
    )

ax.legend()
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_title("Clusters detectados por DBSCAN")
st.pyplot(fig)

# Mostrar resumen
st.write("Número de perfiles atípicos detectados:", sum(df["Cluster"] == -1))
st.write("Número total de clusters (sin contar atípicos):", len(set(labels)) - (1 if -1 in labels else 0))
