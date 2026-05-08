from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# GENERAR DATASET
np.random.seed(42)

horas = np.random.randint(1, 300, 300)
logros = np.random.randint(1, 100, 300)

df = pd.DataFrame({
    "Horas": horas,
    "Logros": logros
})

# NORMALIZACION
X = df[["Horas", "Logros"]]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# METODO DEL CODO
inercias = []

for k in range(1, 11):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    modelo.fit(X_scaled)

    inercias.append(modelo.inertia_)

# GRAFICA CODO
plt.figure(figsize=(8,5))

plt.plot(range(1,11), inercias, marker='o')

plt.title("Método del Codo")
plt.xlabel("Número de Clusters")
plt.ylabel("Inercia")

plt.grid(True)

plt.savefig("static/img/codo.png")

plt.close()

# KMEANS FINAL
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

centroides = kmeans.cluster_centers_

# GRAFICA CLUSTERS
plt.figure(figsize=(9,6))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=clusters,
    cmap='viridis',
    s=70
)

plt.scatter(
    centroides[:,0],
    centroides[:,1],
    c='red',
    s=300,
    marker='X'
)

plt.title("Visualización de Clústeres")

plt.xlabel("Horas jugadas")
plt.ylabel("Logros")

plt.grid(True)

plt.savefig("static/img/clusters.png")

plt.close()

# GRAFICA CENTROIDES
plt.figure(figsize=(7,5))

plt.scatter(
    centroides[:,0],
    centroides[:,1],
    c='red',
    s=400,
    marker='X'
)

for i, c in enumerate(centroides):

    plt.text(
        c[0],
        c[1],
        f'C{i+1}',
        fontsize=12
    )

plt.title("Centroides del modelo")

plt.grid(True)

plt.savefig("static/img/centroides.png")

plt.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dataset")
def dataset():
    tabla = df.head(100).to_dict(orient="records")
    return render_template(
        "dataset.html",
        tabla=tabla
    )

@app.route("/kmeans")
def kmeans_page():

    return render_template(
        "kmeans.html"
    )

if __name__ == "__main__":
    app.run(debug=True)