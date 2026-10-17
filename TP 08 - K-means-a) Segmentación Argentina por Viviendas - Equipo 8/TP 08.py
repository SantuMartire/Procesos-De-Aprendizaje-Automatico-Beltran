# TP 08 - K-Means (a) Segmentacion de jurisdicciones argentinas por viviendas habitadas
# Equipo 8
#
# Objetivo: agrupar las 24 jurisdicciones del pais en 3 segmentos segun la
# cantidad de viviendas habitadas:
#   - Muchas viviendas
#   - Cantidad normal de viviendas
#   - Pocas viviendas

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Viviendas habitadas por jurisdicción
viviendas = {
    "CABA": 1391258, 
    "Buenos Aires": 5970702,
    "Catamarca": 131978,
    "Chaco": 368728,
    "Chubut": 213317,
    "Córdoba": 1378237,
    "Corrientes": 370958,
    "Entre Ríos": 494473,
    "Formosa": 194689,
    "Jujuy": 238141,
    "La Pampa": 140879,
    "La Rioja": 124149,
    "Mendoza": 639467,
    "Misiones": 420101,
    "Neuquén": 254545,
    "Río Negro": 276371,
    "Salta": 404504,
    "San Juan": 241436,
    "San Luis": 182886,
    "Santa Cruz": 118047,
    "Santa Fe": 1273460,
    "Santiago del Estero": 311361,
    "Tierra del Fuego": 65535,
    "Tucumán": 493794,   
}

# Paso 1: Pasar el diccionario a una matriz de datos para scikit-learn.
# K-Means espera una matriz de forma (n_muestras, n_caracteristicas); como
# tenemos una sola caracteristica (la cantidad de viviendas), usamos reshape
# para armar una columna. No hace falta estandarizar: al trabajar con una
# unica variable, escalarla no cambia como quedan formados los grupos.
jurisdicciones = list(viviendas.keys())
X = np.array(list(viviendas.values())).reshape(-1, 1)

# Paso 2: Aplicar K-Means con 3 clusters.
# n_init=10 repite el algoritmo con 10 inicializaciones distintas y se queda
# con la mejor, para no depender del azar de los centroides iniciales.
# random_state=42 fija la semilla para que el resultado sea reproducible.
modelo = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = modelo.fit_predict(X)

centroides = modelo.cluster_centers_.flatten()

# Paso 3: Asignar etiquetas interpretables a los clusters.
# K-Means numera los clusters arbitrariamente (0, 1, 2), asi que ordenamos los
# centroides de menor a mayor y recien ahi les ponemos nombre segun su posicion.
orden = np.argsort(centroides)          # indices de los clusters, de menor a mayor centroide
nombres = ["Pocas viviendas", "Cantidad normal de viviendas", "Muchas viviendas"]
etiquetas = {}
for posicion in range(3):
    etiquetas[orden[posicion]] = nombres[posicion]

# Paso 4: Mostrar a que grupo pertenece cada jurisdiccion.
tabla = pd.DataFrame({
    "Jurisdicción": jurisdicciones,
    "Viviendas habitadas": X.flatten(),
    "Cluster": clusters,
    "Segmento": [etiquetas[c] for c in clusters],
})
tabla = tabla.sort_values("Viviendas habitadas", ascending=False).reset_index(drop=True)

print("=== CENTROIDES DE CADA CLUSTER ===")
for posicion in range(3):
    cluster = orden[posicion]
    cantidad = (clusters == cluster).sum()
    print(f"Cluster {cluster} -> {nombres[posicion]:<28} "
          f"centroide: {centroides[cluster]:>12,.0f} viviendas  ({cantidad} jurisdicciones)")

print("\n=== SEGMENTO DE CADA JURISDICCIÓN ===")
print(tabla.to_string(index=False))

print("\n=== JURISDICCIONES AGRUPADAS POR SEGMENTO ===")
for posicion in range(2, -1, -1):              # de "Muchas" a "Pocas"
    segmento = nombres[posicion]
    grupo = tabla[tabla["Segmento"] == segmento]
    print(f"\n{segmento} ({len(grupo)} jurisdicciones):")
    for nombre, cantidad in zip(grupo["Jurisdicción"], grupo["Viviendas habitadas"]):
        print(f"   - {nombre:<22} {cantidad:>10,} viviendas")

# Paso 5: Visualizar los resultados en un grafico de dispersion.
# En el eje X va cada jurisdiccion (ordenadas de mayor a menor) y en el eje Y
# la cantidad de viviendas habitadas. El color indica el segmento y las lineas
# punteadas marcan el centroide de cada cluster.
colores = {
    "Pocas viviendas": "tab:blue",
    "Cantidad normal de viviendas": "tab:orange",
    "Muchas viviendas": "tab:red",
}

fig, ax = plt.subplots(figsize=(13, 7))

posiciones = np.arange(len(tabla))
for posicion in range(2, -1, -1):
    segmento = nombres[posicion]
    mascara = (tabla["Segmento"] == segmento).to_numpy()
    ax.scatter(posiciones[mascara], tabla["Viviendas habitadas"][mascara],
               s=140, color=colores[segmento], edgecolor="black",
               zorder=3, label=segmento)

for posicion in range(3):
    cluster = orden[posicion]
    ax.axhline(centroides[cluster], color=colores[nombres[posicion]],
               linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)
    ax.text(len(tabla) - 0.4, centroides[cluster],
            f"  centroide: {centroides[cluster]:,.0f}",
            color=colores[nombres[posicion]], va="bottom", fontsize=8)

ax.set_xticks(posiciones)
ax.set_xticklabels(tabla["Jurisdicción"], rotation=60, ha="right", fontsize=9)
ax.set_xlabel("Jurisdicción")
ax.set_ylabel("Viviendas habitadas")
ax.set_title("K-Means (k=3): segmentación de jurisdicciones argentinas\n"
             "según cantidad de viviendas habitadas")
ax.yaxis.set_major_formatter(lambda valor, pos: f"{valor:,.0f}")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(title="Segmento")
plt.tight_layout()

plt.savefig("TP 08 - grafico kmeans.png", dpi=150)
print("\nGráfico guardado en: TP 08 - grafico kmeans.png")
plt.show()
