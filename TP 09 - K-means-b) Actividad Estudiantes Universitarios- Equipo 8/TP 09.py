# TP 09 - K-Means (b) Actividad: Estudiantes Universitarios
# Equipo 8
#
# Objetivo: agrupar un set simulado de 50 estudiantes universitarios en 4
# segmentos segun tres caracteristicas:
#   - Edad
#   - Horas de estudio semanales
#   - Promedio academico
#
# A diferencia del TP 08 (una sola variable), aca conviven tres variables
# medidas en escalas distintas, por lo que estandarizar antes de agrupar es
# obligatorio: sin estandarizar, la edad dominaria el calculo de distancias.

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Datos simulados de estudiantes universitarios (set provisto por la consigna)
estudiantes = [
    {'edad': 42, 'horas_estudio': 2, 'promedio_academico': 8.87},
    {'edad': 26, 'horas_estudio': 8, 'promedio_academico': 9.11},
    {'edad': 56, 'horas_estudio': 5, 'promedio_academico': 7.29},
    {'edad': 29, 'horas_estudio': 14, 'promedio_academico': 7.27},
    {'edad': 43, 'horas_estudio': 8, 'promedio_academico': 8.5},
    {'edad': 34, 'horas_estudio': 10, 'promedio_academico': 4.45},
    {'edad': 51, 'horas_estudio': 3, 'promedio_academico': 6.12},
    {'edad': 22, 'horas_estudio': 12, 'promedio_academico': 4.33},
    {'edad': 47, 'horas_estudio': 6, 'promedio_academico': 7.0},
    {'edad': 38, 'horas_estudio': 9, 'promedio_academico': 4.75},
    {'edad': 60, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 31, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 45, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 3.2},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 36, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 40, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 24, 'horas_estudio': 15, 'promedio_academico': 3.0},
    {'edad': 55, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 33, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 48, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 28, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 52, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 46, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 35, 'horas_estudio': 9, 'promedio_academico': 4.0},
    {'edad': 41, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 23, 'horas_estudio': 14, 'promedio_academico': 1.0},
    {'edad': 53, 'horas_estudio': 3, 'promedio_academico': 8.0},
    {'edad': 32, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 49, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 25, 'horas_estudio': 13, 'promedio_academico': 2.0},
    {'edad': 54, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 37, 'horas_estudio': 8, 'promedio_academico': 4.0},
    {'edad': 39, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 21, 'horas_estudio': 15, 'promedio_academico': 9.0},
    {'edad': 58, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 44, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 29, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 34, 'horas_estudio': 9, 'promedio_academico': 3.0},
    {'edad': 42, 'horas_estudio': 7, 'promedio_academico': 6.0},
    {'edad': 26, 'horas_estudio': 14, 'promedio_academico': 3.0},
    {'edad': 57, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 47, 'horas_estudio': 4, 'promedio_academico': 7.0},
    {'edad': 31, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 45, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 1.0},
    {'edad': 55, 'horas_estudio': 5, 'promedio_academico': 8.0}]

CARACTERISTICAS = ["edad", "horas_estudio", "promedio_academico"]
K = 4                    # cantidad de segmentos pedida por la consigna
ARCHIVO_GRAFICO = "TP 09 - grafico kmeans.png"

# ---------------------------------------------------------------------------
# Paso 0: cargar los datos en un DataFrame y mirarlos antes de agrupar.
# Conocer el rango y el desvio de cada variable es lo que justifica el paso
# siguiente (la estandarizacion), y la correlacion ayuda a interpretar despues
# que significan los grupos que encuentre el algoritmo.
# ---------------------------------------------------------------------------
df = pd.DataFrame(estudiantes)

print("=== DATOS CARGADOS ===")
print(f"{len(df)} estudiantes y {len(CARACTERISTICAS)} caracteristicas por estudiante\n")
print(df[CARACTERISTICAS].describe().round(2).to_string())

print("\n=== CORRELACION ENTRE LAS VARIABLES ===")
print(df[CARACTERISTICAS].corr().round(3).to_string())

# ---------------------------------------------------------------------------
# Paso 1: estandarizar los datos (consigna 1).
# K-Means agrupa usando distancia euclidea, asi que una variable con numeros
# mas grandes pesa mas que las otras solo por su escala. Aca la edad va de 21
# a 60 (desvio ~11,1), las horas de 2 a 15 (desvio ~3,8) y el promedio de 1 a
# 9,11 (desvio ~2,5): sin estandarizar, la edad decidiria practicamente sola
# los grupos. StandardScaler transforma cada variable en z = (x - media) / desvio,
# de modo que las tres queden con media 0 y desvio 1 y pesen lo mismo.
# ---------------------------------------------------------------------------
X = df[CARACTERISTICAS].to_numpy(dtype=float)

escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

print("\n=== ESTANDARIZACION ===")
tabla_escala = pd.DataFrame({
    "Variable": CARACTERISTICAS,
    "Media original": escalador.mean_,
    "Desvío original": np.sqrt(escalador.var_),
    "Media escalada": X_escalado.mean(axis=0),
    "Desvío escalado": X_escalado.std(axis=0),
})
print(tabla_escala.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# Paso 2: aplicar K-Means con 4 clusters (consigna 2).
# n_init=10 corre el algoritmo diez veces con centroides iniciales distintos y
# se queda con la mejor solucion (la de menor inercia), para no depender del
# azar del arranque. random_state=42 fija la semilla para que el resultado sea
# reproducible. El modelo se entrena SOBRE LOS DATOS ESTANDARIZADOS.
# ---------------------------------------------------------------------------
modelo = KMeans(n_clusters=K, n_init=10, random_state=42)
clusters = modelo.fit_predict(X_escalado)

# Centroides: el modelo los devuelve en unidades estandarizadas (utiles para
# comparar variables entre si) y con inverse_transform los volvemos a las
# unidades originales (utiles para interpretarlos y ponerles nombre).
centroides_escalados = modelo.cluster_centers_
centroides = escalador.inverse_transform(centroides_escalados)

# ---------------------------------------------------------------------------
# Paso 3: asignar etiquetas interpretables a los clusters.
# K-Means numera los grupos de manera arbitraria (0, 1, 2, 3) y ese numero no
# dice nada. El nombre se construye leyendo el centroide estandarizado: si una
# coordenada esta a mas de medio desvio por encima de la media general el nivel
# es "alto", a mas de medio desvio por debajo es "bajo", y en el medio "medio".
# Asi la etiqueta sale de los datos y no de un numero de cluster.
# ---------------------------------------------------------------------------
UMBRAL = 0.5             # en desvios estandar

NIVELES_EDAD = {"bajo": "Jóvenes", "medio": "Edad intermedia", "alto": "Adultos"}
NIVELES_HORAS = {"bajo": "poca dedicación", "medio": "dedicación media", "alto": "mucha dedicación"}
NIVELES_PROMEDIO = {"bajo": "bajo rendimiento", "medio": "rendimiento medio", "alto": "alto rendimiento"}


def nivel(z):
    """Traduce una coordenada estandarizada del centroide a bajo / medio / alto."""
    if z >= UMBRAL:
        return "alto"
    if z <= -UMBRAL:
        return "bajo"
    return "medio"


def nombrar(centroide_escalado):
    """Arma el nombre del segmento a partir del centroide estandarizado."""
    z_edad, z_horas, z_promedio = centroide_escalado
    return (f"{NIVELES_EDAD[nivel(z_edad)]} con "
            f"{NIVELES_HORAS[nivel(z_horas)]} y "
            f"{NIVELES_PROMEDIO[nivel(z_promedio)]}")


# Para presentar los resultados ordenamos los segmentos por promedio academico
# del centroide, de mayor a menor, y les damos un codigo corto (S1 ... S4).
orden = np.argsort(centroides[:, 2])[::-1]
nombres = {cluster: nombrar(centroides_escalados[cluster]) for cluster in range(K)}
codigos = {cluster: f"S{posicion + 1}" for posicion, cluster in enumerate(orden)}

df["cluster"] = clusters
df["codigo"] = [codigos[c] for c in clusters]
df["segmento"] = [nombres[c] for c in clusters]

print("\n=== CENTROIDES Y NOMBRE DE CADA SEGMENTO ===")
tabla_centroides = pd.DataFrame({
    "Código": [codigos[c] for c in orden],
    "Cluster": orden,
    "N": [(clusters == c).sum() for c in orden],
    "Edad": centroides[orden, 0],
    "Horas": centroides[orden, 1],
    "Promedio": centroides[orden, 2],
    "Segmento": [nombres[c] for c in orden],
})
print(tabla_centroides.round(2).to_string(index=False))

print("\n=== CENTROIDES EN UNIDADES ESTANDARIZADAS (desvíos respecto de la media general) ===")
tabla_z = pd.DataFrame(centroides_escalados[orden], columns=["Edad (z)", "Horas (z)", "Promedio (z)"])
tabla_z.insert(0, "Código", [codigos[c] for c in orden])
print(tabla_z.round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# Paso 4: mostrar a que segmento pertenece cada estudiante.
# ---------------------------------------------------------------------------
tabla = df.sort_values(["codigo", "edad"]).reset_index(drop=True)

print("\n=== SEGMENTO DE CADA ESTUDIANTE ===")
print(tabla[["codigo", "edad", "horas_estudio", "promedio_academico", "segmento"]]
      .to_string(index=False))

print("\n=== ESTUDIANTES AGRUPADOS POR SEGMENTO ===")
for cluster in orden:
    grupo = df[df["cluster"] == cluster]
    print(f"\n{codigos[cluster]} – {nombres[cluster]}  ({len(grupo)} estudiantes)")
    print(f"   Perfil promedio: {grupo['edad'].mean():.1f} años, "
          f"{grupo['horas_estudio'].mean():.1f} h/semana, "
          f"promedio {grupo['promedio_academico'].mean():.2f}")
    print(f"   Rangos: edad {grupo['edad'].min()}–{grupo['edad'].max()}, "
          f"horas {grupo['horas_estudio'].min()}–{grupo['horas_estudio'].max()}, "
          f"promedio {grupo['promedio_academico'].min():.2f}–{grupo['promedio_academico'].max():.2f}")
    for _, e in grupo.sort_values("edad").iterrows():
        print(f"      - {e['edad']:>2} años | {e['horas_estudio']:>2} h/sem | "
              f"promedio {e['promedio_academico']:.2f}")

# ---------------------------------------------------------------------------
# Paso 5: medir la calidad del agrupamiento.
# La inercia es lo que K-Means minimiza (suma de distancias al cuadrado de cada
# punto a su centroide): siempre baja al agregar clusters, por eso se usa
# buscando el "codo". El coeficiente de silueta va de -1 a 1 y compara, para
# cada estudiante, que tan cerca esta de su propio grupo frente al grupo vecino
# mas proximo; valores cercanos a 1 indican grupos bien separados.
# ---------------------------------------------------------------------------
silueta_global = silhouette_score(X_escalado, clusters)
siluetas = silhouette_samples(X_escalado, clusters)

print("\n=== CALIDAD DEL AGRUPAMIENTO (k = 4) ===")
print(f"Inercia: {modelo.inertia_:.4f}")
print(f"Silueta global: {silueta_global:.4f}")
print(f"Iteraciones hasta converger: {modelo.n_iter_}")
print("\nSilueta por segmento:")
for cluster in orden:
    mascara = clusters == cluster
    print(f"   {codigos[cluster]} – {nombres[cluster]:<55} "
          f"n={mascara.sum():>2}  silueta={siluetas[mascara].mean():.4f}")

print("\n=== COMPARACION DE DISTINTOS VALORES DE k ===")
comparacion = []
for k in range(2, 8):
    prueba = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_escalado)
    tamanos = sorted(((prueba.labels_ == i).sum() for i in range(k)), reverse=True)
    comparacion.append({
        "k": k,
        "Inercia": round(prueba.inertia_, 4),
        "Silueta": round(silhouette_score(X_escalado, prueba.labels_), 4),
        "Tamaños": " – ".join(str(t) for t in tamanos),
    })
print(pd.DataFrame(comparacion).to_string(index=False))

# ---------------------------------------------------------------------------
# Paso 6: visualizar los resultados en un grafico 2D (consigna 3).
# Eje X: edad. Eje Y: promedio academico. El color indica el segmento y la X
# grande marca el centroide de cada cluster. Como el agrupamiento se hizo con
# tres variables pero el grafico solo muestra dos, el tamano del punto se usa
# para representar la tercera (horas de estudio semanales).
# ---------------------------------------------------------------------------
colores = ["tab:green", "tab:blue", "tab:orange", "tab:red"]

fig, ax = plt.subplots(figsize=(12.5, 7.5))

for posicion, cluster in enumerate(orden):
    mascara = (clusters == cluster)
    ax.scatter(df.loc[mascara, "edad"], df.loc[mascara, "promedio_academico"],
               s=df.loc[mascara, "horas_estudio"] * 16,
               color=colores[posicion], edgecolor="black", linewidth=0.6,
               alpha=0.85, zorder=3,
               label=f"{codigos[cluster]} – {nombres[cluster]} (n={mascara.sum()})")

for posicion, cluster in enumerate(orden):
    ax.scatter(centroides[cluster, 0], centroides[cluster, 2],
               marker="X", s=360, color=colores[posicion],
               edgecolor="black", linewidth=1.6, zorder=5)
    ax.annotate(codigos[cluster],
                (centroides[cluster, 0], centroides[cluster, 2]),
                textcoords="offset points", xytext=(0, 16),
                ha="center", fontsize=11, fontweight="bold", zorder=6)

ax.set_xlabel("Edad (años)")
ax.set_ylabel("Promedio académico")
ax.set_title("K-Means (k = 4): segmentación de estudiantes universitarios\n"
             "Edad vs promedio académico – el tamaño del punto indica las horas de estudio semanales")
ax.grid(linestyle=":", alpha=0.5)
ax.set_ylim(0, 10)
ax.legend(title="Segmento (la X marca el centroide)", loc="lower right", fontsize=9)
plt.tight_layout()

plt.savefig(ARCHIVO_GRAFICO, dpi=150)
print(f"\nGráfico guardado en: {ARCHIVO_GRAFICO}")
plt.show()
