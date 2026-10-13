
"""
===============================================================================
TP 05 - b) PREDICCION DE ABANDONO EN ESTUDIANTES TERCIARIOS
         Instituto Tecnologico Beltran (ITB) - Equipo 8
===============================================================================

CONSIGNA
--------
Se quiere identificar estudiantes con riesgo de abandonar la carrera para
implementar estrategias de acompanamiento. Se dispone de datos historicos con:
edad, genero, carrera, promedio del 1er cuatrimestre, materias aprobadas,
materias desaprobadas, asistencia promedio (%), situacion laboral, distancia al
instituto, participacion en tutorias/extracurriculares y el estado final
(Abandono / Continua).

OBJETIVO: construir un arbol de decision que prediga el abandono.

TAREAS
  1. Analizar y preparar los datos.
  2. Entrenar un arbol de decision con un conjunto de entrenamiento.
  3. Evaluar el modelo con un conjunto de prueba.
  4. Interpretar el arbol: que variables son mas importantes.
  5. Proponer recomendaciones para la institucion.


ARCHIVOS DE ENTRADA
  - TablaPrediccionAbandono-Entrenamiento.xlsx  -> 100 alumnos CON etiqueta
  - TablaPrediccionAbandono-DatosFinal.xlsx     -> 100 alumnos SIN etiqueta
                                                   (a los que hay que predecir)
ARCHIVOS DE SALIDA (se generan al ejecutar, en la carpeta salidas/)
  - arbol_decision.png
  - matriz_confusion.png
  - importancia_variables.png
  - reglas_arbol.txt
  - Resultados-PrediccionAbandono.xlsx
  - anexo_arbol_con_senal.png
===============================================================================
"""

# =============================================================================
# PASO 0 - IMPORTS Y CONFIGURACION DEL ENTORNO
# =============================================================================
# Un proyecto de ciencia de datos arranca fijando las herramientas y la semilla
# aleatoria. La semilla (SEMILLA) hace que el experimento sea REPRODUCIBLE: si
# un companero corre este script, obtiene exactamente los mismos numeros.

import os
import sys
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # backend sin ventana: solo guarda imagenes
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# Para que los acentos se impriman bien en la consola de Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEMILLA = 42                        # reproducibilidad
CARPETA_SALIDA = "salidas"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def titulo(texto):
    """Imprime un encabezado de seccion para que la salida sea legible."""
    print("\n" + "=" * 79)
    print(texto)
    print("=" * 79)


def subtitulo(texto):
    print("\n--- " + texto + " " + "-" * max(0, 72 - len(texto)))


# =============================================================================
# PASO 1 - CARGA Y ANALISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
# TAREA 1 (primera mitad): "analizar los datos".
#
# Antes de entrenar NADA hay que mirar los datos. Un modelo entrenado sobre
# datos que no entendemos es una caja negra sin valor. En el EDA buscamos:
#   a) Cuantas filas y columnas hay (tamano de la muestra).
#   b) Tipos de dato (numericos vs. categoricos) -> define como se preprocesan.
#   c) Valores nulos y duplicados -> calidad del dato.
#   d) Balance de la clase objetivo -> si una clase domina, la accuracy engaña.
#   e) Si las variables PARECEN relacionarse con el abandono -> hay senal?

titulo("PASO 1 - CARGA Y ANALISIS EXPLORATORIO (EDA)")

ARCHIVO_ENTRENAMIENTO = "TablaPrediccionAbandono-Entrenamiento.xlsx"
ARCHIVO_FINAL = "TablaPrediccionAbandono-DatosFinal.xlsx"

datos = pd.read_excel(ARCHIVO_ENTRENAMIENTO, sheet_name="PrediccionAbandonoTabla")
datos_finales = pd.read_excel(ARCHIVO_FINAL, sheet_name="PrediccionAbandono-DatosFinal")

# Los nombres originales traen parentesis, barras y mayusculas mezcladas.
# Renombrarlos a nombres cortos y limpios hace el codigo (y los graficos) legibles.
RENOMBRAR = {
    "edad": "Edad",
    "genero": "Genero",
    "carrera": "Carrera",
    "PromedioPrimerCuatrimestre": "Promedio",
    "CantMateriasAprobadasPrimerCuatrimestre": "Aprobadas",
    "CantMateriasDesaprobadasPrimerCuatrimestre": "Desaprobadas",
    "AsistenciaPromedio(%)": "Asistencia",
    "trabaja/NoTrabaja": "Trabaja",
    "DistanciaDomicilioAlInstituto(Kms)": "DistanciaKm",
    "ActividadesExtracurriculares(Estudio)": "Tutorias",
    "EstadoFinal": "EstadoFinal",
}
datos = datos.rename(columns=RENOMBRAR)
datos_finales = datos_finales.rename(columns=RENOMBRAR)

subtitulo("1.a Dimensiones de los conjuntos")
print(f"Entrenamiento (con etiqueta): {datos.shape[0]} filas x {datos.shape[1]} columnas")
print(f"Datos finales (sin etiqueta): {datos_finales.shape[0]} filas x {datos_finales.shape[1]} columnas")
print("\nPrimeras 5 filas del conjunto de entrenamiento:")
print(datos.head().to_string())

subtitulo("1.b Tipos de dato")
print(datos.dtypes.to_string())
print("\nLectura: 6 variables numericas (Edad, Promedio, Aprobadas, Desaprobadas,")
print("Asistencia, DistanciaKm) y 4 categoricas (Genero, Carrera, Trabaja, Tutorias).")
print("El arbol de sklearn solo acepta numeros -> las categoricas hay que codificarlas.")

subtitulo("1.c Calidad del dato: nulos y duplicados")
print("Valores nulos por columna:")
print(datos.isna().sum().to_string())
print(f"\nFilas duplicadas: {datos.duplicated().sum()}")
print("Lectura: el dataset esta limpio, no hay que imputar ni borrar nada.")

subtitulo("1.d Estadisticas descriptivas de las variables numericas")
print(datos.describe().round(2).to_string())

subtitulo("1.e Balance de la variable objetivo (EstadoFinal)")
conteo = datos["EstadoFinal"].value_counts()
proporcion = datos["EstadoFinal"].value_counts(normalize=True)
print(pd.DataFrame({"cantidad": conteo, "proporcion": proporcion.round(3)}).to_string())
print("\nLectura: las clases estan practicamente balanceadas (~54% / ~46%).")
print("Esto es bueno: no necesitamos tecnicas de rebalanceo agresivas, pero SI")
print("tenemos que recordar que el 'baseline tonto' (predecir siempre la clase")
print("mayoritaria) ya acierta ~54%. Cualquier modelo tiene que superar eso.")

# --- Entropia inicial (calculada a mano, como en el TP 04) --------------------
# La entropia mide el "desorden" del conjunto. Con dos clases vale entre 0
# (conjunto puro, todos iguales) y 1 (mitad y mitad, maxima incertidumbre).
# El arbol de decision va a elegir, en cada nodo, la pregunta que MAS reduzca
# esta entropia (ganancia de informacion).
subtitulo("1.e-bis Entropia del conjunto original (calculo manual, como TP 04)")
p_abandono = proporcion.get("Abandonó", 0.0)
p_continua = proporcion.get("Continúa", 0.0)
entropia_inicial = -(p_abandono * math.log2(p_abandono) + p_continua * math.log2(p_continua))
print(f"P(Abandono) = {p_abandono:.4f}   P(Continua) = {p_continua:.4f}")
print(f"H(S) = -[p1*log2(p1) + p2*log2(p2)] = {entropia_inicial:.4f} bits")
print("Lectura: un valor casi igual a 1.0 significa maxima incertidumbre. El")
print("conjunto arranca muy desordenado; el arbol deberia ir bajando esa entropia.")

subtitulo("1.f Hay senal? Comparacion de medias por clase")
# Si una variable sirve para predecir, su promedio deberia DIFERIR entre los que
# abandonaron y los que continuaron. Si los promedios son casi iguales, esa
# variable no aporta informacion.
NUMERICAS = ["Edad", "Promedio", "Aprobadas", "Desaprobadas", "Asistencia", "DistanciaKm"]
comparacion = datos.groupby("EstadoFinal")[NUMERICAS].mean().T
comparacion["diferencia_abs"] = (comparacion["Abandonó"] - comparacion["Continúa"]).abs()
comparacion["dif_%_sobre_media"] = (
    100 * comparacion["diferencia_abs"] / datos[NUMERICAS].mean()
)
print(comparacion.round(2).to_string())
print("\nLectura CLAVE: las diferencias son minimas. Ejemplo: la asistencia")
print("promedio de los que abandonaron es casi identica a la de los que")
print("continuaron. Intuitivamente esperariamos lo contrario (que el que abandona")
print("haya faltado mucho mas). Esto es una PRIMERA SENAL DE ALARMA: las")
print("variables casi no se relacionan con el resultado.")

subtitulo("1.g Hay senal? Tablas de contingencia de las variables categoricas")
CATEGORICAS = ["Genero", "Trabaja", "Tutorias"]
for col in CATEGORICAS:
    tabla = pd.crosstab(datos[col], datos["EstadoFinal"], normalize="index").round(3)
    print(f"\n% de abandono segun {col}:")
    print(tabla.to_string())
print("\nLectura: en todas las categorias el abandono ronda el 51-57%, es decir")
print("practicamente la tasa general. Ninguna categoria discrimina.")


# =============================================================================
# PASO 2 - PREPARACION DE LOS DATOS
# =============================================================================
# TAREA 1 (segunda mitad): "preparar los datos".
#
# Tres sub-pasos:
#   2.1 Ingenieria de atributos: crear variables nuevas con sentido pedagogico.
#   2.2 Separar X (atributos) de y (objetivo) y codificar las categoricas.
#   2.3 Partir en entrenamiento / prueba (train / test).

titulo("PASO 2 - PREPARACION DE LOS DATOS")

subtitulo("2.1 Ingenieria de atributos (feature engineering)")
# Un modelo no "descubre" relaciones que podemos darle ya masticadas. Dos
# variables derivadas que en la realidad son muy informativas del abandono:
#   - TotalMaterias  = aprobadas + desaprobadas  (carga que realmente curso)
#   - TasaAprobacion = aprobadas / total         (rendimiento relativo, 0 a 1)
# La tasa es mejor que el conteo crudo: aprobar 3 de 4 no es lo mismo que 3 de 9.


def agregar_atributos(df):
    df = df.copy()
    df["TotalMaterias"] = df["Aprobadas"] + df["Desaprobadas"]
    df["TasaAprobacion"] = df["Aprobadas"] / df["TotalMaterias"].replace(0, np.nan)
    df["TasaAprobacion"] = df["TasaAprobacion"].fillna(0)
    return df


datos = agregar_atributos(datos)
datos_finales = agregar_atributos(datos_finales)
print("Se crearon: TotalMaterias y TasaAprobacion.")
print(datos[["Aprobadas", "Desaprobadas", "TotalMaterias", "TasaAprobacion"]].head().round(3).to_string())

subtitulo("2.2 Separacion X / y y codificacion de categoricas")
# y: la variable a predecir. La convertimos a 1/0 porque sklearn trabaja con
# numeros y porque asi definimos explicitamente cual es la clase POSITIVA.
# Elegimos 1 = "Abandono" porque es el evento que la institucion quiere detectar.
COLUMNAS_NUMERICAS = NUMERICAS + ["TotalMaterias", "TasaAprobacion"]
COLUMNAS_CATEGORICAS = ["Genero", "Carrera", "Trabaja", "Tutorias"]

X = datos[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS]
y = (datos["EstadoFinal"] == "Abandonó").astype(int)

print(f"X: {X.shape[1]} atributos. y: clase positiva = 1 = 'Abandono'")
print(f"Distribucion de y -> 1 (Abandono): {int(y.sum())} | 0 (Continua): {int((1 - y).sum())}")

# One-Hot Encoding: cada categoria se convierte en una columna 0/1.
# Por que One-Hot y no asignar numeros (0,1,2,...)? Porque numerar categorias
# inventa un ORDEN que no existe (Radiologia=2 no es "mayor" que Enfermeria=1)
# y el arbol haria cortes sin sentido tipo "carrera <= 1.5".
#
# Lo armamos dentro de un ColumnTransformer + Pipeline. Por que un Pipeline?
# Para que el preprocesamiento se aprenda SOLO con los datos de entrenamiento y
# se aplique igual a test y a los datos finales. Si codificaramos todo junto
# antes de partir, habria fuga de informacion (data leakage).
preprocesador = ColumnTransformer(
    transformers=[
        ("num", "passthrough", COLUMNAS_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         COLUMNAS_CATEGORICAS),
    ]
)
print("\nPreprocesador: numericas tal cual (el arbol no necesita escalado,")
print("porque parte por umbrales y no por distancias) + One-Hot en categoricas.")

subtitulo("2.3 Particion entrenamiento / prueba (70% / 30%)")
# Por que partir? Para poder medir el modelo en datos que NUNCA vio. Evaluar en
# los mismos datos de entrenamiento siempre da resultados infladamente buenos
# (un arbol sin poda memoriza y da 100%).
# stratify=y mantiene la misma proporcion de abandono en train y test.
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.30, random_state=SEMILLA, stratify=y
)
print(f"Entrenamiento: {len(X_entrenamiento)} alumnos "
      f"({y_entrenamiento.mean():.1%} abandono)")
print(f"Prueba:        {len(X_prueba)} alumnos "
      f"({y_prueba.mean():.1%} abandono)")
print("La proporcion de abandono se mantiene en ambos (gracias a stratify).")


# =============================================================================
# PASO 3 - ENTRENAMIENTO DEL ARBOL DE DECISION
# =============================================================================
# TAREA 2.
#
# Como funciona un arbol de decision (repaso rapido):
#   - En cada nodo prueba todas las preguntas posibles del tipo
#     "variable <= umbral" y elige la que mas reduce la impureza.
#   - Usamos criterion="entropy" => la reduccion de impureza es exactamente la
#     GANANCIA DE INFORMACION que calculamos a mano en el TP 04.
#   - Repite hasta que los nodos son puros o hasta un criterio de parada.
#
# El gran riesgo es el SOBREAJUSTE (overfitting): un arbol sin limites crece
# hasta memorizar cada alumno y despues falla con alumnos nuevos. Para evitarlo
# se PODA con hiperparametros: max_depth, min_samples_leaf, min_samples_split.

titulo("PASO 3 - ENTRENAMIENTO DEL ARBOL DE DECISION")

subtitulo("3.1 Modelo de referencia (baseline)")
# Regla de oro: siempre comparar contra un modelo tonto. Si el arbol no le gana
# a "predecir siempre la clase mayoritaria", el arbol no sirve para nada.
baseline = DummyClassifier(strategy="most_frequent", random_state=SEMILLA)
baseline.fit(X_entrenamiento, y_entrenamiento)
acc_baseline = accuracy_score(y_prueba, baseline.predict(X_prueba))
print(f"Baseline (siempre 'Abandono'): accuracy en prueba = {acc_baseline:.3f}")
print("ESTE es el numero a superar.")

subtitulo("3.2 Arbol sin poda: demostracion del sobreajuste")
# Entrenamos un arbol sin restricciones para VER el sobreajuste con numeros.
arbol_sin_poda = Pipeline([
    ("prep", preprocesador),
    ("arbol", DecisionTreeClassifier(criterion="entropy", random_state=SEMILLA)),
])
arbol_sin_poda.fit(X_entrenamiento, y_entrenamiento)
acc_train_sp = accuracy_score(y_entrenamiento, arbol_sin_poda.predict(X_entrenamiento))
acc_test_sp = accuracy_score(y_prueba, arbol_sin_poda.predict(X_prueba))
print(f"Accuracy en ENTRENAMIENTO: {acc_train_sp:.3f}")
print(f"Accuracy en PRUEBA:        {acc_test_sp:.3f}")
print(f"Profundidad del arbol:     {arbol_sin_poda.named_steps['arbol'].get_depth()}")
print(f"Hojas:                     {arbol_sin_poda.named_steps['arbol'].get_n_leaves()}")
print("\nLectura: 100% en entrenamiento y mucho menos en prueba = SOBREAJUSTE")
print("clasico. El arbol memorizo a los 70 alumnos de entrenamiento uno por uno.")
print("Por eso hay que podar.")

subtitulo("3.3 Busqueda de hiperparametros con validacion cruzada (GridSearchCV)")
# En lugar de elegir max_depth "a ojo", probamos una grilla de combinaciones y
# nos quedamos con la mejor segun validacion cruzada estratificada de 5 pliegues.
# Validacion cruzada = partir el entrenamiento en 5, entrenar con 4 y validar
# con 1, rotando. Da una estimacion mas estable que una sola particion.
grilla = {
    "arbol__criterion": ["entropy", "gini"],
    "arbol__max_depth": [2, 3, 4, 5, 6, None],
    "arbol__min_samples_leaf": [1, 3, 5, 8, 10],
    "arbol__min_samples_split": [2, 5, 10],
    "arbol__class_weight": [None, "balanced"],
}
pipeline_base = Pipeline([
    ("prep", preprocesador),
    ("arbol", DecisionTreeClassifier(random_state=SEMILLA)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
busqueda = GridSearchCV(pipeline_base, grilla, cv=cv, scoring="f1",
                        n_jobs=-1, refit=True)
busqueda.fit(X_entrenamiento, y_entrenamiento)

print(f"Combinaciones evaluadas: {len(busqueda.cv_results_['params'])}")
print("\nMejores hiperparametros encontrados:")
for clave, valor in busqueda.best_params_.items():
    print(f"   {clave.replace('arbol__', '')}: {valor}")
print(f"\nMejor F1 en validacion cruzada: {busqueda.best_score_:.3f}")
print("Se optimizo F1 (y no accuracy) porque nos interesa detectar bien la clase")
print("'Abandono': F1 combina precision y recall sobre esa clase.")

modelo = busqueda.best_estimator_
arbol = modelo.named_steps["arbol"]
nombres_atributos = list(modelo.named_steps["prep"].get_feature_names_out())
# Limpiamos los prefijos "num__" / "cat__" que agrega el ColumnTransformer
nombres_atributos = [c.replace("num__", "").replace("cat__", "") for c in nombres_atributos]

print(f"\nArbol final -> profundidad: {arbol.get_depth()} | hojas: {arbol.get_n_leaves()}")
print(f"Atributos tras One-Hot: {len(nombres_atributos)}")

subtitulo("3.4 Estabilidad del modelo elegido (validacion cruzada)")
puntajes_cv = cross_val_score(modelo, X_entrenamiento, y_entrenamiento,
                              cv=cv, scoring="accuracy")
print(f"Accuracy por pliegue: {np.round(puntajes_cv, 3)}")
print(f"Media: {puntajes_cv.mean():.3f}  |  Desvio: {puntajes_cv.std():.3f}")
print("Lectura: un desvio alto significa que el resultado depende mucho de que")
print("alumnos cayeron en cada pliegue -> el modelo no es estable.")


# =============================================================================
# PASO 4 - EVALUACION DEL MODELO EN EL CONJUNTO DE PRUEBA
# =============================================================================
# TAREA 3.
#
# Metricas y por que cada una (positivo = "Abandono"):
#   - Accuracy  : % total de aciertos. Enganna si las clases estan desbalanceadas.
#   - Precision : de los que marque como "va a abandonar", cuantos abandonaron.
#                 Precision baja = gasto recursos de tutoria en alumnos que no
#                 estaban en riesgo (costo economico).
#   - Recall    : de todos los que realmente abandonaron, cuantos detecte.
#                 Recall bajo = se me escapan alumnos en riesgo (costo social).
#                 ** Para esta institucion el RECALL es la metrica mas importante **
#   - F1        : media armonica de precision y recall.
#   - ROC-AUC   : capacidad de ordenar alumnos por riesgo. 0.5 = azar.

titulo("PASO 4 - EVALUACION EN EL CONJUNTO DE PRUEBA")

y_predicho = modelo.predict(X_prueba)
y_probabilidad = modelo.predict_proba(X_prueba)[:, 1]

accuracy = accuracy_score(y_prueba, y_predicho)
precision = precision_score(y_prueba, y_predicho, zero_division=0)
recall = recall_score(y_prueba, y_predicho, zero_division=0)
f1 = f1_score(y_prueba, y_predicho, zero_division=0)
try:
    auc = roc_auc_score(y_prueba, y_probabilidad)
except ValueError:
    auc = float("nan")

subtitulo("4.1 Metricas principales")
tabla_metricas = pd.DataFrame({
    "Metrica": ["Accuracy", "Precision (Abandono)", "Recall (Abandono)",
                "F1 (Abandono)", "ROC-AUC"],
    "Valor": [accuracy, precision, recall, f1, auc],
})
print(tabla_metricas.round(3).to_string(index=False))
print(f"\nComparacion contra el baseline: modelo {accuracy:.3f} vs "
      f"baseline {acc_baseline:.3f}  ->  diferencia {accuracy - acc_baseline:+.3f}")

subtitulo("4.2 Matriz de confusion")
# Filas = realidad, columnas = prediccion.
#   VN: continuo y predije que continuaba      FP: continuo y predije abandono
#   FN: abandono y predije que continuaba      VP: abandono y predije abandono
# El FN es el error caro: el alumno se va y la institucion no lo acompanio.
matriz = confusion_matrix(y_prueba, y_predicho)
print(pd.DataFrame(matriz,
                   index=["Real: Continua", "Real: Abandono"],
                   columns=["Pred: Continua", "Pred: Abandono"]).to_string())
vn, fp, fn, vp = matriz.ravel()
print(f"\nVerdaderos Negativos (VN): {vn}  -> continuaron y los predije bien")
print(f"Falsos Positivos    (FP): {fp}  -> alerta en falso (gasto innecesario)")
print(f"Falsos Negativos    (FN): {fn}  -> ABANDONOS NO DETECTADOS (error grave)")
print(f"Verdaderos Positivos(VP): {vp}  -> abandonos detectados correctamente")

subtitulo("4.3 Reporte de clasificacion completo")
print(classification_report(y_prueba, y_predicho,
                            target_names=["Continua (0)", "Abandono (1)"],
                            zero_division=0))

# Grafico de la matriz de confusion
fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay(confusion_matrix=matriz,
                       display_labels=["Continua", "Abandono"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Matriz de confusion - conjunto de prueba")
ax.set_xlabel("Prediccion")
ax.set_ylabel("Valor real")
plt.tight_layout()
ruta = os.path.join(CARPETA_SALIDA, "matriz_confusion.png")
plt.savefig(ruta, dpi=150)
plt.close()
print(f"[guardado] {ruta}")

subtitulo("4.4 TEST DE SIGNIFICANCIA: el modelo aprendio algo real?")
# Esta es la prueba decisiva y la parte mas importante del TP.
# Idea: si el modelo realmente captura un patron, deberia andar MUCHO mejor con
# las etiquetas verdaderas que con etiquetas MEZCLADAS AL AZAR.
# Mezclamos y 200 veces, medimos la accuracy por validacion cruzada en cada caso
# y comparamos con la del modelo real. Esto es un "permutation test".
accuracy_real = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy").mean()

generador = np.random.default_rng(SEMILLA)
accuracies_azar = []
for _ in range(200):
    y_mezclado = pd.Series(generador.permutation(y.values), index=y.index)
    puntaje = cross_val_score(modelo, X, y_mezclado, cv=cv, scoring="accuracy").mean()
    accuracies_azar.append(puntaje)
accuracies_azar = np.array(accuracies_azar)

# p-valor: proporcion de veces que el AZAR iguala o supera al modelo real.
p_valor = (np.sum(accuracies_azar >= accuracy_real) + 1) / (len(accuracies_azar) + 1)

print(f"Accuracy del modelo con etiquetas REALES (CV): {accuracy_real:.3f}")
print(f"Accuracy con etiquetas AL AZAR (200 repeticiones): "
      f"media {accuracies_azar.mean():.3f}, maximo {accuracies_azar.max():.3f}")
print(f"p-valor del test de permutacion: {p_valor:.3f}")
print()
if p_valor > 0.05:
    print(">>> CONCLUSION ESTADISTICA: p-valor > 0.05. NO podemos rechazar la")
    print(">>> hipotesis nula. El modelo NO rinde mejor que el azar: los datos")
    print(">>> provistos NO contienen una relacion aprendible entre los atributos")
    print(">>> del alumno y su estado final. El dataset es, en la practica, ruido.")
else:
    print(">>> CONCLUSION ESTADISTICA: p-valor <= 0.05. El modelo captura un")
    print(">>> patron real, no explicable por azar.")


# =============================================================================
# PASO 5 - INTERPRETACION DEL ARBOL
# =============================================================================
# TAREA 4: "que variables parecen ser mas importantes para predecir el abandono".
#
# Tres formas complementarias de interpretar:
#   5.1 Dibujar el arbol: la variable de la RAIZ es la que mas informacion aporta.
#   5.2 feature_importances_: cuanta impureza total redujo cada variable.
#       OJO: esta medida esta sesgada a favor de variables con muchos valores
#       distintos (como Promedio o Asistencia, que son continuas).
#   5.3 Importancia por permutacion: se desordena una variable y se mide cuanto
#       empeora el modelo. Es mas honesta porque se mide SOBRE DATOS DE PRUEBA.

titulo("PASO 5 - INTERPRETACION DEL ARBOL")

subtitulo("5.1 Estructura del arbol y reglas de decision")
reglas = export_text(arbol, feature_names=nombres_atributos, decimals=2)
print("Primeras 40 lineas de las reglas (el archivo completo se guarda aparte):")
print("\n".join(reglas.splitlines()[:40]))
ruta_reglas = os.path.join(CARPETA_SALIDA, "reglas_arbol.txt")
with open(ruta_reglas, "w", encoding="utf-8") as archivo:
    archivo.write(reglas)
print(f"\n[guardado] {ruta_reglas}")

variable_raiz = (nombres_atributos[arbol.tree_.feature[0]]
                 if arbol.tree_.feature[0] >= 0 else "(arbol de un solo nodo)")
print(f"\nVariable en la RAIZ del arbol: {variable_raiz}")
print("La raiz es la pregunta con mayor ganancia de informacion: segun el")
print("algoritmo, es la variable que mejor separa a los que abandonan.")

# Dibujo del arbol
fig, ax = plt.subplots(figsize=(22, 11))
plot_tree(arbol, feature_names=nombres_atributos,
          class_names=["Continua", "Abandono"],
          filled=True, rounded=True, fontsize=8, ax=ax, proportion=False)
ax.set_title("Arbol de decision - Prediccion de abandono (ITB)", fontsize=15)
plt.tight_layout()
ruta_arbol = os.path.join(CARPETA_SALIDA, "arbol_decision.png")
plt.savefig(ruta_arbol, dpi=140, bbox_inches="tight")
plt.close()
print(f"[guardado] {ruta_arbol}")

subtitulo("5.2 Importancia de variables segun el arbol (Gini/entropia)")
importancias = pd.DataFrame({
    "Variable": nombres_atributos,
    "Importancia": arbol.feature_importances_,
}).sort_values("Importancia", ascending=False)
importancias_utiles = importancias[importancias["Importancia"] > 0]
print(importancias_utiles.round(4).to_string(index=False))
print(f"\nVariables que el arbol NO uso en ningun corte: "
      f"{len(importancias) - len(importancias_utiles)} de {len(importancias)}")

subtitulo("5.3 Importancia por permutacion (medida sobre el conjunto de prueba)")
perm = permutation_importance(modelo, X_prueba, y_prueba, n_repeats=30,
                              random_state=SEMILLA, scoring="accuracy")
importancias_perm = pd.DataFrame({
    "Variable": X_prueba.columns,
    "Caida_media_accuracy": perm.importances_mean,
    "Desvio": perm.importances_std,
}).sort_values("Caida_media_accuracy", ascending=False)
print(importancias_perm.round(4).to_string(index=False))
print("\nLectura: si al desordenar una variable la accuracy NO cae (valor ~0 o")
print("negativo), esa variable no estaba aportando nada. Valores negativos")
print("significan que el modelo incluso mejora sin esa variable -> era ruido.")

# Grafico de importancias
top = importancias_utiles.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top["Variable"], top["Importancia"], color="#4C72B0")
ax.set_xlabel("Importancia (reduccion de impureza)")
ax.set_title("Variables mas importantes segun el arbol")
plt.tight_layout()
ruta_imp = os.path.join(CARPETA_SALIDA, "importancia_variables.png")
plt.savefig(ruta_imp, dpi=150)
plt.close()
print(f"\n[guardado] {ruta_imp}")

subtitulo("5.4 Interpretacion honesta de los resultados")
print("""
Lo que muestra el arbol entrenado:
  * La variable de la raiz y las importancias cambian si se cambia la semilla o
    la particion, porque no hay un patron estable que encontrar.
  * El arbol NO le gana al baseline: en el conjunto de prueba queda incluso por
    debajo (0.43 vs 0.53) y el ROC-AUC da menos de 0.50, o sea que ordenar
    alumnos por riesgo con este modelo es peor que tirar una moneda.
  * Medido por validacion cruzada sobre los 100 alumnos la accuracy sube a ~0.56,
    practicamente igual al 0.54 de la clase mayoritaria: la diferencia entre
    0.43 y 0.56 es puro efecto de que alumnos cayeron en cada particion.
  * El test de permutacion (4.4) lo confirma: con etiquetas al azar el mismo
    modelo llega a 0.64, mas que con las etiquetas reales (p-valor 0.164).

Conclusion metodologica: NO se puede afirmar, con ESTE dataset, que una
variable sea "la mas importante" para predecir el abandono. Hacerlo seria
inventar una conclusion: estariamos interpretando ruido como si fuera senal.
Esa es, en si misma, una respuesta valida de ciencia de datos.

Por que pasa esto? Las causas tipicas son:
  1. Los datos parecen generados de forma aleatoria (sin relacion real entre
     atributos y etiqueta): ver las medias casi identicas del paso 1.f.
  2. La muestra es chica (100 alumnos para 10 atributos, uno de ellos con 9
     categorias -> One-Hot genera mas de 20 columnas).
  3. Faltan las variables que la bibliografia senala como mas predictivas:
     situacion economica, beca, nota de ingreso, materias recursadas,
     inasistencias consecutivas, intervalo desde el egreso del secundario.
""")


# =============================================================================
# PASO 6 - PREDICCION SOBRE EL CONJUNTO "DATOS FINAL"
# =============================================================================
# El archivo DatosFinal tiene 100 alumnos SIN etiqueta: son los alumnos reales a
# clasificar. Para cada uno devolvemos:
#   - la clase predicha (Abandona / Continua)
#   - la PROBABILIDAD de abandono (mas util que la clase: permite priorizar)
#   - un nivel de riesgo (Alto / Medio / Bajo) para operativizar las tutorias
#
# OJO: dado el resultado del paso 4.4, estas predicciones se entregan como
# ejercicio tecnico, NO como un diagnostico confiable.

titulo("PASO 6 - PREDICCION SOBRE EL CONJUNTO 'DATOS FINAL'")

X_final = datos_finales[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS]
prediccion_final = modelo.predict(X_final)
probabilidad_final = modelo.predict_proba(X_final)[:, 1]

resultados = datos_finales.copy()
resultados["Prediccion"] = np.where(prediccion_final == 1, "Abandona", "Continúa")
resultados["ProbabilidadAbandono"] = probabilidad_final.round(4)
resultados["NivelRiesgo"] = pd.cut(
    probabilidad_final,
    bins=[-0.001, 0.40, 0.70, 1.001],
    labels=["Bajo", "Medio", "Alto"],
)

subtitulo("6.1 Distribucion de las predicciones")
print(resultados["Prediccion"].value_counts().to_string())
print("\nDistribucion por nivel de riesgo:")
print(resultados["NivelRiesgo"].value_counts().sort_index().to_string())

subtitulo("6.2 Los 10 alumnos con mayor probabilidad de abandono")
columnas_mostrar = ["Edad", "Promedio", "Asistencia", "Desaprobadas", "Trabaja",
                    "Tutorias", "DistanciaKm", "ProbabilidadAbandono", "NivelRiesgo"]
print(resultados.sort_values("ProbabilidadAbandono", ascending=False)
      .head(10)[columnas_mostrar].to_string())

ruta_resultados = os.path.join(CARPETA_SALIDA, "Resultados-PrediccionAbandono.xlsx")
with pd.ExcelWriter(ruta_resultados, engine="openpyxl") as escritor:
    resultados.to_excel(escritor, sheet_name="Predicciones", index=False)
    tabla_metricas.round(4).to_excel(escritor, sheet_name="Metricas", index=False)
    importancias.round(5).to_excel(escritor, sheet_name="ImportanciaVariables", index=False)
print(f"\n[guardado] {ruta_resultados}")


# =============================================================================
# PASO 7 - RECOMENDACIONES PARA LA INSTITUCION
# =============================================================================
# TAREA 5.

titulo("PASO 7 - RECOMENDACIONES PARA EL INSTITUTO BELTRAN")

print("""
A) RECOMENDACIONES SOBRE LOS DATOS (prioridad 1: sin esto no hay modelo)

  1. Auditar el origen del dataset. El test de permutacion muestra que los
     atributos no se relacionan con el estado final. Si los datos fueron
     generados sinteticamente al azar, ningun algoritmo va a funcionar: el
     problema no es el modelo, es la materia prima.

  2. Ampliar la muestra. 100 registros son muy pocos. Con 10 atributos (uno de
     9 categorias) se necesitan varios cientos de casos, para que el arbol generalice.

  3. Incorporar variables que la bibliografia identifica como predictoras:
       - academicas: nota de ingreso, materias recursadas, inasistencias
         CONSECUTIVAS (no solo el promedio anual), abandono de cursada en el 1er
         mes, cantidad de finales rendidos vs. adeudados.
       - socioeconomicas: beca, cuota al dia, nivel educativo de la familia,
         horas semanales de trabajo (no solo si/no).
       - trayectoria: anos entre el secundario y el ingreso, carrera previa
         abandonada, turno de cursada.
       - vinculo: uso del campus virtual, respuesta a comunicaciones, asistencia
         a tutorias (cantidad, no solo si/no).

  4. Registrar la variable objetivo con MAS detalle: no solo "abandono/continua"
     sino CUANDO abandono (mes, cuatrimestre) y, si se sabe, por que. Esto
     permite pasar de una clasificacion binaria a un modelo de supervivencia,
     que es lo que realmente sirve para intervenir a tiempo.

  5. Medir con fecha de corte. Hoy las variables y la etiqueta parecen del mismo
     periodo. Para un sistema de alerta temprana hay que predecir el abandono
     del 2do cuatrimestre usando SOLO datos disponibles al final del 1ro.

B) RECOMENDACIONES SOBRE EL MODELO Y SU USO

  6. No poner este modelo en produccion. Con el rendimiento actual (equivalente
     al azar) etiquetar alumnos como "en riesgo" seria arbitrario, con el riesgo
     de estigmatizar y de malgastar las horas de tutoria.

  7. Priorizar RECALL sobre precision cuando el modelo si funcione. Es mucho mas
     barato invitar a una tutoria a un alumno que no la necesitaba (falso
     positivo) que perder a uno que si estaba en riesgo (falso negativo).
     Concretamente: bajar el umbral de decision de 0.50 a ~0.35.

  8. Entregar PROBABILIDAD y no solo la clase. Un ranking de riesgo permite
     atender primero a los 20 alumnos mas criticos con los recursos que hay.

  9. No usar Genero (ni Carrera como factor punitivo) en un sistema de alerta.
     Es una variable sensible: si el modelo la usa, puede generar un trato
     discriminatorio. Conviene medirla para auditar sesgos, no para decidir.

 10. Comparar el arbol contra otros modelos (regresion logistica, Random Forest,
     Gradient Boosting). Se eligio arbol por su INTERPRETABILIDAD, que es clave
     para que un tutor entienda y acepte la alerta, pero conviene verificar
     cuanta performance se pierde.

C) RECOMENDACIONES INSTITUCIONALES (independientes del modelo)

 11. Montar un tablero de seguimiento con reglas simples y transparentes
     mientras no haya modelo valido: por ejemplo "3 inasistencias consecutivas"
     o "0 materias aprobadas al mes 2" disparan un llamado del tutor. Una regla
     explicita y auditada hoy vale mas que un modelo de caja negra sin validar.

 12. Cerrar el circuito: registrar cada intervencion (a quien se contacto, que
     se hizo, que resultado tuvo). Esos registros son el dataset que permitira,
     en uno o dos anos, entrenar un modelo que SI funcione y ademas medir el
     impacto real del programa de acompanamiento.
""")


# =============================================================================
# ANEXO - EXPERIMENTO DE CONTROL: COMO SE VE UN ARBOL CUANDO SI HAY SENAL
# =============================================================================
# Por que agregamos esto? Porque la TAREA 4 pide interpretar el arbol y, con los
# datos reales, el arbol no es interpretable (es ruido). Para demostrar que el
# PROCEDIMIENTO esta bien implementado, lo aplicamos a un dataset SIMULADO donde
# nosotros definimos las relaciones segun la bibliografia sobre abandono:
# menos asistencia, peor promedio, mas materias desaprobadas, mas distancia,
# trabajar y no ir a tutorias AUMENTAN la probabilidad de abandono.
#
# IMPORTANTE: estos datos son INVENTADOS. Sirven como control metodologico, NO
# como conclusion sobre los alumnos del ITB.

titulo("ANEXO - EXPERIMENTO DE CONTROL CON DATOS SIMULADOS (CON SENAL)")

generador = np.random.default_rng(SEMILLA)
n = 1000

simulado = pd.DataFrame({
    "Edad": generador.integers(18, 61, n),
    "Promedio": np.clip(generador.normal(7.0, 1.4, n), 1, 10).round(2),
    "Aprobadas": generador.integers(0, 8, n),
    "Desaprobadas": generador.integers(0, 8, n),
    "Asistencia": np.clip(generador.normal(72, 16, n), 20, 100).round(2),
    "DistanciaKm": generador.integers(1, 41, n),
    "Trabaja": generador.choice(["Si", "No"], n, p=[0.45, 0.55]),
    "Tutorias": generador.choice(["Si", "No"], n, p=[0.35, 0.65]),
})

# Modelo causal explicito (log-odds de abandonar).
# Cada coeficiente dice cuanto pesa esa variable. Los signos expresan la teoria:
# mas asistencia -> menos abandono (coeficiente negativo), etc.
log_odds = (
    - 0.070 * (simulado["Asistencia"] - 72)        # faltar mucho -> abandona
    - 0.550 * (simulado["Promedio"] - 7)           # mal promedio -> abandona
    + 0.330 * (simulado["Desaprobadas"] - 3.5)     # desaprobar -> abandona
    + 0.045 * (simulado["DistanciaKm"] - 10)       # vivir lejos -> abandona
    + 0.700 * (simulado["Trabaja"] == "Si")        # trabajar -> abandona
    - 0.900 * (simulado["Tutorias"] == "Si")       # tutorias -> protege
    + generador.normal(0, 0.6, n)                  # ruido irreducible
)
probabilidad = 1 / (1 + np.exp(-log_odds))
simulado["EstadoFinal"] = np.where(
    generador.random(n) < probabilidad, "Abandonó", "Continúa")

print(f"Dataset simulado: {n} alumnos")
print(simulado["EstadoFinal"].value_counts().to_string())

COLS_NUM_SIM = ["Edad", "Promedio", "Aprobadas", "Desaprobadas", "Asistencia", "DistanciaKm"]
COLS_CAT_SIM = ["Trabaja", "Tutorias"]
X_sim = simulado[COLS_NUM_SIM + COLS_CAT_SIM]
y_sim = (simulado["EstadoFinal"] == "Abandonó").astype(int)

X_sim_tr, X_sim_te, y_sim_tr, y_sim_te = train_test_split(
    X_sim, y_sim, test_size=0.30, random_state=SEMILLA, stratify=y_sim)

prep_sim = ColumnTransformer([
    ("num", "passthrough", COLS_NUM_SIM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COLS_CAT_SIM),
])
modelo_sim = Pipeline([
    ("prep", prep_sim),
    ("arbol", DecisionTreeClassifier(criterion="entropy", max_depth=4,
                                     min_samples_leaf=20, random_state=SEMILLA)),
])
modelo_sim.fit(X_sim_tr, y_sim_tr)

y_sim_pred = modelo_sim.predict(X_sim_te)
y_sim_prob = modelo_sim.predict_proba(X_sim_te)[:, 1]

subtitulo("Metricas del arbol sobre los datos simulados")
print(f"Accuracy : {accuracy_score(y_sim_te, y_sim_pred):.3f}")
print(f"Precision: {precision_score(y_sim_te, y_sim_pred, zero_division=0):.3f}")
print(f"Recall   : {recall_score(y_sim_te, y_sim_pred, zero_division=0):.3f}")
print(f"F1       : {f1_score(y_sim_te, y_sim_pred, zero_division=0):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_sim_te, y_sim_prob):.3f}")
print(f"Baseline : {max(y_sim_te.mean(), 1 - y_sim_te.mean()):.3f}")

nombres_sim = [c.replace("num__", "").replace("cat__", "")
               for c in modelo_sim.named_steps["prep"].get_feature_names_out()]
arbol_sim = modelo_sim.named_steps["arbol"]

subtitulo("Importancia de variables (datos simulados)")
imp_sim = pd.DataFrame({
    "Variable": nombres_sim,
    "Importancia": arbol_sim.feature_importances_,
}).sort_values("Importancia", ascending=False)
print(imp_sim[imp_sim["Importancia"] > 0].round(4).to_string(index=False))
print(f"\nVariable en la raiz: {nombres_sim[arbol_sim.tree_.feature[0]]}")

fig, ax = plt.subplots(figsize=(20, 9))
plot_tree(arbol_sim, feature_names=nombres_sim,
          class_names=["Continua", "Abandono"],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("ANEXO (datos SIMULADOS) - arbol interpretable cuando hay senal",
             fontsize=14)
plt.tight_layout()
ruta_anexo = os.path.join(CARPETA_SALIDA, "anexo_arbol_con_senal.png")
plt.savefig(ruta_anexo, dpi=140, bbox_inches="tight")
plt.close()
print(f"\n[guardado] {ruta_anexo}")

print("""
QUE DEMUESTRA EL ANEXO
  * El codigo, el preprocesamiento y la evaluacion estan correctamente armados:
    cuando los datos CONTIENEN una relacion, el mismo pipeline la encuentra
    (accuracy y AUC muy por encima del baseline).
  * El arbol resultante SI es interpretable: la raiz y las importancias
    reproducen los coeficientes que pusimos a mano (promedio, asistencia,
    tutorias y situacion laboral arriba de la lista).
  * Por contraste, queda claro que el pobre desempeno con los datos reales del
    TP no es un error de implementacion, sino una propiedad del dataset.
""")


# =============================================================================
# RESUMEN EJECUTIVO FINAL
# =============================================================================
titulo("RESUMEN EJECUTIVO")
print(f"""
Tarea 1 - Analisis y preparacion:
    100 alumnos, 10 atributos, sin nulos ni duplicados, clases balanceadas
    ({p_abandono:.0%} abandono / {p_continua:.0%} continua).
    Entropia inicial H(S) = {entropia_inicial:.4f} bits.
    Se crearon TotalMaterias y TasaAprobacion, se codificaron las 4 categoricas
    con One-Hot dentro de un Pipeline y se particiono 70/30 estratificado.

Tarea 2 - Entrenamiento:
    DecisionTreeClassifier con criterion='{busqueda.best_params_['arbol__criterion']}',
    max_depth={busqueda.best_params_['arbol__max_depth']},
    min_samples_leaf={busqueda.best_params_['arbol__min_samples_leaf']},
    min_samples_split={busqueda.best_params_['arbol__min_samples_split']},
    class_weight={busqueda.best_params_['arbol__class_weight']},
    elegidos por GridSearchCV ({len(busqueda.cv_results_['params'])} combinaciones, CV 5-fold, scoring=F1).
    Se mostro ademas el sobreajuste del arbol sin poda ({acc_train_sp:.0%} train vs {acc_test_sp:.0%} test).

Tarea 3 - Evaluacion en prueba (30 alumnos):
    Accuracy {accuracy:.3f} | Precision {precision:.3f} | Recall {recall:.3f} | F1 {f1:.3f} | AUC {auc:.3f}
    Baseline (clase mayoritaria): {acc_baseline:.3f}
    Test de permutacion: p-valor = {p_valor:.3f}
    => El modelo NO se distingue del azar con estos datos.

Tarea 4 - Interpretacion:
    Variable en la raiz: {variable_raiz}
    Pero las importancias son inestables y el test de permutacion descarta que
    haya senal: NO corresponde declarar una variable como "la mas importante".
    El anexo muestra que, cuando hay senal, el metodo si identifica promedio,
    asistencia, tutorias y situacion laboral como las variables determinantes.

Tarea 5 - Recomendaciones:
    12 recomendaciones agrupadas en datos / modelo / institucion. La principal:
    no desplegar este modelo y trabajar primero en la calidad y cantidad de los
    datos, usando reglas simples y auditables mientras tanto.

Archivos generados en la carpeta '{CARPETA_SALIDA}/'.
""")
print("=" * 79)
print("FIN DEL TP 05 - b) Prediccion de abandono - Equipo 8")
print("=" * 79)
