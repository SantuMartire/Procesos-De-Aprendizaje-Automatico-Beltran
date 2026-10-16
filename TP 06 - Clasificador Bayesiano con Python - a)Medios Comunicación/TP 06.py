from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

x =[ #Noticias
    "El presidente anunció una nueva reforma educativa",
    "Descubren que la vacuna convierte a las personas en robots",
    "La NASA confirma el hallazgo de agua en Marte",
    "Científicos afirman que la Tierra es plana",
    "El ministerio de salud lanza campaña contra el dengue",
    "Celebridades usan crema milagrosa para rejuvenecer 30 años",
    "Se inaugura el nuevo hospital en la ciudad",
    "Estudio revela que comer chocolate cura el cáncer",
    "Gobierno aprueba ley de protección ambiental",
    "Investigadores aseguran que los teléfonos espían nuestros sueños"]

y = ['real','fake','real', 'fake','real','fake','real','fake','real','fake'] #Etiquetas

# Paso 1: Convertir los textos en vectores numéricos (conteo de palabras).
# GaussianNB no puede trabajar con texto directamente, por eso usamos
# CountVectorizer + MultinomialNB, que es el Naive Bayes pensado para
# conteos de palabras (equivalente a lo hecho a mano en 09 - codigo_BAYES.py,
# con suavizado de Laplace alpha=1).
vectorizador = CountVectorizer(lowercase=True)
X = vectorizador.fit_transform(x)

# Paso 2: Separar en entrenamiento y prueba para evaluar el modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# Paso 3: Evaluar el modelo
y_pred = model.predict(X_test)

print("=== RESULTADOS DEL MODELO ===")
print("Precisión total (accuracy):", round(accuracy_score(y_test, y_pred), 4))

clases = model.classes_
matriz = confusion_matrix(y_test, y_pred, labels=clases)
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print(pd.DataFrame(matriz, index=clases, columns=clases))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, labels=clases, zero_division=0))

# Paso 4: Reentrenar con todos los datos para clasificar las nuevas noticias
model.fit(X, y)

# Nuevas noticias para predecir
nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

X_nuevas = vectorizador.transform(nuevas_noticias)
predicciones = model.predict(X_nuevas)
probabilidades = model.predict_proba(X_nuevas)

print("=== CLASIFICACIÓN DE NUEVAS NOTICIAS ===")
for i in range(len(nuevas_noticias)):
    print(f'\n{i + 1}. "{nuevas_noticias[i]}" -> {predicciones[i]}')
    for j in range(len(model.classes_)):
        print("   Clase:", model.classes_[j], "- Probabilidad:", round(probabilidades[i][j], 4))
