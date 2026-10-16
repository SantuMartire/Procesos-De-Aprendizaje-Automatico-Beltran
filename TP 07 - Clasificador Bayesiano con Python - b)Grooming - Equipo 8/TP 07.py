from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

x =[ #Mensajes de Grooming
    "¿Cuántos años tenés? No le digas a nadie que hablamos.",
    "Hola, ¿cómo estás? ¿Querés jugar algo online?",
    "Sos muy linda, no le muestres esto a tus papás. ",
    "¿Terminaste la tarea de matemáticas?",
    "Podés mandarme una foto tuya, pero que sea solo para mí.",
    "¿Querés que te ayude con el juego que no podías pasar?",
    "No le contés a nadie que hablamos por acá, ¿sí? ",
    "¿Te gustó la película de ayer? ",
    "¿Estás solo en casa ahora?",
    "¿Querés hacer una videollamada para estudiar juntos?",
    "No le digas a tus padres que te escribí, es nuestro secreto?",
    "¿Cómo te fue en el examen de historia? ",
    "Mandame una foto tuya, pero que nadie más la vea.",
    "¿Jugamos Minecraft esta tarde?",
    "Sos muy especial para mí, no le cuentes a nadie lo que hablamos. ",
    "¿Querés que te pase los apuntes de biología?",
    "¿Podés mostrarme cómo estás vestida ahora?",
    "¿Tenés ganas de salir a andar en bici mañana?",
    "No hace falta que le digas a nadie que hablamos tanto.",
    "¿Querés que estudiemos juntos para el parcial?"   
    ]

y = ['grooming','No grooming','grooming','No grooming','grooming','No grooming','grooming','No grooming','grooming','No grooming','grooming','No grooming','grooming','No grooming','grooming','No grooming','grooming','No grooming','grooming','No grooming',] #Etiquetas


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
nuevos_mensajes = [
    "¿Podés mandarme una foto tuya? No se la muestres a nadie.", #grooming
    "¿Jugamos Minecraft esta tarde?", #No grooming
    "No le digas a tus papás que hablamos, ¿sí?", #grooming
    "¿Terminaste el trabajo práctico de historia?" #No grooming
]

X_nuevas = vectorizador.transform(nuevos_mensajes)
predicciones = model.predict(X_nuevas)
probabilidades = model.predict_proba(X_nuevas)

print("=== CLASIFICACIÓN DE NUEVAS NOTICIAS ===")
for i in range(len(nuevos_mensajes)):
    print(f'\n{i + 1}. "{nuevos_mensajes[i]}" -> {predicciones[i]}')
    for j in range(len(model.classes_)):
        print("   Clase:", model.classes_[j], "- Probabilidad:", round(probabilidades[i][j], 4))
