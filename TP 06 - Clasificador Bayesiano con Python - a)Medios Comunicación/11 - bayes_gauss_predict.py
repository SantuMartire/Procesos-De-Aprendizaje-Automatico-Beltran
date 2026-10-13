from sklearn.naive_bayes import GaussianNB

x= [[170,65],[180,80], [160,50]] #altura y peso
y= ['hombre', 'hombre', 'mujer']

model = GaussianNB()
model.fit(x,y)

# Nuevo punto para predecir
nuevo_punto = [[165, 60]]

# Predicción
prediccion = model.predict(nuevo_punto)
print(f"Predicción para el punto {nuevo_punto}: {prediccion[0]}")

# Probabilidades por clase
#La función predict_proba() en scikit-learn se usa para obtener 
#las probabilidades de pertenencia a cada clase para un punto dado, 
#en lugar de solo la clase más probable.
probabilidades = model.predict_proba(nuevo_punto)
print("Probabilidades por clase:")

# Obtener las clases aprendidas por el modelo
clases = model.classes_

# Obtener las probabilidades para el nuevo punto
probs = probabilidades[0]

# Recorrer cada clase y su probabilidad
for i in range(len(clases)):
    clase = clases[i]
    probabilidad = probs[i]
    print("Clase:", clase, "- Probabilidad:", round(probabilidad, 4))
