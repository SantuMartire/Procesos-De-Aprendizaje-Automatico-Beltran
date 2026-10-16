from collections import defaultdict

# Datos de entrenamiento
correos = [
    {"palabras": ["gratis", "oferta"], "clase": "Spam"},
    {"palabras": ["gratis", "urgente"], "clase": "Spam"},
    {"palabras": ["reunión", "proyecto"], "clase": "No Spam"},
    {"palabras": ["proyecto", "entrega"], "clase": "No Spam"},
    {"palabras": ["oferta", "urgente"], "clase": "Spam"},
    {"palabras": ["reunión", "entrega"], "clase": "No Spam"},
]

# Nuevo correo a clasificar
nuevo_correo = ["gratis", "proyecto"]

# Paso 1: Calcular probabilidades previas
#defaultdict es un tipo de dato que pertenece al módulo de colecciones. 
# Es una subclase de diccionario que se utiliza para devolver un objeto
# similar a un diccionario

conteo_clases = defaultdict(int)
for correo in correos:
    conteo_clases[correo["clase"]] += 1

total_correos = len(correos)
P_clase = {clase: conteo / total_correos for clase, conteo in conteo_clases.items()}

# Paso 2: Conteo de palabras por clase
conteo_palabras = defaultdict(lambda: defaultdict(int))
total_palabras_por_clase = defaultdict(int)
vocabulario = set()

for correo in correos:
    clase = correo["clase"]
    for palabra in correo["palabras"]:
        conteo_palabras[clase][palabra] += 1
        total_palabras_por_clase[clase] += 1
        vocabulario.add(palabra)

# Paso 3: Calcular probabilidades con suavizado de Laplace
def probabilidad_palabra_dada_clase(palabra, clase):
    return (conteo_palabras[clase][palabra] + 1) / (total_palabras_por_clase[clase] + len(vocabulario))

# Paso 4: Calcular probabilidad total para cada clase
probabilidades = {}
for clase in P_clase:
    prob = P_clase[clase]
    for palabra in nuevo_correo:
        prob *= probabilidad_palabra_dada_clase(palabra, clase)
    probabilidades[clase] = prob

# Mostrar resultados
print("Probabilidades calculadas para el nuevo correo:")
for clase, prob in probabilidades.items():
    print(f"{clase}: {prob:.4f}")

# Clasificación final
clase_predicha = max(probabilidades, key=probabilidades.get)
print(f"\nEl nuevo correo se clasifica como: {clase_predicha}")

