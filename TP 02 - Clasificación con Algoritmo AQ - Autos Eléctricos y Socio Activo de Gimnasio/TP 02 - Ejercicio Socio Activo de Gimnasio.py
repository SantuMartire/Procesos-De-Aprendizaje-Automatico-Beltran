# Paso 1: Ejemplos positivos
positivos = [
{"edad": 25,"frec_asist": "frecuente","plan": "premium"},
{"edad": 32,"frec_asist": "frecuente" ,"plan": "premium"},
{"edad": 45,"frec_asist": "frecuente","plan": "premium"}
]

# Paso 2: Ejemplos negativos
negativos = [
{"edad": 25,"frec_asist": "ocasional","plan": "basico"},
{"edad": 32,"frec_asist": "ocasional","plan": "estandar"},
{"edad": 38,"frec_asist": "rara","plan": "basico"}
]

# Paso 3: Inducción de reglas 
regla = {}

# Obtener los nombres de los atributos
atributos = []
ejemplo = positivos[0]
for clave in ejemplo:
    print("Clave: ", clave)
    atributos.append(clave)

# Para cada atributo, comparar valores únicos en positivos y negativos
for atributo in atributos:
    valores_pos = []
    valores_neg = []

    # Extraer valores de positivos
    for ej in positivos:
        valor = ej[atributo]
        #print("++valor: ", valor)
        if valor not in valores_pos:
            valores_pos.append(valor)

    # Extraer valores de negativos
    for ej in negativos:
        valor = ej[atributo]
        if valor not in valores_neg:
            valores_neg.append(valor)

    # Comparar y guardar los valores que están en positivos pero no en negativos
    valores_validos = []
    for valor in valores_pos:
        encontrado = False
        for v in valores_neg:
            if valor == v:
                encontrado = True
                break
        if not encontrado:
            valores_validos.append(valor)

    # Si hay valores válidos, los agregamos a la regla
    if len(valores_validos) > 0:
        regla[atributo] = valores_validos

# Paso 4: Mostrar la regla inducida final
print("Regla inducida para identificar a un socio activo:")
for atributo in regla:
    print("-", atributo, "debe ser:", regla[atributo])
