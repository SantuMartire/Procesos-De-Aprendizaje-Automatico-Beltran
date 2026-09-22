#Ejercicio Auto electrico:

# Paso 1: Ejemplos positivos
positivos = [
{"edad": "Joven", "ingreso": "Alto", "tiene_garaje": "Si", "distancia_trabajo": "Corta" },
{ "edad": "Adulto", "ingreso": "Medio", "tiene_garaje": "Si", "distancia_trabajo": "Media"},
{"edad": "Mayor", "ingreso": "Alto", "tiene_garaje": "Si", "distancia_trabajo": "Larga"}]

# Paso 2: Ejemplos negativos
negativos = [
{"edad": "Joven", "ingreso": "Alto", "tiene_garaje": "No", "distancia_trabajo": "Corta" },
{"edad": "Adulto", "ingreso": "Medio", "tiene_garaje": "No", "distancia_trabajo": "Media"},
{"edad": "Mayor", "ingreso": "Bajo", "tiene_garaje": "No", "distancia_trabajo": "Larga"}]

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
    valores_positivos = []
    valores_negativos = []

    # Extraer valores positivos
    for ej in positivos:
        valor = ej[atributo]
        #print("++valor: ", valor)
        if valor not in valores_positivos:
            valores_positivos.append(valor)

    # Extraer valores negativos
    for ej in negativos:
        valor = ej[atributo]
        if valor not in valores_negativos:
            valores_negativos.append(valor)

    # Comparar y guardar los valores que están en positivos pero no en negativos
    valores_validos = []
    for valor in valores_positivos:
        encontrado = False
        for v in valores_negativos:
            if valor == v:
                encontrado = True
                break
        if not encontrado:
            valores_validos.append(valor)

    # Si hay valores válidos, los agregamos a la regla
    if len(valores_validos) > 0:
        regla[atributo] = valores_validos

# Paso 4: Mostrar la regla inducida
print("Regla inducida para identificar a un comprador:")
for atributo in regla:
    print("-", atributo, "debe ser:", regla[atributo])
