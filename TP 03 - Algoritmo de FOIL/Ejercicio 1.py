datos = [
    {"edad": 22, "departamento": "IT", "nivel_educativo": "terciario", "en_formacion": True},
    {"edad": 24, "departamento": "IT", "nivel_educativo": "universitario", "en_formacion": True},
    {"edad": 21, "departamento": "RRHH", "nivel_educativo": "terciario", "en_formacion": True},
    {"edad": 35, "departamento": "IT", "nivel_educativo": "universitario", "en_formacion": False},
    {"edad": 40, "departamento": "Finanzas", "nivel_educativo": "maestría", "en_formacion": False},
    {"edad": 29, "departamento": "RRHH", "nivel_educativo": "universitario", "en_formacion": False},
    {"edad": 23, "departamento": "IT", "nivel_educativo": "terciario", "en_formacion": True},
    {"edad": 38, "departamento": "Finanzas", "nivel_educativo": "universitario", "en_formacion": False}]

# Separar ejemplos positivos y negativos
positivos = [p for p in datos if p["en_formacion"]]
negativos = [p for p in datos if not p["en_formacion"]]

#Para cada atributo, identificar los valores que aparecen en los positivos pero no en los negativos.

def inducir_regla(positivos, negativos):
    atributos = ["edad", "departamento", "nivel_educativo"]
    regla = {}

    for atributo in atributos:
        valores_pos = set(p[atributo] for p in positivos)
        valores_neg = set(p[atributo] for p in negativos)
 
        # Para edad, usamos valores numéricos, así que buscamos intersección mínima
        if atributo == "edad":
            valores_validos = [v for v in valores_pos if v not in valores_neg]
        else:
             valores_validos = list(valores_pos - valores_neg)

        if valores_validos:
            regla[atributo] = valores_validos

    return regla

# Ejecutar el algoritmo

regla_inducida = inducir_regla(positivos, negativos)

# Mostrar la regla

print("Regla inducida para identificar si esta en formación:")
for atributo, valores in regla_inducida.items():
    print(f"- {atributo} debe ser uno de: {valores}")