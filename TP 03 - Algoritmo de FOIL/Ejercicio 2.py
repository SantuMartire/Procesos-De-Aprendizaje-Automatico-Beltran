
# Ejercicio 2:
# a) Desarrollar un programa en Python que obtenga el cálculo del FOIL Gain para la condición
# nivel_educativo == 'terciario', con la siguiente salida:

import math

datos = [
    {"edad": 22, "departamento": "IT", "nivel_educativo": "terciario", "en_formacion": True},
    {"edad": 24, "departamento": "IT", "nivel_educativo": "universitario", "en_formacion": True},
    {"edad": 21, "departamento": "RRHH", "nivel_educativo": "terciario", "en_formacion": True},
    {"edad": 35, "departamento": "IT", "nivel_educativo": "universitario", "en_formacion": False},
    {"edad": 40, "departamento": "Finanzas", "nivel_educativo": "maestría", "en_formacion": False},
    {"edad": 29, "departamento": "RRHH", "nivel_educativo": "universitario", "en_formacion": False},
    {"edad": 23, "departamento": "IT", "nivel_educativo": "terciario", "en_formacion": True},
    {"edad": 38, "departamento": "Finanzas", "nivel_educativo": "universitario", "en_formacion": False}]

# Valores antes de aplicar la condición
P = sum(1 for d in datos if d["en_formacion"])      #4
N = sum(1 for d in datos if not d["en_formacion"])  #4

# Condición: nivel_educativo == 'terciario'
condicion = "nivel_educativo == 'terciario'"

# Aplicar condición: nivel_educativo == 'terciario
filtrados = [d for d in datos if d["nivel_educativo"] == "terciario"]

p = sum(1 for d in filtrados if d["en_formacion"])      #3
n = sum(1 for d in filtrados if not d["en_formacion"]) #0


# Cálculo FOIL Gain
def log2_safe(x):
    return math.log2(x) if x > 0 else float('-inf')

foil_gain = p * (log2_safe(p / (p + n)) - log2_safe(P / (P + N)))

# Mostrar resultados
print("Punto A Terminado:\n")

print(f"P = {P}, N = {N}")
print(f"p = {p}, n = {n}")
print(f"p / (p + n) = {p / (p + n):.3f}")
print(f"P / (P + N) = {P / (P + N):.3f}")
print(f"log2(p / (p + n)) = {log2_safe(p / (p + n)):.3f}")
print(f"log2(P / (P + N)) = {log2_safe(P / (P + N)):.3f}")
print(f"FOIL Gain = {foil_gain:.3f}\n\n")



#-----------------------------------------------------------------------------------------------
#b) Realizar el calculo de FOIL Gain manual para comprobar los cálculos


# Fórmula:
#   FOIL Gain = p * ( log2( p / (p + n) ) - log2( P / (P + N) ) )
#
# Conteos utilizados:
#   Antes de la condición (conjunto completo):
#     Positivos (en_formacion = True):  edad 22, 24, 21, 23  -> P = 4
#     Negativos (en_formacion = False): edad 35, 40, 29, 38  -> N = 4
#   Después de aplicar nivel_educativo == 'terciario':
#     edad 22, IT,   terciario, en_formacion = True   -> positivo
#     edad 21, RRHH, terciario, en_formacion = True   -> positivo
#     edad 23, IT,   terciario, en_formacion = True   -> positivo
#     Ningún ejemplo terciario tiene en_formacion = False
#     -> p = 3, n = 0
#
# 1) Fracción después de aplicar la condición:
#      p / (p + n) = 3 / (3 + 0) = 3 / 3 = 1.000
#
# 2) Fracción antes de aplicar la condición:
#      P / (P + N) = 4 / (4 + 4) = 4 / 8 = 0.500
#
# 3) Logaritmos:
#      log2(1.000) =  0.000
#      log2(0.500) = -1.000
#
# 4) FOIL Gain:
#      FOIL Gain = 3 * ( 0.000 - (-1.000) )
#                = 3 * ( 1.000 )
#                = 3.000
#
# 5) Interpretación:
#      La condición nivel_educativo == 'terciario' cubre 3 ejemplos positivos y
#      ningún negativo, por lo que su pureza pasa de 50% (antes) a 100% (después).
#      La ganancia de 1 bit por ejemplo cubierto, multiplicada por los 3 positivos,
#      da un FOIL Gain de 3.000: es una condición perfectamente discriminante y,
#      por lo tanto, una buena candidata para incorporar a la regla.
#
# Resultado manual: FOIL Gain = 3.000

#-----------------------------------------------------------------------------------------------

#c) Desarrollar un programa en Python que obtenga el cálculo del FOIL Gain para la condición edad ≤ 23, con la siguiente salida:

# Valores antes de aplicar la condición (los mismos de antes)
#P = sum(1 for d in datos if d["en_formacion"])      #4
#N = sum(1 for d in datos if not d["en_formacion"])  #4

# Aplicar condición: edad <= 23
filtrados_edad = [d for d in datos if d["edad"] <= 23]
p_edad = sum(1 for d in filtrados_edad if d["edad"] <= 23)
n_edad = sum(1 for d in filtrados_edad if not d["edad"] <= 23)

# Cálculo FOIL Gain

foil_gain_edad = p_edad * (log2_safe(p_edad / (p_edad + n_edad)) - log2_safe(P / (P + N)))

# Mostrar resultados
print("Punto C Terminado:\n")

print(f"P = {P}, N = {N}")
print(f"p = {p_edad}, n = {n_edad}")
print(f"p / (p + n) = {p_edad / (p_edad + n_edad):.3f}")
print(f"P / (P + N) = {P / (P + N):.3f}")
print(f"log2(p / (p + n)) = {log2_safe(p_edad / (p_edad + n_edad)):.3f}")
print(f"log2(P / (P + N)) = {log2_safe(P / (P + N)):.3f}")
print(f"FOIL Gain = {foil_gain_edad:.3f}\n\n")

#-----------------------------------------------------------------------------------------------
#d) Realizar el calculo de FOIL Gain manual para comprobar los cálculos
# Fórmula:
#   FOIL Gain = p * ( log2( p / (p + n) ) - log2( P / (P + N) ) )
#
# Conteos utilizados:
#   Antes de la condición (conjunto completo):
#     Positivos (en_formacion = True):  edad 22, 24, 21, 23  -> P = 4
#     Negativos (en_formacion = False): edad 35, 40, 29, 38  -> N = 4
#   Después de aplicar edad <= 23:
#     edad 22, IT,   terciario, en_formacion = True   -> positivo
#     edad 21, RRHH, terciario, en_formacion = True   -> positivo
#     edad 23, IT,   terciario, en_formacion = True   -> positivo
#     Ningún ejemplo terciario tiene en_formacion = False
#     -> p = 3, n = 0
#
# 1) Fracción después de aplicar la condición:
#      p / (p + n) = 3 / (3 + 0) = 3 / 3 = 1.000
#
# 2) Fracción antes de aplicar la condición:
#      P / (P + N) = 4 / (4 + 4) = 4 / 8 = 0.500
#
# 3) Logaritmos:
#      log2(1.000) =  0.000
#      log2(0.500) = -1.000
#
# 4) FOIL Gain:
#      FOIL Gain = 3 * ( 0.000 - (-1.000) )
#                = 3 * ( 1.000 )
#                = 3.000
#
# 5) Interpretación:
#      La condición edad <= 23 cubre 3 ejemplos positivos y
#      ningún negativo, por lo que su pureza pasa de 50% (antes) a 100% (después).
#      La ganancia de 1 bit por ejemplo cubierto, multiplicada por los 3 positivos,
#      da un FOIL Gain de 3.000: es una condición perfectamente discriminante y,
#      por lo tanto, una buena candidata para incorporar a la regla.
#
# Resultado manual: FOIL Gain = 3.000