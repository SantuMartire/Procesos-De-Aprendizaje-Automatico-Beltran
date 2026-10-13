import math
""" 
Una empresa de telecomunicaciones quiere predecir si un cliente aceptará una oferta de plan de datos 
móviles. Para ello, se dispone de un conjunto de datos con información de 10 clientes, incluyendo los 
siguientes atributos: 
* Nivel de uso mensual de datos (en GB) 
* Edad (en años) 
* Tiene línea fija (Sí / No) 
* Aceptó la oferta (Sí / No)
 """

# Datos del conjunto original
total = 10
aceptaron = 5
no_aceptaron = 5

#Probabilidades
p_si = aceptaron / total
p_no = no_aceptaron / total

#---------------------------------------------------------------------------------------------

#1. Calcular la entropía del conjunto original.
# Cálculo de entropía       
entropia_original = - (p_si * math.log2(p_si) + p_no * math.log2(p_no))
print("\n\nEntropía del conjunto original: ", entropia_original)


#---------------------------------------------------------------------------------------------


#Evaluar la ganancia de información para los atributos: 
#   1.Edad (agrupada en rangos: Joven ≤30, Adulto 31–50, Mayor >50) 
#   2.Tiene línea fija 
#   3.Uso de datos (agrupado: Bajo ≤3GB, Medio 3.1–6GB, Alto >6GB) 

#SUBCONJUNTO EDAD
# Subconjunto Joven (≤30): IDs 1, 3, 8
total_joven = 3
si_joven = 0
no_joven = 3
p_joven = si_joven / total_joven 
p_no_joven = no_joven / total_joven 
H_joven =  0 if si_joven == 0 or no_joven == 0 else - (p_joven * math.log2(p_joven) + p_no_joven * math.log2(p_no_joven))
print("\n\nEntropía Joven: ", H_joven)

# Subconjunto Adulto (31-50): IDs 2, 4, 6, 7, 9, 10
total_adulto = 6
si_adulto = 4
no_adulto = 2
p_adulto = si_adulto / total_adulto
p_no_adulto = no_adulto / total_adulto
H_adulto = (p_adulto * math.log2(p_adulto) + p_no_adulto * math.log2(p_no_adulto))
print("\n\nEntropía Adulto: ", H_adulto)

# Subconjunto Mayor (>50): ID 5
total_mayor = 1
si_mayor = 1
no_mayor = 0
p_mayor = si_mayor / total_mayor
p_no_mayor = no_mayor / total_mayor
H_mayor = 0 if si_mayor == 0 or no_mayor == 0 else - (p_mayor * math.log2(p_mayor) + p_no_mayor * math.log2(p_no_mayor))
print("\n\nEntropía Mayor: ", H_mayor)

# Entropía ponderada por edad
peso_joven = total_joven / total
peso_adulto = total_adulto / total
peso_mayor = total_mayor / total
entropia_ponderada_edad = (peso_joven * H_joven) + (peso_adulto * H_adulto) + (peso_mayor * H_mayor)
print("\n\nEntropía ponderada por edad: ", entropia_ponderada_edad)

# Ganancia de información por edad
ganancia_edad = entropia_original - entropia_ponderada_edad
print("\n\nGanancia de información por edad: ", ganancia_edad)




#SUBCONJUNTO LINEA FIJA
# Subconjunto con línea fija: IDs 2, 4, 5, 7, 9
total_con_linea = 5
si_con_linea = 5
no_con_linea = 0
p_con_linea = si_con_linea / total_con_linea
p_no_con_linea = no_con_linea / total_con_linea
H_con_linea = 0 if si_con_linea == 0 or no_con_linea == 0 else - (p_con_linea * math.log2(p_con_linea) + p_no_con_linea * math.log2(p_no_con_linea))
print("\n\nEntropía con línea fija: ", H_con_linea)

# Subconjunto sin línea fija: IDs 1, 3, 6, 8, 10
total_sin_linea = 5
si_sin_linea = 0
no_sin_linea = 5
p_sin_linea = si_sin_linea / total_sin_linea
p_no_sin_linea = no_sin_linea / total_sin_linea
H_sin_linea = 0 if si_sin_linea == 0 or no_sin_linea == 0 else - (p_sin_linea * math.log2(p_sin_linea) + p_no_sin_linea * math.log2(p_no_sin_linea))
print("\n\nEntropía sin línea fija: ", H_sin_linea)

# Entropía ponderada por línea fija
peso_con_linea = total_con_linea / total
peso_sin_linea = total_sin_linea / total
entropia_ponderada_linea_fija = (peso_con_linea * H_con_linea) + (peso_sin_linea * H_sin_linea)
print("\n\nEntropía ponderada por línea fija: ", entropia_ponderada_linea_fija)

# Ganancia de información por línea fija
ganancia_linea_fija = entropia_original - entropia_ponderada_linea_fija



#SUBCONJUNTO USO DE DATOS

# Subconjunto Bajo (≤3GB): IDs 1, 3, 8
total_bajo = 3
si_bajo = 0
no_bajo = 3
p_bajo = si_bajo / total_bajo
p_no_bajo = no_bajo / total_bajo
H_bajo = 0 if si_bajo == 0 or no_bajo == 0 else - (p_bajo * math.log2(p_bajo) + p_no_bajo * math.log2(p_no_bajo))
print("\n\nEntropía Bajo: ", H_bajo)

# Subconjunto Medio (3.1–6GB): IDs 2, 6, 7, 10
total_medio = 4
si_medio = 2
no_medio = 2
p_medio = si_medio / total_medio
p_no_medio = no_medio / total_medio
H_medio = 0 if si_medio == 0 or no_medio == 0 else - (p_medio * math.log2(p_medio) + p_no_medio * math.log2(p_no_medio))
print("\n\nEntropía Medio: ", H_medio)


# Subconjunto Alto (>6GB): ID 4, 5, 9
total_alto = 3
si_alto = 3
no_alto = 0
p_alto = si_alto / total_alto
p_no_alto = no_alto / total_alto
H_alto = 0 if si_alto == 0 or no_alto == 0 else - (p_alto * math.log2(p_alto) + p_no_alto * math.log2(p_no_alto))
print("\n\nEntropía Alto: ", H_alto)

# Entropía ponderada por uso de datos
peso_bajo = total_bajo / total
peso_medio = total_medio / total
peso_alto = total_alto / total
entropia_ponderada_uso_datos = (peso_bajo * H_bajo) + (peso_medio * H_medio) + (peso_alto * H_alto)
print("\n\nEntropía ponderada por uso de datos: ", entropia_ponderada_uso_datos)

# Ganancia de información por uso de datos
ganancia_uso_datos = entropia_original - entropia_ponderada_uso_datos
print("\n\nGanancia de información por uso de datos: ", ganancia_uso_datos)


#-------------------------------------------------------------------------

# Comparación de las ganancias de información para los tres atributos
print("\n\nGanancia de información por edad: ", ganancia_edad)
print("Ganancia de información por línea fija: ", ganancia_linea_fija)
print("Ganancia de información por uso de datos: ", ganancia_uso_datos)

mejor_atributo = max(ganancia_edad, ganancia_linea_fija, ganancia_uso_datos)

if mejor_atributo == ganancia_edad:
    print("\n\nEl atributo con mayor ganancia de información es la Edad.")
elif mejor_atributo == ganancia_linea_fija:
    print("\n\nEl atributo con mayor ganancia de información es la Línea Fija.")
else:
    print("\n\nEl atributo con mayor ganancia de información es el Uso de Datos.")

print("\n\nEl atributo con mayor ganancia de información es el que mejor separa los datos y, por lo tanto, es el más relevante para predecir si un cliente aceptará la oferta de plan de datos móviles.")
