def clasificar(accion, duracion):
    if accion == "Combate":
        if duracion <= 120:
            prediccion = "Derrota"
        else:
            prediccion = "Victoria"

    elif accion == "Exploracion":
        if duracion <= 300:
            prediccion = "Sin Hallazgos"
        else:
            prediccion = "Descubrimiento"                
    elif accion == "Interaccion social":
        prediccion = "Mensaje enviado"
    else:
        prediccion = "No clasificado"

    return prediccion

prediccion = clasificar("Combate", 7)
print(prediccion)