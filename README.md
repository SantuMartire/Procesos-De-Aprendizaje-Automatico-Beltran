# Procesamiento de Aprendizaje Automático

Repositorio de apuntes, trabajos prácticos y código de la materia **Procesamiento de Aprendizaje Automático**, correspondiente al ciclo lectivo 2026 de la Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial.

**Institución:** Instituto Superior de Formación Técnica N° 197 — Avellaneda (Instituto Tecnológico Beltrán)
**Carga horaria:** 4 horas módulo semanales — 64 horas módulo anuales
**Modalidad de cursada:** 16 clases

---

## Sobre la materia

La materia introduce los fundamentos teóricos y prácticos del aprendizaje automático, la rama de la inteligencia artificial que permite a los sistemas aprender y mejorar de forma automática a partir de datos. A lo largo del año se recorren los principales algoritmos, las técnicas de procesamiento de datos, los métodos de evaluación de modelos y sus aplicaciones reales, con foco en diseñar e implementar soluciones de manera ética y eficiente.

### Objetivos

**Conceptuales**

- Comprender los fundamentos teóricos del aprendizaje automático y sus paradigmas: supervisado, no supervisado y por refuerzo.
- Identificar las etapas del procesamiento de datos dentro de un sistema de aprendizaje automático.

**Procedimentales**

- Diseñar y entrenar modelos utilizando Python y bibliotecas como Scikit-learn o TensorFlow.
- Evaluar el rendimiento de los modelos mediante métricas adecuadas y técnicas de validación cruzada.

**Actitudinales**

- Sostener una mirada crítica sobre el uso de la IA, considerando sus aspectos éticos y sociales.
- Trabajar de forma colaborativa en proyectos de desarrollo de modelos.

### Aptitudes a desarrollar

| Aptitud | Qué implica |
| --- | --- |
| Comprensión de los algoritmos | Entender cómo funcionan los modelos, desde los más simples como la regresión lineal hasta las redes neuronales profundas. |
| Análisis crítico | Evaluar qué técnica es la más adecuada según el tipo de dato y el problema a resolver. |
| Aplicación práctica | Diseñar, entrenar y evaluar modelos aplicables a salud, finanzas, educación y otros dominios. |
| Fundamento para la innovación | Conocer los procesos internos habilita a mejorar algoritmos existentes o crear nuevos. |
| Ética y responsabilidad | Identificar sesgos en el procesamiento de datos y promover un uso responsable de la IA. |

---

## Contenidos

### Módulo I — Aprendizaje Automático

**Unidad 1: Bloque Aprendizaje Automático**

- Inteligencia Artificial y sus áreas
- Aprendizaje automático (Machine Learning)
- Jerarquías entre Ciencia de la Computación, Ciencia de Datos, IA y Aprendizaje Automático
- Diferencias entre Aprendizaje Automático y Minería de Datos
- Categorías de ML
  - Aprendizaje supervisado: algoritmos de clasificación y de regresión
  - Aprendizaje no supervisado: clustering y reducción de dimensionalidad
  - Aprendizaje por refuerzo: enfoques *value-based*, *policy-based*, *model-based* y variantes modernas
- Aprendizaje inductivo y aprendizaje deductivo

### Módulo II — Aprendizaje Supervisado

**Unidad 2: Bloque Aprendizaje Supervisado**

- Repaso de las categorías de ML
- Algoritmos del aprendizaje supervisado: clasificación y regresión
- Clasificadores
  - Tipos de clasificadores supervisados
  - Importancia y criterios de elección del clasificador
- Aprendizaje de conceptos y de reglas
- Espacio de versiones: componentes clave y aplicación
- Algoritmo AQ y su relación con el espacio de versiones
- Programación lógica inductiva
- Algoritmo FOIL y ganancia de información en FOIL
- Árboles de decisión
  - Pasos del algoritmo
  - Métricas para su construcción: entropía, índice de Gini, ganancia de información
- Aprendizaje basado en instancias
- k-Nearest Neighbors
- Clasificador bayesiano

### Módulo III — Aprendizaje No Supervisado

**Unidad 3: Bloque Aprendizaje No Supervisado**

- Elaboración y comparación de modelos
- Características del aprendizaje no supervisado
- K-Means y método del codo
- Funciones y clases de Scikit-learn aplicadas a K-Means
- Funciones y clases de Matplotlib
- Aprendizaje por refuerzo: algoritmos populares, ventajas y desventajas
- Pasos del algoritmo Q-Learning

---

## Cronograma

| Clase | Tema |
| :---: | --- |
| 1–2 | Unidad 1 |
| 3–8 | Unidad 2 |
| 9–10 | Unidad 3 |
| 11 | Entrega de TP para la promoción |
| 12 | Parcial |
| 13 | Devolución y entrega de notas. Consultas para el recuperatorio |
| 14 | Recuperatorio |
| 15 | Devolución del recuperatorio |
| 16 | Firma de libretas — Modelo de final |

---

## Evaluación

La evaluación se realiza mediante trabajos prácticos, producciones individuales o grupales y una instancia parcial con su correspondiente recuperatorio.

La unidad curricular puede aprobarse por **promoción directa** o mediante **examen final**, según las condiciones de asistencia, calificaciones, regularidad, recuperatorios y promedios establecidas en el Régimen Académico Institucional (RAI) vigente.

Para acceder a la promoción directa hay que alcanzar los requisitos de asistencia, aprobar las actividades y cumplir con el promedio mínimo exigido. Quienes no promocionen pero mantengan la regularidad pueden rendir el examen final. Las situaciones de ausencia, aplazo, recuperación, pérdida de regularidad y recursada se resuelven conforme al RAI.

---

## Herramientas

- Python
- Scikit-learn
- TensorFlow
- Matplotlib
- Jupyter Notebook

---

## Estructura del repositorio

```
.
├── unidad-1/          Fundamentos de IA y categorías de ML
├── unidad-2/          Aprendizaje supervisado
├── unidad-3/          Aprendizaje no supervisado y por refuerzo
├── trabajos-practicos/
├── apuntes/
├── datasets/
└── README.md
```

---

## Bibliografía

- Godoy Amado, M. (s. f.). *Algunos clasificadores bayesianos* [Monografía de trabajo de grado para optar por el título de Matemático, Proyecto Curricular de Matemáticas].
- López Briega, R. E. (s. f.). *Machine Learning*. Comunidad Argentina de Inteligencia Artificial (IAAR). https://iaarbook.github.io/machine-learning/
- Mining, E. (2019). *Machine Learning for Beginners: A Complete and Phased Beginner's Guide to Learning and Understanding Machine Learning and Artificial Intelligence*.
- Morales, E., & Escalante, H. J. (s. f.). *Programación lógica inductiva*. Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE).
- Theobald, O. (2017). *Machine Learning for Absolute Beginners: A Plain English Introduction* (2.ª ed.).
