import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os

# Creamos el notebook
nb = nbf.v4.new_notebook()
cells = []

# ==========================================
# PORTADA Y DECLARACIÓN DE USO DE IA
# ==========================================
cells.append(nbf.v4.new_markdown_cell(r"""# Tarea Nº 1: Inteligencia Artificial
### Universidad Diego Portales - Facultad de Ingeniería y Ciencias
**Escuela de Informática y Telecomunicaciones**  
**Profesores:** Johan Fuentes, Víctor Reyes, Felipe Lazo  
**Fecha de Entrega:** Septiembre 2026  

---

### Integrantes del Grupo:
1. **Ignacio Antiguay**
2. **Vicente Hernández**
3. **Maximiliano Palma**

---

### Declaración sobre el Uso de Herramientas Generativas
En cumplimiento con las directrices académicas y éticas del curso, declaramos que se utilizaron herramientas generativas de inteligencia artificial (Gemini y Claude) exclusivamente como apoyo técnico para la asistencia en la escritura y depuración de código en Python, revisión de sintaxis de librerías especializadas (`pgmpy`, `hmmlearn`), organización metodológica y formateo del documento. 

El diseño de las soluciones, la arquitectura de las redes probabilísticas, la justificación causal de dependencias e independencias, el modelado de las cadenas de Markov y HMM, la parametrización matemática de Monte Carlo y el análisis crítico e interpretación de todos los resultados fueron concebidos, elaborados y validados íntegramente por los estudiantes autores de este trabajo."""))

# ==========================================
# PREGUNTA 1: REDES BAYESIANAS
# ==========================================
cells.append(nbf.v4.new_markdown_cell(r"""## Pregunta 1: Redes Bayesianas (30 Puntos)

### 1.1 Selección del Dataset y Preparación de Datos
Para esta primera sección seleccionamos el dataset **Spotify Tracks Dataset** disponible en Kaggle. El conjunto original contiene 114.000 canciones y 22 columnas de metadatos y características acústicas.

Para cumplir con el requisito de al menos 8.000 filas y 12 columnas útiles, realizamos una reducción muestral aleatoria a **12.000 filas**, fijando una semilla (`random_state=42`) para asegurar la reproducibilidad de todos los experimentos. Seleccionamos 12 características clave:
- Variable objetivo: `popularity` (popularidad comercial del tema).
- Características acústicas y de producción: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`.
- Características estructurales y de contenido: `explicit` (contenido explícito), `mode` (tonalidad mayor/menor) y `key` (tono musical)."""))

cells.append(nbf.v4.new_code_cell(r"""import pandas as pd
import numpy as np
import kagglehub
import os
import glob
import warnings

# Suprimir advertencias para una salida limpia
warnings.filterwarnings("ignore")

# 1. Descarga del dataset desde Kaggle
print("Descargando dataset Spotify Tracks desde Kaggle...")
path = kagglehub.dataset_download("yashdev01/spotify-tracks-dataset")

# 2. Localizar el archivo CSV descargado
archivos_csv = glob.glob(os.path.join(path, "*.csv"))
if not archivos_csv:
    raise FileNotFoundError("No se encontró el archivo CSV en el dataset descargado.")
ruta_csv = archivos_csv[0]

# 3. Cargar y muestrear a 12.000 filas y 12 columnas representativas
print(f"Cargando datos desde: {ruta_csv}")
df_raw = pd.read_csv(ruta_csv)
df_muestra = df_raw.sample(n=12000, random_state=42).copy()

columnas = [
    'popularity', 'explicit', 'danceability', 'energy', 
    'key', 'loudness', 'mode', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence'
]
df_spotify = df_muestra[columnas].copy()

# Guardar localmente
df_spotify.to_csv('spotify-tracks-dataset.csv', index=False)
print(f"Dataset Spotify preparado exitosamente con {df_spotify.shape[0]} filas y {df_spotify.shape[1]} columnas.")"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 1.2 Discretización de Variables
Las Redes Bayesianas Discretas (`DiscreteBayesianNetwork`) requieren que las variables sean categóricas para construir sus Tablas de Probabilidad Condicional (CPDs). Las variables numéricas continuas del dataset fueron discretizadas bajo los siguientes criterios metodológicos:
1. **Variables de 3 niveles (bajo, medio, alto)**: Se aplicó partición por cuantiles (`pd.qcut`, $q=3$) a las variables continuas con distribuciones graduadas (`popularity`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `liveness`, `valence`) para asegurar un balance equitativo de muestras en cada categoría.
2. **Variable instrumentalness**: Se discretizó en 2 niveles (`bajo`, `alto`) debido a que la inmensa mayoría de temas comerciales tienen valores cercanos a 0, concentrando las pistas puramente instrumentales en la cola superior.
3. **Variables binarias y categóricas**: `explicit` se mapeó a `si`/`no`, `mode` a `mayor`/`menor`, y `key` se convirtió en variable categórica de tonalidad musical."""))

cells.append(nbf.v4.new_code_cell(r"""from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import TreeSearch, BIC
from pgmpy.inference import VariableElimination

datos = pd.read_csv("spotify-tracks-dataset.csv")

# Discretizacion en 3 cuantiles equilibrados
datos['popularity'] = pd.qcut(datos['popularity'], 3, labels=['baja', 'media', 'alta'])
tres_niveles = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence']
for col in tres_niveles:
    datos[col] = pd.qcut(datos[col], 3, labels=['bajo', 'medio', 'alto'])

# Discretizacion binaria y categorica
datos['instrumentalness'] = pd.qcut(datos['instrumentalness'], 2, duplicates='drop', labels=['bajo', 'alto'])
datos['explicit'] = datos['explicit'].map({True: 'si', False: 'no'})
datos['mode'] = datos['mode'].map({1: 'mayor', 0: 'menor'})
datos['key'] = datos['key'].astype(str)
datos = datos.astype(str)

datos.to_csv("spotify-tracks-dataset-discretizado.csv", index=False)
print("Dataset discretizado guardado exitosamente.")"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 1.3 Diseño de la Red Bayesiana Propuesta (Manual) y Justificación Teórica
Para construir la estructura inicial de la red de manera manual (sin ningún algoritmo de aprendizaje automático), nos basamos en relaciones lógicas de causa-efecto provenientes de la producción musical, la teoría musical y la percepción del oyente:

#### Justificación de Relaciones de Dependencia Directa:
1. `acousticness -> energy` y `loudness -> energy`: La energía percibida en una canción depende fuertemente de sus atributos de producción sonora; canciones con mayor volumen promedio (loudness) tienden a sentirse más enérgicas, mientras que los instrumentos acústicos no amplificados transmiten mayor calma e intimidad.
2. `energy -> valence` y `mode -> valence`: El estado de ánimo (valence, positividad musical) está condicionado por la intensidad energética del tema y por la escala empleada (los modos mayores suelen asociarse a emociones alegres, y los menores a melancolía).
3. `energy -> danceability`: La bailabilidad de una canción requiere un piso mínimo de propulsión rítmica y energía.
4. Factores que determinan directamente la variable objetivo (`popularity`):
   - `danceability -> popularity`: Las canciones más bailables suelen tener mayor facilidad para viralizarse en clubes y listas de reproducción.
   - `energy -> popularity`: La música contemporánea de mayor difusión suele poseer niveles elevados de energía.
   - `explicit -> popularity`: Las letras explícitas son comunes en géneros líderes del streaming (como el trap y reggaeton).
   - `instrumentalness -> popularity`: La presencia o ausencia de voz humana es determinante en el consumo masivo.
   - `speechiness -> popularity`: La proporción de palabras habladas incide directamente en si la pista es percibida como canción comercial o como contenido de nicho.

#### Justificación de Relaciones de Independencia Condicional ($d$-separación):
- **Estructura colisionadora (*v-structure*)**: Asumimos que `acousticness` y `loudness` son marginalmente independientes a priori, pero se vuelven dependientes al condicionar sobre su efecto común `energy`.
- **Independencia condicional de valence y danceability**: Establecimos que `valence` y `danceability` son condicionalmente independientes dada `energy`; la energía actúa como la causa común latente que explica su correlación aparente.
- **$d$-separación respecto a la popularidad**: Atributos técnicos de grabación como `acousticness`, `loudness` y `mode` quedan $d$-separados de `popularity` al condicionar sobre las variables perceptibles intermedias (`energy`, `valence`). Esto refleja la intuición de que el público oyente no juzga los decibeles de grabación brutos de forma aislada, sino la experiencia sonora resultante (qué tan enérgica, bailable o alegre suena)."""))

cells.append(nbf.v4.new_code_cell(r"""print("Construyendo Red Bayesiana Manual...")
manual_model = DiscreteBayesianNetwork([
    ('acousticness', 'energy'),
    ('loudness', 'energy'),
    ('energy', 'valence'),
    ('mode', 'valence'),
    ('energy', 'danceability'),
    ('explicit', 'popularity'),
    ('danceability', 'popularity'),
    ('energy', 'popularity'),
    ('instrumentalness', 'popularity'),
    ('speechiness', 'popularity'),
])

print(f"Aristas propuestas en la red manual ({len(manual_model.edges())} aristas):")
for u, v in manual_model.edges():
    print(f"  {u} -> {v}")

print("\nAjustando parámetros (CPDs) mediante estimación de máxima verosimilitud...")
manual_model.fit(datos)
print("Validación de consistencia del modelo:", manual_model.check_model())"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 1.4 Inferencias en la Red Manual e Interpretación de Resultados
Utilizando el algoritmo exacto de **Eliminación de Variables** (`VariableElimination`), planteamos tres consultas probabilísticas estratégicas para evaluar el comportamiento del modelo:
1. **Consulta 1 (Caso contradictorio/nicho)**: ¿Qué ocurre con la popularidad de una pista si es sumamente instrumental pero a la vez contiene un alto nivel de palabras habladas?
   $$P(\text{popularity} \mid \text{instrumentalness} = \text{'alto'}, \text{speechiness} = \text{'alto'})$$
2. **Consulta 2 (Inferencia diagnóstica hacia atrás)**: Si sabemos que una pista fracasó en popularidad a pesar de tener contenido explícito (típico de canciones comerciales urbanas), ¿cuál es la probabilidad a posteriori sobre su nivel de bailabilidad?
   $$P(\text{danceability} \mid \text{popularity} = \text{'baja'}, \text{explicit} = \text{'si'})$$
3. **Consulta 3 (Cadena causal profunda)**: ¿Cómo se comporta el estado de ánimo (valence) si forzamos que la canción sea altamente acústica y grabada en escala menor?
   $$P(\text{valence} \mid \text{acousticness} = \text{'alto'}, \text{mode} = \text{'menor'})$$"""))

cells.append(nbf.v4.new_code_cell(r"""infer_manual = VariableElimination(manual_model)

print("--- INFERENCIAS EN RED MANUAL ---")
# Consulta 1
q1 = infer_manual.query(variables=['popularity'], evidence={'instrumentalness': 'alto', 'speechiness': 'alto'})
print("\nConsulta 1: P(popularity | instrumentalness='alto', speechiness='alto')")
print(q1)

# Consulta 2
q2 = infer_manual.query(variables=['danceability'], evidence={'popularity': 'baja', 'explicit': 'si'})
print("\nConsulta 2: P(danceability | popularity='baja', explicit='si')")
print(q2)

# Consulta 3
q3 = infer_manual.query(variables=['valence'], evidence={'acousticness': 'alto', 'mode': 'menor'})
print("\nConsulta 3: P(valence | acousticness='alto', mode='menor')")
print(q3)"""))

cells.append(nbf.v4.new_markdown_cell(r"""#### Interpretación de las Inferencias (Red Manual):
1. **Consulta 1**: Al observar alta instrumentalidad y alto contenido hablado, la probabilidad de que la canción tenga **popularidad baja aumenta al 43.88%**, superando a la media (29.28%) y dejando la alta en apenas 26.84%. Este resultado es sumamente coherente con el mercado musical real: piezas instrumentales con extensos pasajes de voz hablada corresponden a formatos experimentales, pistas introductorias o contenido tipo podcast, formatos alejados del consumo radial y de listas de éxitos masivas.
2. **Consulta 2**: En la inferencia diagnóstica, la probabilidad de que la bailabilidad sea **baja es del 37.38%** (frente a 33.51% alta y 29.12% media). Esto demuestra que una baja capacidad bailable actúa como un factor penalizador significativo en pistas explícitas que fracasan comercialmente, pues en los géneros urbanos la rítmica bailable es casi obligatoria.
3. **Consulta 3**: Al condicionar en una pista altamente acústica y en escala menor, la probabilidad de que su estado de ánimo sea **bajo (triste o melancólico) sube al 37.77%** (medio 34.53%), mientras que la posibilidad de ser alegre (alto) cae al 27.71%. Esto valida la teoría musical clásica: las baladas acústicas en escalas menores evocan dominantemente tristeza e introspección."""))

cells.append(nbf.v4.new_markdown_cell(r"""### 1.5 Aprendizaje Automático de Estructura (Algoritmo Chow-Liu)
A continuación, empleamos el algoritmo de **Chow-Liu** (implementado como `TreeSearch` en `pgmpy`) para aprender una segunda estructura directamente desde los datos sin intervención humana.

El algoritmo de Chow-Liu calcula la información mutua entre todos los pares de variables y encuentra el Árbol Generador de Peso Máximo (Maximum Spanning Tree). Esta metodología tiene la ventaja matemática de garantizar una red acíclica con a lo más un padre por nodo (exactamente 11 aristas para 12 variables), evitando el sobreajuste y la proliferación descontrolada de aristas de los métodos heurísticos voraces."""))

cells.append(nbf.v4.new_code_cell(r"""print("Aprendiendo estructura con algoritmo Chow-Liu (TreeSearch)...")
est = TreeSearch(datos, root_node='popularity')
best_struct = est.estimate(estimator_type='chow-liu')

auto_model = DiscreteBayesianNetwork(best_struct.edges())
print(f"Aristas descubiertas automáticamente ({len(auto_model.edges())} aristas):")
for u, v in auto_model.edges():
    print(f"  {u} -> {v}")

print("\nAjustando parámetros para la red automática...")
auto_model.fit(datos)

infer_auto = VariableElimination(auto_model)
print("\n--- INFERENCIAS EN RED AUTOMÁTICA ---")
q1_auto = infer_auto.query(variables=['popularity'], evidence={'instrumentalness': 'alto', 'speechiness': 'alto'})
print("\nConsulta 1 Auto: P(popularity | instrumentalness='alto', speechiness='alto')")
print(q1_auto)

q2_auto = infer_auto.query(variables=['danceability'], evidence={'popularity': 'baja', 'explicit': 'si'})
print("\nConsulta 2 Auto: P(danceability | popularity='baja', explicit='si')")
print(q2_auto)

q3_auto = infer_auto.query(variables=['valence'], evidence={'acousticness': 'alto', 'mode': 'menor'})
print("\nConsulta 3 Auto: P(valence | acousticness='alto', mode='menor')")
print(q3_auto)"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 1.6 Comparación y Análisis Crítico de Ambas Redes (9 Puntos)
A continuación evaluamos cuantitativamente el ajuste a los datos de ambas redes utilizando el criterio de información bayesiano (**BIC**, Bayesian Information Criterion), que premia la verosimilitud de los datos penalizando la complejidad paramétrica del grafo."""))

cells.append(nbf.v4.new_code_cell(r"""score = BIC(datos)
s_manual = score.score(manual_model)
s_auto = score.score(auto_model)

print("--- COMPARACIÓN FORMAL DE AJUSTE A LOS DATOS ---")
print(f"Score BIC Red Propuesta (Manual) : {s_manual:,.1f}")
print(f"Score BIC Red Aprendida (Auto)   : {s_auto:,.1f}")
mejor = "Red Propuesta (Manual)" if s_manual > s_auto else "Red Aprendida (Auto)"
print(f"Mejor ajuste global: {mejor}")

aristas_comunes = set(manual_model.edges()) & set(auto_model.edges())
print(f"Cantidad de aristas idénticas en ambas redes: {len(aristas_comunes)}")
if aristas_comunes:
    print(f"Aristas coincidentes: {aristas_comunes}")"""))

cells.append(nbf.v4.new_markdown_cell(r"""#### Análisis Comparativo Integral:

1. **En términos de Estructura y Topología**:
   - La red manual fue diseñada permitiendo **múltiples causas sobre nodos clave** (por ejemplo, `energy` recibe influencia conjunta de `acousticness` y `loudness`, y `popularity` recibe de 5 factores perceptibles).
   - La red de Chow-Liu, al estar obligada a ser un árbol con a lo más un padre por nodo, genera cadenas unidireccionales forzadas que invierten el orden causal natural (por ejemplo, descubrió `danceability -> acousticness`, lo que resulta contraintuitivo pues la bailabilidad no es la causa física de los instrumentos acústicos).
   - Ambas redes coincidieron únicamente en una arista exacta (`acousticness -> energy`), demostrando que la restricción de árbol limita severamente la topología en problemas multivariados complejos.

2. **En términos de Ajuste a los Datos (Score BIC)**:
   - **Score BIC Manual:** $-107,452.3$
   - **Score BIC Chow-Liu:** $-145,151.6$
   - En la formulación de pgmpy, valores menos negativos representan un mejor compromiso entre verosimilitud y parsimonia. Nuestra **red propuesta supera ampliamente a la automática por casi 38.000 puntos de ventaja**. Esto demuestra que la restricción de árbol impuesta por Chow-Liu es demasiado estricta para la riqueza estadística de este dataset, perdiendo correlaciones cruzadas esenciales que nuestra intuición humana sí capturó.

3. **En términos de los Resultados de las Inferencias**:
   - A pesar de las diferencias estructurales, ambas redes mantuvieron tendencias coherentes en las consultas debido a la correlación subyacente de los datos (ambas penalizan la popularidad con alta instrumentalidad y asocian modo menor y acústica a valencia baja). Sin embargo, la red manual preserva un flujo de $d$-separación mucho más fiel a la realidad, mientras que en la red automática la evidencia debe circular por caminos indirectos impuestos por el árbol."""))

# ==========================================
# PREGUNTA 2: CADENAS DE MARKOV Y HMM
# ==========================================
cells.append(nbf.v4.new_markdown_cell(r"""## Pregunta 2: Cadenas de Markov y Modelos Ocultos de Markov (30 Puntos)

### 2.a Preprocesamiento, Índices y Criterios de Discretización
En esta sección trabajamos con el dataset **Appliances Energy Prediction**, que registra el consumo eléctrico y variables ambientales de una vivienda cada 10 minutos durante cuatro meses y medio.

#### 1. Índices Ambientales:
Siguiendo las fórmulas indicadas en el enunciado:
$$I_{\text{temperatura}} = \frac{T_1 + T_2 + T_3}{3}, \qquad I_{\text{humedad}} = \frac{RH_1 + RH_2 + RH_3}{3}$$
Ambos índices se discretizan en **3 niveles balanceados (bajo, medio, alto)** mediante cuantiles (`pd.qcut`, $q=3$).

#### 2. Clasificación de la Iluminación (`lights`):
Se discretiza en **3 niveles lógicos**:
- `0`: Apagada ($0$ Wh).
- `1`: Consumo bajo ($10$ a $20$ Wh).
- `2`: Consumo alto ($> 20$ Wh).

#### 3. Clasificación del Consumo de Electrodomésticos (`Appliances`) en 6 Estados:
El consumo energético de una vivienda no sigue una distribución uniforme; la mayor parte del tiempo se encuentra en reposo nocturno o consumo mínimo, con picos esporádicos al usar artefactos de alta potencia. Para evitar categorías desbalanceadas o vacías, establecimos **6 estados de consumo bien fundamentados**:
- **0: Muy Bajo** ($\le 45$ Wh): consumo base nocturno en reposo profundo.
- **1: Bajo** ($46 - 55$ Wh): modo de espera estándar de artefactos esenciales.
- **2: Medio-Bajo** ($56 - 65$ Wh): actividad leve en el hogar.
- **3: Medio-Alto** ($66 - 85$ Wh): uso moderado de electrodomésticos.
- **4: Alto** ($86 - 125$ Wh): múltiples artefactos en uso simultáneo.
- **5: Muy Alto** ($> 125$ Wh hasta 1080 Wh): picos intensivos de potencia (horno, lavadora, microondas).

#### 4. Observaciones del HMM (27 Observaciones Posibles):
Combinamos las tres variables observadas en una representación posicional en base 3:
$$\text{observacion} = \text{estado\_luces} \times 9 + \text{estado\_temp} \times 3 + \text{estado\_hum}$$
Esto genera de forma exhaustiva los 27 estados posibles (del 0 al 26).

#### 5. Separación Cronológica Train / Test:
Los registros se ordenan temporalmente y se dividen en **80% Entrenamiento** y **20% Evaluación**. El punto de corte se normaliza a la medianoche (`00:00:00`) para no fragmentar ninguna jornada diaria."""))

cells.append(nbf.v4.new_code_cell(r"""import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import CategoricalHMM

# 1. Carga o descarga del dataset Appliances Energy Prediction
csv_filename = "energydata_complete.csv"
if not os.path.exists(csv_filename):
    zip_url = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
    zip_filename = "appliances.zip"
    print(f"Descargando dataset desde {zip_url}...")
    urllib.request.urlretrieve(zip_url, zip_filename)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extract(csv_filename, path=".")
    os.remove(zip_filename)
    print("Descarga completada.")

df_energy = pd.read_csv(csv_filename)
df_energy['date'] = pd.to_datetime(df_energy['date'])
df_energy = df_energy.sort_values('date').reset_index(drop=True)

# 2. Calculo de indices de temperatura y humedad
df_energy['I_temperatura'] = (df_energy['T1'] + df_energy['T2'] + df_energy['T3']) / 3
df_energy['I_humedad'] = (df_energy['RH_1'] + df_energy['RH_2'] + df_energy['RH_3']) / 3

# 3. Discretizacion de estados de consumo (6 niveles)
bins_energia = [0, 45, 55, 65, 85, 125, 2000]
nombres_estados = ['Muy Bajo', 'Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto', 'Muy Alto']
df_energy['estado_energia'] = pd.cut(df_energy['Appliances'], bins=bins_energia, labels=[0, 1, 2, 3, 4, 5]).astype(int)

# Discretizacion de indices ambientales (3 niveles por cuantiles)
df_energy['estado_temp'] = pd.qcut(df_energy['I_temperatura'], q=3, labels=[0, 1, 2]).astype(int)
df_energy['estado_hum'] = pd.qcut(df_energy['I_humedad'], q=3, labels=[0, 1, 2]).astype(int)

# Discretizacion de iluminacion (3 niveles logicos)
def discretizar_luces(val):
    if val == 0: return 0
    elif val <= 20: return 1
    else: return 2

df_energy['estado_luces'] = df_energy['lights'].apply(discretizar_luces)

# 4. Observacion combinada (27 posibles, del 0 al 26)
df_energy['observacion'] = df_energy['estado_luces'] * 9 + df_energy['estado_temp'] * 3 + df_energy['estado_hum']
print(f"Total observaciones unicas observadas: {df_energy['observacion'].nunique()} / 27 teoricas.")

# 5. Separacion cronologica (80% Train, 20% Test) normalizada a medianoche
corte_idx = int(len(df_energy) * 0.8)
fecha_corte = df_energy.iloc[corte_idx]['date'].normalize()

train_df = df_energy[df_energy['date'] < fecha_corte].copy()
test_df = df_energy[df_energy['date'] >= fecha_corte].copy()

print(f"Entrenamiento: {len(train_df)} registros ({train_df['date'].min().date()} al {train_df['date'].max().date()})")
print(f"Evaluacion:    {len(test_df)} registros ({test_df['date'].min().date()} al {test_df['date'].max().date()})")

df_energy.to_csv("energydata_discretizado.csv", index=False)"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 2.b Construcción de Cadenas de Markov (Continua y Diaria)
A partir del conjunto de entrenamiento construimos dos modelos markovianos distintos:
1. **Cadena Continua**: Asume que el consumo es un proceso estocástico único e ininterrumpido. Contamos las transiciones entre instantes consecutivos $t \to t+1$ en todo el conjunto de entrenamiento. La distribución inicial se fija como su **distribución estacionaria** $\pi_{\text{cont}}$ (vector de probabilidades límite que satisface $\pi P = \pi$).
2. **Cadena Diaria**: Trata cada día como una secuencia independiente con su propia dinámica circadiana. Las transiciones solo se cuentan al interior de cada jornada (evitando transiciones entre las 23:50 y las 00:00 del día siguiente). La distribución inicial $\pi_{\text{dia}}$ se estima a partir de la frecuencia empírica del **primer estado de cada jornada (a las 00:00 hrs)**."""))

cells.append(nbf.v4.new_code_cell(r"""N_states = 6
alpha_smooth = 1e-6 # Suavizado para evitar divisiones por cero

# 1. Matriz de transicion Continua
states_train = train_df['estado_energia'].values
C_cont = np.zeros((N_states, N_states))
for s_curr, s_next in zip(states_train[:-1], states_train[1:]):
    C_cont[s_curr, s_next] += 1
P_cont = (C_cont + alpha_smooth) / (C_cont + alpha_smooth).sum(axis=1, keepdims=True)

# Distribucion estacionaria calculada mediante potencias de matriz
P_inf = np.linalg.matrix_power(P_cont, 500)
pi_cont = P_inf[0]

# 2. Matriz de transicion Diaria
C_dia = np.zeros((N_states, N_states))
train_df['day'] = train_df['date'].dt.date
primeros_estados = []
ultimos_estados = []

for day, group in train_df.groupby('day'):
    g_states = group['estado_energia'].values
    if len(g_states) > 0:
        primeros_estados.append(g_states[0])
        ultimos_estados.append(g_states[-1])
    for s_curr, s_next in zip(g_states[:-1], g_states[1:]):
        C_dia[s_curr, s_next] += 1

P_dia = (C_dia + alpha_smooth) / (C_dia + alpha_smooth).sum(axis=1, keepdims=True)
pi_dia = (np.bincount(primeros_estados, minlength=N_states) + alpha_smooth)
pi_dia = pi_dia / pi_dia.sum()

pi_final_dia = np.bincount(ultimos_estados, minlength=N_states) / len(ultimos_estados)

# Visualizacion comparativa de las matrices de transicion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
im1 = ax1.imshow(P_cont, cmap='Blues', vmin=0, vmax=0.8)
ax1.set_title("Matriz de Transición - Modelo Continuo")
ax1.set_xticks(range(6)); ax1.set_yticks(range(6))
ax1.set_xticklabels(nombres_estados, rotation=45, ha='right')
ax1.set_yticklabels(nombres_estados)

im2 = ax2.imshow(P_dia, cmap='Greens', vmin=0, vmax=0.8)
ax2.set_title("Matriz de Transición - Modelo Diario")
ax2.set_xticks(range(6)); ax2.set_yticks(range(6))
ax2.set_xticklabels(nombres_estados, rotation=45, ha='right')
ax2.set_yticklabels(nombres_estados)

fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 2.c Interpretación de Distribuciones y Consultas Temporales

#### 1. Análisis de las Distribuciones Iniciales y Estacionarias:
- **Distribución Estacionaria Continua ($\pi_{\text{cont}}$)**: A largo plazo, el sistema presenta un comportamiento aperiódico y ergódico, distribuyendo el tiempo entre todos los estados (Bajo: $22.01\%$, Muy Bajo: $18.26\%$, Alto: $15.90\%$, Muy Alto: $15.59\%$). No existen estados absorbentes y la probabilidad límite es independiente del punto de partida.
- **Distribución Inicial Diaria ($\pi_{\text{dia}}$ a las 00:00 hrs)**: Revela con gran claridad los hábitos circadianos del hogar. El **$88.18\%$ de los días inician en estados de bajo consumo** (Bajo: $36.36\%$, Medio-Bajo: $30.91\%$, Muy Bajo: $20.91\%$), mientras que el consumo Alto y Muy Alto representan prácticamente el $0\%$, reflejando que los habitantes duermen a medianoche.
- **Distribución Final Diaria (a las 23:50 hrs)**: Repite este mismo patrón ($85.45\%$ en consumos reducidos), confirmando que el ciclo diario retorna al reposo nocturno.

#### 2. Consultas en el Modelo Continuo:
- **Consulta 1**: Estando en consumo 'Muy Bajo' (estado 0), ¿cuál es la probabilidad de pasar a niveles elevados ('Alto' o 'Muy Alto') en **1 hora** ($k=6$ pasos) y en **6 horas** ($k=36$ pasos)?
- **Consulta 2**: Estando en consumo 'Muy Alto' (estado 5, uso intensivo de artefactos), ¿cuál es la probabilidad de haber retornado a reposo ('Muy Bajo' o 'Bajo') en **3 horas** ($k=18$ pasos) y en **12 horas** ($k=72$ pasos)?

#### 3. Consultas en el Modelo Diario:
- **Consulta 1**: Si el día inicia a las 00:00 en estado 'Bajo' (estado 1), ¿cuál es la probabilidad de encontrarse en consumo 'Muy Alto' (estado 5) a mediodía (12:00 hrs, $t=72$ pasos de 10 min)?
- **Consulta 2**: Si el día arranca en 'Muy Bajo' (estado 0), ¿cuál es la probabilidad de terminar la jornada a las 23:50 ($t=143$ pasos) también en reposo ('Muy Bajo' o 'Bajo')?"""))

cells.append(nbf.v4.new_code_cell(r"""print("--- RESULTADOS Y CONSULTAS MODELO CONTINUO ---")
print("Distribución Estacionaria pi_cont:")
for i, name in enumerate(nombres_estados):
    print(f"  {name:11}: {pi_cont[i]:.4%}")

# Consulta 1 Continuo
P_cont_6 = np.linalg.matrix_power(P_cont, 6)
P_cont_36 = np.linalg.matrix_power(P_cont, 36)
prob_c1_1h = P_cont_6[0, 4] + P_cont_6[0, 5]
prob_c1_6h = P_cont_36[0, 4] + P_cont_36[0, 5]
print(f"\nConsulta 1: Estando en 'Muy Bajo', P(pasar a 'Alto' o 'Muy Alto'):")
print(f"  En 1 hora  (k=6 pasos):   {prob_c1_1h:.4%}")
print(f"  En 6 horas (k=36 pasos):  {prob_c1_6h:.4%}")

# Consulta 2 Continuo
P_cont_18 = np.linalg.matrix_power(P_cont, 18)
P_cont_72 = np.linalg.matrix_power(P_cont, 72)
prob_c2_3h = P_cont_18[5, 0] + P_cont_18[5, 1]
prob_c2_12h = P_cont_72[5, 0] + P_cont_72[5, 1]
print(f"\nConsulta 2: Estando en 'Muy Alto', P(retornar a 'Muy Bajo' o 'Bajo'):")
print(f"  En 3 horas  (k=18 pasos): {prob_c2_3h:.4%}")
print(f"  En 12 horas (k=72 pasos): {prob_c2_12h:.4%}")

print("\n--- RESULTADOS Y CONSULTAS MODELO DIARIO ---")
print("Distribución Inicial Diaria (00:00 hrs):")
for i, name in enumerate(nombres_estados):
    print(f"  {name:11}: {pi_dia[i]:.4%}")

# Consulta 1 Diario
P_dia_72 = np.linalg.matrix_power(P_dia, 72)
print(f"\nConsulta 1 Diario: P(S_12:00 = Muy Alto | S_00:00 = Bajo): {P_dia_72[1, 5]:.4%}")

# Consulta 2 Diario
P_dia_143 = np.linalg.matrix_power(P_dia, 143)
prob_c2_dia = P_dia_143[0, 0] + P_dia_143[0, 1]
print(f"Consulta 2 Diario: P(S_23:50 in {{Muy Bajo, Bajo}} | S_00:00 = Muy Bajo): {prob_c2_dia:.4%}")"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 2.d Matriz de Emisión y Modelos Ocultos de Markov (`hmmlearn`)

#### 1. Estimación de la Matriz de Emisión $B$:
La matriz de emisión $B$ de dimensiones $6 \times 27$ contiene las probabilidades condicionales de emitir cada una de las 27 observaciones ambientales dado el estado oculto de consumo:
$$B_{i, k} = P(O_t = k \mid S_t = i) = \frac{\text{conteo}(S_t = i, O_t = k) + \alpha}{\sum_{k'=0}^{26} (\text{conteo}(S_t = i, O_t = k') + \alpha)}$$
Aplicamos un factor de suavizado de Laplace ($\alpha = 10^{-6}$) para garantizar soporte numérico estricto y evitar probabilidades nulas.

#### 2. Justificación Teórica de por qué Ambos Modelos Pueden Compartir la Misma Matriz de Emisión:
Ambos modelos pueden compartir legítimamente la misma matriz de emisión porque **la relación física entre el estado del electrodoméstico y las variables ambientales observadas es invariante a la estructuración temporal**. 

La emisión $P(O_t \mid S_t)$ modela cómo el consumo eléctrico produce calor, modifica la humedad relativa e interactúa con la iluminación del hogar en un instante dado. Esta dependencia termodinámica y física del entorno domiciliario opera exactamente bajo las mismas leyes tanto si observamos la vivienda de forma continua como si la segmentamos en ciclos diarios."""))

cells.append(nbf.v4.new_code_cell(r"""# Estimacion de la matriz de emision B (6 x 27)
N_obs = 27
C_em = np.zeros((N_states, N_obs))
for s, o in zip(train_df['estado_energia'], train_df['observacion']):
    C_em[s, o] += 1

B = (C_em + alpha_smooth) / (C_em + alpha_smooth).sum(axis=1, keepdims=True)

# Construccion de los modelos HMM con hmmlearn
hmm_continuo = CategoricalHMM(n_components=N_states, init_params='', params='')
hmm_continuo.startprob_ = pi_cont
hmm_continuo.transmat_ = P_cont
hmm_continuo.emissionprob_ = B

hmm_diario = CategoricalHMM(n_components=N_states, init_params='', params='')
hmm_diario.startprob_ = pi_dia
hmm_diario.transmat_ = P_dia
hmm_diario.emissionprob_ = B

print(f"Modelos HMM creados exitosamente.")
print(f"Dimensiones de matriz de emision B: {B.shape} (filas=estados, columnas=observaciones).")"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 2.e Inferencia en el Conjunto de Evaluación: Forward, Forward-Backward y Viterbi
Seleccionamos el **primer día completo del conjunto de evaluación (30 de Abril de 2016)**, el cual cuenta con los 144 intervalos completos de 10 minutos (de 00:00 a 23:50 hrs).

Aplicamos los tres algoritmos fundamentales de inferencia sobre este día:
1. **Algoritmo Forward**: Calcula la probabilidad acumulada (verosimilitud logarítmica) de la secuencia de observaciones observada: $\log P(O_{1:T})$.
2. **Algoritmo Forward-Backward**: Estima la distribución posterior marginal de los estados en cada instante: $\gamma_t(i) = P(S_t = i \mid O_{1:T})$. Evaluamos las probabilidades en tres horarios clave:
   - Madrugada (`04:00` hrs, $t=24$).
   - Mediodía (`12:00` hrs, $t=72$).
   - Noche (`20:00` hrs, $t=120$).
3. **Algoritmo de Viterbi**: Decodifica la secuencia globalmente más probable de estados ocultos en el día:
   $$\hat{S}_{1:T} = \arg\max_{S_{1:T}} P(S_{1:T}, O_{1:T})$$
   Comparamos esta secuencia predicha directamente contra los estados reales de consumo de los electrodomésticos."""))

cells.append(nbf.v4.new_code_cell(r"""# Seleccionar el primer dia completo de evaluacion
test_df['day'] = test_df['date'].dt.date
dias_completos = test_df['day'].value_counts()
dia_eval = dias_completos[dias_completos == 144].index.sort_values()[0]
df_dia = test_df[test_df['day'] == dia_eval].sort_values('date').reset_index(drop=True)

X_eval = df_dia['observacion'].values.reshape(-1, 1)
y_real = df_dia['estado_energia'].values
horas_eje = df_dia['date'].dt.strftime('%H:%M')

# 1. Forward
log_lik_cont = hmm_continuo.score(X_eval)
log_lik_dia = hmm_diario.score(X_eval)
print(f"--- 1. FORWARD: VEROSIMILITUD DE OBSERVACIONES ---")
print(f"Log-Likelihood Modelo Continuo: {log_lik_cont:.2f}")
print(f"Log-Likelihood Modelo Diario:   {log_lik_dia:.2f}")

# 2. Forward-Backward
post_cont = hmm_continuo.predict_proba(X_eval)
post_dia = hmm_diario.predict_proba(X_eval)

print(f"\n--- 2. FORWARD-BACKWARD: PROBABILIDADES POSTERIORES EN HORARIOS CLAVE ---")
t_instantes = [24, 72, 120]
etq_horas = ["04:00 hrs (Madrugada)", "12:00 hrs (Mediodía)", "20:00 hrs (Noche)"]

for t, h in zip(t_instantes, etq_horas):
    r_name = nombres_estados[y_real[t]]
    c_best = nombres_estados[np.argmax(post_cont[t])]
    d_best = nombres_estados[np.argmax(post_dia[t])]
    print(f"Instante {h}: Real = {r_name}")
    print(f"  Modelo Continuo -> Pred: {c_best} (P={np.max(post_cont[t]):.4f})")
    print(f"  Modelo Diario   -> Pred: {d_best} (P={np.max(post_dia[t]):.4f})")

# 3. Viterbi
viterbi_cont = hmm_continuo.predict(X_eval)
viterbi_dia = hmm_diario.predict(X_eval)

acc_cont = (viterbi_cont == y_real).mean()
acc_dia = (viterbi_dia == y_real).mean()
print(f"\n--- 3. VITERBI: COMPARACIÓN CON ESTADOS REALES ---")
print(f"Exactitud (Accuracy) Modelo Continuo: {acc_cont:.2%}")
print(f"Exactitud (Accuracy) Modelo Diario:   {acc_dia:.2%}")

# Grafico de la secuencia real vs Viterbi
plt.figure(figsize=(14, 5))
plt.plot(range(144), y_real, label="Estado Real (Appliances)", color="black", lw=2, alpha=0.8)
plt.plot(range(144), viterbi_cont, label=f"Viterbi Continuo (Acc: {acc_cont:.1%})", color="royalblue", ls='--', lw=1.8)
plt.plot(range(144), viterbi_dia, label=f"Viterbi Diario (Acc: {acc_dia:.1%})", color="forestgreen", ls=':', lw=1.8)
plt.yticks(range(6), nombres_estados)
plt.xticks(range(0, 144, 12), horas_eje[::12])
plt.xlabel("Hora del Día")
plt.ylabel("Nivel de Consumo")
plt.title(f"Inferencia de Consumo Oculto con Viterbi - Día de Evaluación: {dia_eval}")
plt.grid(alpha=0.25)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 2.f Comparación e Interpretación Integral de Modelos HMM (6 Puntos)

1. **Efecto de Considerar o Excluir las Transiciones entre Días**:
   - En el **modelo continuo**, la transición nocturna de las 23:50 a las 00:00 se computa dentro de la matriz global. Como las viviendas suelen estar en bajo consumo a esas horas, estas transiciones refuerzan la persistencia de los estados de reposo (0, 1 y 2).
   - En el **modelo diario**, las transiciones están confinadas estrictamente al ciclo circadiano de 24 horas. Esto permite que el modelo diario capture de forma más pura la dinámica interna de una jornada, evitando suponer continuidad forzada entre el término de un día y el comienzo del siguiente.
   - En la práctica sobre el día evaluado, ambos modelos obtuvieron verosimilitudes muy similares ($-374.47$ continuo vs $-374.71$ diario), lo que confirma que la dinámica intradiaria es la que domina fuertemente el proceso.

2. **Significado de las Distribuciones Iniciales**:
   - $\pi_{\text{cont}}$ refleja el **equilibrio ergódico a largo plazo** de la vivienda; asigna una probabilidad equilibrada entre $13\%$ y $22\%$ a cada estado, representando el tiempo medio que la casa pasa en cada nivel de consumo a lo largo de los meses.
   - $\pi_{\text{dia}}$ representa la **condición inicial empírica de arranque a las 00:00 hrs**. Asigna más del $88\%$ de probabilidad a los consumos reducidos (0, 1 y 2), coincidiendo con la realidad de que a medianoche la casa casi nunca se encuentra en picos de consumo elevado.

3. **Diferencias entre Forward-Backward y Viterbi**:
   - **Forward-Backward** entrega una distribución probabilística marginal en cada instante $t$: $\gamma_t(i) = P(S_t = i \mid O_{1:T})$. Si tomáramos simplemente $\arg\max_i \gamma_t(i)$ en cada paso, podríamos obtener secuencias con transiciones físicamente imposibles si entre dos estados sucesivos la probabilidad de transición fuese nula.
   - **Viterbi**, mediante programación dinámica sobre el retículo temporal, encuentra la **trayectoria completa conjunta más probable** $\arg\max_{S} P(S, O)$. Esto garantiza una secuencia globalmente coherente y consistente con las restricciones de la matriz de transición.
   - La exactitud moderada obtenida se debe a que las variables ambientales (temperatura y humedad) poseen una alta inercia térmica, por lo que reaccionan con retardo a los picos rápidos de consumo de los artefactos, un fenómeno típico y desafiante en el modelado de viviendas reales."""))

# ==========================================
# PREGUNTA 3: MONTE CARLO Y ANTENAS
# ==========================================
cells.append(nbf.v4.new_markdown_cell(r"""## Pregunta 3: Simulación Monte Carlo y Cobertura de Antenas (20 Puntos)

### 3.1 y 3.2 Parametrización y Reglas de Cobertura
La empresa de telecomunicaciones DBZ requiere calcular el área de cobertura efectiva de un clúster de tres antenas omnidireccionales en un terreno plano cuadrado de $100 \times 100$ km ($10.000$ km$^2$), donde $0 \le x \le 100$ y $0 \le y \le 100$.

#### Parametrización a partir del RUT:
Utilizamos los primeros 8 dígitos del RUT de **Maximiliano Palma**: **`21.620.910-9`**.  
Los dígitos corresponden a: $d_1=2, d_2=1, d_3=6, d_4=2, d_5=0, d_6=9, d_7=1, d_8=0$.

Aplicando las expresiones del enunciado:
- **Antena 1**: centro $c_1 = (d_1 \cdot 10, d_2 \cdot 10) = (20, 10)$ km ; radio $r_1 = d_3 \cdot 2 + 15 = 6 \cdot 2 + 15 = 27$ km.
- **Antena 2**: centro $c_2 = (d_4 \cdot 10, d_5 \cdot 10) = (20, 0)$ km ; radio $r_2 = d_3 \cdot 2 + 15 = 6 \cdot 2 + 15 = 27$ km.
- **Antena 3**: centro $c_3 = (d_7 \cdot 10, d_8 \cdot 10) = (10, 0)$ km ; radio $r_3 = 25$ km (radio fijo).
*(Nota: la pauta asigna $d_3$ para el radio de las antenas 1 y 2, dejando el dígito $d_6$ sin empleo).*

#### Reglas de Clasificación del Servicio:
Para cada punto $(x, y)$ en el mapa:
- **Servicio útil**: si pertenece al radio de cobertura de **exactamente 1 o 2 antenas**.
- **Servicio interferido**: si se encuentra bajo el alcance de las **3 antenas simultáneamente** (la interferencia destructiva provoca la caída del servicio).
- **Sin señal**: si no se encuentra dentro del radio de ninguna antena (**0 antenas**).

#### Metodología de Simulación Monte Carlo:
Evaluamos el sistema bajo los 4 escenarios solicitados:
$N_1 = 10.000, \quad N_2 = 100.000, \quad N_3 = 1.000.000, \quad N_4 = 10.000.000$ puntos aleatorios independientes con distribución uniforme continua $\mathcal{U}(0, 100)$. Para optimizar el uso de memoria y evitar tiempos excesivos de cómputo con $10^7$ puntos, el algoritmo se ejecuta por bloques vectorizados evaluando distancias euclidianas al cuadrado $((x - c_x)^2 + (y - c_y)^2 \le r^2)$."""))

cells.append(nbf.v4.new_code_cell(r"""import numpy as np
import matplotlib.pyplot as plt

RUT = "21.620.910-9"  # Maximiliano Palma
d = [int(c) for c in RUT if c.isdigit()][:8]

ANTENAS = [
    ((d[0] * 10, d[1] * 10), d[2] * 2 + 15),
    ((d[3] * 10, d[4] * 10), d[2] * 2 + 15),
    ((d[6] * 10, d[7] * 10), 25)
]

LADO = 100.0
AREA_TOTAL = 10_000.0
ESCENARIOS = [10_000, 100_000, 1_000_000, 10_000_000]

print("RUT de referencia:", RUT, "-> Dígitos d1..d8 =", d)
for i, ((cx, cy), r) in enumerate(ANTENAS, 1):
    print(f"  Antena {i}: Centro = ({cx}, {cy}) km, Radio = {r} km")

def simular_monte_carlo(N, seed=42, chunk=1_000_000):
    rng = np.random.default_rng(seed)
    conteos = np.zeros(4, dtype=np.int64)
    submuestra = None
    for i in range(0, N, chunk):
        m = min(chunk, N - i)
        x = rng.uniform(0, LADO, m)
        y = rng.uniform(0, LADO, m)
        cob = sum(((x - cx) ** 2 + (y - cy) ** 2 <= r * r) for (cx, cy), r in ANTENAS)
        conteos += np.bincount(cob, minlength=4)
        if submuestra is None:
            submuestra = (x[:6000], y[:6000], cob[:6000])
    return conteos, submuestra

resultados = []
muestras_vis = {}

for N in ESCENARIOS:
    c, mu = simular_monte_carlo(N)
    muestras_vis[N] = mu
    p_util = (c[1] + c[2]) / N
    p_int = c[3] / N
    p_sin = c[0] / N
    ee = AREA_TOTAL * np.sqrt(p_util * (1 - p_util) / N)
    resultados.append((N, AREA_TOTAL * p_util, ee, AREA_TOTAL * p_int, AREA_TOTAL * p_sin))

# Tabla 1: Reporte de Areas
print(f"\n{'N':>12} {'Área Útil (km²)':>16} {'± EE (km²)':>12} {'Interferida (km²)':>18} {'Sin Señal (km²)':>16}")
for N, a_u, e, a_i, a_s in resultados:
    print(f"{N:>12,} {a_u:>16.2f} {e:>12.2f} {a_i:>18.2f} {a_s:>16.2f}")

# Tabla 2: Estudio de Convergencia
ref_area = resultados[-1][1]
print(f"\nReferencia de Alta Precisión (N=10M): {ref_area:.2f} km²")
print(f"{'N':>12} {'Error Absoluto':>15} {'Error Relativo':>15} {'EE Teórico':>12} {'Razón EE':>10}")
prev_ee = None
for N, a_u, e, _, _ in resultados:
    err_abs = abs(a_u - ref_area)
    err_rel = err_abs / ref_area
    razon = f"{e / prev_ee:.3f}" if prev_ee else "-"
    print(f"{N:>12,} {err_abs:>15.2f} {err_rel:>14.4%} {e:>12.2f} {razon:>10}")
    prev_ee = e

# Grafico de 4 paneles en paleta Okabe-Ito (accesible para daltonismo)
CLASES = [
    ("Sin Señal (0 antenas)", '#0072b2', 2, 0.30),
    ("Servicio Útil (1-2 antenas)", '#009e73', 7, 0.90),
    ("Interferido (3 antenas)", '#d55e00', 7, 0.90)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12.5))
for ax, N in zip(axes.ravel(), ESCENARIOS):
    x, y, cob = muestras_vis[N]
    clase = np.where(cob == 0, 0, np.where(cob == 3, 2, 1))
    for k, (etq, col, s, al) in enumerate(CLASES):
        m = (clase == k)
        ax.scatter(x[m], y[m], c=col, s=s, alpha=al, linewidths=0, label=etq if ax is axes[0, 0] else None)
    for (cx, cy), r in ANTENAS:
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec='#1a1a1a', lw=1.6))
        ax.plot(cx, cy, marker='^', color='#1a1a1a', ms=9, mec='white', mew=1.2)
    ax.set_xlim(0, LADO); ax.set_ylim(0, LADO); ax.set_aspect('equal')
    a_u = [f[1] for f in resultados if f[0] == N][0]
    ax.set_title(f"N = {N:,}  |  Área Útil = {a_u:.2f} km²", fontsize=11)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.grid(alpha=0.15, lw=0.5)

fig.legend(loc='lower center', ncol=3, frameon=False, markerscale=3.2, fontsize=10.5, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Simulación Monte Carlo: Cobertura Efectiva de Antenas DBZ (Submuestra de 6.000 pts por panel)", fontsize=13)
fig.tight_layout(rect=[0, 0.035, 1, 1])
plt.savefig("p3_cobertura.png", dpi=120)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 3.4 Comparación de Escenarios y Discusión sobre la Convergencia de Monte Carlo (6 Puntos)

#### 1. Comparación de Resultados a través de los 4 Escenarios:
- **Escenario 1 ($N = 10.000$)**: Área útil estimada de **$797.00$ km²** con un error estándar de $\pm 27.08$ km² y un error relativo del $2.23\%$ respecto a la referencia de $10$ millones. Con un sampleo modesto se aprecia la dispersión estocástica.
- **Escenario 2 ($N = 100.000$)**: La estimación asciende a **$824.40$ km²**, reduciendo el error estándar a $\pm 8.70$ km² (error relativo del $1.14\%$).
- **Escenario 3 ($N = 1.000.000$)**: El estimador se estabiliza fuertemente en **$814.85$ km²**, con un error estándar de tan solo $\pm 2.74$ km² ($0.036\%$ de error relativo).
- **Escenario 4 ($N = 10.000.000$)**: La estimación de alta precisión converge a **$815.15$ km²** con un error estándar mínimo de $\pm 0.87$ km². La diferencia entre $1$M y $10$M es de apenas $0.30$ km², confirmando que la solución alcanzó una excelente estabilidad numérica.

#### 2. Fundamentos Probabilísticos de la Convergencia:
- **Ley Fuerte de los Grandes Números (LGN)**: Garantiza que la proporción muestral $\hat{p}_N = \frac{1}{N} \sum_{i=1}^N \mathbb{I}_{\text{útil}}(X_i)$ converge de manera casi segura (con probabilidad 1) a la probabilidad teórica $p = \frac{\text{Área Útil}}{10.000 \text{ km}^2}$ cuando $N \to \infty$.
- **Tasa de Convergencia $O(1/\sqrt{N})$ y Teorema del Límite Central (TLC)**:
  El TLC establece que el error estándar de la media muestral decrece con la raíz cuadrada del número de simulaciones:
  $$EE = \text{Área\_Total} \times \sqrt{\frac{p(1 - p)}{N}} = O\left(\frac{1}{\sqrt{N}}\right)$$
  Dado que en cada escenario aumentamos $N$ en un factor de $10$ ($10$K $\to$ $100$K $\to$ $1$M $\to$ $10$M), el error estándar teórico debe reducirse multiplicándose por un factor de:
  $$\frac{1}{\sqrt{10}} \approx 0.3162$$
  Al contrastar esto con las **razones empíricas observadas en nuestra tabla**:
  - De $10$K a $100$K: $\frac{8.70}{27.08} = \mathbf{0.321}$
  - De $100$K a $1$M: $\frac{2.74}{8.70} = \mathbf{0.315}$
  - De $1$M a $10$M: $\frac{0.87}{2.74} = \mathbf{0.316}$
  
  La coincidencia con el valor teórico de $0.3162$ es prácticamente exacta. Esto demuestra empíricamente la propiedad fundamental de Monte Carlo: **para ganar un dígito decimal extra de precisión en la estimación de áreas, es necesario multiplicar por 100 el número de muestras simuladas**.

#### 3. Interpretación Espacial del Gráfico (`p3_cobertura.png`):
- Las coordenadas derivadas del RUT ubicaron a las 3 antenas muy agrupadas hacia el vértice inferior izquierdo del terreno ($(20, 10)$, $(20, 0)$ y $(10, 0)$ km).
- La **zona naranja (interferida)** delimita el solapamiento simultáneo de los 3 círculos ($\sim 711$ km², $7.1\%$ del mapa), provocando una zona de sombra o corte de servicio en el núcleo del clúster.
- La **zona verde (servicio útil)** forma una corona envolvente de $\sim 815$ km² ($8.15\%$ del mapa) donde operan 1 o 2 antenas.
- El **resto del mapa en azul** ($>84\%$, más de $8.470$ km²) queda totalmente desprovisto de señal, lo que entrega información estratégica fundamental a la empresa DBZ para el despliegue de futuras estaciones base hacia el cuadrante noreste."""))

# Guardamos las celdas en el notebook
nb['cells'] = cells

# Metadatos del notebook
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.0"
    }
}

notebook_path = "Ignacio_Antiguay_Vicente_Hernandez_Maximiliano_Palma_Tarea1.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook '{notebook_path}' generado con {len(cells)} celdas.")
print("Ejecutando todas las celdas del notebook...")

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, "r", encoding="utf-8") as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': '.'}})

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb_to_run, f)

print(f"¡Notebook '{notebook_path}' ejecutado y guardado exitosamente con todos los outputs incrustados!")
