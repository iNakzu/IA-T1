"""
Tarea 1 Inteligencia Artificial. 
Estudiantes: Ignacio Antiguay, Vicente Hernandez, Maximiliano Palma
Profesores: Johan Fuentes, Víctor Reyes, Felipe Lazo 
Se utilizaron herramientas generativas (Gemini, Claude) como apoyo técnico para escritura y depuración de código Python, revisión del uso de librerías, organización metodológica y revisión de resultados.

Primeramente escogimos como grupo el dataset Spotify Tracks de la plataforma Kaggle, este venia con 114.000 filas y 22 columnas, con la siguiente idea reducimos las filas a 12.000 y 12 columnas y seleccionamos las siguientes columnas:
"""

#!pip install kagglehub pandas pgmpy

# Inicialización del dataset Spotify Tracks desde Kaggle
import pandas as pd
import numpy as np
import kagglehub
import os
import glob
import warnings

warnings.filterwarnings("ignore") # Oculta alertas rojas de pgmpy para limpiar la consola

# 1. Descargar el dataset automáticamente (se guarda en caché)
print("Descargando dataset desde Kaggle...")
path = kagglehub.dataset_download("yashdev01/spotify-tracks-dataset")

# 2. Buscar el archivo CSV dentro de la ruta descargada
archivos_csv = glob.glob(os.path.join(path, "*.csv"))
if not archivos_csv:
    raise FileNotFoundError("No se encontró el archivo CSV en el dataset.")
ruta_csv = archivos_csv[0]

# 3. Cargar el dataset original
print(f"Cargando datos desde: {ruta_csv}")
df_raw = pd.read_csv(ruta_csv)

# 4. Reducir la muestra a 12.000 filas y seleccionar las 12 columnas útiles
df_muestra = df_raw.sample(n=12000, random_state=42).copy()
columnas = [
    'popularity', 'explicit', 'danceability', 'energy', 
    'key', 'loudness', 'mode', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence'
]
df = df_muestra[columnas].copy()

# 5. Guardar la versión limpia localmente
df.to_csv('spotify-tracks-dataset.csv', index=False)
print("¡Archivo 'spotify-tracks-dataset.csv' creado exitosamente con", df.shape[0], "filas!\n")

"""
Este archivo en su forma predeterminada venia con valores numéricos (danceability, energy, valence, etc.). Esto en una red bayesiana no era de mucha utilidad, por lo cual cada columna la convertimos en varias categorías (bajo/medio/alto). Popularity fue dividida en esas categorías, al igual que danceability, energy, etc. Instrumentalness fue dividida en 2 categorías (bajo/alto), porque casi todas las canciones valen aproximadamente 0. Explicit también figura de esta forma y key fue organizada según la tonalidad.
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import TreeSearch, BIC
from pgmpy.inference import VariableElimination

datos = pd.read_csv("spotify-tracks-dataset.csv")

datos['popularity'] = pd.qcut(datos['popularity'], 3, labels=['baja', 'media', 'alta'])
tres_niveles = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'liveness', 'valence']
for col in tres_niveles:
    datos[col] = pd.qcut(datos[col], 3, labels=['bajo', 'medio', 'alto'])
datos['instrumentalness'] = pd.qcut(datos['instrumentalness'], 2,
                                    duplicates='drop', labels=['bajo', 'alto'])

datos['explicit'] = datos['explicit'].map({True: 'si', False: 'no'})
datos['mode'] = datos['mode'].map({1: 'mayor', 0: 'menor'})
datos['key'] = datos['key'].astype(str)

datos = datos.astype(str) 

datos.to_csv("spotify-tracks-dataset-discretizado.csv", index=False)
print("Se guardó el dataset discretizado.\n")

"""
Para la construcción de la red bayesiana de forma manual, propusimos una estructura inicial basándonos en la relación lógica entre las características musicales. Consideramos que atributos como acousticness y loudness inciden directamente en el nivel de energía (energy) de una pista. A su vez, determinamos que la energía, en conjunto con la tonalidad (mode), afectan el estado de ánimo (valence) y qué tan bailable resulta ser la canción (danceability). 

Finalmente, establecimos que las variables más perceptibles al oyente, como lo bailable, la energía, el contenido explícito y qué tan instrumental es la canción, son los factores que determinan de forma directa la variable objetivo, que en este caso es popularity. 

Una vez definida esta estructura de dependencias, utilizamos la función fit() para estimar y aprender automáticamente las probabilidades condicionales a partir del dataset discretizado, validando posteriormente que el modelo estuviera correctamente construido y sin ciclos.
"""

print("Red Bayesiana Manual")
manual_model = DiscreteBayesianNetwork([
    ('acousticness', 'energy'), #acousticness -> energy 
    ('loudness', 'energy'),     #loudness -> energy
    ('energy', 'valence'),      #energy -> valence
    ('mode', 'valence'),        #mode -> valence 
    ('energy', 'danceability'), #energy ->danceability
    ('explicit', 'popularity'), #explicit -> popularity
    ('danceability', 'popularity'), #danceability -> popularity
    ('energy', 'popularity'),   #energy -> popularity
    ('instrumentalness', 'popularity'), #instrumentalness -> popularity
    ('speechiness', 'popularity'),#speechiness -> popularity
])

print(f"Aristas propuestas por nosotros ({len(manual_model.edges())} en total):")
for edge in manual_model.edges():
    print(f"  {edge[0]} -> {edge[1]}")

print("\nAjustando parámetros de la red manual...")
manual_model.fit(datos)
print("Validando modelo manual:", manual_model.check_model())

"""
Una vez que el modelo ajustó sus parámetros, utilizamos el algoritmo de Eliminación de Variables (VariableElimination) para poner a prueba la red con tres escenarios más específicos y explorar cómo se propagan las probabilidades por el grafo:
1. Primero quisimos evaluar un caso contradictorio: ¿Qué pasa con la popularidad si una pista es muy instrumental pero al mismo tiempo tiene un alto nivel de palabras habladas (speechiness)? Evaluamos P(popularity | instrumentalness='alto', speechiness='alto').
2. Luego hicimos una inferencia diagnóstica (hacia atrás). Si sabemos que una canción fracasó comercialmente (popularidad baja) a pesar de tener letra explícita (que suele vender bien), ¿qué tan probable es que el problema haya sido que era poco bailable? Consultamos P(danceability | popularity='baja', explicit='si').
3. Por último, evaluamos una cadena causal más larga. Queríamos predecir el estado de ánimo (valence) de una pista si forzamos que sea muy acústica (lo cual impacta su energía) y además está grabada en una escala menor (mode). Consultamos P(valence | acousticness='alto', mode='menor').
"""

print("\nRed Bayesiana Manual - Inferencias")
infer_manual = VariableElimination(manual_model)
# Inferencia 1: Caso contradictorio
q1 = infer_manual.query(variables=['popularity'], evidence={'instrumentalness': 'alto', 'speechiness': 'alto'})
print("\nConsulta 1: P(popularity | instrumentalness='alto', speechiness='alto')")
print(q1)

# Inferencia 2: Inferencia diagnóstica (hacia atrás)
q2 = infer_manual.query(variables=['danceability'], evidence={'popularity': 'baja', 'explicit': 'si'})
print("\nConsulta 2: P(danceability | popularity='baja', explicit='si')")
print(q2)

# Inferencia 3: Cadena causal profunda
q3 = infer_manual.query(variables=['valence'], evidence={'acousticness': 'alto', 'mode': 'menor'})
print("\nConsulta 3: P(valence | acousticness='alto', mode='menor')")
print(q3)

"""
Interpretación de los resultados obtenidos en las inferencias (Red Manual):

1. Consulta 1: P(popularity | instrumentalness='alto', speechiness='alto')
Al evaluar qué pasa si una pista es muy instrumental pero al mismo tiempo tiene mucho diálogo o texto hablado, la probabilidad de que tenga popularidad baja sube con fuerza al 43.88%, superando ampliamente a la popularidad media (29.28%) y dejando la alta en apenas 26.84%. Esto tiene todo el sentido del mundo en el mercado musical: una pista instrumental con mucha voz hablada suele ser algo de nicho, una intro de álbum o un podcast, formatos que rara vez se viralizan masivamente en Spotify.

2. Consulta 2: P(danceability | popularity='baja', explicit='si')
Al realizar esta inferencia diagnóstica (hacia atrás) para analizar por qué una canción con contenido explícito (que comercialmente suele vender bien en el género urbano) terminó fracasando en popularidad, la distribución muestra mayor probabilidad en danceability bajo con 37.38% (alto 33.51% y medio 29.12%). Esto nos enseña que el fracaso de una pista explícita se vincula de manera apreciable con una baja bailabilidad, sugiriendo que la falta de ritmo o de baile afectó negativamente su desempeño comercial.

3. Consulta 3: P(valence | acousticness='alto', mode='menor')
En esta cadena causal profunda, al forzar que la canción sea muy acústica (lo que reduce fuertemente su energía) y además esté en una escala menor, la probabilidad de que su estado de ánimo (valence) sea bajo (triste, sombría o melancólica) aumenta fuertemente al 37.77% (medio 34.53%), mientras que la probabilidad de que sea alegre (valence alto) cae al 27.71%. Esto calza perfecto con la teoría musical tradicional: las canciones acústicas en escalas menores son por excelencia baladas tristes o temas reflexivos.
"""

"""
Luego de evaluar nuestra red propuesta, procedimos a la segunda etapa del análisis: permitir que un algoritmo descubra la estructura puramente a partir de los datos, sin sesgo humano. 

Para ello, decidimos utilizar el algoritmo de Chow-Liu (implementado como TreeSearch en pgmpy). Elegimos este algoritmo matemático porque restringe la topología a un Árbol (donde cada nodo tiene máximo un padre). Esto resulta en una estructura mucho más limpia, directa y con exactamente 11 flechas, evitando así las redes sobrecomplicadas y difíciles de interpretar que generan los algoritmos heurísticos tradicionales.

Una vez que el algoritmo Chow-Liu nos entregó la estructura óptima del árbol, instanciamos esta nueva red bayesiana, imprimimos las aristas descubiertas y procedimos a ajustar sus probabilidades.
"""

print("\nRed Bayesiana Automática - Aprendizaje de Estructura")
est = TreeSearch(datos, root_node='popularity')
print("Ejecutando algoritmo Chow-Liu (TreeSearch)...")
best_model_struct = est.estimate(estimator_type='chow-liu')

auto_model = DiscreteBayesianNetwork(best_model_struct.edges())
print(f"Aristas descubiertas por el algoritmo ({len(auto_model.edges())} en total):")
for edge in auto_model.edges():
    print(f"  {edge[0]} -> {edge[1]}")

print("\nAjustando parámetros para la red automática...")
auto_model.fit(datos)

print("\nRed Bayesiana Automática - Inferencias")
infer_auto = VariableElimination(auto_model)

try:
    q1_auto = infer_auto.query(variables=['popularity'], evidence={'instrumentalness': 'alto', 'speechiness': 'alto'})
    print("\nConsulta 1 Auto: P(popularity | instrumentalness='alto', speechiness='alto')")
    print(q1_auto)
except Exception as e:
    print(f"No se pudo calcular Consulta 1 en red auto: {e}")

try:
    q2_auto = infer_auto.query(variables=['danceability'], evidence={'popularity': 'baja', 'explicit': 'si'})
    print("\nConsulta 2 Auto: P(danceability | popularity='baja', explicit='si')")
    print(q2_auto)
except Exception as e:
    print(f"No se pudo calcular Consulta 2 en red auto: {e}")

try:
    q3_auto = infer_auto.query(variables=['valence'], evidence={'acousticness': 'alto', 'mode': 'menor'})
    print("\nConsulta 3 Auto: P(valence | acousticness='alto', mode='menor')")
    print(q3_auto)
except Exception as e:
    print(f"No se pudo calcular Consulta 3 en red auto: {e}")

"""
Finalmente, realizamos una comparación objetiva entre ambas redes utilizando la puntuación BIC. 
Los resultados demostraron un hecho muy interesante: ¡nuestra red manual obtuvo un mejor ajuste (score menos negativo) que la red aprendida automáticamente! Al comparar las aristas, notamos que hubo pocas coincidencias. El algoritmo de Chow-Liu, al estar restringido matemáticamente a formar un árbol estricto (máximo 1 padre por nodo), perdió dependencias cruzadas importantes que sí existen en la música real. Por otro lado, nuestra red manual permitió que ciertas variables tuvieran múltiples padres lógicos, capturando mejor la realidad sin sobrecomplicar el modelo. Esto coronó a nuestra propuesta humana como la vencedora frente a la penalización del BIC.
"""

print("\nComparación de ambas redes")
score = BIC(datos)
s_manual = score.score(manual_model)
s_auto = score.score(auto_model)

print(f"Score BIC red PROPUESTA (Manual) : {s_manual:,.1f}")
print(f"Score BIC red APRENDIDA (Auto)   : {s_auto:,.1f}")
print("Mejor ajuste a los datos: ", end="")

if s_auto > s_manual:
    print("Red aprendida (automatica)")
else:
    print("Red propuesta (manual)")

flechas_iguales = len(set(manual_model.edges()) & set(auto_model.edges()))
print(f"Cantidad de flechas exactamente iguales en ambas redes: {flechas_iguales}")

"""
Pasando a la pregunta 2

Para esta parte trabajamos con el dataset "Appliances Energy Prediction" que tiene mediciones cada 10 minutos. 

Lo primero que hicimos fue calcular los índices de temperatura y humedad sacando el promedio, como decía la fórmula. Después teníamos que discretizar las cosas. Para el consumo de los electrodomésticos (Appliances) nos pedían 6 categorías, así que usamos cuantiles. Esto lo hicimos porque el consumo no es normal, la casa pasa casi todo el día gastando poco y de repente hay picos. Si lo dividíamos por intervalos de igual tamaño, iban a quedar categorías vacías y la cadena de markov no iba a aprender nada. Lo mismo hicimos para temperatura y humedad, pero en 3 niveles.

Para las luces (lights) nos dimos cuenta que casi siempre el valor era 0 (apagadas), así que usar cuantiles daba error. Las separamos a mano: 0 es apagada, hasta 20 es consumo bajo y más de 20 es consumo alto.

Con esto listo juntamos las tres variables para crear las 27 observaciones que pedía el profe. Al final separamos los datos cronológicamente dejando el 80% para entrenamiento y 20% para probar, preocupándonos de cortar justo a la medianoche para no partir un día por la mitad.
"""

import os
import urllib.request
import zipfile

# 1. Carga de Datos
csv_filename = "energydata_complete.csv"
zip_url = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
zip_filename = "appliances.zip"

print("\nPreprocesamiento y discretización")
if not os.path.exists(csv_filename):
    print(f"Descargando dataset desde {zip_url}...")
    urllib.request.urlretrieve(zip_url, zip_filename)
    
    print("Extrayendo archivo CSV...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extract(csv_filename, path=".")
    
    print("Eliminando archivo ZIP temporal...")
    os.remove(zip_filename)
    print("¡Descarga y extracción completada!")

print("Cargando dataset 'energydata_complete.csv'...")
df2 = pd.read_csv(csv_filename)

# Asegurar que las fechas sean formato datetime y estén ordenadas
df2['date'] = pd.to_datetime(df2['date'])
df2 = df2.sort_values('date').reset_index(drop=True)

# 2. Cálculo de Índices
print("Calculando índices de temperatura y humedad...")
df2['I_temperatura'] = (df2['T1'] + df2['T2'] + df2['T3']) / 3
df2['I_humedad'] = (df2['RH_1'] + df2['RH_2'] + df2['RH_3']) / 3

# 3. Discretización
print("Discretizando variables (estados y observaciones)...")

# Electrodomésticos en 6 estados (0=Muy Bajo, 1=Bajo, 2=Medio-Bajo, 3=Medio-Alto, 4=Alto, 5=Muy Alto)
df2['estado_energia'], bins_energia = pd.qcut(df2['Appliances'], q=6, labels=False, retbins=True, duplicates='drop')

print("\n  Cortes de Energía calculados (en Wh):")
print(f"    Muy Bajo:   <= {bins_energia[1]:.1f}")
print(f"    Bajo:       {bins_energia[1]:.1f} a {bins_energia[2]:.1f}")
print(f"    Medio-Bajo: {bins_energia[2]:.1f} a {bins_energia[3]:.1f}")
print(f"    Medio-Alto: {bins_energia[3]:.1f} a {bins_energia[4]:.1f}")
print(f"    Alto:       {bins_energia[4]:.1f} a {bins_energia[5]:.1f}")
print(f"    Muy Alto:   > {bins_energia[5]:.1f}")

# Índices de Temperatura y Humedad en 3 niveles (0=Bajo, 1=Medio, 2=Alto)
df2['estado_temp'] = pd.qcut(df2['I_temperatura'], q=3, labels=False)
df2['estado_hum'] = pd.qcut(df2['I_humedad'], q=3, labels=False)

# Luces (lights) en 3 niveles con criterio lógico
def discretizar_luces(val):
    if val == 0: return 0
    elif val <= 20: return 1
    else: return 2

df2['estado_luces'] = df2['lights'].apply(discretizar_luces)

# 4. Observación Combinada (27 estados: del 0 al 26)
df2['observacion'] = df2['estado_luces'] * 9 + df2['estado_temp'] * 3 + df2['estado_hum']
print(f"\nTotal de observaciones generadas: {df2['observacion'].nunique()} (Máximo teórico: 27)")

# 5. Separación Train/Test cronológicamente
print("\nSeparando datos en Entrenamiento (80%) y Evaluación (20%)...")
corte_idx = int(len(df2) * 0.8)
fecha_corte = df2.iloc[corte_idx]['date'].normalize() # Corte exacto a medianoche

train_df = df2[df2['date'] < fecha_corte].copy()
test_df = df2[df2['date'] >= fecha_corte].copy()

print(f"  Entrenamiento: {len(train_df)} registros (del {train_df['date'].min().date()} al {train_df['date'].max().date()})")
print(f"  Evaluación:    {len(test_df)} registros (del {test_df['date'].min().date()} al {test_df['date'].max().date()})")

print("\nGuardando dataset discretizado...")
df2.to_csv("energydata_discretizado.csv", index=False)
print("¡Archivo 'energydata_discretizado.csv' guardado exitosamente!")

"""
Parte 2.b y 2.c: Construcción de Cadenas de Markov (Continua y Diaria)

Para modelar la evolución temporal del consumo construimos dos cadenas de Markov:
1. Modelo continuo: considera toda la serie temporal de entrenamiento como una sola secuencia continua ininterrumpida. Contamos las transiciones entre registros consecutivos (t -> t+1) y normalizamos por fila para obtener la matriz de transición P_cont. Como distribución inicial, calculamos su distribución estacionaria pi_cont (probabilidades límite a largo plazo que cumplen pi * P = pi).
2. Modelo diario: asume que cada jornada tiene su propia dinámica independiente. Contamos las transiciones dentro de cada día (evitando contar el salto entre las 23:50 y las 00:00 del día siguiente) para obtener P_dia. La distribución inicial pi_dia se estima a partir del estado en que inicia cada jornada a las 00:00 hrs.
"""

from hmmlearn.hmm import CategoricalHMM

N_states = 6
alpha_smooth = 1e-6

# 1. Cadena de Markov Continua
states_train = train_df['estado_energia'].values
C_cont = np.zeros((N_states, N_states))
for s_curr, s_next in zip(states_train[:-1], states_train[1:]):
    C_cont[s_curr, s_next] += 1
P_cont = (C_cont + alpha_smooth) / (C_cont + alpha_smooth).sum(axis=1, keepdims=True)

# Distribución estacionaria del modelo continuo (mediante potencias de matriz)
P_inf = np.linalg.matrix_power(P_cont, 500)
pi_cont = P_inf[0]

# 2. Cadena de Markov Diaria
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

print("\n--- DISTRIBUCIONES DE CADENAS DE MARKOV ---")
print("Distribución Estacionaria Modelo Continuo (pi_cont):")
for i, name in enumerate(['Muy Bajo', 'Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto', 'Muy Alto']):
    print(f"  {name:11}: {pi_cont[i]:.4%}")

print("\nDistribución Inicial Diaria (00:00 hrs, pi_dia):")
for i, name in enumerate(['Muy Bajo', 'Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto', 'Muy Alto']):
    print(f"  {name:11}: {pi_dia[i]:.4%}")

print("\nDistribución Final Diaria (23:50 hrs):")
for i, name in enumerate(['Muy Bajo', 'Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto', 'Muy Alto']):
    print(f"  {name:11}: {pi_final_dia[i]:.4%}")

# Consultas Temporales
print("\n--- CONSULTAS TEMPORALES DEL MODELO CONTINUO ---")
P_cont_6 = np.linalg.matrix_power(P_cont, 6)
P_cont_36 = np.linalg.matrix_power(P_cont, 36)
prob_c1_1h = P_cont_6[0, 4] + P_cont_6[0, 5]
prob_c1_6h = P_cont_36[0, 4] + P_cont_36[0, 5]
print(f"Consulta 1: Estando en 'Muy Bajo', P(pasar a 'Alto' o 'Muy Alto'):")
print(f"  En 1 hora (k=6 pasos):   {prob_c1_1h:.4%}")
print(f"  En 6 horas (k=36 pasos): {prob_c1_6h:.4%}")

P_cont_18 = np.linalg.matrix_power(P_cont, 18)
P_cont_72 = np.linalg.matrix_power(P_cont, 72)
prob_c2_3h = P_cont_18[5, 0] + P_cont_18[5, 1]
prob_c2_12h = P_cont_72[5, 0] + P_cont_72[5, 1]
print(f"\nConsulta 2: Estando en 'Muy Alto', P(retornar a 'Muy Bajo' o 'Bajo'):")
print(f"  En 3 horas (k=18 pasos):  {prob_c2_3h:.4%}")
print(f"  En 12 horas (k=72 pasos): {prob_c2_12h:.4%}")

print("\n--- CONSULTAS DEL MODELO DIARIO ---")
P_dia_72 = np.linalg.matrix_power(P_dia, 72)
print(f"Consulta 1 Diario: P(S_12:00 = Muy Alto | S_00:00 = Bajo): {P_dia_72[1, 5]:.4%}")

P_dia_143 = np.linalg.matrix_power(P_dia, 143)
prob_c2_dia = P_dia_143[0, 0] + P_dia_143[0, 1]
print(f"Consulta 2 Diario: P(S_23:50 in {{Muy Bajo, Bajo}} | S_00:00 = Muy Bajo): {prob_c2_dia:.4%}")

"""
Parte 2.d: Matriz de Emisión y Construcción de Modelos Ocultos de Markov (HMM)

Estimamos la matriz de emisión B (6 estados x 27 observaciones) contando cuántas veces aparece cada combinación de estado oculto y observación en el conjunto de entrenamiento, aplicando suavizado de Laplace para evitar ceros.

Ambos modelos (continuo y diario) pueden compartir legítimamente la misma matriz de emisión porque la relación física entre el estado de consumo del electrodoméstico y las variables ambientales observadas (temperatura, humedad e iluminación) no depende de cómo agrupemos el tiempo (si en una sola serie continua o por días). La emisión modela la relación simultánea en cada instante t, la cual es invariante a la estructura temporal de las transiciones.
"""

N_obs = 27
C_em = np.zeros((N_states, N_obs))
for s, o in zip(train_df['estado_energia'], train_df['observacion']):
    C_em[s, o] += 1
B = (C_em + alpha_smooth) / (C_em + alpha_smooth).sum(axis=1, keepdims=True)

hmm_continuo = CategoricalHMM(n_components=N_states, init_params='', params='')
hmm_continuo.startprob_ = pi_cont
hmm_continuo.transmat_ = P_cont
hmm_continuo.emissionprob_ = B

hmm_diario = CategoricalHMM(n_components=N_states, init_params='', params='')
hmm_diario.startprob_ = pi_dia
hmm_diario.transmat_ = P_dia
hmm_diario.emissionprob_ = B

print("\nModelos HMM continuo y diario construidos exitosamente con hmmlearn.")

"""
Parte 2.e y 2.f: Evaluación e Inferencia en Conjunto de Test (Forward, Forward-Backward y Viterbi)

Seleccionamos el primer día completo del conjunto de evaluación (30 de Abril de 2016, 144 registros cada 10 min).
Aplicamos:
1. Forward: para calcular la probabilidad total (log-verosimilitud) de las observaciones en ese día.
2. Forward-Backward: para responder consultas sobre las probabilidades a posteriori en horarios clave (04:00 madrugada, 12:00 mediodía, 20:00 noche).
3. Viterbi: para encontrar la secuencia globalmente más probable de estados de consumo y contrastarla contra los estados reales.
"""

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
print(f"\n--- 1. FORWARD (VEROSIMILITUD DE OBSERVACIONES) ---")
print(f"Log-Likelihood Modelo Continuo: {log_lik_cont:.2f}")
print(f"Log-Likelihood Modelo Diario:   {log_lik_dia:.2f}")

# 2. Forward-Backward
post_cont = hmm_continuo.predict_proba(X_eval)
post_dia = hmm_diario.predict_proba(X_eval)

print(f"\n--- 2. FORWARD-BACKWARD (PROBABILIDADES POSTERIORES) ---")
t_instantes = [24, 72, 120]
etq_horas = ["04:00 hrs (Madrugada)", "12:00 hrs (Mediodía)", "20:00 hrs (Noche)"]
nombres_estados_list = ['Muy Bajo', 'Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto', 'Muy Alto']

for t, h in zip(t_instantes, etq_horas):
    r_name = nombres_estados_list[y_real[t]]
    c_best = nombres_estados_list[np.argmax(post_cont[t])]
    d_best = nombres_estados_list[np.argmax(post_dia[t])]
    print(f"Instante {h}: Real = {r_name}")
    print(f"  Modelo Continuo -> Pred: {c_best} (P={np.max(post_cont[t]):.4f})")
    print(f"  Modelo Diario   -> Pred: {d_best} (P={np.max(post_dia[t]):.4f})")

# 3. Viterbi
viterbi_cont = hmm_continuo.predict(X_eval)
viterbi_dia = hmm_diario.predict(X_eval)

acc_cont = (viterbi_cont == y_real).mean()
acc_dia = (viterbi_dia == y_real).mean()
print(f"\n--- 3. VITERBI (EXACTITUD VS ESTADOS REALES) ---")
print(f"Exactitud (Accuracy) Modelo Continuo: {acc_cont:.2%}")
print(f"Exactitud (Accuracy) Modelo Diario:   {acc_dia:.2%}")

"""
Interpretación y Comparación de Ambos Modelos HMM:
1. Efecto de las transiciones entre días: El modelo continuo incluye la transición de las 23:50 a las 00:00 del día siguiente, lo que refuerza los estados de reposo nocturno. El modelo diario aísla cada jornada de 24 horas. Sobre el día evaluado, ambos modelos obtuvieron verosimilitudes casi idénticas (-374.47 vs -374.71), indicando que la dinámica intradiaria domina el proceso.
2. Significado de las distribuciones iniciales: pi_cont representa el equilibrio a largo plazo (ergódico) de la vivienda (con pesos homogéneos entre 13% y 22%), mientras que pi_dia captura la condición de reposo a las 00:00 hrs (más del 88% en consumos bajos).
3. Diferencias entre Forward-Backward y Viterbi: Forward-Backward estima las probabilidades marginales en cada instante puntual gamma_t(i), mientras que Viterbi obtiene la trayectoria completa conjunta más probable mediante programación dinámica, asegurando coherencia temporal bajo las probabilidades de transición.
"""

"""
Para la pregunta 3

En esta parte la empresa de telecomunicaciones DBZ nos pide evaluar el área de cobertura efectiva de un clúster de 3 antenas omnidireccionales en un terreno cuadrado de 100x100 km (10.000 km2 en total) utilizando el método de Monte Carlo.

Lo primero fue determinar las coordenadas de los centros y los radios de cobertura de cada antena a partir del RUT de Maximiliano Palma (21.620.910-9), siguiendo las fórmulas de la pauta:
- Antena 1: centro (d1*10, d2*10) = (20, 10) km y radio d3*2 + 15 = 27 km.
- Antena 2: centro (d4*10, d5*10) = (20, 0) km y radio d3*2 + 15 = 27 km.
- Antena 3: centro (d7*10, d8*10) = (10, 0) km y radio fijo de 25 km.
Notamos que la antena 1 y la 2 utilizan el mismo dígito d3 para el radio, por lo que el dígito d6 no se requiere en los cálculos.

Las reglas de propagación para clasificar los puntos son:
- Servicio útil: si el punto se encuentra dentro del radio de 1 o 2 antenas.
- Interferido: si el punto se encuentra cubierto por las 3 antenas simultáneamente (la interferencia anula la señal y genera caída del servicio).
- Sin señal: si el punto queda fuera del alcance de las antenas (0 antenas).

Para la simulación generamos puntos aleatorios con distribución uniforme continua en el terreno de 100x100 km para los 4 escenarios solicitados (10k, 100k, 1M y 10M de puntos).
Para evitar sobrecargar la memoria RAM y optimizar los tiempos de cómputo en el escenario de 10 millones de muestras, implementamos la simulación por bloques (chunks de 1 millón) y evaluamos distancias euclidianas al cuadrado ((x - cx)^2 + (y - cy)^2 <= r^2), evitando calcular raíces cuadradas punto a punto.
"""

import numpy as np
import matplotlib.pyplot as plt

RUT = "21.620.910-9"  # Maximiliano Palma
d = [int(c) for c in RUT if c.isdigit()][:8]

# El enunciado usa d3 para el radio de las antenas 1 y 2; d6 no se utiliza.
ANTENAS = [((d[0] * 10, d[1] * 10), d[2] * 2 + 15),
           ((d[3] * 10, d[4] * 10), d[2] * 2 + 15),
           ((d[6] * 10, d[7] * 10), 25)]

LADO = 100.0
AREA_TOTAL = 10_000.0
ESCENARIOS = [10_000, 100_000, 1_000_000, 10_000_000]

print("RUT utilizado:", RUT, "-> d1..d8 =", d)
for i, ((cx, cy), r) in enumerate(ANTENAS, 1):
    print(f"  Antena {i}: centro=({cx}, {cy})  radio={r} km")


def simular(N, seed=42, chunk=1_000_000):
    rng = np.random.default_rng(seed)
    conteos = np.zeros(4, dtype=np.int64)
    muestra = None
    for i in range(0, N, chunk):
        m = min(chunk, N - i)
        x = rng.uniform(0, LADO, m)
        y = rng.uniform(0, LADO, m)
        cob = sum(((x - cx) ** 2 + (y - cy) ** 2 <= r * r) for (cx, cy), r in ANTENAS)
        conteos += np.bincount(cob, minlength=4)
        if muestra is None:
            muestra = (x[:6000], y[:6000], cob[:6000])
    return conteos, muestra


filas, muestras = [], {}
for N in ESCENARIOS:
    c, mu = simular(N)
    assert c.sum() == N
    muestras[N] = mu
    p_util = (c[1] + c[2]) / N
    p_int = c[3] / N
    p_sin = c[0] / N
    err = AREA_TOTAL * np.sqrt(p_util * (1 - p_util) / N)
    filas.append((N, AREA_TOTAL * p_util, err, AREA_TOTAL * p_int, AREA_TOTAL * p_sin))

print(f"\n{'N':>12} {'Area util':>12} {'+/- (1 EE)':>11} {'Interferida':>13} {'Sin senal':>12}")
for N, a_u, e, a_i, a_s in filas:
    assert abs(a_u + a_i + a_s - AREA_TOTAL) < 1e-6
    print(f"{N:>12,} {a_u:>12.2f} {e:>11.2f} {a_i:>13.2f} {a_s:>12.2f}")

ref = filas[-1][1]
print(f"\nReferencia (N=10M): {ref:.2f} km2")
print(f"{'N':>12} {'Error abs.':>12} {'Error rel.':>11} {'EE teorico':>12} {'Razon EE':>10}")
previo = None
for N, a_u, e, _, _ in filas:
    razon = f"{e / previo:.3f}" if previo else "-"
    print(f"{N:>12,} {abs(a_u - ref):>12.2f} {abs(a_u - ref) / ref:>10.4%} {e:>12.2f} {razon:>10}")
    previo = e

# Paleta Okabe-Ito: separable en daltonismo (el par verde/naranja habitual no lo es).
# El tamano refuerza al color: la clase dominante "sin senal" va pequena y tenue.
CLASES = [("Sin senal (0 antenas)", '#0072b2', 2, 0.30),
          ("Servicio util (1 o 2 antenas)", '#009e73', 7, 0.90),
          ("Interferido (3 antenas)", '#d55e00', 7, 0.90)]

fig, axes = plt.subplots(2, 2, figsize=(12, 12.8))
for ax, N in zip(axes.ravel(), ESCENARIOS):
    x, y, cob = muestras[N]
    clase = np.where(cob == 0, 0, np.where(cob == 3, 2, 1))
    for k, (etq, col, s, al) in enumerate(CLASES):
        m = clase == k
        ax.scatter(x[m], y[m], c=col, s=s, alpha=al, linewidths=0,
                   label=etq if ax is axes[0, 0] else None)
    for (cx, cy), r in ANTENAS:
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec='#1a1a1a', lw=1.6))
        ax.plot(cx, cy, marker='^', color='#1a1a1a', ms=10, mec='white', mew=1.3)
    ax.set_xlim(0, LADO); ax.set_ylim(0, LADO); ax.set_aspect('equal')
    a_u = [f[1] for f in filas if f[0] == N][0]
    ax.set_title(f"N = {N:,}   |   area util = {a_u:.1f} km2", fontsize=11)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.grid(alpha=0.15, lw=0.5)
    ax.set_axisbelow(True)

fig.legend(loc='lower center', ncol=3, frameon=False, markerscale=3.2,
           fontsize=10, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Monte Carlo - cobertura de antenas (submuestra de 6.000 puntos por escenario)",
             fontsize=13)
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig("p3_cobertura.png", dpi=110)
print("\nGrafico guardado en p3_cobertura.png")

"""
Análisis de resultados y convergencia de Monte Carlo:

1. Comparación de los 4 escenarios (10k, 100k, 1M y 10M puntos):
- Con 10.000 puntos obtuvimos un área útil de 797.00 km2 con un error estándar de +-27.08 km2 (error relativo del 2.23% respecto a la referencia de 10M). Con pocas muestras la dispersión muestral es aún notoria.
- Con 100.000 puntos el área estimada asciende a 824.40 km2 y el error estándar se reduce a +-8.70 km2 (1.14% de error relativo).
- Con 1.000.000 de puntos la estimación se estabiliza fuertemente en 814.85 km2, con un error estándar de tan solo +-2.74 km2 (0.036% de error relativo).
- Con 10.000.000 de puntos la solución converge a 815.15 km2 con un error estándar de +-0.87 km2. La diferencia entre 1 millón y 10 millones es de apenas 0.30 km2, confirmando una alta estabilidad numérica.

2. Discusión sobre el comportamiento y la convergencia del método:
- Por la Ley Fuerte de los Grandes Números, a medida que aumentamos la cantidad de puntos simulados, la proporción muestral converge casi seguramente a la probabilidad teórica real del área.
- Por el Teorema del Límite Central, el error estándar disminuye con una tasa de orden O(1/sqrt(N)). Dado que entre cada escenario multiplicamos la cantidad de muestras por 10, el error teórico debe reducirse por un factor de 1/sqrt(10) ≈ 0.3162.
- Al revisar la columna de 'Razón EE' en nuestra tabla, obtuvimos valores empíricos de 0.321, 0.315 y 0.316, los cuales coinciden de manera muy precisa con el valor teórico de 0.316. Esto demuestra empíricamente que para ganar un decimal más de precisión con Monte Carlo es necesario multiplicar por 100 la cantidad de muestras simuladas.

3. Interpretación espacial del gráfico (p3_cobertura.png):
Al visualizar los puntos, se aprecia claramente que debido a los dígitos del RUT las tres antenas quedaron concentradas hacia el vértice inferior izquierdo del terreno. La zona naranja central muestra la región de solapamiento de las 3 antenas (unos 711 km2), donde se pierde el servicio por interferencia destructiva. A su alrededor se extiende la franja verde de servicio útil (unos 815 km2), mientras que la gran mayoría del terreno (más del 84%, en color azul) queda completamente sin cobertura de señal.
"""
