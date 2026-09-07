"""
Tarea 1 Inteligencia Artificial. 
Estudiantes: Ignacio Antiguay, Vicente Hernandez 
Profesores: Victor Reyes, Felipe Lazo 
Se utilizaron herramientas generativas (Gemini,Claude) como apoyo técnico para escritura y depuración de código Python, revisión del uso de librerías, organización metodológica y revisión de resultados.

Primeramente escogimos como grupo el dataset Spotify Tracks de la plataforma Kaggle, este venia con 114.000 filas y 22 columnas, con la siguiente idea reducimos las filas a 12.000 y 12 columnas y seleccionamos las siguientes columnas:
"""

#!pip install kagglehub pandas pgmpy

# Inicialización del dataset Spotify Tracks desde Kaggle
import pandas as pd
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