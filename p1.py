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

print("\nComparación de ambas redes")
score = BIC(datos)
s_manual = score.score(manual_model)
s_auto = score.score(auto_model)

print(f"Score BIC red PROPUESTA (Manual) : {s_manual:,.1f}")
print(f"Score BIC red APRENDIDA (Auto)   : {s_auto:,.1f}")
print("Mejor ajuste a los datos:", "APRENDIDA" if s_auto > s_manual else "PROPUESTA")

flechas_iguales = len(set(manual_model.edges()) & set(auto_model.edges()))
print(f"Cantidad de flechas exactamente iguales en ambas redes: {flechas_iguales}")