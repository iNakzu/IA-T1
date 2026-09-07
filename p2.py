import pandas as pd
import numpy as np
import warnings

# Ignorar warnings de pandas
warnings.filterwarnings("ignore")

print("--- PARTE A: PREPROCESAMIENTO Y DISCRETIZACIÓN ---")

import os
import urllib.request
import zipfile

# 1. Carga de Datos
csv_filename = "energydata_complete.csv"
zip_url = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
zip_filename = "appliances.zip"

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
df = pd.read_csv(csv_filename)

# Asegurar que las fechas sean formato datetime y estén ordenadas
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2. Cálculo de Índices
print("Calculando índices de temperatura y humedad...")
df['I_temperatura'] = (df['T1'] + df['T2'] + df['T3']) / 3
df['I_humedad'] = (df['RH_1'] + df['RH_2'] + df['RH_3']) / 3

# 3. Discretización
print("Discretizando variables (estados y observaciones)...")

# Electrodomésticos (Appliances) en 6 estados usando cuantiles para asegurar balanceo
# Estados: 0=Muy Bajo, 1=Bajo, 2=Medio-Bajo, 3=Medio-Alto, 4=Alto, 5=Muy Alto
# pd.qcut puede fallar si hay muchos valores iguales en los bordes, usamos duplicates='drop' por seguridad
df['estado_energia'], bins_energia = pd.qcut(df['Appliances'], q=6, labels=False, retbins=True, duplicates='drop')

# Mostrar los cortes matemáticos
print("\n  Cortes de Energía calculados (en Wh):")
print(f"    Muy Bajo:   <= {bins_energia[1]:.1f}")
print(f"    Bajo:       {bins_energia[1]:.1f} a {bins_energia[2]:.1f}")
print(f"    Medio-Bajo: {bins_energia[2]:.1f} a {bins_energia[3]:.1f}")
print(f"    Medio-Alto: {bins_energia[3]:.1f} a {bins_energia[4]:.1f}")
print(f"    Alto:       {bins_energia[4]:.1f} a {bins_energia[5]:.1f}")
print(f"    Muy Alto:   > {bins_energia[5]:.1f}")

# Índices de Temperatura y Humedad en 3 niveles (0=Bajo, 1=Medio, 2=Alto)
df['estado_temp'] = pd.qcut(df['I_temperatura'], q=3, labels=False)
df['estado_hum'] = pd.qcut(df['I_humedad'], q=3, labels=False)

# Luces (lights) en 3 niveles con criterio lógico
# 0 = Apagada (0 Wh), 1 = Bajo (10-20 Wh), 2 = Alto (>= 30 Wh)
def discretizar_luces(val):
    if val == 0:
        return 0
    elif val <= 20:
        return 1
    else:
        return 2

df['estado_luces'] = df['lights'].apply(discretizar_luces)

# 4. Observación Combinada (27 estados: del 0 al 26)
# Usamos una fórmula posicional (base 3) para mapear 3x3x3 a un número único
df['observacion'] = df['estado_luces'] * 9 + df['estado_temp'] * 3 + df['estado_hum']

print(f"\nTotal de observaciones generadas: {df['observacion'].nunique()} (Máximo teórico: 27)")

# 5. Separación Train/Test cronológicamente
print("\nSeparando datos en Entrenamiento y Evaluación...")
# Calculamos el índice que representa el 80%
corte_idx = int(len(df) * 0.8)

# Buscamos la fecha de ese índice y le cortamos la hora (normalize)
# para que la separación sea un día exacto a las 00:00:00
fecha_corte = df.iloc[corte_idx]['date'].normalize()

train_df = df[df['date'] < fecha_corte].copy()
test_df = df[df['date'] >= fecha_corte].copy()

print(f"  Entrenamiento: {len(train_df)} registros (del {train_df['date'].min().date()} al {train_df['date'].max().date()})")
print(f"  Evaluación:    {len(test_df)} registros (del {test_df['date'].min().date()} al {test_df['date'].max().date()})")

print("\nGuardando dataset discretizado...")
df.to_csv("energydata_discretizado.csv", index=False)
print("¡Archivo 'energydata_discretizado.csv' guardado exitosamente!")
