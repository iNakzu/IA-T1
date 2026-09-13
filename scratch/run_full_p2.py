import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import CategoricalHMM

# 1. Carga de datos
csv_filename = "energydata_complete.csv"
if not os.path.exists(csv_filename):
    zip_url = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
    zip_filename = "appliances.zip"
    urllib.request.urlretrieve(zip_url, zip_filename)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extract(csv_filename, path=".")
    os.remove(zip_filename)

df = pd.read_csv(csv_filename)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2. Indices
df['I_temperatura'] = (df['T1'] + df['T2'] + df['T3']) / 3
df['I_humedad'] = (df['RH_1'] + df['RH_2'] + df['RH_3']) / 3

# 3. Discretizacion
# 6 estados de consumo Appliances
bins_energia = [0, 45, 55, 65, 85, 125, 2000]
nombres_estados = ['Muy Bajo', 'Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto', 'Muy Alto']
df['estado_energia'] = pd.cut(df['Appliances'], bins=bins_energia, labels=[0, 1, 2, 3, 4, 5]).astype(int)

# 3 niveles para temperatura y humedad
df['estado_temp'] = pd.qcut(df['I_temperatura'], q=3, labels=[0, 1, 2]).astype(int)
df['estado_hum'] = pd.qcut(df['I_humedad'], q=3, labels=[0, 1, 2]).astype(int)

# 3 niveles para lights: 0=apagada, 1=bajo (10-20), 2=alto (>20)
def discretizar_luces(val):
    if val == 0: return 0
    elif val <= 20: return 1
    else: return 2

df['estado_luces'] = df['lights'].apply(discretizar_luces)

# 27 observaciones combinadas: base 3
df['observacion'] = df['estado_luces'] * 9 + df['estado_temp'] * 3 + df['estado_hum']

# 4. Separacion Train / Test cronologica (80% / 20%)
corte_idx = int(len(df) * 0.8)
fecha_corte = df.iloc[corte_idx]['date'].normalize()
train_df = df[df['date'] < fecha_corte].copy()
test_df = df[df['date'] >= fecha_corte].copy()

print(f"Train: {len(train_df)} registros ({train_df['date'].min().date()} a {train_df['date'].max().date()})")
print(f"Test:  {len(test_df)} registros ({test_df['date'].min().date()} a {test_df['date'].max().date()})")

# PARTE B: Cadenas de Markov (continua y diaria)
N_states = 6
alpha = 1e-6 # Suavizado

# Cadena Continua
states_train = train_df['estado_energia'].values
C_cont = np.zeros((N_states, N_states))
for s_curr, s_next in zip(states_train[:-1], states_train[1:]):
    C_cont[s_curr, s_next] += 1
P_cont = (C_cont + alpha) / (C_cont + alpha).sum(axis=1, keepdims=True)

# Distribucion estacionaria
vals, vecs = np.linalg.eig(P_cont.T)
idx = np.argmin(np.abs(vals - 1.0))
pi_cont = np.real(vecs[:, idx])
pi_cont = np.maximum(pi_cont, 1e-6)
pi_cont = pi_cont / pi_cont.sum()

# Cadena Diaria
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

P_dia = (C_dia + alpha) / (C_dia + alpha).sum(axis=1, keepdims=True)
pi_dia = (np.bincount(primeros_estados, minlength=N_states) + alpha)
pi_dia = pi_dia / pi_dia.sum()

pi_final_dia = np.bincount(ultimos_estados, minlength=N_states) / len(ultimos_estados)

print("\n--- DISTRIBUCIONES INICIALES ---")
print("Distribucion estacionaria pi_cont:")
for i, name in enumerate(nombres_estados):
    print(f"  {name:11}: {pi_cont[i]:.4f}")

print("\nDistribucion inicial diaria pi_dia (00:00 hrs):")
for i, name in enumerate(nombres_estados):
    print(f"  {name:11}: {pi_dia[i]:.4f}")

print("\nDistribucion final diaria (23:50 hrs):")
for i, name in enumerate(nombres_estados):
    print(f"  {name:11}: {pi_final_dia[i]:.4f}")

# PARTE C: Consultas
print("\n--- CONSULTAS MODELO CONTINUO ---")
# Consulta 1: Si S_t = Muy Bajo (0), P(S_{t+k} in {Alto, Muy Alto}) para 1h (k=6) y 6h (k=36)
P_cont_6 = np.linalg.matrix_power(P_cont, 6)
P_cont_36 = np.linalg.matrix_power(P_cont, 36)
prob_c1_1h = P_cont_6[0, 4] + P_cont_6[0, 5]
prob_c1_6h = P_cont_36[0, 4] + P_cont_36[0, 5]
print(f"Consulta 1: Estando en 'Muy Bajo', P(pasar a 'Alto' o 'Muy Alto'):")
print(f"  En 1 hora (k=6 pasos):  {prob_c1_1h:.4%}")
print(f"  En 6 horas (k=36 pasos): {prob_c1_6h:.4%}")

# Consulta 2: Si S_t = Muy Alto (5), P(retornar a reposo {Muy Bajo, Bajo}) para 3h (k=18) y 12h (k=72)
P_cont_18 = np.linalg.matrix_power(P_cont, 18)
P_cont_72 = np.linalg.matrix_power(P_cont, 72)
prob_c2_3h = P_cont_18[5, 0] + P_cont_18[5, 1]
prob_c2_12h = P_cont_72[5, 0] + P_cont_72[5, 1]
print(f"Consulta 2: Estando en 'Muy Alto', P(volver a reposo 'Muy Bajo' o 'Bajo'):")
print(f"  En 3 horas (k=18 pasos):  {prob_c2_3h:.4%}")
print(f"  En 12 horas (k=72 pasos): {prob_c2_12h:.4%}")

print("\n--- CONSULTAS MODELO DIARIO ---")
# Consulta 1 Diario: Si empieza en Bajo (1) a las 00:00, prob de estar en Muy Alto (5) a mediodia (12:00, k=72)
P_dia_72 = np.linalg.matrix_power(P_dia, 72)
print(f"Consulta 1 Diario: P(S_12:00 = Muy Alto | S_00:00 = Bajo): {P_dia_72[1, 5]:.4%}")

# Consulta 2 Diario: Si empieza en Muy Bajo (0), prob de terminar la jornada (23:50, k=143) en reposo (0 o 1)
P_dia_143 = np.linalg.matrix_power(P_dia, 143)
prob_c2_dia = P_dia_143[0, 0] + P_dia_143[0, 1]
print(f"Consulta 2 Diario: P(S_23:50 in {{Muy Bajo, Bajo}} | S_00:00 = Muy Bajo): {prob_c2_dia:.4%}")

# PARTE D: Matriz de Emision y Construccion HMM
N_obs = 27
C_em = np.zeros((N_states, N_obs))
for s, o in zip(train_df['estado_energia'], train_df['observacion']):
    C_em[s, o] += 1
B = (C_em + alpha) / (C_em + alpha).sum(axis=1, keepdims=True)

hmm_cont = CategoricalHMM(n_components=N_states, init_params='', params='')
hmm_cont.startprob_ = pi_cont
hmm_cont.transmat_ = P_cont
hmm_cont.emissionprob_ = B

hmm_dia = CategoricalHMM(n_components=N_states, init_params='', params='')
hmm_dia.startprob_ = pi_dia
hmm_dia.transmat_ = P_dia
hmm_dia.emissionprob_ = B

print("\n--- HMM CONSTRUIDOS ---")
print("Modelo continuo y diario creados con hmmlearn.")

# PARTE E: Evaluacion en 1 dia de test
test_df['day'] = test_df['date'].dt.date
dias_test = test_df['day'].value_counts()
dia_sel = dias_test[dias_test == 144].index.sort_values()[0]
df_dia = test_df[test_df['day'] == dia_sel].sort_values('date').reset_index(drop=True)
print(f"\nDia seleccionado para evaluacion: {dia_sel} (144 mediciones)")

X_eval = df_dia['observacion'].values.reshape(-1, 1)
y_real = df_dia['estado_energia'].values

# 1. Forward
log_lik_cont = hmm_cont.score(X_eval)
log_lik_dia = hmm_dia.score(X_eval)
print(f"Forward Log-Likelihood Modelo Continuo: {log_lik_cont:.2f}")
print(f"Forward Log-Likelihood Modelo Diario:   {log_lik_dia:.2f}")

# 2. Forward-Backward
post_cont = hmm_cont.predict_proba(X_eval)
post_dia = hmm_dia.predict_proba(X_eval)

# Consultas en distintos instantes (04:00, 12:00, 20:00)
# Indices: 04:00 -> t = 4*6 = 24; 12:00 -> t = 12*6 = 72; 20:00 -> t = 20*6 = 120
t_indices = [24, 72, 120]
horas = ["04:00", "12:00", "20:00"]
print("\nProbabilidades posteriores (Forward-Backward) en instantes clave:")
for t, hora in zip(t_indices, horas):
    obs_t = df_dia.iloc[t]['observacion']
    real_t = nombres_estados[y_real[t]]
    pred_c = nombres_estados[np.argmax(post_cont[t])]
    pred_d = nombres_estados[np.argmax(post_dia[t])]
    print(f"  Hora {hora} (Obs={obs_t}, Real={real_t}):")
    print(f"    Continuo -> Estado mas probable: {pred_c} (Prob={np.max(post_cont[t]):.4f})")
    print(f"    Diario   -> Estado mas probable: {pred_d} (Prob={np.max(post_dia[t]):.4f})")

# 3. Viterbi
viterbi_cont = hmm_cont.predict(X_eval)
viterbi_dia = hmm_dia.predict(X_eval)

acc_cont = (viterbi_cont == y_real).mean()
acc_dia = (viterbi_dia == y_real).mean()
print(f"\nPrecision Viterbi vs Estados Reales:")
print(f"  Modelo Continuo: Accuracy = {acc_cont:.2%}")
print(f"  Modelo Diario:   Accuracy = {acc_dia:.2%}")
