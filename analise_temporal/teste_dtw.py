import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.clustering import TimeSeriesKMeans

PATH = "../AccReplay/zolder/1969213/"

print("A carregar ficheiros de telemetria...")
try:
    with open(PATH + 'telemetry.json', 'r', encoding='utf-8') as f:
        telemetry_data = json.load(f)
    with open(PATH + 'sections.json', 'r', encoding='utf-8') as f:
        sections_data = json.load(f)
except FileNotFoundError as e:
    print(f"Erro: {e}")
    exit()

distances = telemetry_data['distances']
lap_channels = telemetry_data['channels'][0]

df = pd.DataFrame({
    'distance': distances,
    'speed': lap_channels['speed'],
    'brake': lap_channels['brake'],
    'throttle': lap_channels['throttle'],
    'steering': lap_channels['steering']
})

def get_section_info(dist, sections):
    for sec in sections:
        if sec['splineStart'] <= dist <= sec['splineEnd']:
            return sec['description']
    return None

df['section'] = df['distance'].apply(lambda x: get_section_info(x, sections_data))
df_curves = df.dropna(subset=['section'])

# --- PASSO 1: FÍSICA COMPLETA ---
features_ts = df_curves.groupby('section').agg({
    'speed': list,
    'brake': list,
    'throttle': list,
    'steering': list
}).reset_index()

series_list = []
for index, row in features_ts.iterrows():
    # Agora a matriz tem 4 dimensões de telemetria
    curve_data = np.column_stack((row['speed'], row['brake'], row['throttle'], row['steering']))
    series_list.append(curve_data)

X_ts = to_time_series_dataset(series_list)
scaler = TimeSeriesScalerMinMax()
X_scaled = scaler.fit_transform(X_ts)

print("\nA executar TimeSeriesKMeans com DTW (Física Completa)...")
# Usamos n_jobs=-1 para usar todos os núcleos do processador (DTW é pesado!)
model = TimeSeriesKMeans(n_clusters=4, metric="dtw", random_state=42, n_jobs=-1)
features_ts['cluster'] = model.fit_predict(X_scaled)

features_ts['tamanho_pontos'] = features_ts['speed'].apply(len)
print("\n--- Resultado (DTW) ---")
print(features_ts[['section', 'tamanho_pontos', 'cluster']].sort_values('cluster').to_string(index=False))

# --- PASSO 3: VISUALIZAÇÃO GRÁFICA ---
print("\nA gerar gráficos dos Baricentros (Curvas Médias)...")

# O model.cluster_centers_ tem o formato (n_clusters, tamanho_tempo, n_features)
centros = model.cluster_centers_

plt.figure(figsize=(14, 10))
plt.suptitle("Assinaturas Médias de Telemetria por Cluster (Normalizadas 0 a 1)", fontsize=16)

cores = ['blue', 'red', 'green', 'purple']
labels = ['Velocidade', 'Travão', 'Acelerador', 'Volante']

for i in range(4):
    plt.subplot(2, 2, i + 1)
    for j in range(4):
        # Desenha a linha para cada feature (velocidade, travão, etc.) do cluster i
        plt.plot(centros[i, :, j], label=labels[j], color=cores[j], linewidth=2)
    
    plt.title(f"Cluster {i}")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
nome_ficheiro = "visualizacao_clusters_dtw.png"
plt.savefig(nome_ficheiro)
print(f"Gráfico guardado com sucesso como '{nome_ficheiro}' na pasta atual!")