import os
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

# Focaremos apenas na pasta de Zolder
TRACK_PATH = "../AccReplay/zolder/"
print(f"Iniciando coleta de dados em: {TRACK_PATH}")

# --- 1. Função do Método do Cotovelo para DTW ---
def melhor_k_cotovelo_dtw(X_ts, max_k=7):
    n_amostras = len(X_ts)
    limite_k = min(max_k + 1, n_amostras)
    if n_amostras < 4:
        return n_amostras
        
    inercias = []
    ks = list(range(2, limite_k))
    
    print("\nCalculando inércias para o Método do Cotovelo...")
    for k in ks:
        # n_jobs=-1 para usar todos os núcleos do processador
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42, n_jobs=-1)
        model.fit(X_ts)
        inercias.append(model.inertia_)
        print(f"  -> K={k} processado (Inércia: {model.inertia_:.2f})")
        
    p1 = np.array([ks[0], inercias[0]])
    p2 = np.array([ks[-1], inercias[-1]])
    
    maior_distancia = 0
    melhor_k = ks[0]
    
    for i in range(len(ks)):
        p3 = np.array([ks[i], inercias[i]])
        distancia = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        if distancia > maior_distancia:
            maior_distancia = distancia
            melhor_k = ks[i]
            
    # Garantimos pelo menos 3 clusters para evitar o underfitting extremo
    return max(3, melhor_k)

all_curves_data = []
curve_sections_labels = []

# --- 2. Coleta de dados e Normalização Global ---
for run_id in os.listdir(TRACK_PATH):
    run_path = os.path.join(TRACK_PATH, run_id)
    if not os.path.isdir(run_path):
        continue
        
    telemetry_file = os.path.join(run_path, 'telemetry.json')
    sections_file = os.path.join(run_path, 'sections.json')
    
    if not (os.path.exists(telemetry_file) and os.path.exists(sections_file)):
        continue
        
    try:
        with open(telemetry_file, 'r', encoding='utf-8') as f:
            telemetry_data = json.load(f)
        with open(sections_file, 'r', encoding='utf-8') as f:
            sections_data = json.load(f)
        
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

        grouped = df_curves.groupby('section')
        for section_name, group in grouped:
            
            # Normalização Física Global
            speed_norm = np.clip(group['speed'].values / 300.0, 0, 1)
            brake_norm = np.clip(group['brake'].values / 100.0, 0, 1)
            throttle_norm = np.clip(group['throttle'].values / 100.0, 0, 1)
            steering_norm = np.clip(group['steering'].abs().values / 180.0, 0, 1)

            curve_matrix = np.column_stack((speed_norm, brake_norm, throttle_norm, steering_norm))
            all_curves_data.append(curve_matrix)
            curve_sections_labels.append(section_name)

    except Exception as e:
        continue

print(f"\nTotal de instâncias de curvas coletadas: {len(all_curves_data)}")

X_ts = to_time_series_dataset(all_curves_data)

# --- 3. Descoberta do Melhor K e Clusterização ---
melhor_k = melhor_k_cotovelo_dtw(X_ts)
print(f"\n✅ O Método do Cotovelo definiu o K ideal como: {melhor_k}")

print("\nExecutando TimeSeriesKMeans Final...")
model = TimeSeriesKMeans(n_clusters=melhor_k, metric="dtw", random_state=42, n_jobs=-1)
labels = model.fit_predict(X_ts)

# --- 4. Resultados por Moda ---
results_df = pd.DataFrame({
    'section': curve_sections_labels,
    'cluster': labels
})

final_classification = results_df.groupby('section')['cluster'].agg(lambda x: x.mode()[0]).reset_index()
print("\n--- Classificação Final da Pista (Zolder) ---")
print(final_classification.sort_values('cluster').to_string(index=False))

# --- 5. Visualização Dinâmica dos Baricentros ---
centros = model.cluster_centers_

# Calcula quantas linhas o gráfico vai precisar (ex: se K=5, precisa de 3 linhas e 2 colunas)
cols = 2
rows = math.ceil(melhor_k / cols)

plt.figure(figsize=(15, 4 * rows))
plt.suptitle(f"Assinaturas de Telemetria (K={melhor_k} escolhido pelo Cotovelo)", fontsize=16)

labels_plot = ['Velocidade', 'Freio', 'Acelerador', 'Volante']
cores = ['blue', 'red', 'green', 'purple']

for i in range(melhor_k):
    plt.subplot(rows, cols, i + 1)
    for j in range(4):
        plt.plot(centros[i, :, j], label=labels_plot[j], color=cores[j], linewidth=2)
    plt.title(f"Cluster {i}")
    
    # Coloca a legenda apenas no primeiro gráfico para não poluir os outros
    if i == 0:
        plt.legend(loc="upper right", fontsize='small')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"zolder_dtw_K{melhor_k}.png")
print(f"\nGráfico salvo como 'zolder_dtw_K{melhor_k}.png'!")