import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.clustering import TimeSeriesKMeans

# Focaremos apenas na pasta de Zolder
TRACK_PATH = "../AccReplay/zolder/"
print(f"Iniciando coleta de dados em: {TRACK_PATH}")

all_curves_data = []
curve_sections_labels = []

# 1. Coleta de dados de TODOS os pilotos/corridas em Zolder
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
        # Pegamos a volta de referência (costuma ser a melhor volta)
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

        # Extraímos a série temporal de cada curva deste piloto
        grouped = df_curves.groupby('section')
        for section_name, group in grouped:
            
            # --- NORMALIZAÇÃO FÍSICA GLOBAL ---
            # Dividimos pelos limites máximos teóricos do carro/jogo.
            # O np.clip garante que nenhum bug de telemetria ultrapasse o limite de 0 a 1.
            
            # Velocidade: Assumindo 300 km/h como teto máximo para a escala
            speed_norm = np.clip(group['speed'].values / 300.0, 0, 1)
            
            # Pedais: Vão de 0 a 100 no AccReplay
            brake_norm = np.clip(group['brake'].values / 100.0, 0, 1)
            throttle_norm = np.clip(group['throttle'].values / 100.0, 0, 1)
            
            # Volante: Aplicamos o .abs() igual ao seu K-Means anterior e dividimos pelo esterço máximo
            # Assumindo 180 graus como limite de esterço prático na pista
            steering_norm = np.clip(group['steering'].abs().values / 180.0, 0, 1)

            # Matriz: [velocidade, freio, acelerador, volante] JÁ NORMALIZADA
            curve_matrix = np.column_stack((speed_norm, brake_norm, throttle_norm, steering_norm))
            
            all_curves_data.append(curve_matrix)
            curve_sections_labels.append(section_name)

    except Exception as e:
        print(f"Erro ao processar {run_id}: {e}")

print(f"Total de instâncias de curvas coletadas: {len(all_curves_data)}")

# 2. Preparação (SEM O MINMAX SCALER)
X_ts = to_time_series_dataset(all_curves_data)

# 3. Clusterização com DTW 
print("\nExecutando TimeSeriesKMeans (DTW) - Conjunto de Pilotos...")
# Passamos o X_ts DIRETAMENTE para o modelo
model = TimeSeriesKMeans(n_clusters=4, metric="dtw", random_state=42, n_jobs=-1)
labels = model.fit_predict(X_ts)

# 4. Resultados e Classificação das Curvas por Moda
results_df = pd.DataFrame({
    'section': curve_sections_labels,
    'cluster': labels
})

# Classificação final da pista (o que a maioria dos pilotos faz em cada curva)
final_classification = results_df.groupby('section')['cluster'].agg(lambda x: x.mode()[0]).reset_index()

print("\n--- Classificação Final da Pista (Zolder) ---")
print(final_classification.sort_values('cluster'))

# 5. Visualização dos Baricentros (Assinaturas Médias)
centros = model.cluster_centers_
plt.figure(figsize=(15, 10))
plt.suptitle("Assinaturas de Telemetria Consolidadas - Pista Zolder", fontsize=16)

labels_plot = ['Velocidade', 'Freio', 'Acelerador', 'Volante']
cores = ['blue', 'red', 'green', 'purple']

for i in range(4):
    plt.subplot(2, 2, i + 1)
    for j in range(4):
        plt.plot(centros[i, :, j], label=labels_plot[j], color=cores[j], linewidth=2)
    plt.title(f"Cluster {i}")
    plt.legend(loc="upper right", fontsize='small')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("zolder_dtw_sem_normalizar.png")
print("\nGráfico salvo como 'zolder_dtw_consolidado.png'. Analise o Cluster 2 novamente!")