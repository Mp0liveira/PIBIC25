import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Caminho base onde estão todas as pistas
BASE_PATH = "../AccReplay/"

# Lista para guardar os DataFrames de cada corrida antes de juntar tudo
all_features = []

print("A iniciar o processamento em lote...")

# 1. Iterar sobre todas as diretorias de pistas (ex: zolder, monza, spa)
for track_name in os.listdir(BASE_PATH):
    track_path = os.path.join(BASE_PATH, track_name)
    if not os.path.isdir(track_path):
        continue # Ignorar ficheiros soltos, procurar apenas pastas
        
    # 2. Iterar sobre os IDs das corridas (ex: 1969213)
    for run_id in os.listdir(track_path):
        run_path = os.path.join(track_path, run_id)
        if not os.path.isdir(run_path):
            continue
            
        telemetry_file = os.path.join(run_path, 'telemetry.json')
        sections_file = os.path.join(run_path, 'sections.json')
        
        # Verificar se os ficheiros JSON existem nesta pasta
        if not (os.path.exists(telemetry_file) and os.path.exists(sections_file)):
            print(f"Aviso: Ficheiros não encontrados em {run_path}")
            continue
            
        try:
            with open(telemetry_file, 'r', encoding='utf-8') as f:
                telemetry_data = json.load(f)
            
            with open(sections_file, 'r', encoding='utf-8') as f:
                sections_data = json.load(f)
        except Exception as e:
            print(f"Erro ao ler {run_path}: {e}")
            continue

        # Validar se existem canais de telemetria
        if len(telemetry_data.get('channels', [])) == 0:
            continue
            
        distances = telemetry_data['distances']
        lap_channels = telemetry_data['channels'][0]

        # ------------- Preparação do DataFrame --------------------
        df = pd.DataFrame({
            'distance': distances,
            'speed': lap_channels['speed'],
            'throttle': lap_channels['throttle'],
            'brake': lap_channels['brake'],
            'steering': lap_channels['steering'],
            'gear': lap_channels['gears'],
            'rpm': lap_channels['engineRPM']
        })

        df['delta_dist'] = df['distance'].diff()
        df['delta_steering'] = df['steering'].diff()
        df['speed_ms'] = df['speed'] / 3.6
        df['delta_time'] = df.apply(
            lambda row: row['delta_dist'] / row['speed_ms'] if (pd.notnull(row['speed_ms']) and row['speed_ms'] > 1) else 0.1, 
            axis=1
        )
        df['steering_velocity'] = df['delta_steering'] / df['delta_time']
        df.fillna(0, inplace=True)

        # Mapeamento das curvas
        def get_section_info(dist, sections):
            for sec in sections:
                if sec['splineStart'] <= dist <= sec['splineEnd']:
                    return sec['description']
            return None

        df['section'] = df['distance'].apply(lambda x: get_section_info(x, sections_data))
        df_curves = df.dropna(subset=['section'])

        # Se não houver curvas mapeadas, avançar para a próxima corrida
        if df_curves.empty:
            continue

        # Extração de Features
        features = df_curves.groupby('section').agg({
            'speed': ['min', 'mean', 'max'],
            'brake': ['max', 'mean'],
            'throttle': ['mean'],
            'gear': ['min'],
            'steering': ['std', lambda x: x.abs().max()], 
            'steering_velocity': [lambda x: x.abs().mean(), lambda x: x.abs().max()],
            'distance': lambda x: x.max() - x.min()
        }).reset_index()

        # Ajuste dos nomes das colunas
        features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in features.columns.values]
        features.rename(columns={
            'steering_<lambda_0>': 'steering_angle_max',
            'steering_velocity_<lambda_0>': 'hand_speed_mean',
            'steering_velocity_<lambda_1>': 'hand_speed_max',
            'distance_<lambda>': 'curve_length'
        }, inplace=True)
        
        # --- NOVO: Adicionar Identificadores ---
        features['track_name'] = track_name
        features['run_id'] = run_id
        
        all_features.append(features)

# ------------- Agrupar e Clusterizar Tudo --------------------
if not all_features:
    print("Erro: Não foi possível processar nenhuma corrida. Verifique os caminhos e ficheiros.")
    exit()

# Concatenar todos os resultados parciais num único DataFrame gigante
final_df = pd.concat(all_features, ignore_index=True)
print(f"\nTotal de curvas processadas: {len(final_df)}")

# Definir as colunas que vão alimentar a IA (ignorando textos/IDs)
cols_for_ai = [c for c in final_df.columns if c not in ['section', 'track_name', 'run_id']]

# Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(final_df[cols_for_ai])

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
final_df['cluster'] = kmeans.fit_predict(X_scaled)

# Guardar o ficheiro final
output_file = 'resultados_todas_corridas.csv'
final_df.to_csv(output_file, index=False)
print(f"Clusterização concluída com sucesso! Ficheiro guardado em: {output_file}")