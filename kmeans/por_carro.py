# Esse arquivo faz a classificação das curvas por pista
# A classificação é feita de duas formas, global e por carro
import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

def encontrar_melhor_k_cotovelo(X_scaled, max_k=8):
    n_amostras = len(X_scaled)
    limite_k = min(max_k + 1, n_amostras)
    
    if n_amostras < 4:
        return n_amostras
        
    inercias = []
    ks = list(range(2, limite_k))
    
    # 1. Calcula a inércia para cada K
    for k in ks:
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(X_scaled)
        inercias.append(kmeans_temp.inertia_)
        
    # 2. Matemática para achar o "Cotovelo" (Ponto mais distante da reta)
    # Reta que liga o primeiro ponto (K=2) ao último ponto (K=max_k)
    p1 = np.array([ks[0], inercias[0]])
    p2 = np.array([ks[-1], inercias[-1]])
    
    maior_distancia = 0
    melhor_k = ks[0]
    
    for i in range(len(ks)):
        p3 = np.array([ks[i], inercias[i]])
        # Fórmula da distância de um ponto até uma reta
        distancia = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        
        if distancia > maior_distancia:
            maior_distancia = distancia
            melhor_k = ks[i]
            
    # Garantir que não caia em K=2 se a pista for complexa
    return max(3, melhor_k)

BASE_PATH = "../AccReplay/"
# Nome da pasta principal mais limpo
OUTPUT_DIR = "Resultados_Clusterizacao/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Iniciando o processamento híbrido (Global e Por Carro)...")

# 1. Iterar sobre todas as pistas
for track_name in os.listdir(BASE_PATH):
    track_path = os.path.join(BASE_PATH, track_name)
    if not os.path.isdir(track_path):
        continue
        
    print(f"\nProcessando pista: {track_name}")
    track_features_list = []
    
    # 2. Iterar sobre todos os pilotos/corridas dessa pista
    for run_id in os.listdir(track_path):
        run_path = os.path.join(track_path, run_id)
        if not os.path.isdir(run_path):
            continue
            
        telemetry_file = os.path.join(run_path, 'telemetry.json')
        sections_file = os.path.join(run_path, 'sections.json')
        info_file = os.path.join(run_path, 'info_lap.json')
        
        if not (os.path.exists(telemetry_file) and os.path.exists(sections_file) and os.path.exists(info_file)):
            continue
            
        try:
            with open(telemetry_file, 'r', encoding='utf-8') as f:
                telemetry_data = json.load(f)
            with open(sections_file, 'r', encoding='utf-8') as f:
                sections_data = json.load(f)
            with open(info_file, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
                
            # Extrair o modelo do carro
            car_model = info_data.get('car', {}).get('model', 'Desconhecido')
            
        except Exception as e:
            print(f"  Erro ao ler arquivos do run {run_id}: {e}")
            continue

        if len(telemetry_data.get('channels', [])) == 0:
            continue
            
        distances = telemetry_data['distances']
        lap_channels = telemetry_data['channels'][0]

        df = pd.DataFrame({
            'distance': distances,
            'speed': lap_channels['speed'],
            'throttle': lap_channels['throttle'],
            'brake': lap_channels['brake'],
            'steering': lap_channels['steering'],
            'gear': lap_channels['gears']
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

        def get_section_info(dist, sections):
            for sec in sections:
                if sec['splineStart'] <= dist <= sec['splineEnd']:
                    return sec['description']
            return None

        df['section'] = df['distance'].apply(lambda x: get_section_info(x, sections_data))
        df_curves = df.dropna(subset=['section'])

        if df_curves.empty:
            continue

        features = df_curves.groupby('section').agg({
            'speed': ['min', 'mean', 'max'],
            'brake': ['max', 'mean'],
            'throttle': ['mean'],
            'gear': ['min'],
            'steering': ['std', lambda x: x.abs().max()], 
            'steering_velocity': [lambda x: x.abs().mean(), lambda x: x.abs().max()],
            'distance': lambda x: x.max() - x.min()
        }).reset_index()

        features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in features.columns.values]
        features.rename(columns={
            'steering_<lambda_0>': 'steering_angle_max',
            'steering_velocity_<lambda_0>': 'hand_speed_mean',
            'steering_velocity_<lambda_1>': 'hand_speed_max',
            'distance_<lambda>': 'curve_length'
        }, inplace=True)
        
        features['car_model'] = car_model
        track_features_list.append(features)

    if not track_features_list:
        print(f"  Nenhum dado válido extraído para {track_name}.")
        continue

    all_runs_df = pd.concat(track_features_list, ignore_index=True)
    
    # --- NOVO: Cria a subpasta específica para a pista atual ---
    track_output_dir = os.path.join(OUTPUT_DIR, track_name)
    os.makedirs(track_output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # PARTE A: CLUSTERIZAÇÃO GLOBAL (Todos os carros misturados)
    # ---------------------------------------------------------
    global_track_df = all_runs_df.drop(columns=['car_model']).groupby('section').mean().reset_index()
    
    cols_for_ai = [c for c in global_track_df.columns if c != 'section']
    scaler = StandardScaler()
    X_scaled_global = scaler.fit_transform(global_track_df[cols_for_ai])

    best_k_global = encontrar_melhor_k_cotovelo(X_scaled_global)
    print(f"    -> Melhor número de clusters GLOBAL para {track_name}: {best_k_global}")

    kmeans_global = KMeans(n_clusters=best_k_global, random_state=42)
    global_track_df['cluster'] = kmeans_global.fit_predict(X_scaled_global)

    global_track_df = global_track_df.sort_values('cluster')
    
    # Salva o arquivo GLOBAL dentro da subpasta da pista
    output_global = os.path.join(track_output_dir, 'clusters_GLOBAL.csv')
    global_track_df.to_csv(output_global, index=False)
    print(f"  -> Salvo Global: {output_global}")

    # ---------------------------------------------------------
    # PARTE B: CLUSTERIZAÇÃO ESPECÍFICA (Por modelo de carro)
    # ---------------------------------------------------------
    car_models = all_runs_df['car_model'].unique()
    
    for car in car_models:
        car_df = all_runs_df[all_runs_df['car_model'] == car]
        car_track_df = car_df.drop(columns=['car_model']).groupby('section').mean().reset_index()
        
        if len(car_track_df) < 4:
            print(f"  -> Ignorado: {car} (Curvas insuficientes para clusterizar)")
            continue
            
        X_scaled_car = scaler.fit_transform(car_track_df[cols_for_ai])

        best_k_car = encontrar_melhor_k_cotovelo(X_scaled_car)
        print(f"    -> Melhor K para {car} em {track_name}: {best_k_car}")
        
        kmeans_car = KMeans(n_clusters=best_k_car, random_state=42)
        car_track_df['cluster'] = kmeans_car.fit_predict(X_scaled_car)
        
        car_track_df = car_track_df.sort_values('cluster')
        
        safe_car_name = car.replace(' ', '_').replace('/', '-')
        # Salva o arquivo DO CARRO dentro da subpasta da pista
        output_car = os.path.join(track_output_dir, f'clusters_{safe_car_name}.csv')
        
        car_track_df.to_csv(output_car, index=False)
        print(f"  -> Salvo Específico: {output_car}")

print("\nProcessamento finalizado com sucesso!")