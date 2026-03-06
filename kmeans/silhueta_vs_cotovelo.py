import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_PATH = "../AccReplay/"
MAX_K_TESTE = 8 # Limite máximo de clusters para testar

# --- FUNÇÃO 1: Silhouette Score ---
def melhor_k_silhueta(X_scaled, max_k=MAX_K_TESTE):
    n_amostras = len(X_scaled)
    limite_k = min(max_k + 1, n_amostras)
    if n_amostras < 4:
        return n_amostras 
        
    melhor_k = 2
    melhor_score = -1
    
    for k in range(4, limite_k):
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        labels = kmeans_temp.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > melhor_score:
            melhor_score = score
            melhor_k = k
            
    return melhor_k

# --- FUNÇÃO 2: Método do Cotovelo (Distância à Reta) ---
def melhor_k_cotovelo(X_scaled, max_k=MAX_K_TESTE):
    n_amostras = len(X_scaled)
    limite_k = min(max_k + 1, n_amostras)
    if n_amostras < 4:
        return n_amostras
        
    inercias = []
    ks = list(range(2, limite_k))
    
    for k in ks:
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(X_scaled)
        inercias.append(kmeans_temp.inertia_)
        
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
            
    # Forçamos o cotovelo a ser no mínimo 3 para evitar o underfitting extremo
    return max(3, melhor_k)

print("Iniciando o comparativo de métricas (Silhueta vs Cotovelo)...\n" + "="*60)

for track_name in os.listdir(BASE_PATH):
    track_path = os.path.join(BASE_PATH, track_name)
    if not os.path.isdir(track_path):
        continue
        
    track_features_list = []
    
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
                
            car_model = info_data.get('car', {}).get('model', 'Desconhecido')
        except:
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
            lambda row: row['delta_dist'] / row['speed_ms'] if (pd.notnull(row['speed_ms']) and row['speed_ms'] > 1) else 0.1, axis=1)
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
        continue

    all_runs_df = pd.concat(track_features_list, ignore_index=True)
    scaler = StandardScaler()
    
    print(f"📍 PISTA: {track_name.upper()}")
    
    # --- AVALIAÇÃO GLOBAL ---
    global_track_df = all_runs_df.drop(columns=['car_model']).groupby('section').mean().reset_index()
    cols_for_ai = [c for c in global_track_df.columns if c != 'section']
    X_scaled_global = scaler.fit_transform(global_track_df[cols_for_ai])
    
    k_sil_global = melhor_k_silhueta(X_scaled_global)
    k_cot_global = melhor_k_cotovelo(X_scaled_global)
    
    print(f"   [GLOBAL]     -> Silhueta: K={k_sil_global} | Cotovelo: K={k_cot_global}")

    # --- AVALIAÇÃO POR CARRO ---
    car_models = all_runs_df['car_model'].unique()
    for car in car_models:
        car_df = all_runs_df[all_runs_df['car_model'] == car]
        car_track_df = car_df.drop(columns=['car_model']).groupby('section').mean().reset_index()
        
        if len(car_track_df) < 4:
            continue
            
        X_scaled_car = scaler.fit_transform(car_track_df[cols_for_ai])
        
        k_sil_car = melhor_k_silhueta(X_scaled_car)
        k_cot_car = melhor_k_cotovelo(X_scaled_car)
        
        print(f"   [{car}] -> Silhueta: K={k_sil_car} | Cotovelo: K={k_cot_car}")
        
    print("-" * 60)

print("Diagnóstico finalizado!")