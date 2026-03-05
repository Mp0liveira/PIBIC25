import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans


PATH = "AccReplay/zolder/1969213/"

try:
    with open(PATH + 'telemetry.json', 'r') as f:
        telemetry_data = json.load(f)
    
    with open(PATH + 'sections.json', 'r') as f:
        sections_data = json.load(f)
    print("Arquivos carregados com sucesso.")
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos: {e}")
    exit()

# ------------- Preparação do DataFrame --------------------
distances = telemetry_data['distances']

if len(telemetry_data['channels']) > 0:
    lap_channels = telemetry_data['channels'][0]
else:
    print("Erro: Nenhuma volta encontrada.")
    exit()

df = pd.DataFrame({
    'distance': distances,
    'speed': lap_channels['speed'],
    'throttle': lap_channels['throttle'],
    'brake': lap_channels['brake'],
    'steering': lap_channels['steering'],
    'gear': lap_channels['gears'],
    'rpm': lap_channels['engineRPM']
})

# --------------  Criação de novas features (velocidade angular do volante) -------------

# Variação de distância (m) e volante (graus)
df['delta_dist'] = df['distance'].diff()
df['delta_steering'] = df['steering'].diff()

# Velocidade em m/s
df['speed_ms'] = df['speed'] / 3.6

# Cálculo do Tempo (dt = d / v)
# Tratamos divisão por zero ou velocidades muito baixas substituindo por um valor mínimo seguro
df['delta_time'] = df.apply(
    lambda row: row['delta_dist'] / row['speed_ms'] if (pd.notnull(row['speed_ms']) and row['speed_ms'] > 1) else 0.1, 
    axis=1
)

# Cálculo da Velocidade Angular (Graus por Segundo)
df['steering_velocity'] = df['delta_steering'] / df['delta_time']

# Limpeza de dados (NaNs gerados pelo .diff() na primeira linha)
df.fillna(0, inplace=True)


# ------------- Inserindo o trecho da curva no dataframe ---------------
def get_section_info(dist, sections):
    for sec in sections:
        if sec['splineStart'] <= dist <= sec['splineEnd']:
            return sec['description']
    return None

df['section'] = df['distance'].apply(lambda x: get_section_info(x, sections_data))
df_curves = df.dropna(subset=['section'])

print(f"Pontos de telemetria mapeados em curvas: {len(df_curves)}")


# ------- Variações de features já existentes (máximo, mínimo, média) ----------
features = df_curves.groupby('section').agg({

    'speed': ['min', 'mean', 'max'],       # Velocidades mínimas, média e máxima durante a curva
    'brake': ['max', 'mean'],              # Máximo e média da pisada no freio
    'throttle': ['mean'],                  # Média de quanto pisa no acelerador
    'gear': ['min'],                       # Marcha mais baixa
    
    # Volante: variação da angulação (std) e angulação máxima (abs max)
    'steering': ['std', lambda x: x.abs().max()], 
    # Velocidade angular média e máxima
    'steering_velocity': [lambda x: x.abs().mean(), lambda x: x.abs().max()],
    
    'distance': lambda x: x.max() - x.min() # Tamanho da curva
}).reset_index()

# Arrumando nomes das colunas
features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in features.columns.values]

features.rename(columns={
    'steering_<lambda_0>': 'steering_angle_max',
    'steering_velocity_<lambda_0>': 'hand_speed_mean',
    'steering_velocity_<lambda_1>': 'hand_speed_max',
    'distance_<lambda>': 'curve_length'
}, inplace=True)


# --- 6. Clusterização ---
#scaler = MinMaxScaler()
scaler = StandardScaler()
cols_for_ai = [c for c in features.columns if c != 'section']
X_scaled = scaler.fit_transform(features[cols_for_ai])

print(X_scaled)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
features['cluster'] = kmeans.fit_predict(X_scaled)

features.to_csv('clu04_standard.csv', index=False)

# ------ Resultados ------
summary = features.sort_values('cluster')

print("\n--- Classificação das Curvas ---")
print(summary)

"""# --- 6. Clusterização Avançada com ColumnTransformer ---

# 1. Definir quais colunas vão para qual Scaler
features_minmax = ['brake_max', 'brake_mean', 'throttle_mean']
features_standard = [
    'speed_min', 'speed_mean', 'speed_max', 
    'steering_angle_max', 'steering_std',
    'hand_speed_mean', 'hand_speed_max',
    'curve_length', 'gear_min'
]


# 2. Usar normalizações diferentes para dados diferentes
preprocessor = ColumnTransformer(
    transformers=[
        ('minmax', MinMaxScaler(), features_minmax),
        ('standard', StandardScaler(), features_standard)
    ]
)

# 3. Aplicar a transformação
X_scaled = preprocessor.fit_transform(features)

# 4. Rodar o K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
features['cluster'] = kmeans.fit_predict(X_scaled)


# --- 7. Resultados ---
summary = features.sort_values('cluster')
print("\n--- Classificação com ColumnTransformer ---")
print(summary)"""
