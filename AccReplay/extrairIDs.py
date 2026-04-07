import requests
import json
import os
from tqdm import tqdm # Apenas para uma barra de progresso (pip install tqdm)

print("Iniciando busca de IDs de voltas via API (mais rápido)...")

# 1. Obter a lista de todas as pistas
try:
    # Pode usar lastPatchOnly=false para pegar o máximo de pistas
    records_url = "https://www.accreplay.com/api/leaderboards/records/top?group=ALL&lastPatchOnly=true&count=3"
    response = requests.get(records_url, timeout=10)
    response.raise_for_status()
    tracks_data = response.json()
    
    # Extrai apenas os trackId únicos
    track_ids = sorted(list(set(track['trackId'] for track in tracks_data)))
    print(f"Encontrados {len(track_ids)} IDs de pistas.")
    
except requests.exceptions.RequestException as e:
    print(f"Erro ao buscar lista de pistas: {e}")
    exit()

all_lap_ids = set()

# 2. Iterar sobre cada pista e buscar suas voltas
for track_id in tqdm(track_ids, desc="Processando pistas"):
    try:
        # Pode usar lastPatchOnly=false e group=All para pegar o máximo de voltas
        laps_url = f"https://www.accreplay.com/api/leaderboards/laps?trackId={track_id}&lastPatchOnly=true&group=All"
        response = requests.get(laps_url, timeout=10)
        
        if response.status_code == 200:
            laps_json = response.json()
            for lap in laps_json:
                all_lap_ids.add(lap['lapId'])
        
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar voltas da pista {track_id}: {e}")

print("\n--- Coleta concluída ---")
print(f"Total de IDs de voltas únicos encontrados: {len(all_lap_ids)}")

# 3. Salvar a lista de IDs em um arquivo
output_file = "lap_ids_list.json"
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(list(all_lap_ids), f)
    print(f"Lista de IDs salva em: {output_file}")
except IOError as e:
    print(f"Erro ao salvar arquivo JSON: {e}")