import requests
import os
import json
import time

with open("lap_ids_list.json", "r", encoding="utf-8") as f:
    valid_ids = json.load(f)

    for i in valid_ids:

        lap_id = i
        print(f"\n--- Processando Lap ID: {lap_id} ---")

        # JSON com as infos da pista (Track ID)
        info_lap_url = f"https://www.accreplay.com/api/laps/{lap_id}"

        dir_name = ""
        track_id = ""
        track_short_name = ""

        try:
            # 1. FAZER A REQUISIÇÃO CRÍTICA
            response_info = requests.get(info_lap_url, timeout=10)

            # 2. VERIFICAR O STATUS DA REQUISIÇÃO CRÍTICA
            if response_info.status_code != 200:
                # Se a volta não existe (404) ou outro erro, pule para a próxima
                print(f"Info da volta {lap_id} não encontrada. Status: {response_info.status_code}. Pulando...")
                continue
            
            # 3. PROCESSAR OS DADOS E CRIAR A PASTA
            info_lap_json = response_info.json()

            track_id = info_lap_json["session"]["trackId"]
            track_short_name = info_lap_json["session"]["trackShortName"]
            dir_name = f"{track_short_name}/{lap_id}"
            
            os.makedirs(dir_name, exist_ok=True)

            # 4. SALVAR ARQUIVO DE INFO
            info_lap_file = "info_lap.json"
            info_lap_path = os.path.join(dir_name, info_lap_file)
            with open(info_lap_path, "w", encoding="utf-8") as f:
                json.dump(info_lap_json, f, ensure_ascii=False, indent=4)
            print(f"Info da volta salva em: {info_lap_path}")

            # --- 5. FAZER AS OUTRAS REQUISIÇÕES (AGORA QUE SABEMOS QUE A VOLTA É VÁLIDA) ---

            # JSON com a telemetria
            try:
                telemetry_url = f"https://www.accreplay.com/api/telemetry?lapId={lap_id}"
                response = requests.get(telemetry_url, timeout=10)
                if response.status_code == 200:
                    telemetry_path = os.path.join(dir_name, "telemetry.json")
                    with open(telemetry_path, "w", encoding="utf-8") as f:
                        json.dump(response.json(), f, ensure_ascii=False, indent=4)
                    print("  - Telemetria salva.")
                else:
                    print(f"  - Telemetria não encontrada (Status: {response.status_code})")
                    continue
            except Exception as e:
                print(f"  - Erro ao baixar telemetria: {e}")

            # JSON das Seções
            try:
                sections_url = f"https://www.accreplay.com/api/tracks/{track_id}/sections"
                response = requests.get(sections_url, timeout=10)
                if response.status_code == 200:
                    sections_path = os.path.join(dir_name, "sections.json")
                    with open(sections_path, "w", encoding="utf-8") as f:
                        json.dump(response.json(), f, ensure_ascii=False, indent=4)
                    print("  - Seções salvas.")
                else:
                    print(f"  - Seções não encontradas (Status: {response.status_code})")
                    continue
            except Exception as e:
                print(f"  - Erro ao baixar seções: {e}")

            # JSON Limite "Inner"
            try:
                inner_url = f"https://www.accreplay.com/api/tracks/{track_short_name}/limits?type=inner"
                response = requests.get(inner_url, timeout=10)
                if response.status_code == 200:
                    inner_path = os.path.join(dir_name, "inner.json")
                    with open(inner_path, "w", encoding="utf-8") as f:
                        json.dump(response.json(), f, ensure_ascii=False, indent=4)
                    print("  - Limite 'inner' salvo.")
                else:
                    print(f"  - Limite 'inner' não encontrado (Status: {response.status_code})")
                    continue
            except Exception as e:
                print(f"  - Erro ao baixar limite 'inner': {e}")
                
            # JSON Limite "Outer"
            try:
                outer_url = f"https://www.accreplay.com/api/tracks/{track_short_name}/limits?type=outer"
                response = requests.get(outer_url, timeout=10)
                if response.status_code == 200:
                    outer_path = os.path.join(dir_name, "outer.json")
                    with open(outer_path, "w", encoding="utf-8") as f:
                        json.dump(response.json(), f, ensure_ascii=False, indent=4)
                    print("  - Limite 'outer' salvo.")
                else:
                    print(f"  - Limite 'outer' não encontrado (Status: {response.status_code})")
            except Exception as e:
                print(f"  - Erro ao baixar limite 'outer': {e}")

        # 6. CAPTURAR ERROS DE REDE (Timeout, DNS, etc.)
        except requests.exceptions.RequestException as e:
            print(f"Erro de rede para lap_id {lap_id}: {e}. Pulando...")
            continue
        except Exception as e:
            print(f"Erro inesperado processando {lap_id}: {e}. Pulando...")
            continue

        # 7. SER GENTIL COM O SERVIDOR
        time.sleep(0.1)