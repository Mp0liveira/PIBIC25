import json

PATH = "AccReplay/zolder/1969213/"

with open(PATH + 'telemetry.json', 'r') as f:
    telemetry_data = json.load(f)

dados = {}

for i in telemetry_data:
    dados[i] = telemetry_data[i]

for key in dados.keys():
    print(key)

"""
Nos dá todos os parâmetros do item "channel"
lapId, speed, throttle, brake, steering, engineRPM, gears, deltaT

lapId -> inútil
speed (km/h) -> útil
throttle (quanto foi pressionado em %) -> útil
brake (quanto foi pressionado em %) -> úti
steering (quanto o volante girou em °) -> útil
engineRPM -> pode ser útil
gears -> pode ser útil
deltaT -> não descobri o que é (todos seus valores são nulos, vamos descartar de qualquer forma)

monza
spa
suzuka
"""
json_channels = json.loads(str(dados["channels"][0]).replace("'", '"'))

for i in json_channels:
    print(i)