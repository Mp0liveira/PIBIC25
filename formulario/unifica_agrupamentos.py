import pandas as pd
import re
from collections import defaultdict
from itertools import combinations

# 1. Carregar os dados
caminho_arquivo = 'Classificação de curvas por IA (respostas) - Respostas ao formulário 1.csv'
df = pd.read_csv(caminho_arquivo)

# 2. Separar as colunas por pista
pistas = ["Monza", "Spa-Francorchamps", "Suzuka"]
colunas_por_pista = {p: [] for p in pistas}
indice_pista = -1

for col in df.columns:
    # Pega apenas as colunas de curvas individuais
    if 'Categorize as curvas [' in col:
        # Verifica se é a curva T1 para identificar a troca de pista no formulário
        if re.search(r'\[T1\](?:\.\d+)?$', col):
            indice_pista += 1
        
        if 0 <= indice_pista < len(pistas):
            colunas_por_pista[pistas[indice_pista]].append(col)

# 3. Analisar cada pista separadamente
for pista, colunas in colunas_por_pista.items():
    print("\n" + "="*70)
    print(f"🏎️  PISTA: {pista.upper()}")
    print("="*70)
    
    usuarios_grupos = {}
    
    for index, row in df.iterrows():
        usuario_id = f"Usuário {index + 1}"
        grupos_do_usuario = defaultdict(set)
        
        for col in colunas:
            resposta = str(row[col]).strip()
            
            if resposta != 'nan' and resposta != '':
                # Extrair o nome limpo da curva
                match = re.search(r'\[(T\d+)\]', col)
                if match:
                    nome_curva = match.group(1)
                    grupos_do_usuario[resposta].add(nome_curva)
                    
        if grupos_do_usuario:
            usuarios_grupos[usuario_id] = grupos_do_usuario

    # --- ANÁLISE 1: GRUPOS IDÊNTICOS ---
    print("\n--- ANÁLISE 1: GRUPOS IDÊNTICOS (1 Grupo == 1 Grupo) ---")
    agrupamentos_exatos = defaultdict(list)
    
    for usuario, grupos in usuarios_grupos.items():
        for nome_grupo, curvas in grupos.items():
            # Removida a trava: agora inclui grupos com apenas 1 curva (curvas soltas)
            conjunto_curvas = frozenset(curvas)
            agrupamentos_exatos[conjunto_curvas].append(f"{usuario} (chamou de '{nome_grupo}')")

    contador_padrao = 1
    for curvas, lista_usuarios in agrupamentos_exatos.items():
        if len(lista_usuarios) > 1:
            curvas_ordenadas = ", ".join(sorted(list(curvas), key=lambda x: int(x[1:])))
            print(f"\n[PADRÃO {contador_padrao}] -> Curvas: {curvas_ordenadas}")
            for u in lista_usuarios:
                print(f"  -> {u}")
            contador_padrao += 1

    if contador_padrao == 1:
        print("\nNenhum grupo exato 100% igual foi encontrado entre usuários diferentes nesta pista.")

    # --- ANÁLISE 2: COMBINAÇÕES DE GRUPOS ---
    print("\n--- ANÁLISE 2: COMBINAÇÕES DE GRUPOS (G_A + G_B == G_C + G_D) ---")
    agrupamentos_combinados = defaultdict(list)
    
    for usuario, grupos in usuarios_grupos.items():
        nomes_dos_grupos = list(grupos.keys())
        
        for g1, g2 in combinations(nomes_dos_grupos, 2):
            conjunto_unido = frozenset(grupos[g1].union(grupos[g2]))
            
            # Filtro para não pegar uniões que englobem quase a pista inteira
            if 3 <= len(conjunto_unido) <= 8:
                nome_combinado = f"{g1} + {g2}"
                agrupamentos_combinados[conjunto_unido].append(f"{usuario} (juntou '{nome_combinado}')")

    contador_combinacao = 1
    for curvas, lista_usuarios in agrupamentos_combinados.items():
        # Só nos interessa se usuários DIFERENTES chegaram à mesma combinação
        usuarios_unicos = set([u.split(" (")[0] for u in lista_usuarios])
        
        if len(usuarios_unicos) > 1:
            curvas_ordenadas = ", ".join(sorted(list(curvas), key=lambda x: int(x[1:])))
            print(f"\n[COMBINAÇÃO {contador_combinacao}] -> Curvas unidas: {curvas_ordenadas}")
            for u in lista_usuarios:
                print(f"  -> {u}")
            contador_combinacao += 1

    if contador_combinacao == 1:
        print("\nNenhuma combinação de 2 grupos bateu entre os usuários nesta pista.")