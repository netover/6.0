import math
import os
import re

def organizar_e_dividir(ficheiro_entrada="analysis_report.txt"):
    # Verifica se o ficheiro original existe
    if not os.path.exists(ficheiro_entrada):
        print(f"Erro: O ficheiro '{ficheiro_entrada}' não foi encontrado.")
        return
        
    # Lê o conteúdo do ficheiro
    with open(ficheiro_entrada, 'r', encoding='utf-8', errors='ignore') as f:
        conteudo = f.read()
        
    # 1. Utiliza regex para detetar todas as secções dinamicamente
    # Vai encontrar padrões como "=== 1. RUFF ===", "=== 3. PYLINT ===", etc.
    seccoes = re.split(r'(=== \d+\. .*? ===)', conteudo)
    
    linhas_finais = []
    
    # Preserva o cabeçalho principal do documento (antes da 1ª secção)
    if seccoes[0].strip():
        linhas_finais.extend(seccoes[0].strip().split('\n'))
        
    # 2. Processa, agrupa e junta cada secção
    # O re.split cria uma lista onde os ímpares são os títulos e os pares os conteúdos
    for i in range(1, len(seccoes), 2):
        cabecalho = seccoes[i]
        conteudo_seccao = seccoes[i+1] if i+1 < len(seccoes) else ""
        
        linhas_finais.append(f"\n\n{cabecalho} [AGRUPADO POR FICHEIRO/ERRO]\n")
        
        # Limpa as linhas em branco e ordena para agrupar por caminho do ficheiro
        linhas_conteudo = [linha.strip() for linha in conteudo_seccao.split('\n') if linha.strip()]
        linhas_conteudo.sort() 
        
        linhas_finais.extend(linhas_conteudo)

    # Limpar linhas vazias extra para não criar lixo nos .txt finais
    linhas_finais = [linha for linha in linhas_finais if linha.strip() != ""]
    
    # 3. Calcular o tamanho exato para dividir em 10 blocos
    num_partes = 10
    tamanho_bloco = math.ceil(len(linhas_finais) / num_partes)
    
    # 4. Criar e guardar os 10 ficheiros .txt independentes
    for i in range(num_partes):
        inicio = i * tamanho_bloco
        fim = inicio + tamanho_bloco
        bloco_atual = linhas_finais[inicio:fim]
        
        if bloco_atual: # Só cria se houver conteúdo no bloco
            nome_ficheiro_txt = f"erros_parte_{i+1:02d}.txt"
            conteudo_bloco = "\n".join(bloco_atual)
            
            # Guarda diretamente na pasta atual
            with open(nome_ficheiro_txt, 'w', encoding='utf-8') as out_f:
                out_f.write(conteudo_bloco)
                
            print(f"Ficheiro gerado com sucesso: {nome_ficheiro_txt}")
            
    print("\nProcesso concluído! Todos os erros (Mypy, Pylint, Radon, etc.) foram organizados e divididos em 10 ficheiros txt.")

# Executa a função
organizar_e_dividir()