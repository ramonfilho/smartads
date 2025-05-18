#!/usr/bin/env python
"""
Script para localizar o arquivo 10_gmm_calibrated.joblib 
"""
import os

def find_file(root_dir, filename):
    """Encontra um arquivo em qualquer lugar da estrutura de diretórios."""
    matches = []
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            matches.append(os.path.join(root, filename))
    return matches

# Diretório raiz do projeto
project_root = "/Users/ramonmoreira/desktop/smart_ads"

# Arquivo a procurar
target_file = "10_gmm_calibrated.joblib"

# Procurar o arquivo
matches = find_file(project_root, target_file)

# Imprimir os resultados
if matches:
    print(f"Arquivo encontrado em {len(matches)} locais:")
    for path in matches:
        print(f"  - {path}")
        # Verificar se o arquivo existe e pode ser lido
        if os.path.exists(path) and os.access(path, os.R_OK):
            print(f"    Arquivo existe e pode ser lido")
            print(f"    Tamanho: {os.path.getsize(path)} bytes")
        else:
            print(f"    AVISO: Arquivo não pode ser acessado")
else:
    print(f"Arquivo {target_file} não encontrado em nenhum lugar dentro de {project_root}")