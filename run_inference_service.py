#!/usr/bin/env python
"""
Script para iniciar o serviço de inferência GMM.
Serve como ponto de entrada para o container Docker no Cloud Run.
"""

import os
import sys
import argparse

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from src.inference.gmm_inference_service import start_service

def main():
    """Função principal para iniciar o serviço."""
    parser = argparse.ArgumentParser(description='Iniciar o serviço de inferência GMM')
    
    # Usar variáveis de ambiente para configuração no Cloud Run
    default_port = int(os.environ.get('PORT', 8080))  # Cloud Run usa a variável PORT
    default_host = os.environ.get('HOST', '0.0.0.0')
    
    parser.add_argument('--host', default=default_host, help='Endereço de host')
    parser.add_argument('--port', type=int, default=default_port, help='Porta do serviço')
    
    args = parser.parse_args()
    
    print(f"Iniciando serviço de inferência GMM em {args.host}:{args.port}...")
    start_service(host=args.host, port=args.port)

if __name__ == '__main__':
    main()