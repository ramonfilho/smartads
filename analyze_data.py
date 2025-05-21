# analyze_data.py
import pandas as pd
import json
import random

# Caminho para o arquivo de treino
file_path = "data/01_split/train.csv"

def analyze_training_data():
    # Carregar dados de treino
    try:
        df = pd.read_csv(file_path)
        print(f"Arquivo carregado com sucesso: {file_path}")
        print(f"Dimensões: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Mostrar as colunas (campos esperados pelo modelo)
        print("\n=== COLUNAS DO DATASET ===")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
            
        # Verificar dados da variável target (se existir)
        if 'target' in df.columns:
            pos_count = df['target'].sum()
            print(f"\nDistribuição da variável target:")
            print(f"  - Positivos (1): {pos_count} ({pos_count/len(df)*100:.2f}%)")
            print(f"  - Negativos (0): {len(df) - pos_count} ({(len(df) - pos_count)/len(df)*100:.2f}%)")
        
        # Verificar UTM campos
        utm_cols = [col for col in df.columns if 'utm' in col.lower()]
        if utm_cols:
            print("\n=== EXEMPLOS DE UTM ===")
            for col in utm_cols:
                unique_values = df[col].dropna().unique()
                print(f"{col}: {len(unique_values)} valores únicos")
                print(f"Exemplos: {list(unique_values[:5])}")
        
        # Gerar exemplos de requisições
        print("\n=== EXEMPLOS DE REQUISIÇÕES ===")
        
        # Selecionar algumas amostras aleatórias
        sample_indices = random.sample(range(len(df)), min(3, len(df)))
        for i, idx in enumerate(sample_indices):
            sample = df.iloc[idx].dropna()
            
            # Criar payload para API
            payload = {"data": {}}
            for col in sample.index:
                if col != 'target':  # Não incluir a target na requisição
                    value = sample[col]
                    # Converter para tipos primitivos Python para JSON
                    if isinstance(value, (int, float)):
                        payload["data"][col] = value
                    else:
                        payload["data"][col] = str(value)
            
            # Mostrar exemplo
            print(f"\nExemplo {i+1}:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            
            # Mostrar comando curl
            print("\nComando curl:")
            curl_cmd = f'curl -X POST https://smartads-api-12955519745.southamerica-east1.run.app/predict \\\n  -H "Content-Type: application/json" \\\n  -d \'{json.dumps(payload, ensure_ascii=False)}\''
            print(curl_cmd)
            
            # Se houver target, mostrar o valor real
            if 'target' in df.columns:
                print(f"\nValor real (target): {df.iloc[idx]['target']}")
                
        # Criar exemplo JSON minimalista com os campos principais
        min_payload = {"data": {}}
        essential_fields = [col for col in df.columns[:20] if col != 'target']
        for col in essential_fields:
            sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if isinstance(sample_value, (int, float)):
                min_payload["data"][col] = sample_value
            else:
                min_payload["data"][col] = str(sample_value)
        
        print("\n=== EXEMPLO MINIMALISTA ===")
        print(json.dumps(min_payload, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")

if __name__ == "__main__":
    analyze_training_data()