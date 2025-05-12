import pandas as pd
import numpy as np
import os
import sys

# Definir a classe GMM_Wrapper diretamente no script de teste
class GMM_Wrapper:
    """
    Classe wrapper para o GMM que implementa a API sklearn para calibração.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pca_model = pipeline['pca_model'] if 'pca_model' in pipeline else None
        self.gmm_model = pipeline['gmm_model'] if 'gmm_model' in pipeline else None
        self.scaler_model = pipeline['scaler_model'] if 'scaler_model' in pipeline else None
        self.cluster_models = pipeline['cluster_models'] if 'cluster_models' in pipeline else {}
        self.n_clusters = pipeline['n_clusters'] if 'n_clusters' in pipeline else 3
        self.threshold = pipeline.get('threshold', 0.15)
        
        # Adicionar atributos necessários para a API sklearn
        self.classes_ = np.array([0, 1])  # Classes binárias
        self._fitted = True  # Marcar como já ajustado
        self._estimator_type = "classifier"  # Indicar explicitamente que é um classificador
        
    def fit(self, X, y):
        # Como o modelo já está treinado, apenas verificamos as classes
        self.classes_ = np.unique(y)
        self._fitted = True
        return self
        
    def predict_proba(self, X):
        # Implementação simplificada para compatibilidade
        return np.zeros((len(X), 2))  # Será substituído pela implementação real
    
    def predict(self, X):
        # Implementação simplificada para compatibilidade
        return np.zeros(len(X))  # Será substituído pela implementação real

# Importar a pipeline de inferência
from src.inference.gmm_inference_pipeline import GMMInferencePipeline

# 1. Carregar dados crus (exemplo com dados de survey originais)
# Substitua pelo caminho correto dos seus dados originais
original_data_path = "/Users/ramonmoreira/desktop/smart_ads/data/01_split/test.csv"
raw_df = pd.read_csv(original_data_path)
print(f"Dados originais carregados: {raw_df.shape[0]} registros")

# 2. Preparar um subconjunto desses dados para teste
# Podemos usar uma amostra aleatória ou simplesmente os primeiros N registros
test_sample = raw_df.sample(n=min(100, len(raw_df)), random_state=42)
# Ou: test_sample = raw_df.head(100)
print(f"Amostra para teste: {test_sample.shape[0]} registros")

# 3. Carregar resultados conhecidos para esses registros específicos
# Vamos criar uma correspondência com os IDs ou emails desses registros
if 'email' in raw_df.columns:
    # Identificador para correspondência
    id_column = 'email'
elif 'ID' in raw_df.columns:
    id_column = 'ID'
else:
    # Use qualquer outra coluna única como identificador
    id_column = raw_df.columns[0]
    print(f"Usando {id_column} como identificador para correspondência")

# Carregar resultados finais conhecidos (após todas as transformações da pipeline original)
final_results_path = "/Users/ramonmoreira/desktop/smart_ads/reports/calibration_validation_three_models/20250512_073645/gmm_test_results.csv"
if os.path.exists(final_results_path):
    known_results = pd.read_csv(final_results_path)
    
    # Se possível, faça a correspondência entre os resultados conhecidos e os registros de teste
    print(f"Resultados conhecidos carregados: {known_results.shape[0]} registros")
    has_known_results = True
else:
    print("Arquivo de resultados conhecidos não encontrado")
    has_known_results = False

# 4. Inicializar a pipeline de inferência
inference_pipeline = GMMInferencePipeline()
print("Pipeline de inferência inicializada")

# 5. Executar a inferência na amostra de dados originais
print("Executando inferência nos dados originais...")
inference_results = inference_pipeline.predict(test_sample)
results_df = inference_results['results']
print(f"Inferência concluída: {results_df.shape[0]} resultados")

# 6. Avaliar os resultados

# 6.1 Verificar se os dados foram processados sem erros
if len(results_df) == len(test_sample):
    print("✅ Todos os registros foram processados corretamente")
else:
    print(f"⚠️ Discrepância no número de registros: {len(test_sample)} entrada, {len(results_df)} saída")

# 6.2 Se tivermos identificadores comuns, comparar com resultados conhecidos
if has_known_results and id_column in test_sample.columns and id_column in known_results.columns:
    # Mesclar por identificador para comparar
    comparison = test_sample[[id_column]].copy()
    comparison['new_prediction'] = results_df['prediction'].values
    comparison['new_probability'] = results_df['probability'].values
    
    # Mesclar com resultados conhecidos
    comparison = comparison.merge(
        known_results[[id_column, 'prediction', 'probability']],
        on=id_column,
        how='left',
        suffixes=('', '_known')
    )
    
    # Calcular métricas de correspondência
    matches = (comparison['new_prediction'] == comparison['prediction_known'])
    match_rate = matches.mean() * 100
    
    print(f"\nTaxa de correspondência das predições: {match_rate:.2f}%")
    
    # Comparar probabilidades
    prob_diff = np.abs(comparison['new_probability'] - comparison['probability_known'])
    avg_diff = prob_diff.mean()
    max_diff = prob_diff.max()
    
    print(f"Diferença média nas probabilidades: {avg_diff:.6f}")
    print(f"Diferença máxima nas probabilidades: {max_diff:.6f}")
else:
    print("\nNão foi possível fazer uma comparação direta com resultados conhecidos")

# 7. Mostrar estatísticas gerais das predições
positive_rate = results_df['prediction'].mean() * 100
print(f"\nTaxa de predições positivas: {positive_rate:.2f}%")
print(f"Probabilidade média: {results_df['probability'].mean():.4f}")
print(f"Distribuição de decis: {results_df['decile'].value_counts().sort_index().to_dict()}")

print("\nTeste de validação concluído!")