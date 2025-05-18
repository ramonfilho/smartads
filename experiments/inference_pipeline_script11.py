#!/usr/bin/env python
"""
Pipeline completa de inferência para o projeto Smart Ads.
Esta versão implementa a abordagem do script 11_final_evaluation.py,
seguindo fielmente a função evaluate_gmm_model() para garantir 
máxima consistência com o modelo de referência.

IMPORTANTE: A definição da classe GMM_Wrapper é feita no nível do módulo principal
para garantir que o joblib possa carregar o modelo corretamente.
"""

import os
import sys
import traceback
import numpy as np

print("SCRIPT INICIADO!")
print(f"Diretório de trabalho atual: {os.getcwd()}")
print(f"Python path: {sys.path}")

# CRÍTICO: Definir a classe GMM_Wrapper AQUI no nível do módulo
# Esta definição precisa estar exatamente igual à do script original
# que foi usado para salvar o modelo
class GMM_Wrapper:
    """
    Classe wrapper para o GMM que implementa a API sklearn para calibração.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pca_model = pipeline['pca_model']
        self.gmm_model = pipeline['gmm_model'] 
        self.scaler_model = pipeline['scaler_model']
        self.cluster_models = pipeline['cluster_models']
        self.n_clusters = pipeline.get('n_clusters', 3)
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
        # Preparar os dados para o modelo GMM
        X_numeric = X.select_dtypes(include=['number'])
        
        # Substituir valores NaN por 0
        X_numeric = X_numeric.fillna(0)
        
        # Aplicar o scaler
        if hasattr(self.scaler_model, 'feature_names_in_'):
            # Garantir que temos exatamente as features esperadas pelo scaler
            scaler_features = self.scaler_model.feature_names_in_
            
            # Identificar features em X_numeric que não estão no scaler
            unseen_features = [col for col in X_numeric.columns if col not in scaler_features]
            if unseen_features:
                # print(f"Removendo {len(unseen_features)} features não vistas durante treinamento")
                X_numeric = X_numeric.drop(columns=unseen_features, errors='ignore')
            
            # Identificar features que faltam em X_numeric mas estão no scaler
            missing_features = [col for col in scaler_features if col not in X_numeric.columns]
            if missing_features:
                # print(f"Adicionando {len(missing_features)} features ausentes vistas durante treinamento")
                for col in missing_features:
                    X_numeric[col] = 0.0
            
            # Garantir a ordem correta das colunas
            X_numeric = X_numeric[scaler_features]
        
        # Verificar novamente por NaNs após o ajuste de colunas
        X_numeric = X_numeric.fillna(0)
        
        X_scaled = self.scaler_model.transform(X_numeric)
        
        # Verificar por NaNs no array após scaling
        if np.isnan(X_scaled).any():
            # Se ainda houver NaNs, substitua-os por zeros
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Aplicar PCA
        X_pca = self.pca_model.transform(X_scaled)
        
        # Aplicar GMM para obter cluster labels e probabilidades
        cluster_labels = self.gmm_model.predict(X_pca)
        
        # Inicializar array de probabilidades
        n_samples = len(X)
        y_pred_proba = np.zeros((n_samples, 2), dtype=float)
        
        # Para cada cluster, fazer previsões
        for cluster_id, model_info in self.cluster_models.items():
            # Converter cluster_id para inteiro se for string
            cluster_id_int = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
            
            # Selecionar amostras deste cluster
            cluster_mask = (cluster_labels == cluster_id_int)
            
            if not any(cluster_mask):
                continue
            
            # Obter modelo específico do cluster
            model = model_info['model']
            
            # Detectar features necessárias
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                
                # Criar um DataFrame temporário com as features corretas
                X_temp = X.copy()
                
                # Lidar com features ausentes ou extras
                missing_features = [col for col in expected_features if col not in X.columns]
                for col in missing_features:
                    X_temp[col] = 0.0
                
                # Garantir a ordem correta das colunas
                features_to_use = [col for col in expected_features if col in X_temp.columns]
                X_cluster = X_temp.loc[cluster_mask, features_to_use].astype(float)
                
                # Substituir NaNs por zeros
                X_cluster = X_cluster.fillna(0)
            else:
                # Usar todas as features numéricas disponíveis
                X_cluster = X.loc[cluster_mask].select_dtypes(include=['number']).fillna(0)
            
            if len(X_cluster) > 0:
                # Fazer previsões
                try:
                    proba = model.predict_proba(X_cluster)
                    
                    # Armazenar resultados
                    y_pred_proba[cluster_mask] = proba
                except Exception as e:
                    print(f"ERRO ao fazer previsões para o cluster {cluster_id_int}: {e}")
                    # Em caso de erro, usar probabilidades default
                    y_pred_proba[cluster_mask, 0] = 0.9  # classe negativa (majoritária)
                    y_pred_proba[cluster_mask, 1] = 0.1  # classe positiva (minoritária)
        
        return y_pred_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

try:
    # Adicionar diretório raiz ao path para importar módulos do projeto
    project_root = "/Users/ramonmoreira/desktop/smart_ads"
    sys.path.insert(0, project_root)
    print(f"Diretório raiz adicionado: {project_root}")
    
    # Importações básicas
    import pandas as pd
    import joblib
    from datetime import datetime
    import argparse
    print("Importações básicas bem-sucedidas")
    
    # Importar NLTK para processamento de texto, se necessário
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Baixando recursos NLTK necessários...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    # Importar módulos da pipeline com melhor tratamento de erros
    try:
        from inference.modules.script2_module import apply_script2_transformations
        print("Módulo script2_module importado com sucesso")
    except Exception as e:
        print(f"ERRO AO IMPORTAR script2_module: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)
    
    script3_module_imported = False
    try:
        from inference.modules.script3_module import apply_script3_transformations
        script3_module_imported = True
        print("Módulo script3_module importado com sucesso")
    except Exception as e:
        print(f"AVISO: Não foi possível importar script3_module: {str(e)}")
        print(traceback.format_exc())
    
    script4_module_imported = False
    try:
        from inference.modules.script4_module import apply_script4_transformations
        script4_module_imported = True
        print("Módulo script4_module importado com sucesso")
    except Exception as e:
        print(f"AVISO: Não foi possível importar script4_module: {str(e)}")
    
    def apply_gmm_script11_style(df, params_dir=None):
        """
        Aplica o modelo GMM seguindo EXATAMENTE a implementação do script 11_final_evaluation.py.
        Copia diretamente o código da função evaluate_gmm_model() com mínimas adaptações.
        
        Args:
            df: DataFrame com features processadas
            params_dir: Parâmetro mantido por compatibilidade, mas não usado aqui
            
        Returns:
            DataFrame com predições adicionadas
        """
        print("\n=== Aplicando GMM no estilo do script 11 ===")
        print(f"Processando {len(df)} amostras...")
        
        # Copiar o DataFrame para não modificar o original
        result_df = df.copy()
        
        try:
            # ====== INÍCIO DO CÓDIGO COPIADO DO SCRIPT 11 ======
            # Caminho para o modelo calibrado (script 11)
            GMM_CALIB_DIR = "/Users/ramonmoreira/desktop/smart_ads/models/calibrated/gmm_calibrated_20250518_152543"
            model_path = os.path.join(GMM_CALIB_DIR, "gmm_calibrated.joblib")
            threshold_path = os.path.join(GMM_CALIB_DIR, "threshold.txt")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
            
            # Carregar threshold
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    gmm_threshold = float(f.read().strip())
                print(f"Threshold carregado: {gmm_threshold}")
            else:
                gmm_threshold = 0.1
                print(f"Arquivo de threshold não encontrado. Usando valor padrão: {gmm_threshold}")
            
            print(f"Carregando modelo GMM calibrado de: {model_path}")
            gmm_model = joblib.load(model_path)
            
            # Separar features e target como no script 11
            # Mesma operação exata do script 11
            X_gmm_test = result_df.copy()
            
            # Preservar o email ou outro identificador se existir
            has_email = 'email' in X_gmm_test.columns
            has_email_norm = 'email_norm' in X_gmm_test.columns
            
            print(f"Gerando probabilidades para {len(X_gmm_test)} instâncias...")
            
            # *** EXATAMENTE A MESMA CHAMADA DO SCRIPT 11 ***
            gmm_probs = gmm_model.predict_proba(X_gmm_test)[:, 1]
            gmm_preds = (gmm_probs >= gmm_threshold).astype(int)
            # ====== FIM DO CÓDIGO COPIADO DO SCRIPT 11 ======
            
            # Adicionar resultados ao DataFrame
            result_df['probability'] = gmm_probs
            result_df['prediction'] = gmm_preds
            
            # Calcular estatísticas
            prediction_counts = dict(zip(*np.unique(gmm_preds, return_counts=True)))
            print(f"  Distribuição de predições: {prediction_counts}")
            
            if 1 in prediction_counts:
                positive_rate = prediction_counts[1] / len(df)
                print(f"  Taxa de positivos: {positive_rate:.4f} ({prediction_counts.get(1, 0)} de {len(df)})")
            
            # Tempo de processamento
            print(f"Predições concluídas.")
            
        except Exception as e:
            print(f"  ERRO durante predição: {e}")
            print(traceback.format_exc())
            
            # Em caso de erro, adicionar colunas com valores default
            result_df['prediction'] = 0
            result_df['probability'] = 0.0
        
        return result_df
    
    def run_full_pipeline(input_path, output_path, params_dir, until_step=4):
        """
        Executa a pipeline completa de inferência ou até uma etapa específica.
        """
        print(f"Iniciando pipeline com: {input_path}")
        print(f"Etapas solicitadas: até {until_step}")
        
        # Carregar dados de entrada
        try:
            input_data = pd.read_csv(input_path)
            print(f"Dados carregados: {input_data.shape}")
        except Exception as e:
            print(f"Erro ao carregar dados: {str(e)}")
            sys.exit(1)
        
        # Variável para armazenar o último resultado processado
        latest_result = input_data
        
        # Etapa 1: Pré-processamento básico (script 2)
        if until_step >= 1:
            print("\n--- Etapa 1: Pré-processamento básico ---")
            params_path = os.path.join(params_dir, "02_params.joblib")  # Corrigido nome do arquivo
            if not os.path.exists(params_path):
                print(f"ERRO: Arquivo de parâmetros não encontrado: {params_path}")
                sys.exit(1)
            
            latest_result = apply_script2_transformations(input_data, params_path)
            
            # Se solicitado apenas até a etapa 1, salvar e retornar
            if until_step == 1:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                latest_result.to_csv(output_path, index=False)
                print(f"Pipeline finalizada na etapa 1. Resultado salvo em: {output_path}")
                return latest_result
        
        # Etapa 2: Processamento de texto avançado (script 3)
        step2_complete = False
        if until_step >= 2 and script3_module_imported:
            print("\n--- Etapa 2: Processamento de texto avançado ---")
            try:
                print(f"Chamando apply_script3_transformations()...")
                params_path = os.path.join(params_dir, "dummy.joblib")  # Caminho base apenas
                latest_result = apply_script3_transformations(latest_result, params_path)
                step2_complete = True
                print(f"Transformações da etapa 2 concluídas com sucesso")
                
                # Se solicitado apenas até a etapa 2, salvar e retornar
                if until_step == 2:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    latest_result.to_csv(output_path, index=False)
                    print(f"Pipeline finalizada na etapa 2. Resultado salvo em: {output_path}")
                    return latest_result
            except Exception as e:
                print(f"ERRO durante a etapa 2: {str(e)}")
                print(traceback.format_exc())
                # Continuar com o último resultado válido
        
        # Etapa 3: Features de motivação profissional (script 4)
        step3_complete = False
        if until_step >= 3 and script4_module_imported:
            print("\n--- Etapa 3: Features de motivação profissional ---")
            try:
                params_path = os.path.join(params_dir, "04_params.joblib")
                latest_result = apply_script4_transformations(latest_result, params_path)
                step3_complete = True
                
                if until_step == 3:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    latest_result.to_csv(output_path, index=False)
                    print(f"Pipeline finalizada na etapa 3. Resultado salvo em: {output_path}")
                    return latest_result
            except Exception as e:
                print(f"ERRO durante a etapa 3: {str(e)}")
                print(traceback.format_exc())
                # Continuar com o último resultado válido
        
        # Etapa 4: Aplicação de GMM com código exato do script 11
        step4_complete = False
        if until_step >= 4:
            print("\n--- Etapa 4: Aplicação de GMM (estilo script 11) ---")
            try:
                latest_result = apply_gmm_script11_style(latest_result, params_dir)
                step4_complete = True
            except Exception as e:
                print(f"ERRO durante a etapa 4: {str(e)}")
                print(traceback.format_exc())
                # Continuar com o último resultado válido
        
        # Salvar o resultado final
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        latest_result.to_csv(output_path, index=False)
        
        # Mensagem de conclusão apropriada
        highest_step = 0
        if step4_complete:
            highest_step = 4
        elif step3_complete:
            highest_step = 3
        elif step2_complete:
            highest_step = 2
        else:
            highest_step = 1
        
        print(f"\n=== Pipeline de inferência concluída até a etapa {highest_step} ===")
        print(f"Resultado salvo em: {output_path}")
        return latest_result
    
    def main():
        """Função principal."""
        parser = argparse.ArgumentParser(description="Pipeline completa de inferência (estilo script 11)")
        parser.add_argument("--input", type=str, default="/Users/ramonmoreira/desktop/smart_ads/data/01_split/test.csv",
                          help="Caminho para dados de entrada")
        parser.add_argument("--output", type=str, help="Caminho para salvar resultado")
        parser.add_argument("--params-dir", type=str, default="/Users/ramonmoreira/desktop/smart_ads/inference/params",
                          help="Diretório com parâmetros salvos")
        parser.add_argument("--until-step", type=int, choices=[1, 2, 3, 4], default=4,
                          help="Executar até qual etapa (1-4)")
        
        args = parser.parse_args()
        print(f"Argumentos: {args}")
        
        # Definir caminho de saída padrão se não especificado
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "/Users/ramonmoreira/desktop/smart_ads/inference/output"
            os.makedirs(output_dir, exist_ok=True)
            
            if args.until_step < 4:
                args.output = os.path.join(output_dir, f"inference_step{args.until_step}_{timestamp}.csv")
            else:
                args.output = os.path.join(output_dir, f"predictions_script11_style_{timestamp}.csv")
        
        # Executar pipeline
        run_full_pipeline(
            input_path=args.input,
            output_path=args.output,
            params_dir=args.params_dir,
            until_step=args.until_step
        )
        
        print(f"\n=== Pipeline de inferência concluída com sucesso! ===")
        print(f"Resultado salvo em: {args.output}")
    
    if __name__ == "__main__":
        print("Chamando função main()...")
        main()
        print("Função main() concluída.")

except Exception as e:
    print(f"ERRO GERAL: {str(e)}")
    print(traceback.format_exc())