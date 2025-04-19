"""
Script para análise de erros do modelo Random Forest, adaptado para garantir
compatibilidade com as transformações aplicadas nos scripts de pré-processamento
e seleção de features.
"""

import os
import sys
import argparse
import mlflow
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Adicionar diretório raiz ao path para importar módulos do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Importar a função exata de sanitização que foi usada no treinamento
from src.evaluation.baseline_model import sanitize_column_names

# Importar apenas os módulos necessários
from src.evaluation.error_analysis import (
    analyze_high_value_false_negatives,
    cluster_errors,
    analyze_feature_distributions_by_error_type,
    error_analysis_from_validation_data
)

def get_model_feature_names(model):
    """
    Obtém os nomes das features utilizadas pelo modelo.
    
    Args:
        model: Modelo treinado
        
    Returns:
        Lista de nomes de features
    """
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_
    elif hasattr(model, 'feature_names_'):
        return model.feature_names_
    elif hasattr(model, 'feature_name_'):
        return model.feature_name_
    elif hasattr(model, 'get_booster'):
        booster = model.get_booster()
        return booster.feature_names
    else:
        return None

def custom_error_analysis(val_df, model, target_col, threshold, output_dir):
    """
    Versão customizada da análise de erros que garante compatibilidade com o modelo.
    
    Args:
        val_df: DataFrame com dados de validação
        model: Modelo treinado
        target_col: Nome da coluna target
        threshold: Limiar de classificação
        output_dir: Diretório para salvar resultados
        
    Returns:
        Dicionário com resultados da análise
    """
    # Separar features e target
    feature_cols = [col for col in val_df.columns if col != target_col]
    
    # Obter nomes de features do modelo
    model_feature_names = get_model_feature_names(model)
    if model_feature_names is None:
        print("AVISO: Não foi possível obter nomes de features do modelo")
        model_feature_names = feature_cols
    
    # Verificar a correspondência entre as features do dataset e do modelo
    print(f"Verificando compatibilidade de features:")
    print(f"- Features no modelo: {len(model_feature_names)}")
    print(f"- Features no dataset: {len(feature_cols)}")
    
    # Criar cópia do dataframe antes de modificá-lo
    df_copy = val_df.copy()
    
    # Se os nomes das features não correspondem, aplicar a mesma sanitização
    # que foi usada durante o treinamento do modelo
    if set(feature_cols) != set(model_feature_names):
        print("ATENÇÃO: Diferença entre features do dataset e do modelo.")
        print("Aplicando a mesma sanitização de colunas usada durante o treinamento...")
        
        # Aplicar sanitização usando a mesma função do treinamento
        sanitize_column_names(df_copy)
        
        # Verificar se todos os nomes de features do modelo estão presentes
        sanitized_cols = [col for col in df_copy.columns if col != target_col]
        common_cols = set(sanitized_cols).intersection(set(model_feature_names))
        
        print(f"- Features comuns após sanitização: {len(common_cols)}")
        
        # Se ainda faltam features, adicionar colunas ausentes com valores zero
        missing_cols = set(model_feature_names) - set(sanitized_cols)
        if missing_cols:
            print(f"ATENÇÃO: Adicionando {len(missing_cols)} colunas ausentes com valor zero")
            for col in missing_cols:
                df_copy[col] = 0
    
    # Garantir que os dados estejam no formato correto (convertendo int para float)
    X_val = df_copy[[col for col in model_feature_names if col in df_copy.columns]].copy()
    for col in X_val.columns:
        if pd.api.types.is_integer_dtype(X_val[col].dtype):
            X_val.loc[:, col] = X_val[col].astype(float)
    
    y_val = df_copy[target_col]
    
    # Gerar previsões
    print("Gerando previsões para análise...")
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Análise de falsos negativos de alto valor
    fn_data = analyze_high_value_false_negatives(
        val_df=df_copy,
        y_val=y_val,
        y_pred_val=y_pred,
        y_pred_prob_val=y_pred_prob,
        target_col=target_col,
        results_dir=output_dir
    )
    
    # Análise de distribuições por tipo de erro
    error_distributions = analyze_feature_distributions_by_error_type(
        val_df=df_copy,
        y_val=y_val,
        y_pred_val=y_pred,
        y_pred_prob_val=y_pred_prob,
        target_col=target_col,
        results_dir=output_dir
    )
    
    # Preparar DataFrame para clustering
    analysis_df = df_copy.copy()
    analysis_df['error_type'] = 'correct'
    fn_mask = (y_val == 1) & (y_pred == 0)
    fp_mask = (y_val == 0) & (y_pred == 1)
    analysis_df.loc[fn_mask, 'error_type'] = 'false_negative'
    analysis_df.loc[fp_mask, 'error_type'] = 'false_positive'
    analysis_df['probability'] = y_pred_prob
    
    # Realizar clustering
    top_features = []
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': model_feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        top_features = importance['feature'].head(30).tolist()
    else:
        numeric_cols = error_distributions.get('numeric_cols', [])
        categorical_cols = error_distributions.get('categorical_cols', [])
        top_features = numeric_cols[:20] + categorical_cols[:10]
    
    # Garantir que as features de clustering existam no DataFrame
    available_features = [f for f in top_features if f in analysis_df.columns]
    if len(available_features) == 0:
        print("ALERTA: Nenhuma das top features está disponível para clustering")
        print("Usando todas as colunas numéricas disponíveis")
        available_features = [col for col in analysis_df.columns 
                             if pd.api.types.is_numeric_dtype(analysis_df[col]) 
                             and col not in ['error_type', 'probability']]
    
    clustered_df = cluster_errors(
        analysis_df=analysis_df, 
        features=available_features, 
        n_clusters=5
    )
    
    # Salvar dados com clusters
    clustered_df.to_csv(os.path.join(output_dir, "leads_clusters.csv"), index=False)
    
    # Criar dataframe com resultados
    metrics = {
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'fn': ((y_val == 1) & (y_pred == 0)).sum(),
        'fp': ((y_val == 0) & (y_pred == 1)).sum(),
        'tp': ((y_val == 1) & (y_pred == 1)).sum(),
        'tn': ((y_val == 0) & (y_pred == 0)).sum()
    }
    
    # Adicionar métricas adicionais
    metrics['fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
    
    results = {
        'metrics': metrics,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'clustered_df': clustered_df,
        'fn_data': fn_data,
        'error_distributions': error_distributions
    }
    
    # Gerar análise de resumo
    with open(os.path.join(output_dir, "error_analysis_summary.md"), "w") as f:
        f.write("# Análise de Erros do Modelo Random Forest\n\n")
        
        # Estatísticas gerais
        f.write("## Estatísticas Gerais\n\n")
        f.write(f"- Total de previsões analisadas: {len(y_val)}\n")
        f.write(f"- Previsões corretas: {metrics['tp'] + metrics['tn']} ({(metrics['tp'] + metrics['tn'])/len(y_val)*100:.2f}%)\n")
        f.write(f"- Falsos negativos: {metrics['fn']} ({metrics['fn']/len(y_val)*100:.2f}%)\n")
        f.write(f"- Falsos positivos: {metrics['fp']} ({metrics['fp']/len(y_val)*100:.2f}%)\n\n")
        
        # Métricas
        f.write("## Métricas de Desempenho\n\n")
        f.write(f"- Precision: {metrics['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['recall']:.4f}\n")
        f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"- Taxa de Falsos Positivos: {metrics['fpr']:.4f}\n\n")
        
        # Matriz de confusão
        f.write("### Matriz de Confusão\n\n")
        f.write("| | Previsto: Não Converteu | Previsto: Converteu |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Real: Não Converteu** | {metrics['tn']} | {metrics['fp']} |\n")
        f.write(f"| **Real: Converteu** | {metrics['fn']} | {metrics['tp']} |\n\n")
        
        # Falsos negativos
        if 'fn_data' in locals() and len(fn_data) > 0:
            f.write("## Análise de Falsos Negativos\n\n")
            f.write(f"- Total de falsos negativos: {len(fn_data)}\n")
            f.write("- Foram identificados padrões nos falsos negativos (veja os arquivos de análise detalhada)\n\n")
        
        # Conclusões 
        f.write("## Recomendações para Feature Engineering\n\n")
        f.write("1. **Tratamento de Texto**: \n")
        f.write("   - Refinar pesos TF-IDF para dar mais importância aos termos identificados\n")
        f.write("   - Criar embeddings para capturar melhor o contexto semântico\n")
        f.write("   - Analisar tópicos latentes usando LDA ou NMF\n\n")
        
        f.write("2. **Features de Interação**: \n")
        f.write("   - Diferença logarítmica entre salário desejado e atual\n")
        f.write("   - Interações entre país e características demográficas\n")
        f.write("   - Interações entre dia/horário e outras features\n\n")
    
    return results

def load_model_and_data(random_forest_run_id, mlflow_dir, model_path, train_path, val_path):
    """
    Carrega o modelo e os dados de treino e validação.
    
    Args:
        random_forest_run_id: ID do run do Random Forest no MLflow
        mlflow_dir: Diretório do MLflow
        model_path: Caminho para o arquivo do modelo
        train_path: Caminho para o dataset de treino
        val_path: Caminho para o dataset de validação
        
    Returns:
        Tupla (modelo, dados de treino, dados de validação)
    """
    # 1. Carregar o modelo
    model = None
    
    # Tentar carregar do MLflow primeiro
    if os.path.exists(mlflow_dir):
        print(f"Usando diretório MLflow: {mlflow_dir}")
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        
        try:
            # Carregar o modelo Random Forest diretamente pelo run_id
            model_uri = f"runs:/{random_forest_run_id}/random_forest"
            print(f"Carregando modelo Random Forest: {model_uri}")
            
            # Usar sklearn para carregar o modelo Random Forest
            model = mlflow.sklearn.load_model(model_uri)
            print("Modelo Random Forest carregado com sucesso!")
            
        except Exception as e:
            print(f"Erro ao carregar modelo Random Forest do MLflow: {str(e)}")
            print("Tentaremos usar o modelo fornecido via arquivo.")
    
    # Se não conseguiu do MLflow, tentar carregar do arquivo
    if model is None:
        if model_path and os.path.exists(model_path):
            print(f"Carregando modelo do arquivo: {model_path}")
            try:
                model = joblib.load(model_path)
                print("Modelo carregado com sucesso usando joblib")
            except Exception as joblib_error:
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    print("Modelo carregado com sucesso usando pickle")
                except Exception as pickle_error:
                    raise ValueError(f"Não foi possível carregar o modelo. Erros: Joblib: {joblib_error}, Pickle: {pickle_error}")
        else:
            model_path = input("Random Forest não encontrado. Por favor, forneça o caminho completo para o arquivo do modelo (.joblib, .pkl): ")
            try:
                model = joblib.load(model_path)
                print(f"Modelo carregado com sucesso de: {model_path}")
            except Exception as e:
                raise ValueError(f"Não foi possível carregar o modelo de {model_path}: {str(e)}")
    
    # 2. Carregar dados de treino e validação do diretório "../data/feature_selection"
    if not train_path or not os.path.exists(train_path):
        train_path = os.path.join(project_root, "data", "feature_selection", "train.csv")
        
        if not os.path.exists(train_path):
            train_path = input("Por favor, forneça o caminho completo para o arquivo de treino (.csv): ")
    
    if not val_path or not os.path.exists(val_path):
        val_path = os.path.join(project_root, "data", "feature_selection", "validation.csv")
        
        if not os.path.exists(val_path):
            val_path = input("Por favor, forneça o caminho completo para o arquivo de validação (.csv): ")
    
    print(f"Carregando dados de treino: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Carregando dados de validação: {val_path}")
    val_df = pd.read_csv(val_path)
    
    return model, train_df, val_df

def run_targeted_error_analysis(
    model_path=None,
    train_path=None, 
    val_path=None, 
    target_col="target",
    experiment_name="smart_ads_baseline_20250419_183736", 
    random_forest_run_id="2f474252aaf440dd8548514857ab1ab9",
    threshold=0.13
):
    """
    Executa análise de erros focada em falsos negativos de alto valor e clusters.
    
    Args:
        model_path: Caminho para o modelo treinado (arquivo .joblib, .pkl, etc.)
        train_path: Caminho para o dataset de treino
        val_path: Caminho para o dataset de validação
        target_col: Nome da coluna target
        experiment_name: Nome do experimento no MLflow
        random_forest_run_id: ID do run do Random Forest
        threshold: Limiar de classificação a ser usado
    """
    # 1. Configurar diretório de saída
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(reports_dir, f"error_analysis_randomforest_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Resultados serão salvos em: {output_dir}")
    
    # 2. Carregar modelo e dados
    mlflow_dir = os.path.join(project_root, "models", "mlflow")
    model, train_df, val_df = load_model_and_data(
        random_forest_run_id=random_forest_run_id,
        mlflow_dir=mlflow_dir,
        model_path=model_path,
        train_path=train_path,
        val_path=val_path
    )
    
    # 3. Obter nomes das features usadas pelo modelo
    model_features = get_model_feature_names(model)
    if model_features is not None:
        print(f"Modelo utiliza {len(model_features)} features")
        print("Exemplos de features do modelo:")
        for i, feat in enumerate(model_features[:5]):
            print(f"  {i+1}. {feat}")
    
    # 4. Executar análise customizada
    print("\n=== Executando análise de erros do modelo Random Forest ===")
    results = custom_error_analysis(
        val_df=val_df,
        model=model,
        target_col=target_col,
        threshold=threshold,
        output_dir=output_dir
    )
    
    print("\n=== Análise de erros concluída! ===")
    print(f"Todos os resultados foram salvos em: {output_dir}")
    
    return output_dir, results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Executar análise de erros com foco em falsos negativos e clusters')
    parser.add_argument('--model_path', help='Caminho para o arquivo do modelo treinado (opcional)')
    parser.add_argument('--train_path', help='Caminho para o dataset de treino (opcional)')
    parser.add_argument('--val_path', help='Caminho para o dataset de validação (opcional)')
    parser.add_argument('--target_col', default='target', help='Nome da coluna target')
    parser.add_argument('--random_forest_run_id', default='2f474252aaf440dd8548514857ab1ab9', help='ID do run do Random Forest no MLflow')
    parser.add_argument('--threshold', type=float, default=0.13, help='Limiar de classificação (0.13 para Random Forest)')
    
    args = parser.parse_args()
    
    print("=== Iniciando Análise de Erros do Modelo Random Forest ===")
    
    output_dir, results = run_targeted_error_analysis(
        model_path=args.model_path,
        train_path=args.train_path,
        val_path=args.val_path,
        target_col=args.target_col,
        random_forest_run_id=args.random_forest_run_id,
        threshold=args.threshold
    )
    
    print(f"Análise finalizada. Relatórios disponíveis em: {output_dir}")