import os
import sys
import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# 1. Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 2. Importar funções necessárias
from src.evaluation.baseline_model import sanitize_column_names, get_latest_random_forest_run

# 3. Carregar modelo do MLflow
def load_model_from_mlflow(model_uri):
    """Carrega um modelo a partir do MLflow usando seu URI."""
    try:
        print(f"Carregando modelo de: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Modelo carregado com sucesso!")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {str(e)}")
        return None

# 4. Configurar caminhos e carregar dados
base_dir = os.path.expanduser("~")
mlflow_dir = os.path.join(base_dir, "desktop/smart_ads/models/mlflow")
train_path = os.path.join(base_dir, "desktop/smart_ads/data/03_feature_selection_text_code6/train.csv")
val_path = os.path.join(base_dir, "desktop/smart_ads/data/03_feature_selection_text_code6/validation.csv")

# Verificar se os arquivos existem
for path, name in [(train_path, "treino"), (val_path, "validação")]:
    if not os.path.exists(path):
        print(f"ERRO: Arquivo de {name} não encontrado: {path}")
        alternative_path = input(f"Por favor, forneça o caminho correto para o arquivo de {name}: ")
        if name == "treino":
            train_path = alternative_path
        else:
            val_path = alternative_path

# 5. Carregar dados
print(f"Carregando dados de treino: {train_path}")
train_data = pd.read_csv(train_path)

print(f"Carregando dados de validação: {val_path}")
val_data = pd.read_csv(val_path)

print(f"Dados carregados: treino {train_data.shape}, validação {val_data.shape}")

# 6. Carregar o modelo principal do MLflow
run_id, threshold, model_uri = get_latest_random_forest_run(mlflow_dir)
if not model_uri:
    print("ERRO: Não foi possível encontrar um modelo RandomForest. Abortando.")
    sys.exit(1)

main_model = load_model_from_mlflow(model_uri)
if main_model is None:
    print("ERRO: Falha ao carregar o modelo principal. Abortando.")
    sys.exit(1)

# 7. Preparar dados
def prepare_data_exactly_like_original(df, target_col="target"):
    """
    Prepara os dados da mesma forma que o script original fez,
    usando todas as colunas não-target como features.
    """
    # Criar cópia para evitar modificações no original
    df_copy = df.copy()
    
    # Sanitizar nomes de colunas como no original
    column_mapping = sanitize_column_names(df_copy)
    
    # Identificar coluna target
    target = target_col if target_col in df_copy.columns else column_mapping.get(target_col, target_col)
    
    # Usar todas as colunas não-target como features, exatamente como no original
    feature_cols = [col for col in df_copy.columns if col != target]
    
    # Extrair X e y
    X = df_copy[feature_cols]
    y = df_copy[target] if target in df_copy.columns else None
    
    # Converter inteiros para float como no original
    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col].dtype):
            X.loc[:, col] = X[col].astype(float)
    
    print(f"Dados preparados: {len(X.columns)} features, incluindo todas as colunas não-target")
    
    return X, y, feature_cols

X_train, y_train, feature_cols = prepare_data_exactly_like_original(train_data)
X_val, y_val, _ = prepare_data_exactly_like_original(val_data)

# 8. Garantir que as features correspondam exatamente
if hasattr(main_model, 'feature_names_in_'):
    expected_features = set(main_model.feature_names_in_)
    actual_features = set(X_val.columns)
    
    missing_features = expected_features - actual_features
    if missing_features:
        print(f"AVISO: Faltam {len(missing_features)} features que o modelo espera")
        print(f"  Exemplos: {list(missing_features)[:5]}")
        
        # Adicionar features faltantes com zeros
        for feat in missing_features:
            X_train[feat] = 0
            X_val[feat] = 0
    
    extra_features = actual_features - expected_features
    if extra_features:
        print(f"AVISO: Removendo {len(extra_features)} features extras que o modelo não espera")
        print(f"  Exemplos: {list(extra_features)[:5]}")
        X_train = X_train.drop(columns=list(extra_features))
        X_val = X_val.drop(columns=list(extra_features))
    
    # Garantir a ordem exata das colunas
    X_train = X_train[main_model.feature_names_in_]
    X_val = X_val[main_model.feature_names_in_]

# 9. Obter previsões do modelo principal usando o threshold do MLflow
print(f"Gerando previsões com o modelo principal usando threshold={threshold}...")
train_probs = main_model.predict_proba(X_train)[:, 1]
train_preds = (train_probs >= threshold).astype(int)

val_probs = main_model.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= threshold).astype(int)

# 10. Avaliar modelo principal com o threshold correto
baseline_precision = precision_score(y_val, val_preds)
baseline_recall = recall_score(y_val, val_preds)
baseline_f1 = f1_score(y_val, val_preds)

print(f"\nMétricas do modelo principal com threshold={threshold}:")
print(f"  Precision: {baseline_precision:.4f}")
print(f"  Recall: {baseline_recall:.4f}")
print(f"  F1: {baseline_f1:.4f}")

# 11. NOVA ABORDAGEM: Ajuste direto de probabilidades para aumentar o recall
print("\nImplementando abordagem de ajuste de probabilidade direta...")

# 11.1 Criar um dataframe com as probabilidades e targets para treinamento
print("Preparando dados para calibração...")
prob_df = pd.DataFrame({
    'probability': train_probs,
    'target': y_train
})

# 11.2 Análise da distribuição das probabilidades
print("\nAnálise da distribuição de probabilidades:")
pos_probs = prob_df[prob_df['target'] == 1]['probability']
neg_probs = prob_df[prob_df['target'] == 0]['probability']

print(f"Exemplos positivos: {len(pos_probs)}")
print(f"  Min prob: {pos_probs.min():.4f}, Max prob: {pos_probs.max():.4f}")
print(f"  Média: {pos_probs.mean():.4f}, Mediana: {pos_probs.median():.4f}")
print(f"  % abaixo do threshold ({threshold}): {(pos_probs < threshold).mean() * 100:.2f}%")

print(f"Exemplos negativos: {len(neg_probs)}")
print(f"  Min prob: {neg_probs.min():.4f}, Max prob: {neg_probs.max():.4f}")
print(f"  Média: {neg_probs.mean():.4f}, Mediana: {neg_probs.median():.4f}")
print(f"  % acima do threshold ({threshold}): {(neg_probs >= threshold).mean() * 100:.2f}%")

# 11.3 Testar thresholds variados para encontrar o melhor equilíbrio
print("\nTestando diferentes thresholds para o modelo principal:")
precision_list, recall_list, thresholds_list = precision_recall_curve(y_val, val_probs)
f1_scores = 2 * precision_list * recall_list / (precision_list + recall_list + 1e-10)

best_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds_list[best_idx] if best_idx < len(thresholds_list) else 0.5
best_f1_precision = precision_list[best_idx]
best_f1_recall = recall_list[best_idx]
best_f1_score = f1_scores[best_idx]

print(f"Melhor threshold para F1: {best_f1_threshold:.4f}")
print(f"  Precision: {best_f1_precision:.4f}, Recall: {best_f1_recall:.4f}, F1: {best_f1_score:.4f}")

# 11.4 Configurar thresholds específicos para maximizar recall (0.95 precision) e precision (0.95 recall)
high_precision_idx = np.where(precision_list >= 0.95)[0][-1] if len(np.where(precision_list >= 0.95)[0]) > 0 else 0
high_precision_threshold = thresholds_list[high_precision_idx] if high_precision_idx < len(thresholds_list) else 0.5
high_precision_recall = recall_list[high_precision_idx]

max_recall = max(recall_list)
target_recall = min(0.95, max_recall)
high_recall_idx = np.where(recall_list >= target_recall)[0][0]
high_recall_threshold = thresholds_list[high_recall_idx] if high_recall_idx < len(thresholds_list) else 0.1
high_recall_precision = precision_list[high_recall_idx]

print(f"Threshold para ≥95% precisão: {high_precision_threshold:.4f}, resultando em {high_precision_recall*100:.2f}% recall")
print(f"Threshold para máximo recall: {high_recall_threshold:.4f}, resultando em {high_recall_precision*100:.2f}% precisão")

# 11.5 Implementação alternativa: Ajuste direto de probabilidades
print("\nImplementando ajuste direto de probabilidades...")

# Definir função para ajustar probabilidades
def probability_booster(probs, boost_factor=1.5, min_prob=0.01):
    """
    Aplica um boost às probabilidades para aumentar as chances de exemplos positivos
    sem alterar a ordem relativa das previsões.
    """
    # Aplicar o boost (garantindo que não exceda 1.0)
    boosted_probs = np.minimum(probs * boost_factor, 1.0)
    
    # Garantir um mínimo para evitar confiança zero
    boosted_probs = np.maximum(boosted_probs, min_prob)
    
    return boosted_probs

# Testar diferentes fatores de boost
boost_factors = [1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
best_boost_f1 = 0
best_boost_factor = 1
best_boost_threshold = threshold

print("\nTestando diferentes fatores de ajuste de probabilidade:")

for boost in boost_factors:
    # Aplicar boost às probabilidades de validação
    boosted_val_probs = probability_booster(val_probs, boost)
    
    # Calcular curva precision-recall
    precision_boost, recall_boost, thresholds_boost = precision_recall_curve(y_val, boosted_val_probs)
    f1_scores_boost = 2 * precision_boost * recall_boost / (precision_boost + recall_boost + 1e-10)
    
    # Encontrar melhor threshold
    best_boost_idx = np.argmax(f1_scores_boost)
    boost_threshold = thresholds_boost[best_boost_idx] if best_boost_idx < len(thresholds_boost) else 0.5
    boost_precision = precision_boost[best_boost_idx]
    boost_recall = recall_boost[best_boost_idx]
    boost_f1 = f1_scores_boost[best_boost_idx]
    
    print(f"Boost {boost:.2f}x, Threshold={boost_threshold:.4f}:")
    print(f"  Precision: {boost_precision:.4f}, Recall: {boost_recall:.4f}, F1: {boost_f1:.4f}")
    
    if boost_f1 > best_boost_f1:
        best_boost_f1 = boost_f1
        best_boost_factor = boost
        best_boost_threshold = boost_threshold
        best_boost_precision = boost_precision
        best_boost_recall = boost_recall

# Aplicar o melhor fator de boost
print(f"\nMelhor fator de boost: {best_boost_factor:.2f}x, Threshold={best_boost_threshold:.4f}")
print(f"  Precision: {best_boost_precision:.4f}, Recall: {best_boost_recall:.4f}, F1: {best_boost_f1:.4f}")

boosted_val_probs = probability_booster(val_probs, best_boost_factor)
boosted_val_preds = (boosted_val_probs >= best_boost_threshold).astype(int)

# 11.6 Implementar IsotonicRegression para calibração
print("\nImplementando calibração com Isotonic Regression...")

try:
    # Treinar calibrador usando IsotonicRegression
    isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
    isotonic_calibrator.fit(train_probs.reshape(-1, 1), y_train)
    
    # Aplicar calibração
    calibrated_val_probs = isotonic_calibrator.transform(val_probs.reshape(-1, 1))
    
    # Encontrar melhor threshold
    precision_cal, recall_cal, thresholds_cal = precision_recall_curve(y_val, calibrated_val_probs)
    f1_scores_cal = 2 * precision_cal * recall_cal / (precision_cal + recall_cal + 1e-10)
    
    best_cal_idx = np.argmax(f1_scores_cal)
    best_cal_threshold = thresholds_cal[best_cal_idx] if best_cal_idx < len(thresholds_cal) else 0.5
    best_cal_precision = precision_cal[best_cal_idx]
    best_cal_recall = recall_cal[best_cal_idx]
    best_cal_f1 = f1_scores_cal[best_cal_idx]
    
    # Aplicar threshold para previsões
    cal_preds = (calibrated_val_probs >= best_cal_threshold).astype(int)
    
    print(f"Calibração Isotônica - Threshold={best_cal_threshold:.4f}:")
    print(f"  Precision: {best_cal_precision:.4f}, Recall: {best_cal_recall:.4f}, F1: {best_cal_f1:.4f}")
except Exception as e:
    print(f"Erro na calibração Isotônica: {str(e)}")
    isotonic_calibrator = None
    cal_preds = None
    best_cal_threshold = None
    best_cal_precision = 0
    best_cal_recall = 0
    best_cal_f1 = 0

# 11.7 Comparar todas as abordagens
print("\nComparação das abordagens:")
print(f"1. Modelo original (threshold={threshold}):")
print(f"  Precision: {baseline_precision:.4f}, Recall: {baseline_recall:.4f}, F1: {baseline_f1:.4f}")

print(f"2. Ajuste de probabilidade (boost={best_boost_factor:.2f}x, threshold={best_boost_threshold:.4f}):")
print(f"  Precision: {best_boost_precision:.4f}, Recall: {best_boost_recall:.4f}, F1: {best_boost_f1:.4f}")

if isotonic_calibrator is not None:
    print(f"3. Calibração Isotônica (threshold={best_cal_threshold:.4f}):")
    print(f"  Precision: {best_cal_precision:.4f}, Recall: {best_cal_recall:.4f}, F1: {best_cal_f1:.4f}")

# 12. Analisar os falsos negativos corrigidos
print("\nAnálise de falsos negativos corrigidos:")
false_negatives_original = (val_preds == 0) & (y_val == 1)

# Para o modelo com boost
corrected_fn_boost = false_negatives_original & (boosted_val_preds == 1)
added_fp_boost = (val_preds == 0) & (y_val == 0) & (boosted_val_preds == 1)

print(f"Falsos negativos no modelo original: {sum(false_negatives_original)}")
print(f"Modelo com boost de probabilidade:")
print(f"  Falsos negativos corrigidos: {sum(corrected_fn_boost)} ({sum(corrected_fn_boost)/sum(false_negatives_original)*100:.2f}%)")
print(f"  Falsos positivos adicionados: {sum(added_fp_boost)} ({sum(added_fp_boost)/sum((y_val == 0))*100:.2f}% dos negativos)")

# Para o modelo com calibração isotônica
if isotonic_calibrator is not None and cal_preds is not None:
    corrected_fn_cal = false_negatives_original & (cal_preds == 1)
    added_fp_cal = (val_preds == 0) & (y_val == 0) & (cal_preds == 1)
    
    print(f"Modelo com calibração isotônica:")
    print(f"  Falsos negativos corrigidos: {sum(corrected_fn_cal)} ({sum(corrected_fn_cal)/sum(false_negatives_original)*100:.2f}%)")
    print(f"  Falsos positivos adicionados: {sum(added_fp_cal)} ({sum(added_fp_cal)/sum((y_val == 0))*100:.2f}% dos negativos)")

# 13. Escolher a melhor abordagem
best_method = ""
best_method_threshold = 0
best_method_precision = 0
best_method_recall = 0
best_method_f1 = 0

if isotonic_calibrator is not None and best_cal_f1 > best_boost_f1 and best_cal_f1 > baseline_f1:
    best_method = "isotonic"
    best_method_threshold = best_cal_threshold
    best_method_precision = best_cal_precision
    best_method_recall = best_cal_recall
    best_method_f1 = best_cal_f1
elif best_boost_f1 > baseline_f1:
    best_method = "boost"
    best_method_threshold = best_boost_threshold
    best_method_precision = best_boost_precision
    best_method_recall = best_boost_recall
    best_method_f1 = best_boost_f1
else:
    best_method = "original"
    best_method_threshold = threshold
    best_method_precision = baseline_precision
    best_method_recall = baseline_recall
    best_method_f1 = baseline_f1

print(f"\nMelhor método: {best_method}")
print(f"  Threshold: {best_method_threshold:.4f}")
print(f"  Precision: {best_method_precision:.4f}")
print(f"  Recall: {best_method_recall:.4f}")
print(f"  F1: {best_method_f1:.4f}")

# 14. Salvar o modelo calibrado
model_dir = os.path.join(project_root, "models/enhanced_probability_model")
os.makedirs(model_dir, exist_ok=True)

enhanced_model_params = {
    'main_model_uri': model_uri,
    'main_model_threshold': threshold,
    'best_method': best_method,
    'threshold': best_method_threshold,
    'boost_factor': best_boost_factor if best_method == "boost" else None,
    'metrics': {
        'original': {
            'precision': baseline_precision,
            'recall': baseline_recall,
            'f1': baseline_f1
        },
        'enhanced': {
            'precision': best_method_precision,
            'recall': best_method_recall,
            'f1': best_method_f1
        }
    }
}

# Salvar o modelo e os parâmetros
if isotonic_calibrator is not None:
    joblib.dump(isotonic_calibrator, f"{model_dir}/isotonic_calibrator.joblib")

joblib.dump(enhanced_model_params, f"{model_dir}/enhanced_model_params.joblib")

print(f"\nModelo aprimorado e parâmetros salvos em {model_dir}")

# 15. Criar classe para o modelo aprimorado
class EnhancedProbabilityModel:
    def __init__(self, main_model, method="original", isotonic_calibrator=None, boost_factor=1.0, threshold=0.5):
        self.main_model = main_model
        self.method = method
        self.isotonic_calibrator = isotonic_calibrator
        self.boost_factor = boost_factor
        self.threshold = threshold
    
    def predict(self, X):
        # Obter probabilidades do modelo principal
        probs = self.main_model.predict_proba(X)[:, 1]
        
        # Aplicar método de aprimoramento
        if self.method == "isotonic" and self.isotonic_calibrator is not None:
            enhanced_probs = self.isotonic_calibrator.transform(probs.reshape(-1, 1))
        elif self.method == "boost":
            enhanced_probs = np.minimum(probs * self.boost_factor, 1.0)
        else:
            enhanced_probs = probs
        
        # Aplicar threshold
        return (enhanced_probs >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        # Obter probabilidades do modelo principal
        probs = self.main_model.predict_proba(X)[:, 1]
        
        # Aplicar método de aprimoramento
        if self.method == "isotonic" and self.isotonic_calibrator is not None:
            enhanced_probs = self.isotonic_calibrator.transform(probs.reshape(-1, 1))
        elif self.method == "boost":
            enhanced_probs = np.minimum(probs * self.boost_factor, 1.0)
        else:
            enhanced_probs = probs
        
        # Retornar no formato esperado por sklearn (duas colunas: 1-p, p)
        return np.vstack((1-enhanced_probs, enhanced_probs)).T

# Criar o modelo aprimorado
enhanced_model = EnhancedProbabilityModel(
    main_model=main_model,
    method=best_method,
    isotonic_calibrator=isotonic_calibrator,
    boost_factor=best_boost_factor,
    threshold=best_method_threshold
)

# Salvar o modelo aprimorado
joblib.dump(enhanced_model, f"{model_dir}/enhanced_model.joblib")

# 16. Registrar com MLflow
try:
    # Criar ou usar experimento existente
    experiment_name = "smart_ads_enhanced_model"
    
    try:
        # Verificar se experimento já existe
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            # Criar novo experimento
            experiment_id = mlflow.create_experiment(experiment_name)
        
        print(f"Usando experimento MLflow: {experiment_name} (ID: {experiment_id})")
        
        # Iniciar run com ID de experimento
        with mlflow.start_run(run_name="enhanced_probability_model", experiment_id=experiment_id):
            mlflow.log_params({
                "main_model_run_id": run_id,
                "main_model_threshold": threshold,
                "enhancement_method": best_method,
                "best_threshold": best_method_threshold,
                "boost_factor": best_boost_factor if best_method == "boost" else None
            })
            
            mlflow.log_metrics({
                "original_precision": baseline_precision,
                "original_recall": baseline_recall,
                "original_f1": baseline_f1,
                "enhanced_precision": best_method_precision,
                "enhanced_recall": best_method_recall,
                "enhanced_f1": best_method_f1,
                "recall_improvement": best_method_recall - baseline_recall,
                "f1_improvement": best_method_f1 - baseline_f1
            })
            
            # Log do modelo para MLflow
            mlflow.sklearn.log_model(enhanced_model, "enhanced_model")
            
            # Registrar JSON com informações completas
            import json
            enhancement_info_path = os.path.join(model_dir, "enhancement_info.json")
            with open(enhancement_info_path, "w") as f:
                json.dump({
                    "main_model": {
                        "run_id": run_id,
                        "uri": model_uri,
                        "threshold": threshold
                    },
                    "enhanced_model": {
                        "method": best_method,
                        "threshold": best_method_threshold,
                        "boost_factor": best_boost_factor if best_method == "boost" else None,
                        "metrics": enhanced_model_params["metrics"]
                    }
                }, f, indent=4)
            
            mlflow.log_artifact(enhancement_info_path)
            
            print(f"Experimento MLflow registrado com sucesso: {experiment_name}")
    
    except Exception as e:
        print(f"Erro específico ao criar experimento MLflow: {str(e)}")

except Exception as e:
    print(f"Erro ao registrar com MLflow: {str(e)}")
    print("Continuando sem registro MLflow...")

print("\nProcesso concluído com sucesso!")