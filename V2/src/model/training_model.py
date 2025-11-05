"""
Módulo para treino e registro do modelo - PIPELINE DE TREINO.

Reproduz a célula de modelagem do notebook DevClub.
Treina modelo RandomForest e salva artefatos.
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sklearn
import logging

logger = logging.getLogger(__name__)


def registrar_features_e_modelo_devclub(dataset_devclub_encoded: pd.DataFrame, dataset_devclub_original: pd.DataFrame) -> dict:
    """
    Registra features e salva modelo DevClub para produção.

    Reproduz a lógica da célula de modelagem do notebook DevClub.

    Args:
        dataset_devclub_encoded: DataFrame V1 DevClub encodado
        dataset_devclub_original: DataFrame V1 DevClub original (com Data)

    Returns:
        Dicionário com resultados do registro
    """
    print("REGISTRO DE FEATURES E MODELO DEVCLUB PARA PRODUÇÃO")
    print("=" * 52)

    # 1. PREPARAR DADOS E TREINAR MODELO FINAL
    print("\n1. PREPARANDO DADOS E TREINANDO MODELO FINAL")
    print("-" * 50)

    # Dataset encodado
    dataset_final = dataset_devclub_encoded.copy()

    # Dataset original para extrair datas
    dataset_original = dataset_devclub_original.copy()

    print(f"Dataset: {len(dataset_final):,} registros")
    print(f"Colunas totais: {len(dataset_final.columns)}")

    # Colunas a EXCLUIR do treinamento (não usar como features)
    colunas_excluir_treino = ['target']

    # Verificar quais colunas existem
    colunas_excluir_existentes = [col for col in colunas_excluir_treino if col in dataset_final.columns]

    print(f"Colunas excluídas do treinamento:")
    for col in colunas_excluir_existentes:
        print(f"  - {col}")

    # Preparar features e target
    X = dataset_final.drop(columns=colunas_excluir_existentes)
    y = dataset_final['target']

    # Limpar nomes das colunas
    X_clean = X.copy()
    X_clean.columns = X_clean.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    X_clean.columns = X_clean.columns.str.replace('__+', '_', regex=True)
    X_clean.columns = X_clean.columns.str.strip('_')

    # Reordenar features telefone_comprimento para ordem crescente DEPOIS da limpeza
    colunas_telefone = [col for col in X_clean.columns if col.startswith('telefone_comprimento_')]
    if colunas_telefone:
        colunas_telefone_ordenadas = sorted(colunas_telefone, key=lambda x: int(x.split('_')[-1]))
        outras_colunas = [col for col in X_clean.columns if not col.startswith('telefone_comprimento_')]

        # Encontrar posição das colunas telefone na ordem original
        primeira_pos_telefone = min(X_clean.columns.get_loc(col) for col in colunas_telefone)

        # Reconstruir ordem: colunas antes + telefone ordenado + colunas depois
        colunas_antes = [col for col in X_clean.columns[:primeira_pos_telefone] if col not in colunas_telefone]
        colunas_depois = [col for col in X_clean.columns[primeira_pos_telefone:] if col not in colunas_telefone]

        nova_ordem = colunas_antes + colunas_telefone_ordenadas + colunas_depois
        X_clean = X_clean[nova_ordem]

    print(f"Features para treinamento: {len(X_clean.columns)}")
    print(f"Target: {y.sum():,} positivos ({y.mean()*100:.2f}%)")

    # Split temporal 70/30
    data_dt = pd.to_datetime(dataset_original['Data'], errors='coerce')
    data_min = data_dt.min()
    data_max = data_dt.max()

    dias_totais = (data_max - data_min).days
    dias_treino = int(dias_totais * 0.7)
    data_corte = data_min + pd.Timedelta(days=dias_treino)

    mask_treino = data_dt <= data_corte
    mask_teste = data_dt > data_corte

    X_train = X_clean[mask_treino]
    X_test = X_clean[mask_teste]
    y_train = y[mask_treino]
    y_test = y[mask_teste]

    print(f"Split temporal:")
    print(f"  Período: {data_min.strftime('%Y-%m-%d')} a {data_max.strftime('%Y-%m-%d')}")
    print(f"  Data corte: {data_corte.strftime('%Y-%m-%d')}")
    print(f"  Treino: {len(X_train):,} registros")
    print(f"  Teste: {len(X_test):,} registros")
    print(f"  Taxa treino: {y_train.mean()*100:.2f}%")
    print(f"  Taxa teste: {y_test.mean()*100:.2f}%")

    # Treinar modelo Random Forest
    modelo_final = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    modelo_final.fit(X_train, y_train)
    y_prob = modelo_final.predict_proba(X_test)[:, 1]
    auc_final = roc_auc_score(y_test, y_prob)

    print(f"Modelo treinado - AUC: {auc_final:.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature_original': X.columns,
        'feature_clean': X_clean.columns,
        'importance': modelo_final.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"Feature importance calculada para {len(feature_importance)} features")

    # 2. CRIAR REGISTRY DE FEATURES
    print("\n2. CRIANDO FEATURE REGISTRY")
    print("-" * 50)

    # Categorizar features
    features_utm = []
    features_pesquisa = []
    features_derivadas = []
    features_outras = []

    for col in X.columns:
        if any(utm in col for utm in ['Source_', 'Medium_', 'Term_']):
            features_utm.append(col)
        elif any(pesq in col for pesq in ['gênero', 'idade', 'faz', 'faixa', 'cartão', 'estudou', 'faculdade', 'evento']):
            features_pesquisa.append(col)
        elif any(deriv in col for deriv in ['nome_', 'email_', 'telefone_', 'dia_semana']):
            features_derivadas.append(col)
        else:
            features_outras.append(col)

    # Mapeamento nome original -> nome limpo
    mapeamento_nomes = {}
    for orig, limpo in zip(X.columns, X_clean.columns):
        mapeamento_nomes[orig] = limpo

    # Criar registry completo
    feature_registry = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "model_name": "v1_devclub_rf_temporal_single",
            "dataset_name": "dataset_devclub_rf_temporal",
            "total_features": len(X.columns),
            "total_records": len(dataset_final),
            "target_column": "target",
            "model_type": "RandomForestClassifier",
            "split_type": "temporal",
            "sklearn_version": sklearn.__version__
        },
        "data_split": {
            "method": "temporal",
            "train_records": len(X_train),
            "test_records": len(X_test),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "cut_date": data_corte.strftime('%Y-%m-%d')
        },
        "feature_categories": {
            "utm_features": {
                "count": len(features_utm),
                "description": "Features derived from UTM parameters (Source, Medium, Term)",
                "features": features_utm
            },
            "survey_features": {
                "count": len(features_pesquisa),
                "description": "Features from lead survey responses",
                "features": features_pesquisa
            },
            "derived_features": {
                "count": len(features_derivadas),
                "description": "Features engineered from raw data (name, email, phone, temporal)",
                "features": features_derivadas
            },
            "other_features": {
                "count": len(features_outras),
                "description": "Additional features not in main categories",
                "features": features_outras
            }
        },
        "feature_transformations": {
            "description": "Mapping from original feature names to model-ready names",
            "name_mapping": mapeamento_nomes,
            "encoding_applied": {
                "categorical_encoding": "one-hot",
                "ordinal_features": ["Qual a sua idade?", "Atualmente, qual a sua faixa salarial?"],
                "binary_features": ["dia_semana"],
                "removed_reference_categories": ["telefone_comprimento_8"]
            },
            "column_cleaning": {
                "regex_pattern": "[^A-Za-z0-9_] -> _",
                "multiple_underscores": "__ -> _",
                "strip_underscores": "leading/trailing removed"
            }
        },
        "feature_importance": {
            "description": "Feature importance from trained RandomForestClassifier",
            "top_10_features": [
                {
                    "rank": i+1,
                    "feature_original": row['feature_original'],
                    "feature_clean": row['feature_clean'],
                    "importance": float(row['importance'])
                }
                for i, (_, row) in enumerate(feature_importance.head(10).iterrows())
            ],
            "utm_total_importance": float(
                feature_importance[
                    feature_importance['feature_original'].str.contains('Source_|Medium_|Term_', case=False, na=False)
                ]['importance'].sum()
            )
        },
        "expected_dtypes": {
            feature: str(dataset_final[feature].dtype) if feature in dataset_final.columns else "float64"
            for feature in X.columns
        },
        "validation_rules": {
            "required_features": list(X.columns),
            "optional_features": [],
            "total_expected_features": len(X.columns),
            "target_required": True,
            "missing_value_strategy": "model_will_fail_if_missing_features"
        }
    }

    print(f"Feature registry criado:")
    print(f"  - Features UTM: {len(features_utm)}")
    print(f"  - Features Pesquisa: {len(features_pesquisa)}")
    print(f"  - Features Derivadas: {len(features_derivadas)}")
    print(f"  - Features Outras: {len(features_outras)}")

    # 3. CRIAR METADADOS DO MODELO
    print("\n3. CRIANDO METADADOS DO MODELO")
    print("-" * 50)

    # Calcular métricas detalhadas
    df_analise = pd.DataFrame({
        'probabilidade': y_prob,
        'target_real': y_test.reset_index(drop=True)
    })

    df_analise['decil'] = pd.qcut(
        df_analise['probabilidade'],
        q=10,
        labels=[f'D{i}' for i in range(1, 11)],
        duplicates='drop'
    )

    analise_decis = df_analise.groupby('decil', observed=True).agg({
        'target_real': ['count', 'sum', 'mean']
    }).round(4)

    analise_decis.columns = ['total_leads', 'conversoes', 'taxa_conversao']
    analise_decis['pct_total_conversoes'] = (
        analise_decis['conversoes'] / analise_decis['conversoes'].sum() * 100
    ).round(2)

    taxa_base = y_test.mean()
    analise_decis['lift'] = (analise_decis['taxa_conversao'] / taxa_base).round(2)

    top3_conversoes = analise_decis.tail(3)['pct_total_conversoes'].sum()
    top5_conversoes = analise_decis.tail(5)['pct_total_conversoes'].sum()
    lift_maximo = analise_decis['lift'].max()

    # Monotonia
    taxas = analise_decis['taxa_conversao'].values
    crescimentos = sum(1 for i in range(1, len(taxas)) if taxas[i] >= taxas[i-1])
    monotonia = (crescimentos / (len(taxas) - 1)) * 100 if len(taxas) > 1 else 100.0

    # Metadados do modelo
    model_metadata = {
        "model_info": {
            "model_name": "v1_devclub_rf_temporal_single",
            "model_type": "RandomForestClassifier",
            "split_type": "temporal",
            "library": "scikit-learn",
            "library_version": sklearn.__version__,
            "trained_at": datetime.now().isoformat(),
            "training_duration_info": "Trained with temporal split"
        },
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        },
        "training_data": {
            "dataset_name": "dataset_devclub_rf_temporal",
            "total_records": len(dataset_final),
            "training_records": len(X_train),
            "test_records": len(X_test),
            "features_count": len(X_clean.columns),
            "target_distribution": {
                "training_positive_rate": float(y_train.mean()),
                "test_positive_rate": float(y_test.mean()),
                "training_positive_count": int(y_train.sum()),
                "test_positive_count": int(y_test.sum())
            },
            "temporal_split": {
                "period_start": data_min.strftime('%Y-%m-%d'),
                "period_end": data_max.strftime('%Y-%m-%d'),
                "cut_date": data_corte.strftime('%Y-%m-%d'),
                "training_days": dias_treino,
                "total_days": dias_totais
            }
        },
        "performance_metrics": {
            "auc": float(auc_final),
            "top3_decil_concentration": float(top3_conversoes),
            "top5_decil_concentration": float(top5_conversoes),
            "lift_maximum": float(lift_maximo),
            "monotonia_percentage": float(monotonia),
            "baseline_conversion_rate": float(taxa_base)
        },
        "decil_analysis": {
            f"decil_{i+1}": {
                "total_leads": int(row['total_leads']),
                "conversions": int(row['conversoes']),
                "conversion_rate": float(row['taxa_conversao']),
                "pct_total_conversions": float(row['pct_total_conversoes']),
                "lift": float(row['lift'])
            }
            for i, (_, row) in enumerate(analise_decis.iterrows())
        },
        "production_notes": {
            "use_case": "Lead scoring for DevClub products with budget allocation optimization",
            "prediction_interpretation": "Higher probability = higher priority for budget allocation",
            "calibration_status": "Not calibrated - use for ranking only",
            "recommended_deployment": "Batch scoring with validation on future launches",
            "monitoring_requirements": "Track performance degradation and feature drift",
            "model_limitations": f"Monotonia at {monotonia:.1f}% - investigate if < 80%"
        }
    }

    print(f"Metadados do modelo criados:")
    print(f"  - AUC: {auc_final:.3f}")
    print(f"  - Top 3 decis: {top3_conversoes:.1f}%")
    print(f"  - Lift máximo: {lift_maximo:.1f}x")
    print(f"  - Monotonia: {monotonia:.1f}%")

    # 4. SALVAR ARQUIVOS
    print("\n4. SALVANDO ARQUIVOS")
    print("-" * 50)

    # Criar pasta com timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'files/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Salvar feature registry
    registry_filename = f'{output_dir}/feature_registry_v1_devclub_rf_temporal_single.json'
    with open(registry_filename, 'w', encoding='utf-8') as f:
        json.dump(feature_registry, f, indent=2, ensure_ascii=False)
    print(f"✓ {registry_filename} salvo")

    # Salvar metadados do modelo
    metadata_filename = f'{output_dir}/model_metadata_v1_devclub_rf_temporal_single.json'
    with open(metadata_filename, 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ {metadata_filename} salvo")

    # Salvar modelo
    model_filename = f'{output_dir}/modelo_lead_scoring_v1_devclub_rf_temporal_single.pkl'
    joblib.dump(modelo_final, model_filename)
    print(f"✓ {model_filename} salvo")

    # Salvar features ordenadas
    features_filename = f'{output_dir}/features_ordenadas_v1_devclub_rf_temporal_single.json'
    features_ordenadas = {
        "feature_names": list(X_clean.columns),
        "feature_count": len(X_clean.columns),
        "created_at": datetime.now().isoformat(),
        "model_name": "v1_devclub_rf_temporal_single"
    }
    with open(features_filename, 'w', encoding='utf-8') as f:
        json.dump(features_ordenadas, f, indent=2, ensure_ascii=False)
    print(f"✓ {features_filename} salvo")

    # Salvar test set com predições
    test_set_filename = f'{output_dir}/test_set_predictions.csv'
    df_test_predictions = X_test.copy()
    df_test_predictions['target_real'] = y_test.values
    df_test_predictions['probabilidade'] = y_prob
    df_test_predictions.to_csv(test_set_filename, index=False)
    print(f"✓ {test_set_filename} salvo")

    # 5. RESUMO FINAL
    print(f"\n" + "=" * 50)
    print("MODELO DEVCLUB REGISTRADO COM SUCESSO")
    print("=" * 50)
    print(f"Modelo: v1_devclub_rf_temporal_single")
    print(f"Algoritmo: RandomForestClassifier")
    print(f"Split: temporal")
    print(f"AUC: {auc_final:.3f}")
    print(f"Features: {len(X_clean.columns)}")
    print(f"Arquivos salvos: 5")
    print(f"Pasta: {output_dir}")

    logger.info(f"✅ Modelo registrado com sucesso em {output_dir}")

    resultado_final = {
        "modelo": "v1_devclub_rf_temporal_single",
        "algoritmo": "RandomForestClassifier",
        "split": "temporal",
        "auc": auc_final,
        "top3": top3_conversoes,
        "lift": lift_maximo,
        "monotonia": monotonia,
        "features_count": len(X_clean.columns),
        "output_dir": output_dir
    }

    return resultado_final
