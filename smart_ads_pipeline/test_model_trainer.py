# test_model_trainer.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/ramonmoreira/desktop/smart_ads')

from smart_ads_pipeline.components.model_trainer import ModelTrainer

def test_model_trainer():
    """Testa o ModelTrainer com dados do pipeline."""
    
    print("=== Teste do ModelTrainer ===\n")
    
    # Carregar dados já processados
    data_dir = Path("/Users/ramonmoreira/desktop/smart_ads/data/new/04_feature_selection")
    
    print("1. Carregando dados...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # Separar features e target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_val = val_df.drop('target', axis=1)
    y_val = val_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    print(f"   Taxa conversão train: {y_train.mean():.4f}")
    
    # Criar e treinar modelo
    print("\n2. Treinando modelo...")
    trainer = ModelTrainer()
    
    metrics = trainer.train_with_ranking(
        X_train, y_train,
        X_val, y_val,
        check_column_names=True
    )
    
    print("\n3. Métricas de treino:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    # Testar predição
    print("\n4. Testando predições...")
    probs, deciles = trainer.predict_with_ranking(X_val.head(10))
    
    print("   Primeiras 10 predições:")
    for i, (p, d) in enumerate(zip(probs[:10], deciles[:10])):
        print(f"   {i+1}: prob={p:.4f}, decil={d}")
    
    # Validar no teste
    print("\n5. Validando no conjunto de teste...")
    test_metrics = trainer.validate_on_test_set(X_test, y_test)
    
    print(f"\n   Sucesso geral: {test_metrics['all_criteria_passed']}")
    
    # Salvar artefatos
    print("\n6. Salvando artefatos...")
    output_dir = "/tmp/smart_ads_model_test"
    trainer.save_artifacts(output_dir)
    
    print(f"   Artefatos salvos em: {output_dir}")
    
    # Testar carregamento
    print("\n7. Testando carregamento de artefatos...")
    new_trainer = ModelTrainer()
    new_trainer.load_artifacts(output_dir)
    
    # Verificar que funciona
    probs2, deciles2 = new_trainer.predict_with_ranking(X_val.head(5))
    print("   ✅ Modelo carregado funcionando!")
    
    return trainer

if __name__ == "__main__":
    trainer = test_model_trainer()
    print("\n✅ Teste concluído com sucesso!")