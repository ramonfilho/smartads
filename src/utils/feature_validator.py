import numpy as np

# src/utils/feature_validator.py
class FeatureValidator:
    def __init__(self):
        self.validation_results = {}
    
    def validate_feature_consistency(self, train_df, val_df, test_df):
        """Valida consistência de features entre conjuntos."""
        results = {
            'missing_in_val': [],
            'missing_in_test': [],
            'zero_features_val': [],
            'zero_features_test': [],
            'type_mismatches': []
        }
        
        train_cols = set(train_df.columns)
        val_cols = set(val_df.columns)
        test_cols = set(test_df.columns)
        
        # Verificar colunas faltantes
        results['missing_in_val'] = list(train_cols - val_cols)
        results['missing_in_test'] = list(train_cols - test_cols)
        
        # Verificar features com todos zeros
        for col in val_df.select_dtypes(include=[np.number]).columns:
            if (val_df[col] == 0).all():
                results['zero_features_val'].append(col)
        
        for col in test_df.select_dtypes(include=[np.number]).columns:
            if (test_df[col] == 0).all():
                results['zero_features_test'].append(col)
        
        # Verificar tipos de dados
        for col in train_cols.intersection(val_cols):
            if train_df[col].dtype != val_df[col].dtype:
                results['type_mismatches'].append({
                    'column': col,
                    'train_type': str(train_df[col].dtype),
                    'val_type': str(val_df[col].dtype)
                })
        
        self.validation_results = results
        return results
    
    def generate_report(self):
        """Gera relatório de validação."""
        print("\n" + "="*60)
        print("RELATÓRIO DE VALIDAÇÃO DE FEATURES")
        print("="*60)
        
        for key, value in self.validation_results.items():
            if value:
                print(f"\n{key.upper()}:")
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        for item in value[:5]:
                            print(f"  - {item}")
                    else:
                        for item in value[:10]:
                            print(f"  - {item}")
                    if len(value) > 10:
                        print(f"  ... e mais {len(value)-10} items")