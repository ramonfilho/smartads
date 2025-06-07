# src/utils/parameter_manager.py
import joblib
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, List

class ParameterManager:
    """Gerenciador centralizado de parâmetros do pipeline ML.
    
    Responsável por:
    - Armazenar todos os parâmetros de transformação de forma estruturada
    - Garantir consistência entre fit e transform
    - Facilitar debug e rastreabilidade
    - Permitir versionamento de parâmetros
    """
    
    def __init__(self):
        self.params = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'last_updated': None
            },
            'preprocessing': {
                'quality_columns': {},
                'missing_values': {},
                'outliers': {},
                'normalization': {},
                'categorical_encoding': {},
                'temporal_features': {},
                'column_classifications': {}
            },
            'text_processing': {
                'tfidf_vectorizers': {},
                'discriminative_terms': {},
                'text_columns_processed': []
            },
            'professional_features': {
                'career_tfidf_vectorizers': {},
                'lda_models': {},
                'motivation_keywords': {},
                'aspiration_phrases': {},
                'commitment_phrases': {},
                'career_terms': {},
                'professional_columns_processed': []
            },
            'feature_engineering': {
                'excluded_columns': [],
                'preserved_columns': {},
                'created_features': [],
                'interaction_features': []
            },
            'feature_selection': {
                'selected_features': [],
                'removed_features': [],
                'importance_scores': {}
            }
        }
        self._validation_enabled = True
    
    # === MÉTODOS PRINCIPAIS ===
    
    def save_vectorizer(self, vectorizer_data: Dict, name: str, category: str = 'tfidf') -> None:
        """Salva vetorizador em local padronizado.
        
        Args:
            vectorizer_data: Dict contendo 'vectorizer' e 'feature_names'
            name: Nome/identificador do vetorizador
            category: Categoria ('tfidf', 'career_tfidf')
        """
        if category == 'tfidf':
            self.params['text_processing']['tfidf_vectorizers'][name] = vectorizer_data
            if name not in self.params['text_processing']['text_columns_processed']:
                self.params['text_processing']['text_columns_processed'].append(name)
        elif category == 'career_tfidf':
            self.params['professional_features']['career_tfidf_vectorizers'][name] = vectorizer_data
            if name not in self.params['professional_features']['professional_columns_processed']:
                self.params['professional_features']['professional_columns_processed'].append(name)
        else:
            raise ValueError(f"Categoria desconhecida: {category}")
        
        self._update_timestamp()
    
    def get_vectorizer(self, name: str, category: str = 'tfidf') -> Optional[Dict]:
        """Recupera vetorizador de local padronizado.
        
        Args:
            name: Nome/identificador do vetorizador
            category: Categoria ('tfidf', 'career_tfidf')
            
        Returns:
            Dict com vetorizador ou None se não encontrado
        """
        if category == 'tfidf':
            return self.params['text_processing']['tfidf_vectorizers'].get(name)
        elif category == 'career_tfidf':
            return self.params['professional_features']['career_tfidf_vectorizers'].get(name)
        else:
            raise ValueError(f"Categoria desconhecida: {category}")
    
    def save_lda_model(self, model_data: Dict, name: str) -> None:
        """Salva modelo LDA."""
        self.params['professional_features']['lda_models'][name] = model_data
        self._update_timestamp()
    
    def get_lda_model(self, name: str) -> Optional[Dict]:
        """Recupera modelo LDA."""
        return self.params['professional_features']['lda_models'].get(name)
    
    def save_preprocessing_params(self, param_type: str, params: Dict) -> None:
        """Salva parâmetros de pré-processamento."""
        valid_types = ['quality_columns', 'missing_values', 'outliers', 
                      'normalization', 'categorical_encoding', 'temporal_features']
        
        if param_type not in valid_types:
            raise ValueError(f"Tipo inválido: {param_type}. Válidos: {valid_types}")
        
        self.params['preprocessing'][param_type] = params
        self._update_timestamp()
    
    def get_preprocessing_params(self, param_type: str) -> Dict:
        """Recupera parâmetros de pré-processamento."""
        return self.params['preprocessing'].get(param_type, {})
    
    def save_professional_params(self, param_type: str, params: Any) -> None:
        """Salva parâmetros de features profissionais."""
        valid_types = ['motivation_keywords', 'aspiration_phrases', 
                      'commitment_phrases', 'career_terms']
        
        if param_type not in valid_types:
            raise ValueError(f"Tipo inválido: {param_type}")
        
        self.params['professional_features'][param_type] = params
        self._update_timestamp()
    
    def get_professional_params(self, param_type: str) -> Any:
        """Recupera parâmetros de features profissionais."""
        return self.params['professional_features'].get(param_type)
    
    # === MÉTODOS DE RASTREAMENTO ===
    
    def track_created_features(self, features: List[str]) -> None:
        """Rastreia features criadas durante o processamento."""
        self.params['feature_engineering']['created_features'].extend(features)
        self.params['feature_engineering']['created_features'] = list(
            set(self.params['feature_engineering']['created_features'])
        )
        self._update_timestamp()
    
    def track_excluded_columns(self, columns: List[str]) -> None:
        """Rastreia colunas excluídas do processamento."""
        self.params['feature_engineering']['excluded_columns'].extend(columns)
        self.params['feature_engineering']['excluded_columns'] = list(
            set(self.params['feature_engineering']['excluded_columns'])
        )
        self._update_timestamp()
    
    # === MÉTODOS DE PERSISTÊNCIA ===
    
    def save(self, filepath: str) -> None:
        """Salva todos os parâmetros em arquivo."""
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Atualizar timestamp
        self._update_timestamp()
        
        # Salvar
        joblib.dump(self.params, filepath)
        
        # Criar arquivo de metadados legível
        metadata_path = filepath.replace('.joblib', '_metadata.json')
        self.save_metadata(metadata_path)
        
        print(f"✓ Parâmetros salvos em: {filepath}")
    
    def load(self, filepath: str) -> None:
        """Carrega parâmetros de arquivo."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        self.params = joblib.load(filepath)
        print(f"✓ Parâmetros carregados de: {filepath}")
    
    def save_metadata(self, filepath: str) -> None:
        """Salva metadados em formato legível (JSON)."""
        metadata = {
            'metadata': self.params['metadata'],
            'summary': {
                'n_tfidf_vectorizers': len(self.params['text_processing']['tfidf_vectorizers']),
                'n_career_tfidf_vectorizers': len(self.params['professional_features']['career_tfidf_vectorizers']),
                'n_lda_models': len(self.params['professional_features']['lda_models']),
                'n_text_columns': len(self.params['text_processing']['text_columns_processed']),
                'n_created_features': len(self.params['feature_engineering']['created_features']),
                'n_selected_features': len(self.params['feature_selection']['selected_features'])
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # === MÉTODOS DE VALIDAÇÃO ===
    
    def validate_consistency(self) -> Dict[str, List[str]]:
        """Valida consistência dos parâmetros salvos."""
        issues = {
            'warnings': [],
            'errors': []
        }
        
        # Verificar se existem vetorizadores sem feature_names
        for name, data in self.params['text_processing']['tfidf_vectorizers'].items():
            if 'feature_names' not in data:
                issues['errors'].append(f"TF-IDF '{name}' sem feature_names")
        
        # Verificar se existem modelos LDA sem parâmetros essenciais
        for name, data in self.params['professional_features']['lda_models'].items():
            if 'n_topics' not in data:
                issues['errors'].append(f"LDA '{name}' sem n_topics")
        
        return issues
    
    # === MÉTODOS AUXILIARES ===
    
    def _update_timestamp(self) -> None:
        """Atualiza timestamp de última modificação."""
        self.params['metadata']['last_updated'] = datetime.now().isoformat()
    
    def get_summary(self) -> None:
        """Imprime resumo dos parâmetros armazenados."""
        print("\n" + "="*60)
        print("RESUMO DOS PARÂMETROS")
        print("="*60)
        print(f"Criado em: {self.params['metadata']['created_at']}")
        print(f"Última atualização: {self.params['metadata']['last_updated']}")
        print(f"\nVetorizadores TF-IDF: {len(self.params['text_processing']['tfidf_vectorizers'])}")
        print(f"Vetorizadores Career TF-IDF: {len(self.params['professional_features']['career_tfidf_vectorizers'])}")
        print(f"Modelos LDA: {len(self.params['professional_features']['lda_models'])}")
        print(f"Features criadas: {len(self.params['feature_engineering']['created_features'])}")
        print(f"Features selecionadas: {len(self.params['feature_selection']['selected_features'])}")
        print("="*60 + "\n")