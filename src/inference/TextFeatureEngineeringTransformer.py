# src/inference/TextFeatureEngineeringTransformer.py
#!/usr/bin/env python
"""
Transformador para processamento de texto conforme implementado
nos scripts 02_preprocessing.py e 03_feature_engineering.py.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Importar módulos de processamento de texto
from src.preprocessing.text_processing import text_feature_engineering
from src.preprocessing.advanced_feature_engineering import advanced_feature_engineering
from src.preprocessing.professional_motivation_features import (
    enhance_professional_features,
    analyze_aspiration_sentiment,
    detect_commitment_expressions,
    create_career_term_detector,
    create_professional_motivation_score
)

class TextFeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que aplica todas as etapas de processamento de texto
    implementadas nos scripts de treinamento.
    """
    
    def __init__(self, params_path=None, script03_params_path=None):
        """
        Inicializa o transformador.
        
        Args:
            params_path: Caminho para o arquivo joblib com parâmetros gerais salvos
            script03_params_path: Caminho para o arquivo joblib com parâmetros específicos do script 3
        """
        self.params_path = params_path
        self.script03_params_path = script03_params_path
        self.params = None
        self.script03_params = None
        self.feature_names = None
        
        # Colunas de texto relevantes
        self.text_cols = [
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
            '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?', 
            'Déjame un mensaje',
            '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?'
        ]
        
        # Mapeamento de colunas originais para os nomes usados nos vetorizadores
        self.column_mapping = {
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?': 'cuando_hables_inglés_con_fluid',
            '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?': 'qué_esperas_aprender_en_la',
            'Déjame un mensaje': 'déjame_un_mensaje',
            '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?': 'qué_esperas_aprender_en_la'  # Usando o mesmo vetorizador para coluna similar
        }
        
    def fit(self, X, y=None):
        """
        Carrega os parâmetros salvos durante o treino.
        
        Args:
            X: DataFrame de entrada (não utilizado)
            y: Target (não utilizado)
            
        Returns:
            self
        """
        # Carregar parâmetros principais
        if self.params_path and os.path.exists(self.params_path):
            self.params = joblib.load(self.params_path)
            print(f"Parâmetros gerais carregados de {self.params_path}")
        else:
            raise ValueError(f"Arquivo de parâmetros gerais não encontrado em {self.params_path}")
        
        # Carregar parâmetros do script 3 se especificados
        if self.script03_params_path and os.path.exists(self.script03_params_path):
            self.script03_params = joblib.load(self.script03_params_path)
            print(f"Parâmetros do script 3 carregados de {self.script03_params_path}")
            
            # Adicionar parâmetros do script 3 ao dicionário de parâmetros principal
            if self.script03_params:
                # Verificar se já existe uma chave para os parâmetros do script 3
                if 'script03_features' not in self.params:
                    self.params['script03_features'] = self.script03_params
        
        return self
    
    def transform(self, X):
        """
        Aplica processamento de texto aos dados.
        
        Args:
            X: DataFrame de entrada
            
        Returns:
            DataFrame com features de texto adicionadas
        """
        if self.params is None:
            raise ValueError("O transformador precisa ser ajustado com fit() antes de usar transform()")
        
        print(f"Aplicando processamento de texto ao DataFrame: {X.shape}")
        df_result = X.copy()
        
        # 1. Features textuais básicas
        print("1. Processando features textuais básicas...")
        text_params = self.params.get('text_processing', {})
        df_result, _ = text_feature_engineering(df_result, fit=False, params=text_params)
        
        # 2. Features avançadas
        print("2. Aplicando feature engineering avançada para texto...")
        advanced_params = self.params.get('advanced_features', {})
        df_result, _ = advanced_feature_engineering(df_result, fit=False, params=advanced_params)
        
        # 3. Features de motivação profissional - usar parâmetros do script 3 se disponíveis
        print("3. Criando features de motivação profissional...")
        
        # Primeiro verifica se temos parâmetros específicos do script 3
        if self.script03_params:
            # Verifica cada conjunto de parâmetros no script03_params
            text_cols = [col for col in self.text_cols if col in df_result.columns]
            
            if text_cols:
                # Usar os parâmetros do script 3 diretamente
                print("   Usando parâmetros específicos do script 3 para processamento de texto...")
                
                # Aplicar TF-IDF vetorizadores diretamente
                if 'vectorizers' in self.script03_params:
                    vectorizers = self.script03_params.get('vectorizers', {})
                    
                    # Processar cada coluna de texto com seu vetorizador correspondente
                    for col in text_cols:
                        # Usar o mapeamento para obter o nome correto da coluna para o vetorizador
                        mapped_col = self.column_mapping.get(col)
                        
                        if mapped_col and mapped_col in vectorizers:
                            print(f"   Aplicando vetorizador para '{mapped_col}'")
                            vectorizer = vectorizers[mapped_col]
                            
                            # Processar e aplicar o vetorizador
                            texts = df_result[col].fillna('')
                            
                            try:
                                # Transformar textos usando o vetorizador
                                tfidf_matrix = vectorizer.transform(texts)
                                feature_names = vectorizer.get_feature_names_out()
                                
                                # Converter para array para manipulação
                                tfidf_array = tfidf_matrix.toarray()
                                
                                # Adicionar colunas para cada feature do vetorizador
                                for i, term in enumerate(feature_names):
                                    feature_name = f"{mapped_col}_tfidf_{term}"
                                    df_result[feature_name] = tfidf_array[:, i]
                                
                                print(f"      Adicionadas {len(feature_names)} features TF-IDF para '{mapped_col}'")
                            except Exception as e:
                                print(f"      ERRO ao aplicar vetorizador para '{mapped_col}': {e}")
                        else:
                            print(f"   Vetorizador não encontrado para '{mapped_col if mapped_col else col}', pulando...")
                
                # Aplicar o score de motivação profissional diretamente
                if 'professional_motivation' in self.script03_params:
                    try:
                        prof_mot_params = self.script03_params.get('professional_motivation', {})
                        print("   Aplicando score de motivação profissional diretamente...")
                        df_result_prof, _ = create_professional_motivation_score(
                            df_result, text_cols, fit=False, params=prof_mot_params
                        )
                        # Adicionar colunas sem duplicação
                        for col in df_result_prof.columns:
                            if col not in df_result.columns:
                                df_result[col] = df_result_prof[col]
                        print(f"      Adicionadas {len(df_result_prof.columns)} features de motivação profissional")
                    except Exception as e:
                        print(f"      ERRO ao aplicar score de motivação profissional: {e}")
                
                # Aplicar aspiration_sentiment
                if 'aspiration_sentiment' in self.script03_params:
                    try:
                        asp_sent_params = self.script03_params.get('aspiration_sentiment', {})
                        print("   Aplicando análise de sentimento de aspiração diretamente...")
                        df_result_asp, _ = analyze_aspiration_sentiment(
                            df_result, text_cols, fit=False, params=asp_sent_params
                        )
                        # Adicionar colunas sem duplicação
                        for col in df_result_asp.columns:
                            if col not in df_result.columns:
                                df_result[col] = df_result_asp[col]
                        print(f"      Adicionadas {len(df_result_asp.columns)} features de sentimento de aspiração")
                    except Exception as e:
                        print(f"      ERRO ao aplicar análise de sentimento de aspiração: {e}")
                
                # Aplicar commitment
                if 'commitment' in self.script03_params:
                    try:
                        commitment_params = self.script03_params.get('commitment', {})
                        print("   Aplicando detecção de expressões de compromisso diretamente...")
                        df_result_comm, _ = detect_commitment_expressions(
                            df_result, text_cols, fit=False, params=commitment_params
                        )
                        # Adicionar colunas sem duplicação
                        for col in df_result_comm.columns:
                            if col not in df_result.columns:
                                df_result[col] = df_result_comm[col]
                        print(f"      Adicionadas {len(df_result_comm.columns)} features de expressões de compromisso")
                    except Exception as e:
                        print(f"      ERRO ao aplicar detecção de expressões de compromisso: {e}")
                
                # Aplicar career
                if 'career' in self.script03_params:
                    try:
                        career_params = self.script03_params.get('career', {})
                        print("   Aplicando detector de termos de carreira diretamente...")
                        df_result_career, _ = create_career_term_detector(
                            df_result, text_cols, fit=False, params=career_params
                        )
                        # Adicionar colunas sem duplicação
                        for col in df_result_career.columns:
                            if col not in df_result.columns:
                                df_result[col] = df_result_career[col]
                        print(f"      Adicionadas {len(df_result_career.columns)} features de termos de carreira")
                    except Exception as e:
                        print(f"      ERRO ao aplicar detector de termos de carreira: {e}")
            
            # IMPORTANTE: Não use enhance_professional_features aqui, pois já aplicamos
            # todas as funções diretamente e evitamos a duplicação
        
        # Use o enhance_professional_features como fallback apenas se não tivermos os parâmetros do script 3
        elif 'professional_features' in self.params:
            print("   Usando parâmetros gerais de professional_features como fallback...")
            prof_params = self.params.get('professional_features', {})
            # Filtrar colunas de texto existentes
            text_cols = [col for col in self.text_cols if col in df_result.columns]
            if text_cols:
                df_result, _ = enhance_professional_features(df_result, text_cols, fit=False, params=prof_params)
        
        # Verificar o número total de features geradas por coluna de texto
        if self.script03_params:
            for col in text_cols:
                mapped_col = self.column_mapping.get(col, "unknown")
                feature_count = sum(1 for f in df_result.columns if mapped_col in f)
                print(f"   Total de features para '{mapped_col}': {feature_count}")
        
        # Guardar nomes das features para referência
        self.feature_names = df_result.columns.tolist()
        
        print(f"Processamento de texto concluído! Dimensões finais: {df_result.shape}")
        return df_result
    
    def get_feature_names_out(self, input_features=None):
        """
        Retorna os nomes das features após a transformação.
        
        Args:
            input_features: Lista das features de entrada (não utilizado)
            
        Returns:
            Lista com nomes das features de saída
        """
        if self.feature_names is None:
            raise ValueError("O método transform deve ser chamado antes de get_feature_names_out")
        return self.feature_names