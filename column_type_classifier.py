#!/usr/bin/env python
"""
Módulo de classificação automática de tipos de colunas com suporte a LLM.

Este módulo identifica automaticamente os tipos de colunas em um DataFrame
usando regras heurísticas e, opcionalmente, LLM local para casos ambíguos.

Tipos suportados:
- Numérica (inteiro, float)
- Categórica (baixa cardinalidade)
- Texto (alta cardinalidade, texto livre)
- Data/Tempo
- Email
- Telefone
- URL
- ID/Código
- Booleana
- Mista

Autor: Smart Ads Team
Versão: 2.0 (com suporte LLM)
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
from collections import Counter
import json
import requests
import time
import os
import hashlib

warnings.filterwarnings('ignore')


class ColumnTypeClassifier:
    """
    Classificador robusto de tipos de colunas com suporte opcional a LLM e cache.
    """
    
    # Tipos de colunas suportados
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    TEXT = 'text'
    DATETIME = 'datetime'
    EMAIL = 'email'
    PHONE = 'phone'
    URL = 'url'
    ID = 'id'
    BOOLEAN = 'boolean'
    MIXED = 'mixed'
    UNKNOWN = 'unknown'
    
    # Versão do classificador (para invalidação de cache)
    VERSION = '2.1'
    
    def __init__(self, 
                 categorical_threshold: int = 100,
                 text_min_avg_length: int = 40,
                 text_min_unique_ratio: float = 0.7,
                 text_min_words: int = 5,
                 numeric_string_threshold: float = 0.9,
                 date_detection_threshold: float = 0.8,
                 sample_size: int = 1000,
                 confidence_threshold: float = 0.7,
                 # Parâmetros LLM
                 use_llm: bool = True,  # MUDADO PARA TRUE POR PADRÃO
                 llm_model: str = "phi3:mini",
                 ollama_host: str = "http://localhost:11434",
                 llm_confidence_threshold: float = 0.75,
                 llm_sample_size: int = 100,
                 cache_dir: str = "/Users/ramonmoreira/desktop/smart_ads/cache",
                 # Parâmetros de cache do classificador
                 use_classification_cache: bool = True,
                 classification_cache_path: Optional[str] = None,
                 fail_on_llm_error: bool = True):  # NOVO PARÂMETRO
        """
        Inicializa o classificador.
        
        Args:
            categorical_threshold: Número máximo de valores únicos para considerar categórica
            text_min_avg_length: Comprimento médio mínimo para considerar texto
            text_min_unique_ratio: Razão mínima de valores únicos para considerar texto
            text_min_words: Número mínimo de palavras médias para considerar texto
            numeric_string_threshold: Proporção mínima de strings numéricas
            date_detection_threshold: Proporção mínima de datas válidas
            sample_size: Tamanho da amostra para análise
            confidence_threshold: Confiança mínima para classificação
            use_llm: Se True, usa LLM para casos ambíguos
            llm_model: Modelo Ollama a usar
            ollama_host: URL do servidor Ollama
            llm_confidence_threshold: Threshold para usar LLM
            llm_sample_size: Amostra para enviar à LLM
            cache_dir: Diretório para cache de classificações
            use_classification_cache: Se True, usa cache de classificações do dataset
            classification_cache_path: Caminho específico para o cache de classificações
        """
        self.categorical_threshold = categorical_threshold
        self.text_min_avg_length = text_min_avg_length
        self.text_min_unique_ratio = text_min_unique_ratio
        self.text_min_words = text_min_words
        self.numeric_string_threshold = numeric_string_threshold
        self.date_detection_threshold = date_detection_threshold
        self.sample_size = sample_size
        self.confidence_threshold = confidence_threshold
        
        # Configurações LLM
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.ollama_host = ollama_host
        self.llm_confidence_threshold = llm_confidence_threshold
        self.llm_sample_size = llm_sample_size
        self.cache_dir = cache_dir
        self.fail_on_llm_error = fail_on_llm_error
        
        # Configurações de cache de classificação
        self.use_classification_cache = use_classification_cache
        self.classification_cache_path = classification_cache_path or os.path.join(cache_dir, "column_classifications.json")
        self.classification_cache = None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Criar diretório de cache
        os.makedirs(cache_dir, exist_ok=True)
        
        # Verificar LLM se habilitado
        if self.use_llm:
            self._check_ollama_connection()
        
        # Padrões regex compilados
        self._compile_patterns()
        
        # Conhecimento de domínio para Smart Ads
        self._setup_domain_knowledge()
        
        # Parâmetros do classificador para hash
        self._classifier_params = {
            'version': self.VERSION,
            'categorical_threshold': categorical_threshold,
            'text_min_avg_length': text_min_avg_length,
            'text_min_unique_ratio': text_min_unique_ratio,
            'text_min_words': text_min_words,
            'numeric_string_threshold': numeric_string_threshold,
            'date_detection_threshold': date_detection_threshold
        }
        
    def _compile_patterns(self):
        """Compila padrões regex para reutilização."""
        # Email pattern
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            re.IGNORECASE
        )
        
        # Phone patterns
        self.phone_patterns = [
            re.compile(r'^\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}$'),
            re.compile(r'^\(\d{2,3}\)\s?\d{4,5}-?\d{4}$'),
            re.compile(r'^\d{10,15}$'),
        ]
        
        # URL pattern
        self.url_pattern = re.compile(
            r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)$',
            re.IGNORECASE
        )
        
        # ID patterns
        self.id_patterns = [
            re.compile(r'^[A-Z0-9]{8,}$'),
            re.compile(r'^\d{6,}$'),
            re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'),
        ]
        
        # Numeric string pattern
        self.numeric_string_pattern = re.compile(r'^-?\d+\.?\d*$')
        
        # Date patterns ESPECÍFICOS
        self.strict_date_patterns = [
            (re.compile(r'^\d{4}-\d{2}-\d{2}$'), 'YYYY-MM-DD'),
            (re.compile(r'^\d{2}/\d{2}/\d{4}$'), 'DD/MM/YYYY'),
            (re.compile(r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$'), 'YYYY-MM-DD HH:MM:SS'),
            (re.compile(r'^\d{2}-\d{2}-\d{4}\s\d{2}:\d{2}$'), 'DD-MM-YYYY HH:MM'),
            (re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'), 'ISO8601'),
        ]
        
    def _setup_domain_knowledge(self):
        """Configura conhecimento específico do domínio Smart Ads."""
        # Listas negras - colunas que NÃO são do tipo especificado
        self.datetime_blacklist = [
            'utm_', 'gclid', 'fbclid', 'qualidade', 'quality',
            'nombre', 'name', 'género', 'gender', 'edad', 'age',
            'país', 'country', 'profesión', 'profession', 'sueldo', 'salary',
            'ganar', 'earn', 'razones', 'reasons', 'email', 'e-mail',
            'telefone', 'phone', 'instagram', 'whatsapp'
        ]
        
        # Listas brancas - colunas que SÃO do tipo especificado
        self.datetime_whitelist = [
            'data', 'date', 'fecha', 'timestamp', 'temporal',
            'created', 'updated', 'modified', 'cadastro'
        ]
        
        # Mapeamentos específicos
        self.utm_keywords = [
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_campaing',  # typo comum
            'utm_content', 'utm_term', 'gclid', 'fbclid', 'msclkid'
        ]
        
        self.text_keywords = [
            'mensaje', 'message', 'descripción', 'description',
            'observación', 'observation', 'comentario', 'comment',
            'oportunidades', 'opportunities', 'esperas', 'expect',
            'cambiará', 'change', 'profesión', 'profession'
        ]
    
    def _check_ollama_connection(self):
        """Verifica se o Ollama está acessível (apenas se use_llm=True)."""
        if not self.use_llm:
            return
            
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.llm_model not in model_names:
                    error_msg = f"""
❌ ERRO: Modelo '{self.llm_model}' não encontrado no Ollama.

Modelos disponíveis: {model_names}

Para instalar o modelo:
  ollama pull {self.llm_model}
"""
                    if self.fail_on_llm_error:
                        raise RuntimeError(error_msg)
                    else:
                        print(error_msg)
                        self.use_llm = False
                else:
                    print(f"✓ Ollama conectado. Modelo '{self.llm_model}' disponível.")
            else:
                raise ConnectionError(f"Ollama retornou status {response.status_code}")
                
        except requests.exceptions.ConnectionError as e:
            error_msg = f"""
❌ ERRO: Não foi possível conectar ao Ollama em {self.ollama_host}

Por favor, certifique-se de que o Ollama está rodando:
  1. Abra um novo terminal
  2. Execute: ollama serve
  3. Tente novamente

Erro técnico: {str(e)}
"""
            if self.fail_on_llm_error:
                raise RuntimeError(error_msg)
            else:
                print(error_msg)
                print("⚠️ Continuando sem LLM...")
                self.use_llm = False
        except Exception as e:
            error_msg = f"❌ Erro inesperado ao verificar Ollama: {e}"
            if self.fail_on_llm_error:
                raise RuntimeError(error_msg)
            else:
                print(error_msg)
                self.use_llm = False
    
    def classify_column(self, series: pd.Series, column_name: str = None) -> Dict[str, Any]:
        """
        Classifica o tipo de uma coluna.
        """
        # Primeiro, usar classificação baseada em regras
        base_result = self._classify_with_rules(series, column_name)
        
        # Se LLM habilitado e confiança baixa, usar LLM
        if self.use_llm and base_result['confidence'] < self.llm_confidence_threshold:
            llm_result = self._classify_with_llm(series, column_name, base_result)
            if llm_result:
                return llm_result
        
        return base_result
    
    def _classify_with_rules(self, series: pd.Series, column_name: str = None) -> Dict[str, Any]:
        """
        Classifica usando regras heurísticas (método original).
        """
        # Amostragem
        if len(series) > self.sample_size:
            sample = series.dropna().sample(min(self.sample_size, len(series.dropna())), random_state=42)
        else:
            sample = series.dropna()
        
        if len(sample) == 0:
            return {
                'type': self.UNKNOWN,
                'confidence': 0.0,
                'metadata': {'reason': 'all_null'}
            }
        
        # Análise inicial
        dtype = str(series.dtype)
        unique_count = series.nunique()
        total_count = len(series)
        null_ratio = series.isna().sum() / total_count
        
        # 1. Verificar tipos numéricos nativos
        if pd.api.types.is_numeric_dtype(series):
            if unique_count <= 2 and set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                return {
                    'type': self.BOOLEAN,
                    'confidence': 0.95,
                    'metadata': {
                        'subtype': 'numeric_boolean',
                        'unique_values': list(series.dropna().unique())
                    }
                }
            return {
                'type': self.NUMERIC,
                'confidence': 0.99,
                'metadata': {
                    'subtype': 'float' if series.dtype == float else 'integer',
                    'range': [series.min(), series.max()],
                    'mean': series.mean() if not series.empty else None
                }
            }
        
        # 2. Para tipos object, fazer análise detalhada
        if dtype == 'object':
            str_sample = sample.astype(str)
            
            # ORDEM CRÍTICA DE VERIFICAÇÕES:
            
            # 2.1 Verificar datetime PRIMEIRO (antes de UTM)
            date_result = self._check_datetime_strict(str_sample, column_name)
            if date_result['is_datetime']:
                return {
                    'type': self.DATETIME,
                    'confidence': date_result['confidence'],
                    'metadata': date_result['metadata']
                }
            
            # 2.2 Verificar UTM/Tracking
            utm_result = self._check_utm_or_tracking(str_sample, column_name, unique_count, total_count)
            if utm_result.get('is_utm'):
                return {
                    'type': self.CATEGORICAL,
                    'confidence': utm_result['confidence'],
                    'metadata': utm_result['metadata']
                }
            
            # 2.3 Verificar booleano
            bool_result = self._check_boolean(str_sample, series)
            if bool_result['is_boolean']:
                return {
                    'type': self.BOOLEAN,
                    'confidence': bool_result['confidence'],
                    'metadata': bool_result['metadata']
                }
            
            # 2.4 Verificar email
            email_result = self._check_email(str_sample, column_name)
            if email_result['is_email']:
                return {
                    'type': self.EMAIL,
                    'confidence': email_result['confidence'],
                    'metadata': email_result['metadata']
                }
            
            # 2.5 Verificar telefone
            phone_result = self._check_phone(str_sample, column_name)
            if phone_result['is_phone']:
                return {
                    'type': self.PHONE,
                    'confidence': phone_result['confidence'],
                    'metadata': phone_result['metadata']
                }
            
            # 2.6 Verificar URL
            url_result = self._check_url(str_sample)
            if url_result['is_url']:
                return {
                    'type': self.URL,
                    'confidence': url_result['confidence'],
                    'metadata': url_result['metadata']
                }
            
            # 2.7 Verificar numérico disfarçado
            numeric_result = self._check_numeric_string(str_sample)
            if numeric_result['is_numeric']:
                return {
                    'type': self.NUMERIC,
                    'confidence': numeric_result['confidence'],
                    'metadata': numeric_result['metadata']
                }
            
            # 2.8 Verificar ID
            id_result = self._check_id(str_sample, column_name, unique_count, total_count)
            if id_result['is_id']:
                return {
                    'type': self.ID,
                    'confidence': id_result['confidence'],
                    'metadata': id_result['metadata']
                }
            
            # 2.9 Texto vs Categórica
            text_result = self._check_text_vs_categorical(
                series, str_sample, unique_count, total_count, column_name
            )
            return text_result
        
        # 3. Tipo datetime nativo
        if pd.api.types.is_datetime64_any_dtype(series):
            return {
                'type': self.DATETIME,
                'confidence': 0.99,
                'metadata': {
                    'subtype': 'pandas_datetime',
                    'format': 'native'
                }
            }
        
        return {
            'type': self.UNKNOWN,
            'confidence': 0.0,
            'metadata': {'dtype': dtype}
        }
    
    # ===== MÉTODOS DE VERIFICAÇÃO (mantidos do código original) =====
    
    def _check_utm_or_tracking(self, sample: pd.Series, column_name: Optional[str],
                              unique_count: int, total_count: int) -> Dict[str, Any]:
        """Verifica se é uma coluna de tracking/UTM."""
        if not column_name:
            return {'is_utm': False}
        
        col_lower = column_name.lower()
        
        for utm_keyword in self.utm_keywords:
            if utm_keyword in col_lower:
                return {
                    'is_utm': True,
                    'confidence': 0.95,
                    'metadata': {
                        'type': 'tracking_parameter',
                        'matched_keyword': utm_keyword,
                        'unique_values': unique_count,
                        'unique_ratio': unique_count / total_count
                    }
                }
        
        return {'is_utm': False}
    
    def _check_datetime_strict(self, sample: pd.Series, column_name: Optional[str]) -> Dict[str, Any]:
        """Verifica datetime com rigor."""
        # Verificar se os dados parecem datas
        date_like_count = 0
        sample_str = sample.astype(str).head(20)
        
        for val in sample_str:
            has_numbers = any(c.isdigit() for c in val)
            has_date_separators = any(sep in val for sep in ['-', '/', ':'])
            looks_like_date = has_numbers and has_date_separators and len(val) >= 8
            
            if looks_like_date:
                date_like_count += 1
        
        if date_like_count < len(sample_str) * 0.5:
            return {'is_datetime': False}
        
        # Verificar nome da coluna
        confidence_boost = 0
        if column_name:
            col_lower = column_name.lower()
            
            if any(white in col_lower for white in ['data', 'date', 'timestamp', 'temporal']):
                confidence_boost = 0.1
            
            if any(black in col_lower for black in ['edad', 'sueldo', 'género', 'país']):
                return {'is_datetime': False}
        
        # Verificar padrões
        for pattern, format_name in self.strict_date_patterns:
            matches = sample.astype(str).str.match(pattern)
            match_ratio = matches.sum() / len(sample)
            
            if match_ratio > 0.8:
                return {
                    'is_datetime': True,
                    'confidence': min(1.0, match_ratio + confidence_boost),
                    'metadata': {
                        'format': format_name,
                        'match_ratio': match_ratio,
                        'detected_by': 'pattern_matching'
                    }
                }
        
        return {'is_datetime': False}
    
    def _detect_multiple_choice_patterns(self, series: pd.Series, sample: pd.Series) -> Dict[str, Any]:
        """Detecta padrões de múltipla escolha."""
        value_counts = series.value_counts()
        top_values = value_counts.head(20)
        
        yes_no_patterns = {
            'sí', 'si', 'no', 'tal vez', 'talvez', 'quizás', 'quizas',
            'definitivamente', 'probablemente', 'nunca', 'siempre',
            'yes', 'no', 'maybe', 'definitely', 'probably'
        }
        
        time_patterns = {
            'año', 'años', 'mes', 'meses', 'semana', 'semanas',
            'día', 'días', 'hora', 'horas', 'minuto', 'minutos',
            'siempre', 'nunca', 'a veces', 'frecuentemente'
        }
        
        range_patterns = {
            'más de', 'menos de', 'entre', 'hasta', 'desde',
            '-', 'a', 'o más', 'o menos'
        }
        
        unique_values_lower = [str(v).lower().strip() for v in series.dropna().unique()]
        
        # Verificar respostas sim/não
        yes_no_matches = sum(1 for val in unique_values_lower if val in yes_no_patterns)
        if yes_no_matches >= len(unique_values_lower) * 0.5:
            return {
                'is_multiple_choice': True,
                'pattern': 'yes_no',
                'confidence': 0.95
            }
        
        # Verificar ranges
        range_count = sum(1 for val in unique_values_lower 
                         if any(pattern in val for pattern in range_patterns))
        if range_count >= len(unique_values_lower) * 0.5:
            return {
                'is_multiple_choice': True,
                'pattern': 'numeric_ranges',
                'confidence': 0.9
            }
        
        # Verificar tempo
        time_count = sum(1 for val in unique_values_lower 
                        if any(pattern in val for pattern in time_patterns))
        if time_count >= len(unique_values_lower) * 0.5:
            return {
                'is_multiple_choice': True,
                'pattern': 'time_periods',
                'confidence': 0.9
            }
        
        # Opções limitadas
        avg_length = sample.str.len().mean()
        if len(unique_values_lower) <= 20 and avg_length < 30:
            return {
                'is_multiple_choice': True,
                'pattern': 'limited_options',
                'confidence': 0.8
            }
        
        return {'is_multiple_choice': False}
    
    def _check_text_vs_categorical(self, series: pd.Series, sample: pd.Series,
                                  unique_count: int, total_count: int,
                                  column_name: Optional[str]) -> Dict[str, Any]:
        """Decide entre texto e categórica."""
        # Verificar múltipla escolha
        mc_result = self._detect_multiple_choice_patterns(series, sample)
        if mc_result.get('is_multiple_choice'):
            return {
                'type': self.CATEGORICAL,
                'confidence': mc_result['confidence'],
                'metadata': {
                    'reason': 'multiple_choice_pattern',
                    'pattern': mc_result['pattern'],
                    'unique_count': unique_count,
                    'categories': list(series.value_counts().head(10).index),
                    'distribution': series.value_counts(normalize=True).head(10).to_dict()
                }
            }
        
        # Métricas
        unique_ratio = unique_count / total_count
        avg_length = sample.str.len().mean()
        max_length = sample.str.len().max()
        min_length = sample.str.len().min()
        length_std = sample.str.len().std()
        
        word_counts = sample.str.split().str.len()
        avg_words = word_counts.mean() if not word_counts.empty else 0
        max_words = word_counts.max() if not word_counts.empty else 0
        
        has_punctuation = sample.str.contains(r'[.!?;,:]', regex=True).sum() / len(sample)
        has_multiple_sentences = sample.str.contains(r'[.!?]\s+[A-Z]', regex=True).sum() / len(sample)
        
        # Decisões baseadas em características
        
        if avg_words >= self.text_min_words and has_punctuation > 0.3:
            return {
                'type': self.TEXT,
                'confidence': 0.9,
                'metadata': {
                    'reason': 'high_word_count_with_punctuation',
                    'avg_words': avg_words,
                    'has_punctuation_ratio': has_punctuation,
                    'unique_ratio': unique_ratio
                }
            }
        
        if avg_length >= self.text_min_avg_length and length_std > 20:
            return {
                'type': self.TEXT,
                'confidence': 0.85,
                'metadata': {
                    'reason': 'long_variable_responses',
                    'avg_length': avg_length,
                    'length_std': length_std,
                    'unique_ratio': unique_ratio
                }
            }
        
        if has_multiple_sentences > 0.2:
            return {
                'type': self.TEXT,
                'confidence': 0.9,
                'metadata': {
                    'reason': 'multiple_sentences',
                    'multiple_sentence_ratio': has_multiple_sentences,
                    'avg_words': avg_words
                }
            }
        
        if unique_count < 10 and avg_words <= 3:
            return {
                'type': self.CATEGORICAL,
                'confidence': 0.95,
                'metadata': {
                    'reason': 'low_unique_short_answers',
                    'unique_count': unique_count,
                    'avg_words': avg_words,
                    'categories': list(series.value_counts().head(10).index)
                }
            }
        
        top_values = series.value_counts().head(10)
        top_values_ratio = top_values.sum() / total_count
        
        if top_values_ratio > 0.7 and avg_words < self.text_min_words:
            return {
                'type': self.CATEGORICAL,
                'confidence': 0.85,
                'metadata': {
                    'reason': 'concentrated_distribution',
                    'top_10_coverage': top_values_ratio,
                    'unique_count': unique_count,
                    'distribution': series.value_counts(normalize=True).head(10).to_dict()
                }
            }
        
        if unique_count <= self.categorical_threshold and avg_words < self.text_min_words:
            return {
                'type': self.CATEGORICAL,
                'confidence': 0.8,
                'metadata': {
                    'reason': 'moderate_cardinality_short_text',
                    'unique_count': unique_count,
                    'avg_words': avg_words,
                    'avg_length': avg_length
                }
            }
        
        if unique_ratio > 0.9 and length_std < 5:
            return {
                'type': self.CATEGORICAL,
                'confidence': 0.7,
                'metadata': {
                    'reason': 'possible_id_field',
                    'unique_ratio': unique_ratio,
                    'length_consistency': length_std
                }
            }
        
        if (avg_words >= self.text_min_words or 
            avg_length >= self.text_min_avg_length or 
            unique_ratio >= self.text_min_unique_ratio):
            return {
                'type': self.TEXT,
                'confidence': 0.75,
                'metadata': {
                    'reason': 'text_characteristics',
                    'avg_words': avg_words,
                    'avg_length': avg_length,
                    'unique_ratio': unique_ratio,
                    'has_punctuation': has_punctuation
                }
            }
        
        return {
            'type': self.CATEGORICAL,
            'confidence': 0.7,
            'metadata': {
                'reason': 'default_categorical',
                'unique_count': unique_count,
                'avg_length': avg_length,
                'avg_words': avg_words
            }
        }
    
    def _check_boolean(self, sample: pd.Series, original_series: pd.Series) -> Dict[str, Any]:
        """Verifica se é booleano."""
        bool_values = {
            'true', 'false', 't', 'f', 'yes', 'no', 'y', 'n',
            'sim', 'não', 'nao', 's', 'n', '1', '0',
            'verdadero', 'falso', 'v', 'f', 'si', 'sí'
        }
        
        lower_sample = sample.str.lower().str.strip()
        unique_lower = set(lower_sample.unique())
        
        if len(unique_lower) <= 2 and unique_lower.issubset(bool_values):
            return {
                'is_boolean': True,
                'confidence': 0.95,
                'metadata': {
                    'unique_values': list(original_series.dropna().unique()),
                    'value_counts': original_series.value_counts().to_dict()
                }
            }
        
        return {'is_boolean': False}
    
    def _check_email(self, sample: pd.Series, column_name: Optional[str]) -> Dict[str, Any]:
        """Verifica emails."""
        if column_name:
            email_keywords = ['email', 'e-mail', 'mail', 'correo', 'correio']
            if any(keyword in column_name.lower() for keyword in email_keywords):
                email_match_ratio = sample.str.match(self.email_pattern).sum() / len(sample)
                if email_match_ratio > 0.5:
                    return {
                        'is_email': True,
                        'confidence': 0.9,
                        'metadata': {
                            'match_ratio': email_match_ratio,
                            'detected_by': 'name_and_pattern'
                        }
                    }
        
        email_match_ratio = sample.str.match(self.email_pattern).sum() / len(sample)
        if email_match_ratio > 0.8:
            return {
                'is_email': True,
                'confidence': email_match_ratio,
                'metadata': {
                    'match_ratio': email_match_ratio,
                    'detected_by': 'pattern'
                }
            }
        
        return {'is_email': False}
    
    def _check_phone(self, sample: pd.Series, column_name: Optional[str]) -> Dict[str, Any]:
        """Verifica telefones."""
        if column_name:
            phone_keywords = ['phone', 'telefone', 'tel', 'celular', 'mobile', 'whatsapp', 'fone']
            if any(keyword in column_name.lower() for keyword in phone_keywords):
                confidence_boost = 0.2
            else:
                confidence_boost = 0.0
        else:
            confidence_boost = 0.0
        
        cleaned_sample = sample.str.replace(r'[\s\-\(\)\.]+', '', regex=True)
        
        max_match_ratio = 0
        for pattern in self.phone_patterns:
            match_ratio = cleaned_sample.str.match(pattern).sum() / len(cleaned_sample)
            max_match_ratio = max(max_match_ratio, match_ratio)
        
        adjusted_confidence = min(1.0, max_match_ratio + confidence_boost)
        
        if adjusted_confidence > 0.6:
            return {
                'is_phone': True,
                'confidence': adjusted_confidence,
                'metadata': {
                    'match_ratio': max_match_ratio,
                    'name_boost': confidence_boost > 0
                }
            }
        
        return {'is_phone': False}
    
    def _check_url(self, sample: pd.Series) -> Dict[str, Any]:
        """Verifica URLs."""
        url_match_ratio = sample.str.match(self.url_pattern).sum() / len(sample)
        
        if url_match_ratio < 0.7:
            domain_pattern = re.compile(r'^[\w\-]+(\.[\w\-]+)+(/.*)?$')
            domain_match_ratio = sample.str.match(domain_pattern).sum() / len(sample)
            
            if domain_match_ratio > 0.8:
                return {
                    'is_url': True,
                    'confidence': domain_match_ratio * 0.8,
                    'metadata': {
                        'match_ratio': domain_match_ratio,
                        'type': 'domain_only'
                    }
                }
        
        if url_match_ratio > 0.7:
            return {
                'is_url': True,
                'confidence': url_match_ratio,
                'metadata': {
                    'match_ratio': url_match_ratio,
                    'type': 'full_url'
                }
            }
        
        return {'is_url': False}
    
    def _check_numeric_string(self, sample: pd.Series) -> Dict[str, Any]:
        """Verifica strings numéricas."""
        cleaned_sample = sample.str.strip()
        numeric_match_ratio = cleaned_sample.str.match(self.numeric_string_pattern).sum() / len(cleaned_sample)
        
        if numeric_match_ratio > self.numeric_string_threshold:
            try:
                numeric_values = pd.to_numeric(cleaned_sample, errors='coerce')
                conversion_success_ratio = numeric_values.notna().sum() / len(cleaned_sample)
                
                if conversion_success_ratio > 0.9:
                    return {
                        'is_numeric': True,
                        'confidence': conversion_success_ratio,
                        'metadata': {
                            'match_ratio': numeric_match_ratio,
                            'conversion_ratio': conversion_success_ratio,
                            'subtype': 'numeric_string'
                        }
                    }
            except:
                pass
        
        return {'is_numeric': False}
    
    def _check_id(self, sample: pd.Series, column_name: Optional[str], 
                  unique_count: int, total_count: int) -> Dict[str, Any]:
        """Verifica IDs."""
        if column_name:
            id_keywords = ['id', 'code', 'codigo', 'key', 'identifier', 'uuid', 'guid', 'sku']
            if any(keyword in column_name.lower() for keyword in id_keywords):
                confidence_boost = 0.3
            else:
                confidence_boost = 0.0
        else:
            confidence_boost = 0.0
        
        uniqueness_ratio = unique_count / total_count
        
        if uniqueness_ratio < 0.8:
            return {'is_id': False}
        
        max_match_ratio = 0
        for pattern in self.id_patterns:
            match_ratio = sample.str.match(pattern).sum() / len(sample)
            max_match_ratio = max(max_match_ratio, match_ratio)
        
        lengths = sample.str.len()
        same_length_ratio = (lengths == lengths.mode()[0]).sum() / len(lengths)
        
        confidence = (uniqueness_ratio * 0.4 + 
                     max_match_ratio * 0.3 + 
                     same_length_ratio * 0.3 + 
                     confidence_boost)
        
        if confidence > 0.7:
            return {
                'is_id': True,
                'confidence': min(1.0, confidence),
                'metadata': {
                    'uniqueness_ratio': uniqueness_ratio,
                    'pattern_match_ratio': max_match_ratio,
                    'same_length_ratio': same_length_ratio
                }
            }
        
        return {'is_id': False}
    
    # ===== MÉTODOS LLM =====
    
    def _get_cache_key(self, column_name: str, sample_data: List[str]) -> str:
        """Gera chave para cache."""
        data_str = f"{column_name}:{':'.join(sorted(set(str(s) for s in sample_data[:20])))}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Carrega do cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, classification: Dict[str, Any]):
        """Salva no cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_path, 'w') as f:
            json.dump(classification, f)
    
    def _prepare_llm_sample(self, series: pd.Series) -> Tuple[List[str], Dict[str, Any]]:
        """Prepara amostra para LLM."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return [], {'total_values': 0, 'null_count': len(series)}
        
        stats = {
            'total_values': len(series),
            'null_count': series.isna().sum(),
            'unique_count': series.nunique(),
            'unique_ratio': series.nunique() / len(series)
        }
        
        if len(clean_series) <= self.llm_sample_size:
            sample = clean_series.astype(str).tolist()
        else:
            value_counts = clean_series.value_counts()
            n_frequent = min(len(value_counts), self.llm_sample_size // 2)
            frequent_values = value_counts.head(n_frequent).index.tolist()
            
            remaining = self.llm_sample_size - n_frequent
            random_values = clean_series[~clean_series.isin(frequent_values)].sample(
                min(remaining, len(clean_series) - n_frequent),
                random_state=42
            ).tolist()
            
            sample = [str(v) for v in frequent_values + random_values]
        
        if series.dtype == 'object':
            str_series = clean_series.astype(str)
            stats['avg_length'] = str_series.str.len().mean()
            stats['avg_words'] = str_series.str.split().str.len().mean()
            stats['has_punctuation_ratio'] = str_series.str.contains(r'[.!?;,:]', regex=True).sum() / len(str_series)
        
        return sample, stats
    
    def _create_llm_prompt(self, column_name: str, sample_data: List[str], 
                          stats: Dict[str, Any]) -> str:
        """Cria prompt para LLM."""
        prompt = f"""You are a data type classifier for a machine learning pipeline. Analyze the following column and classify its type.

Column Name: {column_name}

Statistics:
- Total values: {stats['total_values']}
- Null values: {stats['null_count']}
- Unique values: {stats['unique_count']}
- Unique ratio: {stats.get('unique_ratio', 0):.2f}
- Average length: {stats.get('avg_length', 'N/A')}
- Average words: {stats.get('avg_words', 'N/A')}

Sample Values (showing {len(sample_data)} examples):
{json.dumps(sample_data[:50], indent=2, ensure_ascii=False)}

CRITICAL CLASSIFICATION RULES (IN ORDER OF PRIORITY):

1. **datetime** - If values contain dates in ANY format (dd/mm/yyyy, yyyy-mm-dd, timestamps), classify as "datetime"
   - Column name "DATA" or "Marca temporal" → ALWAYS "datetime"
   - Values like "15/01/2024" or "2024-01-15" → ALWAYS "datetime"
   - Timestamp values → ALWAYS "datetime"

2. **text** - High cardinality free text responses
   - Names of people (¿Cómo te llamas?) with thousands of unique values → "text"
   - Instagram handles → "text"
   - Professions with high variety → "text"
   - Open-ended questions with long answers → "text"
   - Messages or comments → "text"

3. **categorical** - Limited set of predefined options
   - UTM parameters → "categorical"
   - Multiple choice questions → "categorical"
   - Yes/No questions → "categorical"
   - Age ranges, salary ranges → "categorical"
   - Countries, genders → "categorical"

4. **numeric** - Pure numbers
5. **email** - Email addresses
6. **phone** - Phone numbers
7. **boolean** - Binary values (Yes/No, True/False)
8. **url** - Web addresses
9. **id** - Unique identifiers

IMPORTANT: Look at the actual data values, not just the column name!

Respond with a JSON object containing:
{{
  "type": "the_column_type",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Only respond with the JSON, no additional text."""
        
        return prompt
    
    def _query_llm(self, prompt: str) -> Dict[str, Any]:
        """Consulta LLM."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "seed": 42
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                    return classification
                else:
                    print(f"⚠️ Resposta sem JSON válido")
                    return None
            else:
                print(f"❌ Erro na API: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Erro LLM: {e}")
            return None
    
    def _classify_with_llm(self, series: pd.Series, column_name: str,
                          base_classification: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Classifica usando LLM."""
        if not self.use_llm:
            return None
            
        print(f"\n🤖 Usando LLM para '{column_name}' (confiança: {base_classification['confidence']:.2f})")
        
        sample_data, stats = self._prepare_llm_sample(series)
        if not sample_data:
            return None
        
        # Cache
        cache_key = self._get_cache_key(column_name, sample_data)
        cached = self._load_from_cache(cache_key)
        if cached:
            print("  ✓ Usando cache")
            return cached
        
        # Prompt
        prompt = self._create_llm_prompt(column_name, sample_data, stats)
        
        # Query
        print(f"  🔄 Consultando {self.llm_model}...")
        start_time = time.time()
        llm_result = self._query_llm(prompt)
        elapsed = time.time() - start_time
        print(f"  ⏱️ {elapsed:.1f}s")
        
        if llm_result and all(k in llm_result for k in ['type', 'confidence']):
            valid_types = [
                'categorical', 'text', 'datetime', 'numeric', 
                'email', 'phone', 'boolean', 'url', 'id'
            ]
            
            if llm_result['type'] in valid_types:
                final_result = {
                    'type': llm_result['type'],
                    'confidence': float(llm_result['confidence']),
                    'metadata': {
                        'llm_model': self.llm_model,
                        'llm_reasoning': llm_result.get('reasoning', ''),
                        'base_type': base_classification['type'],
                        'base_confidence': base_classification['confidence'],
                        'response_time': elapsed
                    }
                }
                
                print(f"  ✓ LLM: {llm_result['type']} ({llm_result['confidence']:.2f})")
                self._save_to_cache(cache_key, final_result)
                return final_result
        
        return None
    
    def _compute_dataset_hash(self, df: pd.DataFrame) -> str:
        """Computa hash do dataset para detectar mudanças."""
        # Incluir: shape, nomes das colunas, tipos
        dataset_info = {
            'shape': df.shape,
            'columns': sorted(df.columns.tolist()),
            'dtypes': {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Adicionar amostra de dados de cada coluna
        data_samples = {}
        for col in df.columns:
            # Pegar primeiros e últimos valores únicos
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                sample_vals = sorted([str(v) for v in unique_vals[:5]])
                data_samples[col] = sample_vals
        
        dataset_info['data_samples'] = data_samples
        
        # Criar hash
        info_str = json.dumps(dataset_info, sort_keys=True)
        return hashlib.md5(info_str.encode()).hexdigest()
    
    def _compute_column_hash(self, series: pd.Series) -> str:
        """Computa hash de uma coluna específica."""
        col_info = {
            'dtype': str(series.dtype),
            'shape': series.shape,
            'nunique': int(series.nunique()),  # Converter para int
            'null_count': int(series.isna().sum()),  # Converter para int
            'value_counts': series.value_counts().head(20).to_dict() if series.nunique() <= 100 else None,
            'stats': {
                'mean': float(series.mean()) if pd.api.types.is_numeric_dtype(series) else None,
                'std': float(series.std()) if pd.api.types.is_numeric_dtype(series) else None
            }
        }
        
        # Converter valores do value_counts para tipos serializáveis
        if col_info['value_counts']:
            col_info['value_counts'] = {
                str(k): int(v) if isinstance(v, (np.integer, np.int64)) else float(v)
                for k, v in col_info['value_counts'].items()
            }
        
        info_str = json.dumps(col_info, sort_keys=True, default=str)
        return hashlib.md5(info_str.encode()).hexdigest()
    
    def _compute_params_hash(self) -> str:
        """Computa hash dos parâmetros do classificador."""
        params_str = json.dumps(self._classifier_params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _load_classification_cache(self) -> bool:
        """Carrega cache de classificações se existir e for válido."""
        if not self.use_classification_cache:
            return False
            
        try:
            if os.path.exists(self.classification_cache_path):
                with open(self.classification_cache_path, 'r') as f:
                    self.classification_cache = json.load(f)
                print(f"✓ Cache de classificações carregado de: {self.classification_cache_path}")
                return True
            else:
                print(f"ℹ️ Cache não encontrado em: {self.classification_cache_path}")
                return False
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache: {e}. Continuando sem cache...")
            self.classification_cache = None
            return False
    
    def _save_classification_cache(self, df: pd.DataFrame, classifications: Dict[str, Dict[str, Any]]):
        """Salva cache de classificações."""
        if not self.use_classification_cache:
            return
            
        try:
            # Preparar cache
            cache_data = {
                'version': self.VERSION,
                'classifier_params': self._classifier_params,
                'params_hash': self._compute_params_hash(),
                'dataset_hash': self._compute_dataset_hash(df),
                'timestamp': datetime.now().isoformat(),
                'classifications': {}
            }
            
            # Função para converter tipos numpy para Python nativos
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj
            
            # Adicionar classificações com hash de cada coluna
            for col_name, classification in classifications.items():
                # Converter tipos numpy
                clean_classification = convert_numpy_types(classification)
                cache_data['classifications'][col_name] = {
                    **clean_classification,
                    'column_hash': self._compute_column_hash(df[col_name])
                }
            
            # Salvar
            with open(self.classification_cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            print(f"✓ Cache de classificações salvo em: {self.classification_cache_path}")
            
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")
    
    def _validate_cache_entry(self, cache_data: Dict, df: pd.DataFrame, column_name: str) -> bool:
        """Valida se uma entrada do cache ainda é válida."""
        try:
            # Verificar versão
            if cache_data.get('version') != self.VERSION:
                return False
            
            # Verificar parâmetros
            if cache_data.get('params_hash') != self._compute_params_hash():
                return False
            
            # Verificar se a coluna existe no cache
            if column_name not in cache_data.get('classifications', {}):
                return False
            
            # Verificar hash da coluna
            cached_column = cache_data['classifications'][column_name]
            current_hash = self._compute_column_hash(df[column_name])
            
            if cached_column.get('column_hash') != current_hash:
                return False
            
            return True
            
        except Exception:
            return False
    
    def classify_dataframe(self, df: pd.DataFrame, 
                          parallel: bool = False,
                          force_reclassify: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Classifica todas as colunas de um DataFrame.
        
        Args:
            df: DataFrame a classificar
            parallel: Se True, usa processamento paralelo
            force_reclassify: Se True, ignora cache e reclassifica tudo
        """
        results = {}
        llm_count = 0
        cache_used = False
        
        print("\n" + "="*60)
        print("CLASSIFICAÇÃO DE TIPOS DE COLUNAS")
        if self.use_llm:
            print(f"Modo: Híbrido (Regras + LLM)")
            print(f"LLM: {self.llm_model}")
        else:
            print("Modo: Apenas Regras")
        
        # Tentar carregar cache
        if not force_reclassify and self._load_classification_cache():
            if self.classification_cache:
                # Verificar se o cache é válido para este dataset
                dataset_hash = self._compute_dataset_hash(df)
                cached_dataset_hash = self.classification_cache.get('dataset_hash')
                
                if dataset_hash == cached_dataset_hash:
                    print("✓ Cache válido para o dataset completo")
                    cache_used = True
                else:
                    print("⚠️ Dataset mudou, cache será atualizado")
        
        print("="*60)
        
        # Processar cada coluna
        for col in df.columns:
            # Verificar cache primeiro
            if cache_used and self._validate_cache_entry(self.classification_cache, df, col):
                # Usar classificação do cache
                cached_result = self.classification_cache['classifications'][col].copy()
                # Remover column_hash do resultado
                cached_result.pop('column_hash', None)
                cached_result['metadata'] = cached_result.get('metadata', {})
                cached_result['metadata']['source'] = 'cache'
                results[col] = cached_result
                self.cache_hits += 1
                print(f"✓ {col}: {cached_result['type']} (cache)")
            else:
                # Classificar normalmente
                self.cache_misses += 1
                results[col] = self.classify_column(df[col], col)
                if 'llm_model' in results[col].get('metadata', {}):
                    llm_count += 1
                print(f"→ {col}: {results[col]['type']} (classificado)")
        
        # Salvar novo cache se houve classificações
        if self.cache_misses > 0:
            self._save_classification_cache(df, results)
        
        # Resumo
        print(f"\n📊 Resumo:")
        print(f"   Total: {len(df.columns)} colunas")
        if self.use_classification_cache:
            print(f"   Do cache: {self.cache_hits}")
            print(f"   Classificadas: {self.cache_misses}")
        if self.use_llm:
            print(f"   Com LLM: {llm_count}")
        
        return results
    
    def get_summary(self, classifications: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Cria resumo das classificações."""
        summary_data = []
        
        for col_name, info in classifications.items():
            summary_data.append({
                'column': col_name,
                'type': info['type'],
                'confidence': info['confidence'],
                'used_llm': 'llm_model' in info.get('metadata', {}),
                'metadata': str(info.get('metadata', {}))[:100] + '...' 
                          if len(str(info.get('metadata', {}))) > 100 else str(info.get('metadata', {}))
            })
        
        return pd.DataFrame(summary_data).sort_values('column')
    
    def save_to_disk(self, filepath: str):
        """
        Salva o classificador e seu cache em disco.
        Útil para salvar junto com o modelo.
        """
        save_data = {
            'classifier_params': self._classifier_params,
            'classification_cache_path': self.classification_cache_path,
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses
            }
        }
        
        # Se tiver cache carregado, incluir
        if self.classification_cache:
            save_data['classification_cache'] = self.classification_cache
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"✓ Classificador salvo em: {filepath}")
    
    @classmethod
    def load_from_disk(cls, filepath: str, **override_params):
        """
        Carrega classificador do disco.
        
        Args:
            filepath: Caminho do arquivo
            **override_params: Parâmetros para sobrescrever
        """
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        # Extrair parâmetros
        params = save_data['classifier_params'].copy()
        params.pop('version', None)  # Remover versão dos params
        
        # Aplicar overrides
        params.update(override_params)
        
        # Criar instância
        classifier = cls(**params)
        
        # Restaurar cache se disponível
        if 'classification_cache' in save_data:
            classifier.classification_cache = save_data['classification_cache']
            print(f"✓ Cache de classificações restaurado")
        
        return classifier


# Função de conveniência
def auto_detect_column_types(df: pd.DataFrame, 
                           classifier: Optional[ColumnTypeClassifier] = None,
                           use_llm: bool = False,
                           use_cache: bool = True,
                           cache_path: Optional[str] = None,
                           verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Detecta tipos de colunas automaticamente.
    
    Args:
        df: DataFrame a analisar
        classifier: Instância do classificador (cria nova se None)
        use_llm: Se True, usa LLM para casos ambíguos
        use_cache: Se True, usa cache de classificações
        cache_path: Caminho específico para o cache
        verbose: Se True, imprime resumo
    """
    if classifier is None:
        classifier = ColumnTypeClassifier(
            use_llm=use_llm,
            use_classification_cache=use_cache,
            classification_cache_path=cache_path
        )
    
    classifications = classifier.classify_dataframe(df)
    
    if verbose:
        type_counts = Counter(info['type'] for info in classifications.values())
        
        print("\n" + "="*60)
        print("RESUMO DA CLASSIFICAÇÃO")
        print("="*60)
        print(f"Total de colunas: {len(classifications)}")
        print("\nDistribuição de tipos:")
        for col_type, count in type_counts.most_common():
            print(f"  {col_type:15} : {count:3} colunas")
        
        # Estatísticas de cache
        if classifier.use_classification_cache:
            cache_stats = classifier.get_cache_statistics()
            print(f"\nEstatísticas de Cache:")
            print(f"  Taxa de acerto: {cache_stats['hit_rate']:.1f}%")
            print(f"  Hits: {cache_stats['cache_hits']}")
            print(f"  Misses: {cache_stats['cache_misses']}")
        
        low_confidence = [
            (col, info['confidence']) 
            for col, info in classifications.items() 
            if info['confidence'] < 0.7
        ]
        
        if low_confidence:
            print(f"\nBaixa confiança (<0.7): {len(low_confidence)} colunas")
            for col, conf in sorted(low_confidence, key=lambda x: x[1]):
                print(f"  {col}: {conf:.2f}")
        
        print("="*60)
    
    return classifications


# Teste
if __name__ == "__main__":
    print("Testando classificador com cache...")
    
    try:
        # Tentar dados reais
        df = pd.read_csv("/Users/ramonmoreira/desktop/smart_ads/data/new/01_split/train.csv")
        print(f"✓ Dados carregados: {df.shape}")
        
        # Limpar cache antigo E cache da LLM
        cache_path = "/Users/ramonmoreira/desktop/smart_ads/cache/column_classifications.json"
        llm_cache_path = "/Users/ramonmoreira/desktop/smart_ads/cache/llm_classifications"
        
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("✓ Cache antigo removido")
            
        # Limpar cache da LLM também
        if os.path.exists(llm_cache_path):
            import shutil
            shutil.rmtree(llm_cache_path)
            print("✓ Cache LLM removido")
            os.makedirs(llm_cache_path, exist_ok=True)
        
        # Teste 1: Primeira execução (sem cache, COM LLM)
        print("\n1. PRIMEIRA EXECUÇÃO (sem cache, COM LLM):")
        classifier = ColumnTypeClassifier(
            use_llm=True,  # MUDADO PARA TRUE
            use_classification_cache=True,
            classification_cache_path=cache_path,
            fail_on_llm_error=False  # Continua se LLM falhar
        )
        
        import time
        start_time = time.time()
        results1 = classifier.classify_dataframe(df)
        time1 = time.time() - start_time
        print(f"⏱️ Tempo: {time1:.2f}s")
        
        # Teste 2: Segunda execução (com cache)
        print("\n2. SEGUNDA EXECUÇÃO (com cache):")
        classifier2 = ColumnTypeClassifier(
            use_llm=True,  # MUDADO PARA TRUE
            use_classification_cache=True,
            classification_cache_path=cache_path,
            fail_on_llm_error=False
        )
        
        start_time = time.time()
        results2 = classifier2.classify_dataframe(df)
        time2 = time.time() - start_time
        print(f"⏱️ Tempo: {time2:.2f}s")
        print(f"🚀 Speedup: {time1/time2:.1f}x mais rápido")
        
        # Verificar consistência
        print("\n3. VERIFICAÇÃO DE CONSISTÊNCIA:")
        inconsistencies = 0
        for col in df.columns:
            if results1[col]['type'] != results2[col]['type']:
                print(f"❌ Inconsistência em {col}: {results1[col]['type']} vs {results2[col]['type']}")
                inconsistencies += 1
        
        if inconsistencies == 0:
            print("✓ Resultados idênticos entre execuções")
        
        # Mostrar classificações problemáticas
        print("\n4. CLASSIFICAÇÕES ATUAIS:")
        problem_cols = ['DATA', 'Marca temporal', '¿Cómo te llamas?', '¿Cuál es tu instagram?', 
                       '¿Cuál es tu profesión?', 'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida?',
                       '¿Qué esperas aprender en el evento Cero a Inglés Fluido?', 'Déjame un mensaje']
        
        for col in problem_cols:
            if col in results1:
                print(f"{col}: {results1[col]['type']} (confiança: {results1[col]['confidence']:.2f})")
        
        # Teste 5: Forçar reclassificação
        print("\n5. FORÇAR RECLASSIFICAÇÃO:")
        start_time = time.time()
        results3 = classifier2.classify_dataframe(df, force_reclassify=True)
        time3 = time.time() - start_time
        print(f"⏱️ Tempo: {time3:.2f}s")
        
        # Teste 6: Com LLM (se disponível)
        print("\n6. TESTE COM LLM:")
        # Iniciar Ollama se não estiver rodando
        print("Para testar com LLM, execute em outro terminal: ollama serve")
        
        classifier_llm = ColumnTypeClassifier(
            use_llm=True,
            llm_model="phi3:mini",
            llm_confidence_threshold=0.8,
            use_classification_cache=False
        )
        
        if classifier_llm.use_llm:  # Só testa se LLM estiver disponível
            print("✓ LLM disponível, testando classificações...")
            
            # Testar apenas algumas colunas
            test_cols = ['DATA', 'Marca temporal', '¿Cómo te llamas?']
            for col in test_cols:
                if col in df.columns:
                    result = classifier_llm.classify_column(df[col], col)
                    print(f"{col}: {result['type']} (confiança: {result['confidence']:.2f})")
                    if 'llm_model' in result.get('metadata', {}):
                        print(f"  → Usado LLM: {result['metadata']['llm_reasoning']}")
        else:
            print("⚠️ LLM não disponível. Execute 'ollama serve' em outro terminal.")
        
        # Salvar classificador
        print("\n7. SALVANDO CLASSIFICADOR:")
        classifier2.save_to_disk("/Users/ramonmoreira/desktop/smart_ads/cache/classifier_config.json")
        
        # Estatísticas finais
        print("\n8. ESTATÍSTICAS FINAIS:")
        
        # Criar novo classificador para obter estatísticas corretas
        final_classifier = ColumnTypeClassifier(
            use_llm=False,
            use_classification_cache=True,
            classification_cache_path=cache_path
        )
        
        # Carregar cache
        final_classifier._load_classification_cache()
        
        # Mostrar estatísticas
        print(f"Classificações em cache: {len(final_classifier.classification_cache.get('classifications', {})) if final_classifier.classification_cache else 0}")
        print(f"Cache path: {cache_path}")
        
        # Mostrar classificações finais corretas/incorretas
        print("\n9. VERIFICAÇÃO FINAL DAS CLASSIFICAÇÕES:")
        expected = {
            'DATA': 'datetime',
            'Marca temporal': 'datetime',
            '¿Cómo te llamas?': 'text',
            '¿Cuál es tu instagram?': 'text',
            '¿Cuál es tu profesión?': 'text',
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida?': 'text',
            '¿Qué esperas aprender en el evento Cero a Inglés Fluido?': 'text',
            'Déjame un mensaje': 'text'
        }
        
        for col, expected_type in expected.items():
            if col in results1:
                actual_type = results1[col]['type']
                status = "✓" if actual_type == expected_type else "❌"
                print(f"{status} {col}: {actual_type} (esperado: {expected_type})")
        
    except FileNotFoundError:
        print("Arquivo não encontrado. Usando dados de exemplo...")
        
        # Dados de exemplo
        test_data = {
            'DATA': ['2024-01-15', '2024-01-16'] * 50,
            'UTM_SOURCE': ['google', 'facebook'] * 50,
            '¿Cómo te llamas?': ['Juan Pérez García', 'María López Fernández'] * 50,
            '¿Cuál es tu edad?': ['25-34', '35-44'] * 50,
            'Cuando hables inglés con fluidez': [
                'Podré conseguir un mejor trabajo y viajar más. También podré comunicarme mejor.',
                'Tendré más oportunidades laborales y podré estudiar en el extranjero.'
            ] * 50,
            'target': [0, 1] * 50
        }
        
        df = pd.DataFrame(test_data)
        
        # Testar com cache
        print("\nTestando com dados de exemplo...")
        classifier = ColumnTypeClassifier(
            use_llm=False,
            use_classification_cache=True
        )
        
        # Primeira execução
        print("\nPrimeira execução:")
        results = auto_detect_column_types(df, classifier=classifier)