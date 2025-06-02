#!/usr/bin/env python
"""
Módulo de classificação automática de tipos de colunas - VERSÃO CORRIGIDA.

Correções principais:
1. Detecção de datetime menos agressiva
2. Suporte específico para UTMs
3. Ordem de verificação otimizada
4. Conhecimento de domínio para Smart Ads
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
from collections import Counter

warnings.filterwarnings('ignore')


class ColumnTypeClassifierFixed:
    """
    Classificador robusto de tipos de colunas - Versão Corrigida.
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
    
    def __init__(self, 
                 categorical_threshold: int = 100,  # Aumentado de 50 para 100
                 text_min_avg_length: int = 40,    # Aumentado de 20 para 40
                 text_min_unique_ratio: float = 0.7,  # Aumentado de 0.5 para 0.7
                 text_min_words: int = 5,  # Novo parâmetro
                 numeric_string_threshold: float = 0.9,
                 date_detection_threshold: float = 0.8,
                 sample_size: int = 1000,
                 confidence_threshold: float = 0.7):
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
        """
        self.categorical_threshold = categorical_threshold
        self.text_min_avg_length = text_min_avg_length
        self.text_min_unique_ratio = text_min_unique_ratio
        self.text_min_words = text_min_words
        self.numeric_string_threshold = numeric_string_threshold
        self.date_detection_threshold = date_detection_threshold
        self.sample_size = sample_size
        self.confidence_threshold = confidence_threshold
        
        # Padrões regex compilados
        self._compile_patterns()
        
        # Conhecimento de domínio para Smart Ads
        self._setup_domain_knowledge()
        
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
        
        # Date patterns ESPECÍFICOS (não genéricos)
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
        
    def classify_column(self, series: pd.Series, column_name: str = None) -> Dict[str, Any]:
        """
        Classifica o tipo de uma coluna - VERSÃO CORRIGIDA.
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
        
        # 1. Verificar tipos numéricos nativos PRIMEIRO
        if pd.api.types.is_numeric_dtype(series):
            # Verificar se é booleano disfarçado
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
            
            # 2.1 Verificar UTM/Tracking PRIMEIRO
            utm_result = self._check_utm_or_tracking(str_sample, column_name, unique_count, total_count)
            if utm_result.get('is_utm'):
                return {
                    'type': self.CATEGORICAL,
                    'confidence': utm_result['confidence'],
                    'metadata': utm_result['metadata']
                }
            
            # 2.2 Verificar booleano
            bool_result = self._check_boolean(str_sample, series)
            if bool_result['is_boolean']:
                return {
                    'type': self.BOOLEAN,
                    'confidence': bool_result['confidence'],
                    'metadata': bool_result['metadata']
                }
            
            # 2.3 Verificar email
            email_result = self._check_email(str_sample, column_name)
            if email_result['is_email']:
                return {
                    'type': self.EMAIL,
                    'confidence': email_result['confidence'],
                    'metadata': email_result['metadata']
                }
            
            # 2.4 Verificar telefone
            phone_result = self._check_phone(str_sample, column_name)
            if phone_result['is_phone']:
                return {
                    'type': self.PHONE,
                    'confidence': phone_result['confidence'],
                    'metadata': phone_result['metadata']
                }
            
            # 2.5 Verificar URL
            url_result = self._check_url(str_sample)
            if url_result['is_url']:
                return {
                    'type': self.URL,
                    'confidence': url_result['confidence'],
                    'metadata': url_result['metadata']
                }
            
            # 2.6 Verificar numérico disfarçado
            numeric_result = self._check_numeric_string(str_sample)
            if numeric_result['is_numeric']:
                return {
                    'type': self.NUMERIC,
                    'confidence': numeric_result['confidence'],
                    'metadata': numeric_result['metadata']
                }
            
            # 2.7 Verificar ID
            id_result = self._check_id(str_sample, column_name, unique_count, total_count)
            if id_result['is_id']:
                return {
                    'type': self.ID,
                    'confidence': id_result['confidence'],
                    'metadata': id_result['metadata']
                }
            
            # 2.8 Verificar datetime COM RIGOR
            date_result = self._check_datetime_strict(str_sample, column_name)
            if date_result['is_datetime']:
                return {
                    'type': self.DATETIME,
                    'confidence': date_result['confidence'],
                    'metadata': date_result['metadata']
                }
            
            # 2.9 Decidir entre categórica e texto
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
    
    def _check_utm_or_tracking(self, sample: pd.Series, column_name: Optional[str],
                              unique_count: int, total_count: int) -> Dict[str, Any]:
        """Verifica se é uma coluna de tracking/UTM."""
        if not column_name:
            return {'is_utm': False}
        
        # Verificar nome exato
        col_lower = column_name.lower()
        
        # Match exato com keywords UTM
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
        """Verifica datetime com MUITO mais rigor."""
        # PRIMEIRO: verificar se os dados realmente parecem datas
        # Contar quantos valores têm características de data
        date_like_count = 0
        sample_str = sample.astype(str).head(20)  # Verificar primeiros 20
        
        for val in sample_str:
            # Características de data: tem números E separadores de data
            has_numbers = any(c.isdigit() for c in val)
            has_date_separators = any(sep in val for sep in ['-', '/', ':'])
            looks_like_date = has_numbers and has_date_separators and len(val) >= 8
            
            if looks_like_date:
                date_like_count += 1
        
        # Se menos de 50% parece data, retornar falso
        if date_like_count < len(sample_str) * 0.5:
            return {'is_datetime': False}
        
        # SEGUNDO: verificar nome da coluna apenas como dica adicional
        confidence_boost = 0
        if column_name:
            col_lower = column_name.lower()
            
            # Se tem palavras que indicam data, dar boost
            if any(white in col_lower for white in ['data', 'date', 'timestamp', 'temporal']):
                confidence_boost = 0.1
            
            # Se tem palavras que NÃO indicam data, penalizar
            if any(black in col_lower for black in ['edad', 'sueldo', 'género', 'país']):
                return {'is_datetime': False}
        
        # TERCEIRO: verificar padrões específicos
        for pattern, format_name in self.strict_date_patterns:
            matches = sample.astype(str).str.match(pattern)
            match_ratio = matches.sum() / len(sample)
            
            if match_ratio > 0.8:  # 80% de match
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
        """
        Detecta padrões típicos de perguntas de múltipla escolha.
        """
        # Verificar padrões comuns de respostas categóricas
        value_counts = series.value_counts()
        top_values = value_counts.head(20)
        
        # Padrões de respostas sim/não/talvez
        yes_no_patterns = {
            'sí', 'si', 'no', 'tal vez', 'talvez', 'quizás', 'quizas',
            'definitivamente', 'probablemente', 'nunca', 'siempre',
            'yes', 'no', 'maybe', 'definitely', 'probably'
        }
        
        # Padrões de escalas
        scale_patterns = {
            'muy', 'poco', 'mucho', 'nada', 'bastante', 'algo',
            'totalmente', 'parcialmente', 'completamente'
        }
        
        # Padrões de tempo/frequência
        time_patterns = {
            'año', 'años', 'mes', 'meses', 'semana', 'semanas',
            'día', 'días', 'hora', 'horas', 'minuto', 'minutos',
            'siempre', 'nunca', 'a veces', 'frecuentemente'
        }
        
        # Padrões de quantidade/ranges
        range_patterns = {
            'más de', 'menos de', 'entre', 'hasta', 'desde',
            '-', 'a', 'o más', 'o menos'
        }
        
        # Analisar os valores únicos
        unique_values_lower = [str(v).lower().strip() for v in series.dropna().unique()]
        
        # 1. Verificar respostas sim/não
        yes_no_matches = sum(1 for val in unique_values_lower if val in yes_no_patterns)
        if yes_no_matches >= len(unique_values_lower) * 0.5:
            return {
                'is_multiple_choice': True,
                'pattern': 'yes_no',
                'confidence': 0.95
            }
        
        # 2. Verificar ranges numéricos (salários, idades, etc)
        range_count = sum(1 for val in unique_values_lower 
                         if any(pattern in val for pattern in range_patterns))
        if range_count >= len(unique_values_lower) * 0.5:
            return {
                'is_multiple_choice': True,
                'pattern': 'numeric_ranges',
                'confidence': 0.9
            }
        
        # 3. Verificar padrões de tempo
        time_count = sum(1 for val in unique_values_lower 
                        if any(pattern in val for pattern in time_patterns))
        if time_count >= len(unique_values_lower) * 0.5:
            return {
                'is_multiple_choice': True,
                'pattern': 'time_periods',
                'confidence': 0.9
            }
        
        # 4. Verificar se são opções curtas e limitadas
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
        """Decide entre texto e categórica baseado em CARACTERÍSTICAS DOS DADOS."""
        
        # PRIMEIRO: verificar se é múltipla escolha
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
        
        # Calcular métricas dos dados
        unique_ratio = unique_count / total_count
        avg_length = sample.str.len().mean()
        max_length = sample.str.len().max()
        min_length = sample.str.len().min()
        length_std = sample.str.len().std()
        
        # Análise de palavras
        word_counts = sample.str.split().str.len()
        avg_words = word_counts.mean() if not word_counts.empty else 0
        max_words = word_counts.max() if not word_counts.empty else 0
        
        # Análise de pontuação e características de texto
        has_punctuation = sample.str.contains(r'[.!?;,:]', regex=True).sum() / len(sample)
        has_multiple_sentences = sample.str.contains(r'[.!?]\s+[A-Z]', regex=True).sum() / len(sample)
        
        # Decisão baseada em MÚLTIPLOS FATORES dos dados
        
        # 1. Se tem muitas palavras E pontuação = TEXTO
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
        
        # 2. Se tem respostas longas E alta variabilidade = TEXTO
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
        
        # 3. Se tem múltiplas frases = TEXTO
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
        
        # 4. Se é pergunta sim/não ou múltipla escolha = CATEGÓRICA
        # Verificar se as respostas são curtas e repetitivas
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
        
        # 5. Análise de padrões de resposta
        # Verificar se são respostas padronizadas
        top_values = series.value_counts().head(10)
        top_values_ratio = top_values.sum() / total_count
        
        # Se top 10 valores cobrem >70% dos dados = CATEGÓRICA
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
        
        # 6. Alta cardinalidade mas respostas curtas = CATEGÓRICA
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
        
        # 7. Verificar se são IDs/códigos (alta unicidade, formato consistente)
        if unique_ratio > 0.9 and length_std < 5:
            # Possível ID/código, mas classificar como categórica
            return {
                'type': self.CATEGORICAL,
                'confidence': 0.7,
                'metadata': {
                    'reason': 'possible_id_field',
                    'unique_ratio': unique_ratio,
                    'length_consistency': length_std
                }
            }
        
        # 8. Default: verificar se tem características de texto
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
        
        # 9. Último recurso: CATEGÓRICA
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
        """Verifica se a coluna é booleana."""
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
        """Verifica se a coluna contém emails."""
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
        """Verifica se a coluna contém telefones."""
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
        """Verifica se a coluna contém URLs."""
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
        """Verifica se a coluna contém strings numéricas."""
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
        """Verifica se a coluna contém IDs."""
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
    
    # Manter todos os outros métodos iguais...
    def classify_dataframe(self, df: pd.DataFrame, 
                          parallel: bool = False) -> Dict[str, Dict[str, Any]]:
        """Classifica todas as colunas de um DataFrame."""
        results = {}
        
        if parallel:
            try:
                from joblib import Parallel, delayed
                
                classifications = Parallel(n_jobs=-1)(
                    delayed(self.classify_column)(df[col], col) 
                    for col in df.columns
                )
                
                for col, classification in zip(df.columns, classifications):
                    results[col] = classification
                    
            except ImportError:
                print("Warning: joblib not available. Using sequential processing.")
                parallel = False
        
        if not parallel:
            for col in df.columns:
                results[col] = self.classify_column(df[col], col)
        
        return results
    
    def get_summary(self, classifications: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Cria um resumo das classificações."""
        summary_data = []
        
        for col_name, info in classifications.items():
            summary_data.append({
                'column': col_name,
                'type': info['type'],
                'confidence': info['confidence'],
                'metadata': str(info.get('metadata', {}))[:100] + '...' 
                          if len(str(info.get('metadata', {}))) > 100 else str(info.get('metadata', {}))
            })
        
        return pd.DataFrame(summary_data).sort_values('column')
    
    def get_columns_by_type(self, classifications: Dict[str, Dict[str, Any]], 
                           type_filter: Union[str, List[str]]) -> List[str]:
        """Retorna lista de colunas de um tipo específico."""
        if isinstance(type_filter, str):
            type_filter = [type_filter]
        
        return [
            col for col, info in classifications.items() 
            if info['type'] in type_filter
        ]


# Função de conveniência
def auto_detect_column_types_fixed(df: pd.DataFrame, 
                                  classifier: Optional[ColumnTypeClassifierFixed] = None,
                                  verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Função de conveniência para detecção automática de tipos - VERSÃO CORRIGIDA.
    """
    if classifier is None:
        classifier = ColumnTypeClassifierFixed()
    
    classifications = classifier.classify_dataframe(df)
    
    if verbose:
        type_counts = Counter(info['type'] for info in classifications.values())
        
        print("=" * 60)
        print("RESUMO DA CLASSIFICAÇÃO DE TIPOS (VERSÃO CORRIGIDA)")
        print("=" * 60)
        print(f"Total de colunas: {len(classifications)}")
        print("\nDistribuição de tipos:")
        for col_type, count in type_counts.most_common():
            print(f"  {col_type:15} : {count:3} colunas")
        
        low_confidence = [
            (col, info['confidence']) 
            for col, info in classifications.items() 
            if info['confidence'] < 0.7
        ]
        
        if low_confidence:
            print(f"\nColunas com baixa confiança (<0.7): {len(low_confidence)}")
            for col, conf in sorted(low_confidence, key=lambda x: x[1]):
                print(f"  {col}: {conf:.2f}")
        
        print("=" * 60)
    
    return classifications


# Teste específico para Smart Ads
if __name__ == "__main__":
    print("Testando classificador corrigido com dados do Smart Ads...")
    
    # Teste com DataFrame real
    try:
        import pandas as pd
        df_example = pd.read_csv("/Users/ramonmoreira/desktop/smart_ads/data/new/01_split/train.csv")
        df_test = df_example.copy()
        
        # Classificar
        classifier = ColumnTypeClassifierFixed()
        results = auto_detect_column_types_fixed(df_test, classifier=classifier)
        
        # Mostrar resultados detalhados
        print("\nClassificação detalhada:")
        for col, info in results.items():
            print(f"{col}: {info['type']} (confiança: {info['confidence']:.2f})")
            
    except FileNotFoundError:
        print("\nArquivo train.csv não encontrado. Usando dados simulados...")
        
        # Criar DataFrame de teste que simula os dados reais
        test_data = {
            'DATA': ['2024-01-15', '2024-01-16', '2024-01-17'] * 100,
            'E-MAIL': ['user1@example.com', 'user2@test.com', 'user3@email.com'] * 100,
            'UTM_CAMPAING': ['campaign1', 'campaign2', 'campaign3'] * 100,
            'UTM_SOURCE': ['google', 'facebook', 'instagram'] * 100,
            'UTM_MEDIUM': ['cpc', 'social', 'email'] * 100,
            'UTM_CONTENT': ['banner1', 'banner2', 'banner3'] * 100,
            'UTM_TERM': ['keyword1', 'keyword2', 'keyword3'] * 100,
            'GCLID': ['CjwKCAiA1', 'CjwKCAiA2', 'CjwKCAiA3'] * 100,
            'Marca temporal': ['2024-01-15 10:30:00', '2024-01-15 11:45:00', '2024-01-15 12:00:00'] * 100,
            '¿Cómo te llamas?': ['Juan Pérez', 'María García', 'Carlos López'] * 100,
            '¿Cuál es tu género?': ['Masculino', 'Femenino', 'Masculino'] * 100,
            '¿Cuál es tu edad?': ['25-34', '35-44', '18-24'] * 100,
            '¿Cual es tu país?': ['México', 'Colombia', 'Argentina'] * 100,
            '¿Cuál es tu e-mail?': ['juan@email.com', 'maria@test.com', 'carlos@example.com'] * 100,
            '¿Cual es tu telefono?': ['+52 55 1234 5678', '(57) 1 234 5678', '+54 11 9876 5432'] * 100,
            '¿Cuál es tu instagram?': ['@juanperez', '@mariag', '@carlos_lopez'] * 100,
            '¿Hace quánto tiempo me conoces?': ['1 año', '6 meses', 'Más de 2 años', '3 meses'] * 75,
            '¿Cuál es tu disponibilidad de tiempo para estudiar inglés?': ['1 hora al día', '2 horas al día', '30 minutos al día'] * 100,
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida?': [
                'Podré conseguir un mejor trabajo y viajar más. También podré comunicarme con clientes internacionales.',
                'Tendré más oportunidades laborales y podré comunicarme mejor con personas de otros países.',
                'Mejoraré mi carrera profesional y tendré acceso a mejores oportunidades.'
            ] * 100,
            '¿Cuál es tu profesión?': ['Ingeniero', 'Médico', 'Profesor', 'Abogado', 'Contador'] * 60,
            '¿Cuál es tu sueldo anual? (en dólares)': ['10000-20000', '20000-30000', '30000-50000', 'Más de 50000'] * 75,
            '¿Cuánto te gustaría ganar al año?': ['50000', '75000', '100000', 'Más de 100000'] * 75,
            '¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?': ['Sí', 'No', 'Tal vez'] * 100,
            '¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?': ['Sí', 'No', 'Definitivamente'] * 100,
            '¿Qué esperas aprender en el evento Cero a Inglés Fluido?': [
                'Quiero aprender las bases del inglés y cómo estudiarlo de manera efectiva.',
                'Espero obtener técnicas para mejorar mi pronunciación y comprensión.',
                'Me gustaría conocer un método efectivo para aprender inglés rápidamente.'
            ] * 100,
            'Déjame un mensaje': [
                'Estoy muy emocionado por este curso. Espero aprender mucho.',
                'Gracias por esta oportunidad.',
                'Necesito aprender inglés para mi trabajo y creo que este curso me ayudará mucho.'
            ] * 100,
            '¿Cuáles son tus principales razones para aprender inglés?': [
                'Trabajo', 'Viajes', 'Estudios', 'Desarrollo personal', 'Negocios'
            ] * 60,
            '¿Has comprado algún curso para aprender inglés antes?': ['Sí', 'No'] * 150,
            'Qualidade (Nome)': ['Alta', 'Media', 'Baja'] * 100,
            'Qualidade (Número)': [85.5, 72.3, 91.0] * 100,
            'target': [0, 1, 0] * 100
        }
        
        df_test = pd.DataFrame(test_data)
        
        # Classificar
        classifier = ColumnTypeClassifierFixed()
        results = auto_detect_column_types_fixed(df_test, classifier=classifier)
        
        # Mostrar resultados detalhados
        print("\nClassificação detalhada:")
        for col, info in results.items():
            print(f"{col}: {info['type']} (confiança: {info['confidence']:.2f})")