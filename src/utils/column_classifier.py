# /Users/ramonmoreira/desktop/smart_ads/src/utils/text_detection.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re

class TextColumnDetector:
    """
    Detector unificado para identificar colunas de texto em DataFrames.
    Usa múltiplas heurísticas para detectar colunas textuais.
    """
    
    def __init__(self):
        # Padrões que indicam colunas de texto
        self.text_patterns = [
            # Perguntas em espanhol
            r'¿.*\?',
            r'cuando.*vida',
            r'qué.*aprender',
            r'mensaje',
            r'razones.*para',
            r'disponibilidad.*tiempo',
            
            # Palavras-chave que indicam texto livre
            r'describe',
            r'explica',
            r'comenta',
            r'opinión',
            r'experiencia',
            
            # Padrões genéricos
            r'text',
            r'comment',
            r'description',
            r'message'
        ]
        
        # Colunas conhecidas de texto (fallback)
        self.known_text_columns = {
            'Cuando hables inglés con fluidez, ¿qué cambiará en tu vida? ¿Qué oportunidades se abrirán para ti?',
            '¿Qué esperas aprender en el evento Cero a Inglés Fluido?',
            'Déjame un mensaje',
            '¿Qué esperas aprender en la Semana de Cero a Inglés Fluido?',
            '¿Qué esperas aprender en la Inmersión Desbloquea Tu Inglés En 72 horas?',
            '¿Cuál es tu disponibilidad de tiempo para estudiar inglés?',
            '¿Crees que aprender inglés te acercaría más al salario que mencionaste anteriormente?',
            '¿Crees que aprender inglés puede ayudarte en el trabajo o en tu vida diaria?',
            '¿Cuáles son tus principales razones para aprender inglés?',
            '¿Has comprado algún curso para aprender inglés antes?'
        }
    
    def is_text_column(self, series: pd.Series, column_name: str) -> Tuple[bool, float, str]:
        """
        Determina se uma série é uma coluna de texto.
        
        Returns:
            Tuple[bool, float, str]: (é_texto, confiança, razão)
        """
        # 1. Verificar se é uma coluna conhecida
        if column_name in self.known_text_columns:
            return True, 1.0, "known_text_column"
        
        # 2. Verificar padrões no nome da coluna
        for pattern in self.text_patterns:
            if re.search(pattern, column_name.lower()):
                return True, 0.9, f"name_pattern_match: {pattern}"
        
        # 3. Verificar tipo de dados
        if series.dtype != 'object':
            return False, 0.9, "non_object_dtype"
        
        # 4. Análise de conteúdo
        # Amostra para análise (máximo 1000 registros)
        sample_size = min(1000, len(series))
        if sample_size == 0:
            return False, 0.5, "empty_series"
        
        sample = series.dropna().sample(n=min(sample_size, len(series.dropna())), 
                                       random_state=42) if len(series.dropna()) > 0 else pd.Series()
        
        if len(sample) == 0:
            return False, 0.5, "all_null_values"
        
        # Métricas de texto
        metrics = self._calculate_text_metrics(sample)
        
        # 5. Decisão baseada em métricas
        is_text, confidence = self._evaluate_metrics(metrics)
        
        return is_text, confidence, f"content_analysis: {metrics}"
    
    def _calculate_text_metrics(self, sample: pd.Series) -> Dict[str, float]:
        """Calcula métricas para determinar se é texto."""
        metrics = {
            'avg_length': 0,
            'avg_words': 0,
            'unique_ratio': 0,
            'has_spaces_ratio': 0,
            'numeric_ratio': 0,
            'special_chars_ratio': 0,
            'avg_word_length_variance': 0
        }
        
        lengths = []
        word_counts = []
        has_spaces = 0
        numeric_count = 0
        special_chars_count = 0
        
        for value in sample:
            if pd.isna(value):
                continue
                
            str_val = str(value)
            lengths.append(len(str_val))
            
            # Contagem de palavras
            words = str_val.split()
            word_counts.append(len(words))
            
            # Tem espaços?
            if ' ' in str_val:
                has_spaces += 1
            
            # É puramente numérico?
            if str_val.replace('.', '').replace(',', '').replace('-', '').isdigit():
                numeric_count += 1
            
            # Caracteres especiais
            special_chars = sum(1 for c in str_val if not c.isalnum() and not c.isspace())
            if special_chars > 0:
                special_chars_count += 1
        
        n_valid = len(lengths)
        if n_valid > 0:
            metrics['avg_length'] = np.mean(lengths)
            metrics['avg_words'] = np.mean(word_counts)
            metrics['unique_ratio'] = len(sample.unique()) / len(sample)
            metrics['has_spaces_ratio'] = has_spaces / n_valid
            metrics['numeric_ratio'] = numeric_count / n_valid
            metrics['special_chars_ratio'] = special_chars_count / n_valid
            
            # Variância no comprimento das palavras (textos têm alta variância)
            if metrics['avg_words'] > 1:
                word_length_variances = []
                for value in sample[:100]:  # Amostra menor para performance
                    if pd.notna(value):
                        words = str(value).split()
                        if len(words) > 1:
                            word_lengths = [len(w) for w in words]
                            word_length_variances.append(np.var(word_lengths))
                
                if word_length_variances:
                    metrics['avg_word_length_variance'] = np.mean(word_length_variances)
        
        return metrics
    
    def _evaluate_metrics(self, metrics: Dict[str, float]) -> Tuple[bool, float]:
        """Avalia métricas para determinar se é texto."""
        score = 0.0
        
        # Comprimento médio > 20 caracteres
        if metrics['avg_length'] > 20:
            score += 0.25
        
        # Média de palavras > 3
        if metrics['avg_words'] > 3:
            score += 0.25
        
        # Alta taxa de valores únicos (> 80%)
        if metrics['unique_ratio'] > 0.8:
            score += 0.15
        
        # Maioria tem espaços
        if metrics['has_spaces_ratio'] > 0.7:
            score += 0.20
        
        # Baixa taxa de numéricos
        if metrics['numeric_ratio'] < 0.1:
            score += 0.10
        
        # Variância no comprimento das palavras
        if metrics['avg_word_length_variance'] > 2:
            score += 0.05
        
        # Decisão
        is_text = score >= 0.5
        confidence = min(score, 1.0)
        
        return is_text, confidence
    
    def detect_text_columns(self, df: pd.DataFrame, 
                          confidence_threshold: float = 0.6,
                          exclude_patterns: List[str] = None) -> List[str]:
        """
        Detecta todas as colunas de texto em um DataFrame.
        
        Args:
            df: DataFrame para análise
            confidence_threshold: Confiança mínima para considerar texto
            exclude_patterns: Padrões para excluir (ex: ['_encoded', '_norm'])
            
        Returns:
            Lista de nomes de colunas de texto
        """
        text_columns = []
        exclude_patterns = exclude_patterns or ['_encoded', '_norm', '_clean', '_tfidf']
        
        print("\n🔍 Detectando colunas de texto...")
        
        for col in df.columns:
            # Pular colunas com padrões de exclusão
            if any(pattern in col for pattern in exclude_patterns):
                continue
            
            is_text, confidence, reason = self.is_text_column(df[col], col)
            
            if is_text and confidence >= confidence_threshold:
                text_columns.append(col)
                print(f"  ✓ {col[:60]}... (confiança: {confidence:.2f})")
            elif confidence >= confidence_threshold - 0.2:
                print(f"  ? {col[:60]}... (confiança: {confidence:.2f}, não incluída)")
        
        print(f"\n📊 Total de colunas de texto detectadas: {len(text_columns)}")
        return text_columns


# Instância global para uso consistente
text_detector = TextColumnDetector()


def detect_text_columns(df: pd.DataFrame, **kwargs) -> List[str]:
    """Função wrapper para facilitar o uso."""
    return text_detector.detect_text_columns(df, **kwargs)


def is_text_column(series: pd.Series, column_name: str) -> bool:
    """Função wrapper para verificar uma única coluna."""
    is_text, confidence, _ = text_detector.is_text_column(series, column_name)
    return is_text and confidence >= 0.6