#!/usr/bin/env python
"""test_fast.py - Teste rápido com 1000 amostras"""

from unified_pipeline import unified_data_pipeline

results = unified_data_pipeline(
    test_mode=True,
    max_samples=1000,
    batch_size=500,  # Menor para teste
    apply_feature_selection=False  # Pular para economizar tempo
    
)

print("\n✅ Pipeline executado com sucesso no modo de teste!")