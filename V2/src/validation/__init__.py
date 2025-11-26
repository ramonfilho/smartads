"""
Módulo de validação de performance do modelo de ML.

Este módulo contém ferramentas para validar a performance real do modelo
de lead scoring comparando campanhas COM ML vs SEM ML e analisando a
performance por decil (D1-D10).

Componentes:
- data_loader: Carrega leads e vendas
- campaign_classifier: Classifica campanhas (COM_ML, SEM_ML, EXCLUIR)
- matching: Vincula leads com vendas
- metrics_calculator: Calcula métricas de performance
- report_generator: Gera relatórios Excel
- visualization: Gera gráficos PNG
"""

__version__ = "1.0.0"
