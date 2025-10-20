"""
Configuração da API do Meta (Facebook Ads)
Para produção: Cliente adiciona você como admin no Business Manager
"""

# Credenciais Meta Ads - PRODUÇÃO
META_CONFIG = {
    "access_token": "EAAS9hlWC7lkBPmTFNOvHZBVZAW6ESTsmVCStlrcslFvNLxr2xBkKrI0kTmI6dou1aB5UOJLFwQo9gwAg1NZBCSWZCZAkxflALfnFeZC8nYRJJO5TZAfy1vswWFs0nCsZBpOanId4ULYCJMzPqt7UuhfuNBablHZAIchs1T7vEGWXgk6Sq2t8YirZBIPldNDVtyp7DxYQZDZD",
    "api_version": "v18.0",
    # Account ID: act_188005769808959 (Los Angeles Producciones LTDA)
    # Account ID será passado por Apps Script
}

# Configuração de negócio - TAXAS CORRIGIDAS POR RECALL
BUSINESS_CONFIG = {
    "product_value": 2027.38,  # Baseado em análise de 1.970 vendas DevClub desde 01/03/2025
    "min_roas": 2.0,

    # TAXAS DE CONVERSÃO CORRIGIDAS
    # Recall do matching: 34.4% (678 conversões observadas / 1970 vendas reais)
    # Fator de correção aplicado: 2.906x
    #
    # IMPORTANTE: Estas taxas refletem conversões REAIS estimadas, não apenas as capturadas pelo matching.
    # O matching por email/telefone captura apenas ~34.4% das conversões devido a:
    #   - Emails diferentes entre pesquisa e compra
    #   - Telefones inválidos/incomparáveis
    #   - Dados ausentes
    #
    # Método de correção: Taxa corrigida = Taxa observada / Recall
    # Data da correção: 2025-10-20
    #
    "conversion_rates": {
        "D1": 0.007555,  # 0.76% (era 0.26% observado, +0.50pp)
        "D2": 0.007555,  # 0.76% (era 0.26% observado, +0.50pp)
        "D3": 0.024698,  # 2.47% (era 0.85% observado, +1.62pp)
        "D4": 0.027313,  # 2.73% (era 0.94% observado, +1.79pp)
        "D5": 0.029637,  # 2.96% (era 1.02% observado, +1.94pp)
        "D6": 0.032252,  # 3.23% (era 1.11% observado, +2.12pp)
        "D7": 0.034577,  # 3.46% (era 1.19% observado, +2.27pp)
        "D8": 0.034577,  # 3.46% (era 1.19% observado, +2.27pp)
        "D9": 0.039807,  # 3.98% (era 1.37% observado, +2.61pp)
        "D10": 0.061889,  # 6.19% (era 2.13% observado, +4.06pp)
    }
}

