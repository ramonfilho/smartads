"""
Configuração da API do Meta (Facebook Ads)
Para produção: Cliente adiciona você como admin no Business Manager
"""

# Credenciais Meta Ads - CONTA SANDBOX
META_CONFIG = {
    "access_token": "EAAS9hlWC7lkBPqBZADSlJOZAgXWAfHF8WmGzmZBiZBtbZA8b35asUvx3O6ELvFaMUmoS3Rl2jGjIZAOUsGE2BHfnqs2jZCBJFwjAZBTTq5YS2OQHp5RVnZClaMZCXuNb9jnVbnqh50AjZBFCtoeJpCllpBYh6ATz3DM6ZCFwObefe6khkAAvAkscY8n7s5sl2eAZCDa02t38Yg2Ce",
    "api_version": "v18.0",
    # Account ID: act_1948313086122284 (sandbox "smart_ads")
    # Account ID será passado por Apps Script
}

# Configuração de negócio
BUSINESS_CONFIG = {
    "product_value": 2027.38,  # Baseado em análise de 1.970 vendas DevClub desde 01/03/2025
    "min_roas": 2.0,
    "conversion_rates": {
        "D1": 0.0026,
        "D2": 0.0026,
        "D3": 0.0085,
        "D4": 0.0094,
        "D5": 0.0102,
        "D6": 0.0111,
        "D7": 0.0119,
        "D8": 0.0119,
        "D9": 0.0137,
        "D10": 0.0213
    }
}

