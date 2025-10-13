"""
Script para testar se há dados de insights (spend) nas campanhas da sandbox
"""

import requests
import json
from datetime import datetime, timedelta

ACCESS_TOKEN = "EAAS9hlWC7lkBPqBZADSlJOZAgXWAfHF8WmGzmZBiZBtbZA8b35asUvx3O6ELvFaMUmoS3Rl2jGjIZAOUsGE2BHfnqs2jZCBJFwjAZBTTq5YS2OQHp5RVnZClaMZCXuNb9jnVbnqh50AjZBFCtoeJpCllpBYh6ATz3DM6ZCFwObefe6khkAAvAkscY8n7s5sl2eAZCDa02t38Yg2Ce"
ACCOUNT_ID = "act_1948313086122284"
API_VERSION = "v18.0"

# Testar insights de campanhas
url = f"https://graph.facebook.com/{API_VERSION}/{ACCOUNT_ID}/insights"

since = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
until = datetime.now().strftime('%Y-%m-%d')

params = {
    'access_token': ACCESS_TOKEN,
    'level': 'campaign',
    'fields': 'campaign_name,spend,impressions,clicks',
    'time_range': json.dumps({'since': since, 'until': until}),
    'limit': 10
}

print(f"🔍 Buscando insights de campanhas (últimos 7 dias: {since} até {until})...")
print()

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    insights = data.get('data', [])

    print(f"📊 Total de campanhas com dados: {len(insights)}")
    print()

    if len(insights) == 0:
        print("⚠️  PROBLEMA IDENTIFICADO:")
        print("   Nenhuma campanha tem dados de spend/impressions")
        print("   Isso é esperado porque:")
        print("   1. Campanhas foram criadas com status PAUSED")
        print("   2. Campanhas pausadas não geram insights")
        print("   3. Na sandbox, não há como simular spend diretamente")
        print()
        print("💡 SOLUÇÃO:")
        print("   Para testar com dados reais, você precisa:")
        print("   - Conectar à conta REAL do cliente (não sandbox)")
        print("   - Usar o token de produção do cliente")
        print()
    else:
        for insight in insights:
            name = insight.get('campaign_name', 'N/A')
            spend = insight.get('spend', '0')
            impressions = insight.get('impressions', '0')
            clicks = insight.get('clicks', '0')

            print(f"Campanha: {name[:60]}...")
            print(f"  Spend: ${spend}")
            print(f"  Impressions: {impressions}")
            print(f"  Clicks: {clicks}")
            print()
else:
    print(f"❌ Erro: {response.status_code}")
    print(response.text)
