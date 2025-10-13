Análise da Arquitetura Atual

  ✅ Pontos Fortes (Boa para POC)

  | Aspecto                        | Status       | Comentário                                                            |
  |--------------------------------|--------------|-----------------------------------------------------------------------|
  | Separação de Responsabilidades | ✅ Ótimo      | ML pipeline separado da API, módulos bem organizados                  |
  | Escalabilidade Básica          | ✅ Bom        | Batching de leads (600 por request), Cloud Run escala automaticamente |
  | Facilidade de Uso              | ✅ Excelente  | Interface no Sheets = zero curva de aprendizado para cliente          |
  | Deploy                         | ✅ Bom        | Docker + Cloud Run = deploy rápido e confiável                        |
  | Observabilidade                | ✅ Suficiente | Logs estruturados com logging, request_id para tracking               |

  ---
  ⚠️ Lacunas e Melhorias Necessárias

  SEGURANÇA 🔴

  Problema 1: Token hardcoded no código
  # api/meta_config.py - ATUAL
  META_CONFIG = {
      "access_token": "EAAS9hlWC7lk...",  # ❌ Token visível no código
  }

  Solução: Usar Google Secret Manager
  # RECOMENDADO
  from google.cloud import secretmanager

  def get_meta_token():
      client = secretmanager.SecretManagerServiceClient()
      name = "projects/smart-ads-451319/secrets/meta-access-token/versions/latest"
      response = client.access_secret_version(request={"name": name})
      return response.payload.data.decode('UTF-8')

  META_CONFIG = {
      "access_token": get_meta_token(),  # ✅ Token seguro
  }

  Impacto: 🔴 CRÍTICO - Token exposto no repositório Git = qualquer pessoa com acesso pode usar

  ---
  Problema 2: Autenticação do Apps Script
  # app.py - ATUAL (linhas comentadas)
  # Não há verificação de token OAuth do Apps Script

  Status Atual: Apps Script envia token, mas API não valida (código comentado)

  Solução: Descomentar e implementar validação OAuth
  # app.py
  from google.oauth2 import id_token
  from google.auth.transport import requests as google_requests

  async def verify_apps_script_token(request: Request):
      auth_header = request.headers.get('Authorization')
      if not auth_header:
          raise HTTPException(401, "Token ausente")

      token = auth_header.replace('Bearer ', '')
      try:
          idinfo = id_token.verify_oauth2_token(
              token,
              google_requests.Request(),
              SERVICE_ACCOUNT_EMAIL
          )
          return idinfo
      except ValueError:
          raise HTTPException(401, "Token inválido")

  @app.post("/predict/batch")
  async def predict_batch(request: Request, data: PredictionRequest):
      await verify_apps_script_token(request)  # ✅ Valida antes
      # ... resto do código

  Impacto: 🟡 MÉDIO - API está aberta publicamente (qualquer um pode chamar)

  ---
  PERFORMANCE 🟡

  Problema 3: Meta Ads API é bloqueante
  # app.py linha 496
  costs_by_period = meta_client.get_costs_multiple_periods(
      account_id=request.account_id,
      periods=[1, 3, 7]  # 4 chamadas sequenciais (1D, 3D, 7D, Total)
  )

  Cada período faz 3 chamadas à Meta API (campaign, adset, ad) = 12 chamadas sequenciais total

  Impacto: Tempo de resposta: ~15-30 segundos

  Solução: Usar asyncio para paralelizar
  import asyncio
  import aiohttp

  async def get_costs_async(account_id, periods):
      async with aiohttp.ClientSession() as session:
          tasks = []
          for days in periods:
              for level in ['campaign', 'adset', 'ad']:
                  task = fetch_insights_async(session, account_id, level, days)
                  tasks.append(task)

          results = await asyncio.gather(*tasks)
          # Processar resultados
          return organize_costs(results)

  Ganho esperado: 15-30s → 5-8s

  ---
  Problema 4: Pipeline de ML roda a cada análise
  # app.py - se leads não têm predições, roda todo o pipeline
  if not has_predictions:
      result_df = run_production_pipeline(leads_df, config)

  Impacto: ~5-10s para 1000 leads

  Solução (futuro):
  - Opção 1: Cache de features no BigQuery
  - Opção 2: Predições incrementais (só novos leads)
  - Opção 3: Modelo serverless (Vertex AI Prediction)

  Para POC: ✅ OK como está

  ---
  Problema 5: Sem rate limiting
  # API não tem proteção contra abuso
  @app.post("/predict/batch")
  async def predict_batch(request):
      # ❌ Nenhum rate limit

  Solução: Adicionar middleware
  from slowapi import Limiter, _rate_limit_exceeded_handler
  from slowapi.util import get_remote_address

  limiter = Limiter(key_func=get_remote_address)
  app.state.limiter = limiter

  @app.post("/predict/batch")
  @limiter.limit("10/minute")  # ✅ 10 requests por minuto
  async def predict_batch(request: Request):
      pass

  Impacto: 🟡 MÉDIO - Alguém pode fazer spam na API

  ---
  CONFIABILIDADE 🟡

  Problema 6: Sem retry em chamadas Meta API
  # meta_integration.py linha 62
  response = requests.get(url, params=params)
  response.raise_for_status()  # ❌ Falha na primeira tentativa

  Solução: Adicionar retry com backoff exponencial
  from tenacity import retry, stop_after_attempt, wait_exponential

  @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
  def get_insights(self, account_id, level, days):
      response = requests.get(url, params=params)
      response.raise_for_status()
      return response.json()

  Impacto: 🟡 MÉDIO - Rate limit temporário do Meta derruba análise inteira

  ---
  Problema 7: Sem monitoramento de erros

  Não há integração com:
  - Google Cloud Monitoring
  - Sentry / Datadog
  - Alertas de falhas

  Solução: Adicionar Sentry
  import sentry_sdk
  from sentry_sdk.integrations.fastapi import FastApiIntegration

  sentry_sdk.init(
      dsn="https://xxxxx@sentry.io/xxxxx",
      integrations=[FastApiIntegration()],
      traces_sample_rate=0.1,
  )

  ---
  DADOS E MODELO 🟢

  Problema 8: Modelo não é versionado
  # src/model/prediction.py
  model_path = "arquivos_modelo/model_metadata_v1_devclub_rf_temporal_single-3.json"

  Para POC: ✅ OK

  Para produção: Usar MLflow ou Vertex AI Model Registry

  ---
  Problema 9: Sem A/B testing de modelos

  Impossível comparar modelo antigo vs novo em produção

  Para POC: ✅ OK (não necessário)

  Para produção: Implementar feature flag para % de tráfego por modelo

  ---
  📊 Resumo: Status Para POC

  | Categoria        | Status           | Prioridade de Fix         |
  |------------------|------------------|---------------------------|
  | Funcionalidade   | ✅ Excelente      | -                         |
  | Performance      | 🟡 Aceitável     | Baixa (otimizar depois)   |
  | Segurança        | 🔴 Tem lacunas   | 🔴 ALTA (token hardcoded) |
  | Confiabilidade   | 🟡 Pode melhorar | Média (retry Meta API)    |
  | Escalabilidade   | ✅ Boa            | Baixa                     |
  | Manutenibilidade | ✅ Boa            | -                         |

  Veredicto: ✅ Para POC está legal! Mas precisa resolver segurança antes de produção real.

  ---