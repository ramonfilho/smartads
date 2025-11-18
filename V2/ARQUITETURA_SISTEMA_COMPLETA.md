# MAPEAMENTO COMPLETO DO SISTEMA SMART ADS V2

> **DOCUMENTO CRÍTICO**: Leia este documento no início de TODA sessão de desenvolvimento.
> Última atualização: 2025-11-18

---

## STATUS DA CONEXÃO COM BANCO DE DADOS

### RESOLVIDO (18/11/2025)

A conexão com Cloud SQL PostgreSQL está **FUNCIONANDO** corretamente via Unix socket.

**Solução implementada:**
- `database.py` agora detecta `CLOUD_SQL_CONNECTION_NAME` e usa pg8000 driver
- Conexão via `URL.create()` com `unix_sock` parameter
- Password configurado via `DB_PASSWORD` environment variable

**Variáveis de ambiente necessárias:**
```bash
CLOUD_SQL_CONNECTION_NAME=smart-ads-451319:us-central1:smart-ads-db
DB_NAME=smart_ads
DB_USER=postgres
DB_PASSWORD=SmartAds2025!
```

**Nota sobre Landing Page:** A página de captura usa `fetch` com `keepalive: true` (não `sendBeacon`).

---

## Estrutura Geral do Projeto

```
/Users/ramonmoreira/Desktop/smart_ads/V2/
├── api/                          # API REST (FastAPI) - Predições e CAPI
├── src/                          # Código ML - Pipeline de treino e produção
│   ├── data_processing/          # Pré-processamento e engenharia de dados
│   ├── features/                 # Engenharia e encoding de features
│   ├── matching/                 # Matching entre leads e vendas
│   ├── model/                    # Treinamento e predição
│   ├── production_pipeline.py    # Pipeline de produção
│   └── train_pipeline.py         # Pipeline de treino
├── landing_page_capi/            # Frontend - Landing Page + captura de leads CAPI
├── files/                        # Artefatos do modelo
├── configs/                      # Configurações (active_model.yaml, devclub.yaml)
├── arquivos_modelo/              # Modelos salvos (.pkl, .json)
├── mlruns/                       # MLflow experiment tracking
└── tests/                        # Testes unitários
```

---

## PARTE 1: API REST (FastAPI)

### Arquivo: `api/app.py` (1709 linhas)

**Propósito:** API principal que orquestra predições de lead scoring, captura de leads CAPI, análise UTM e integração com Meta Ads.

**Framework:** FastAPI + Uvicorn
**Porta:** 8000 (desenvolvimento) ou Cloud Run (produção)
**URL Base Produção:** `https://smart-ads-api-12955519745.us-central1.run.app`

#### Endpoints Principais

| Endpoint | Método | Propósito |
|----------|--------|-----------|
| `/health` | GET | Verificar pipeline e modelo |
| `/predict/batch` | POST | Predição em batch via JSON |
| `/webhook/lead_capture` | POST | Captura leads com FBP/FBC |
| `/webhook/lead_capture/stats` | GET | Estatísticas de leads capturados |
| `/webhook/lead_capture/recent` | GET | Últimos N leads capturados |
| `/capi/process_daily_batch` | POST | Processa batch diário CAPI |

#### Modelos Pydantic Principais

```python
class LeadCaptureRequest:
    name: str                      # Nome completo
    first_name: Optional[str]      # Primeiro nome (CAPI)
    last_name: Optional[str]       # Sobrenome (CAPI)
    email: str
    phone: Optional[str]
    fbp: Optional[str]             # Facebook Browser ID (_fbp cookie)
    fbc: Optional[str]             # Facebook Click ID (_fbc cookie)
    event_id: str                  # ID único evento (deduplicação)
    user_agent: Optional[str]
    event_source_url: Optional[str]
    utm_source/medium/campaign/term/content: Optional[str]
```

---

### Arquivo: `api/database.py` (201 linhas)

**Propósito:** Gerenciar conexão PostgreSQL e modelo SQLAlchemy para tabela `leads_capi`.

**Tecnologia:** SQLAlchemy ORM
**Banco:** PostgreSQL (produção) ou SQLite (desenvolvimento)

#### ⚠️ PROBLEMA CRÍTICO DE CONFIGURAÇÃO

```python
def get_database_url() -> str:
    # Opção 1: URL completa
    if os.getenv('DATABASE_URL'):
        return os.getenv('DATABASE_URL')

    # Opção 2: Componentes individuais (Cloud SQL)
    db_host = os.getenv('DB_HOST')
    db_password = os.getenv('DB_PASSWORD')

    if db_host and db_password:  # ← NUNCA É TRUE NO CLOUD RUN!
        return f"postgresql://..."

    # Opção 3: Fallback SQLite ← SEMPRE CAI AQUI EM PRODUÇÃO!
    return "sqlite:////tmp/smart_ads_dev.db"
```

**O Cloud Run tem:**
- `run.googleapis.com/cloudsql-instances: smart-ads-451319:us-central1:smart-ads-db`

**Mas o código NÃO sabe usar isso!** Precisa conectar via:
```
postgresql+pg8000://postgres@/smart_ads?unix_sock=/cloudsql/smart-ads-451319:us-central1:smart-ads-db/.s.PGSQL.5432
```

#### Modelo: LeadCAPI

```python
class LeadCAPI(Base):
    __tablename__ = 'leads_capi'

    id = Column(Integer, primary_key=True)
    email = Column(String(255), nullable=False, index=True)
    name = Column(String(255))
    first_name = Column(String(255))       # Para CAPI
    last_name = Column(String(255))        # Para CAPI
    phone = Column(String(50))
    fbp = Column(String(255))              # Facebook Browser ID
    fbc = Column(String(255))              # Facebook Click ID
    event_id = Column(String(255), unique=True, index=True)
    user_agent = Column(Text)
    client_ip = Column(String(50))
    event_source_url = Column(Text)
    utm_source/medium/campaign/term/content = Column(String(255))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
```

---

### Arquivo: `api/capi_integration.py` (726 linhas)

**Propósito:** Integração com Meta Conversions API (CAPI) para envio server-side de eventos.

**Tecnologia:** facebook-business SDK
**Pixel ID:** 241752320666130

#### Função Principal: `send_lead_qualified_with_value()`

Envia eventos LeadQualified para Meta com:
- UserData (email, phone, nome - hashados SHA256)
- CustomData (valor por decil)
- event_id (deduplicação)
- fbp/fbc (cookies Facebook)

#### Valores por Decil

```python
CONVERSION_RATES = {
    "D1": 0.003836,   # 0.38% → R$ 7.67
    "D2": 0.004933,   # 0.49% → R$ 9.87
    # ...
    "D10": 0.034551,  # 3.45% → R$ 69.10
}
```

---

### Arquivo: `api/business_config.py` (243 linhas)

**Propósito:** Configurações centralizadas de negócio.

```python
PRODUCT_VALUE = 2000.00  # R$ (LTV)
CONVERSION_RATES = {...}  # Taxas por decil
```

---

## PARTE 2: PIPELINE ML (src/)

### Arquivo: `src/production_pipeline.py` (285 linhas)

**Propósito:** Pipeline de produção que orquestra pré-processamento → features → predição.

#### Classe: LeadScoringPipeline

```python
class LeadScoringPipeline:
    def run(filepath: str) -> pd.DataFrame:
        """
        1. load_data()
        2. preprocess() → 10 passos
        3. predict() → modelo RandomForest
        """
```

#### Passos de Pré-processamento

1. remove_duplicates()
2. clean_columns()
3. remove_campaign_features()
4. unify_utm_columns()
5. unify_medium_columns()
6. remove_technical_fields()
7. rename_long_column_names()
8. create_derived_features()
9. apply_categorical_encoding()
10. handle_missing_values()

---

### Arquivo: `src/model/prediction.py` (231 linhas)

**Propósito:** Carregar modelo RandomForest e realizar predições.

#### Arquivos do Modelo

```
arquivos_modelo/ ou files/{timestamp}/
├── modelo_lead_scoring_*.pkl        # RandomForest serializado
├── features_ordenadas_*.json        # Lista de features esperadas
└── model_metadata_*.json            # Métricas, performance
```

---

## PARTE 3: LANDING PAGE CAPI

### Arquivo: `landing_page_capi/codigo_formulario_completo_com_capi.js`

**Propósito:** Capturar formulário no frontend, enviar para API.

> **NOTA (18/11/2025):** Atualmente a página de captura está usando `fetch` com `keepalive: true`, e NÃO `sendBeacon`. A função sendBeacon está no código mas não está ativa.

#### Fluxo de Captura

```javascript
1. getCookie('_fbp')      → Facebook Browser ID
2. getCookie('_fbc')      → Facebook Click ID
3. generateEventID()      → Deduplicação
4. extractUTM()           → utm_source, etc
5. splitName()            → first_name, last_name
6. sendToCapiAPI()        → POST /webhook/lead_capture (fetch com keepalive)
```

---

## PARTE 4: CONFIGURAÇÕES

### `configs/active_model.yaml`

Define qual modelo está em produção:

```yaml
active_model:
  model_name: "v1_devclub_rf_temporal_single"
  model_path: "files/20251111_212345"
```

---

## PARTE 5: FLUXOS PRINCIPAIS

### Fluxo 1: Predição Google Sheets

```
Google Sheets → Apps Script → POST /predict/batch → Pipeline ML → Scores → Sheets
```

### Fluxo 2: Captura CAPI

```
Landing Page (JS) → POST /webhook/lead_capture → PostgreSQL → Batch CAPI → Meta
```

---

## PARTE 6: VARIÁVEIS DE AMBIENTE

### Produção (Cloud Run) - CONFIGURAÇÃO CORRETA

```bash
# Database - VIA UNIX SOCKET (não precisa host/password!)
CLOUD_SQL_CONNECTION_NAME=smart-ads-451319:us-central1:smart-ads-db
DB_NAME=smart_ads
DB_USER=postgres

# Meta CAPI
META_PIXEL_ID=241752320666130
META_ACCESS_TOKEN=xxx

# Google Cloud
GCP_PROJECT_ID=smart-ads-451319
```

### Anotação Cloud Run (já configurada)

```yaml
run.googleapis.com/cloudsql-instances: smart-ads-451319:us-central1:smart-ads-db
```

---

## PARTE 7: CHECKLIST PRÉ-DEPLOY

### ⚠️ ANTES DE QUALQUER DEPLOY, VERIFICAR:

```
[x] Database (RESOLVIDO 18/11/2025)
    [x] database.py suporta CLOUD_SQL_CONNECTION_NAME
    [x] Conexão via pg8000 + Unix socket
    [x] DB_PASSWORD configurado

[ ] Modelo
    [ ] configs/active_model.yaml aponta para modelo correto
    [ ] Arquivos .pkl, .json existem no path

[ ] Meta CAPI
    [ ] META_ACCESS_TOKEN válido (não expirado)
    [ ] Testar evento de teste no Events Manager

[ ] Teste End-to-End
    [ ] POST /webhook/lead_capture → verifica no banco
    [ ] GET /webhook/lead_capture/stats → retorna count > 0
```

---

## PARTE 8: PONTOS CRÍTICOS DE FALHA

| Componente | Risco | Impacto | Solução |
|-----------|-------|---------|---------|
| ~~database.py não conecta ao Cloud SQL~~ | ~~CRÍTICO~~ | ~~Dados perdidos~~ | **RESOLVIDO** - pg8000 + Unix socket |
| Pipeline não carrega | Alto | 503 errors | Validar active_model.yaml |
| Meta token expirado | Alto | CAPI fails | Renovar token (60 dias) |
| FBP/FBC ausentes | Médio | Baixa cobertura | Usar hash como fallback |

---

## HISTÓRICO DE PROBLEMAS

### 18/11/2025 - Dados CAPI perdidos - **RESOLVIDO**

**Sintoma:** Apenas 8 leads no banco após deploy
**Causa:** database.py usando SQLite fallback
**Solução implementada:**
- Adicionado suporte a `CLOUD_SQL_CONNECTION_NAME` em database.py
- Usa pg8000 driver com `URL.create()` e `unix_sock` parameter
- Configurado `DB_PASSWORD=SmartAds2025!`
- Migration aplicada para adicionar colunas `first_name` e `last_name`

### 18/11/2025 - Nome/Sobrenome 0% cobertura

**Sintoma:** Quality Score caiu de 7.4 para 7.1
**Causa:** Página "parabéns" sobrescrevia registro com dados incompletos
**Solução:** Priorizar registros com first_name + fallback de extração do campo name

### 18/11/2025 - Gender enum error

**Sintoma:** Erro ao enviar CAPI
**Causa:** Passando string ao invés de Gender enum
**Solução:** Usar Gender.MALE/Gender.FEMALE diretamente

---

## COMANDOS ÚTEIS

### Ver status do serviço
```bash
gcloud run services describe smart-ads-api --region=us-central1
```

### Ver logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=smart-ads-api" --limit=50
```

### Conectar ao Cloud SQL (local)
```bash
gcloud sql connect smart-ads-db --user=postgres --database=smart_ads
```

### Verificar leads no banco
```sql
SELECT COUNT(*) FROM leads_capi;
SELECT * FROM leads_capi ORDER BY created_at DESC LIMIT 10;
```

---

**Este documento deve ser consultado no início de TODA sessão de desenvolvimento para evitar erros críticos em produção.**
