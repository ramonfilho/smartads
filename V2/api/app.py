"""
API V2 para Lead Scoring - Batch Predictions
Otimizada para Google Sheets + Apps Script + Google Cloud
"""

import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import io
import time
import tempfile
import uuid
from datetime import datetime
import logging

# Adicionar diretório pai ao path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar pipeline V2
from src.production_pipeline import LeadScoringPipeline

# Importar integrações
from api.meta_integration import MetaAdsIntegration, enrich_utm_analysis_with_costs, enrich_utm_with_hierarchy
from api.meta_config import META_CONFIG, BUSINESS_CONFIG
from api.economic_metrics import enrich_utm_with_economic_metrics

# Importar módulos CAPI
from api.database import get_db, init_database, create_lead_capi, count_leads, count_leads_with_fbp, count_leads_with_fbc, get_leads_by_emails
from api.capi_integration import send_batch_events
from fastapi import Depends, Request
from sqlalchemy.orm import Session

# Padrões de UTMs inválidos (bare names e genéricos)
BARE_CAMPAIGN_NAMES = ['DEVLF', 'devlf']                      # Prefixos incompletos
BARE_MEDIUM_NAMES = ['dgen', 'paid']                          # Termos genéricos sem estrutura
GENERIC_TERMS = ['fb', 'ig', 'instagram', 'facebook']         # Apenas redes sociais

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === MODELS ===
class LeadData(BaseModel):
    """Modelo para um lead individual"""
    data: Dict[str, Any]
    email: Optional[str] = None  # Para identificação
    row_id: Optional[str] = None  # ID da linha no Google Sheets

class BatchPredictionRequest(BaseModel):
    """Request para predições em batch"""
    leads: List[LeadData] = Field(..., min_items=1, max_items=600)
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

class PredictionResult(BaseModel):
    """Resultado de uma predição"""
    lead_score: float
    email: Optional[str] = None
    row_id: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Response para predições em batch"""
    request_id: str
    total_leads: int
    predictions: List[PredictionResult]
    processing_time_seconds: float
    timestamp: str

# Inicializar a aplicação FastAPI
app = FastAPI(
    title="Smart Ads Lead Scoring API V2",
    description="API otimizada para predições em batch via Google Sheets",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Adicionar CORS para Google Apps Script
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://script.google.com", "https://script.googleusercontent.com"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Variável global para o pipeline
pipeline = None

def initialize_pipeline():
    """Inicializa o pipeline de lead scoring"""
    global pipeline
    try:
        logger.info("Inicializando pipeline de Lead Scoring...")
        pipeline = LeadScoringPipeline(model_name='v1_devclub_rf_temporal_single')
        logger.info("Pipeline inicializado com sucesso!")
        return True
    except Exception as e:
        logger.error(f"Erro ao inicializar pipeline: {e}")
        return False

# DEPRECATED: Decis agora são calculados por janela de análise
# def convert_decile_to_numeric(decile_str: str) -> int:
#     """Converte D1-D10 para 1-10"""
#     try:
#         return int(decile_str.replace('D', ''))
#     except:
#         return 5

@app.on_event("startup")
async def startup_event():
    """Inicialização da aplicação"""
    logger.info("🚀 Iniciando Smart Ads API V2...")
    if not initialize_pipeline():
        logger.error("❌ Falha ao inicializar pipeline!")
    else:
        logger.info("✅ API V2 pronta para receber requisições!")

    # Inicializar database
    if init_database():
        logger.info("✅ Database inicializado com sucesso")
    else:
        logger.warning("⚠️ Database não inicializado (desenvolvimento sem PostgreSQL?)")

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "Smart Ads Lead Scoring API V2",
        "status": "online",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/batch (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check detalhado"""
    pipeline_status = "healthy" if pipeline is not None else "unhealthy"
    model_loaded = pipeline is not None

    return {
        "status": "healthy",
        "pipeline_status": pipeline_status,
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/model/info")
async def get_model_info():
    """
    Retorna informações sobre o modelo: metadados, performance e feature importances
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    try:
        # Garantir que o modelo está carregado
        if pipeline.predictor.model is None:
            pipeline.predictor.load_model()

        # Obter metadados
        metadata = pipeline.predictor.metadata

        # Obter feature importances (top 20)
        feature_importances = pipeline.predictor.get_feature_importances(top_n=20)

        # Estruturar resposta
        response = {
            "model_info": metadata.get("model_info", {}),
            "training_data": metadata.get("training_data", {}),
            "performance_metrics": metadata.get("performance_metrics", {}),
            "decil_analysis": metadata.get("decil_analysis", {}),
            "feature_importances": feature_importances,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"✅ Informações do modelo retornadas com sucesso")
        return response

    except Exception as e:
        logger.error(f"❌ Erro ao obter informações do modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter informações do modelo: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_json(request: BatchPredictionRequest):
    """
    Predição em batch via JSON
    Otimizado para Google Apps Script
    """
    global pipeline

    # Verificar pipeline
    if pipeline is None:
        if not initialize_pipeline():
            raise HTTPException(status_code=500, detail="Pipeline não inicializado")

    start_time = time.time()
    logger.info(f"📊 Processando {len(request.leads)} leads (Request ID: {request.request_id})")

    temp_file = None

    try:
        # Converter leads para DataFrame
        lead_rows = []
        for i, lead in enumerate(request.leads):
            row = lead.data.copy()
            # Adicionar metadados
            row['_email'] = lead.email
            row['_row_id'] = lead.row_id or str(i)
            lead_rows.append(row)

        df = pd.DataFrame(lead_rows)
        logger.info(f"📋 DataFrame criado: {df.shape}")

        # Criar arquivo temporário para o pipeline
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            # Salvar sem as colunas de metadados para o modelo
            model_df = df.drop(columns=['_email', '_row_id'], errors='ignore')
            model_df.to_csv(tmp, index=False)
            temp_file = tmp.name

        # Executar pipeline
        logger.info("🔄 Executando pipeline...")
        result_df = pipeline.run(temp_file, with_predictions=True)

        if result_df is None or len(result_df) == 0:
            raise HTTPException(status_code=500, detail="Pipeline retornou resultado vazio")

        # Processar resultados
        predictions = []
        for i, (_, row) in enumerate(result_df.iterrows()):
            lead_score = float(row['lead_score'])

            # Recuperar metadados do lead original
            original_lead = request.leads[i] if i < len(request.leads) else None
            email = original_lead.email if original_lead else None
            row_id = original_lead.row_id if original_lead else str(i)

            predictions.append(PredictionResult(
                lead_score=lead_score,
                email=email,
                row_id=row_id
            ))

        processing_time = time.time() - start_time

        logger.info(f"✅ Processamento concluído em {processing_time:.2f}s")
        logger.info(f"📈 Scores: min={min(p.lead_score for p in predictions):.3f}, max={max(p.lead_score for p in predictions):.3f}")

        return BatchPredictionResponse(
            request_id=request.request_id,
            total_leads=len(predictions),
            predictions=predictions,
            processing_time_seconds=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"❌ Erro no processamento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    finally:
        # Limpar arquivo temporário
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/predict/csv")
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Predição em batch via upload CSV
    Para testes ou uploads manuais
    """
    global pipeline

    if pipeline is None:
        if not initialize_pipeline():
            raise HTTPException(status_code=500, detail="Pipeline não inicializado")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Apenas arquivos CSV são aceitos")

    start_time = time.time()
    logger.info(f"📄 Processando arquivo CSV: {file.filename}")

    temp_file = None

    try:
        # Salvar arquivo temporário
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name

        # Executar pipeline
        result_df = pipeline.run(temp_file, with_predictions=True)

        if result_df is None:
            raise HTTPException(status_code=500, detail="Pipeline retornou resultado vazio")

        # Processar resultados
        predictions = []
        for _, row in result_df.iterrows():
            predictions.append({
                "lead_score": float(row['lead_score']),  # Probabilidade
                "email": row.get('E-mail', None),
                "name": row.get('Nome Completo', None)
            })

        processing_time = time.time() - start_time

        logger.info(f"✅ CSV processado: {len(predictions)} leads em {processing_time:.2f}s")

        return {
            "total_leads": len(predictions),
            "predictions": predictions,
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Erro no processamento CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

# === WEBHOOK PARA CAPTURA DE LEADS (CAPI) ===

class LeadCaptureRequest(BaseModel):
    """Dados capturados do lead no frontend"""
    # Dados pessoais
    name: str
    email: str
    phone: Optional[str] = None

    # Dados CAPI
    fbp: Optional[str] = None
    fbc: Optional[str] = None
    event_id: str
    user_agent: Optional[str] = None
    event_source_url: Optional[str] = None

    # UTMs
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_term: Optional[str] = None
    utm_content: Optional[str] = None

    # Outros
    tem_comp: Optional[str] = None

@app.post("/webhook/lead_capture")
async def webhook_lead_capture(
    request: Request,
    lead_data: LeadCaptureRequest,
    db: Session = Depends(get_db)
):
    """
    Webhook para capturar dados de leads com FBP/FBC
    Chamado pelo formulário frontend após envio do lead
    """
    try:
        # Capturar IP do cliente (real, não do proxy Cloud Run)
        client_ip = request.headers.get('X-Forwarded-For', request.client.host).split(',')[0].strip()

        # Preparar dados para banco
        lead_dict = {
            'email': lead_data.email,
            'name': lead_data.name,
            'phone': lead_data.phone,
            'fbp': lead_data.fbp,
            'fbc': lead_data.fbc,
            'event_id': lead_data.event_id,
            'user_agent': lead_data.user_agent,
            'client_ip': client_ip,
            'event_source_url': lead_data.event_source_url,
            'utm_source': lead_data.utm_source,
            'utm_medium': lead_data.utm_medium,
            'utm_campaign': lead_data.utm_campaign,
            'utm_term': lead_data.utm_term,
            'utm_content': lead_data.utm_content,
            'tem_comp': lead_data.tem_comp
        }

        # Salvar no banco
        lead_record = create_lead_capi(db, lead_dict)

        logger.info(f"✅ Lead capturado: {lead_data.email} (ID: {lead_record.id}, Event ID: {lead_data.event_id})")

        return {
            "status": "success",
            "message": "Lead capturado com sucesso",
            "lead_id": lead_record.id,
            "event_id": lead_data.event_id
        }

    except Exception as e:
        logger.error(f"❌ Erro ao capturar lead: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao capturar lead: {str(e)}")

@app.get("/webhook/lead_capture/stats")
async def lead_capture_stats(db: Session = Depends(get_db)):
    """
    Estatísticas de captura de leads CAPI
    Útil para monitoramento e debug
    """
    try:
        total = count_leads(db)
        with_fbp = count_leads_with_fbp(db)
        with_fbc = count_leads_with_fbc(db)

        return {
            "total_leads": total,
            "leads_with_fbp": with_fbp,
            "leads_with_fbc": with_fbc,
            "fbp_fill_rate": round(with_fbp / total * 100, 2) if total > 0 else 0,
            "fbc_fill_rate": round(with_fbc / total * 100, 2) if total > 0 else 0
        }

    except Exception as e:
        logger.error(f"❌ Erro ao obter stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter stats: {str(e)}")

# === CAPI BATCH PROCESSING ===

class CapiBatchRequest(BaseModel):
    """Request para processamento batch CAPI"""
    leads_d10: List[Dict[str, Any]] = Field(..., description="Leads D10 do dia anterior")

@app.post("/capi/process_daily_batch")
async def process_daily_batch_capi(
    request: CapiBatchRequest,
    db: Session = Depends(get_db)
):
    """
    Processa batch diário de CAPI
    Envia eventos LeadQualified para leads D10

    Chamado pelo Apps Script às 00:00 após classificação ML
    """
    try:
        logger.info(f"📊 Processando batch CAPI: {len(request.leads_d10)} leads D10")

        # Extrair emails dos leads
        emails = [lead['email'] for lead in request.leads_d10 if 'email' in lead]

        if not emails:
            logger.warning("⚠️ Nenhum email encontrado nos leads D10")
            return {
                "status": "error",
                "message": "Nenhum email encontrado",
                "total": 0,
                "success": 0,
                "errors": 0
            }

        # Buscar dados CAPI do banco
        leads_capi = get_leads_by_emails(db, emails)

        # Criar mapeamento email → dados CAPI
        capi_map = {lead.email: lead for lead in leads_capi}

        logger.info(f"   {len(capi_map)} leads encontrados no banco CAPI")

        # Enriquecer leads D10 com dados CAPI
        enriched_leads = []
        for lead in request.leads_d10:
            email = lead.get('email')
            if not email:
                continue

            capi_data = capi_map.get(email)

            # Montar dados para CAPI
            lead_capi = {
                'email': email,
                'phone': lead.get('phone'),
                'lead_score': lead['lead_score'],
                'decil': lead['decil'],
                'event_id': capi_data.event_id if capi_data else f"lead_{int(time.time())}_{email[:8]}",
                'fbp': capi_data.fbp if capi_data else None,
                'fbc': capi_data.fbc if capi_data else None,
                'user_agent': capi_data.user_agent if capi_data else None,
                'client_ip': capi_data.client_ip if capi_data else None,
                'event_source_url': capi_data.event_source_url if capi_data else None,
                'event_timestamp': int(pd.to_datetime(lead['data']).timestamp()) if 'data' in lead else int(time.time())
            }

            enriched_leads.append(lead_capi)

        logger.info(f"   {len(enriched_leads)} leads enriquecidos para envio CAPI")

        # Enviar batch
        results = send_batch_events(enriched_leads)

        logger.info(f"✅ Batch CAPI processado: {results['success']}/{results['total']} enviados")

        return {
            "status": "success",
            "total": results['total'],
            "success": results['success'],
            "errors": results['errors'],
            "leads_with_capi_data": len([l for l in enriched_leads if l['fbp'] or l['fbc']]),
            "details": results.get('details', [])
        }

    except Exception as e:
        logger.error(f"❌ Erro no batch CAPI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no batch CAPI: {str(e)}")

# === ANÁLISE UTM COM CUSTOS ===

class UTMAnalysisRequest(BaseModel):
    """Request para análise UTM com custos"""
    leads: List[LeadData] = Field(..., min_items=1)  # Sem limite máximo - batching interno
    account_id: str = Field(..., description="ID da conta Meta Ads (ex: act_123456)")
    product_value: Optional[float] = Field(default=None, description="Valor do produto (padrão: config)")
    min_roas: Optional[float] = Field(default=None, description="ROAS mínimo (padrão: 2.0)")

class UTMDimensionMetrics(BaseModel):
    """Métricas de uma dimensão UTM"""
    campaign: Optional[str] = None  # Para adsets e ads: nome da campanha de origem
    adset: Optional[str] = None     # Para ads: nome do adset de origem
    value: str
    leads: int
    spend: float
    cpl: float
    taxa_proj: float
    receita_proj: float     # Receita projetada (NOVO - Margem de Contribuição)
    margem_contrib: float   # Margem de Contribuição (NOVO - substitui margem%)
    roas_proj: float
    acao: str
    budget_current: float  # Orçamento atual (gasto do período)
    budget_target: float   # Orçamento alvo (baseado na ação)

class UTMPeriodAnalysis(BaseModel):
    """Análise UTM para um período"""
    campaign: List[UTMDimensionMetrics]
    medium: List[UTMDimensionMetrics]
    ad: List[UTMDimensionMetrics]
    google_ads: List[UTMDimensionMetrics]
    # Metadados do período
    period_start: str  # Data/hora do lead mais antigo (ISO format)
    period_end: str    # Data/hora do lead mais recente (ISO format)
    total_leads: int   # Total de leads analisados
    meta_leads: int    # Leads do Meta/Facebook
    google_leads: int  # Leads do Google Ads

class UTMAnalysisResponse(BaseModel):
    """Response completa da análise UTM"""
    request_id: str
    periods: Dict[str, UTMPeriodAnalysis]  # '1D', '3D', '7D', 'Total'
    config: Dict[str, Any]  # product_value, min_roas usado
    processing_time_seconds: float
    timestamp: str

@app.post("/analyze_utms_with_costs", response_model=UTMAnalysisResponse)
async def analyze_utms_with_costs(request: UTMAnalysisRequest):
    """
    Análise UTM enriquecida com custos do Meta Ads e métricas econômicas

    Fluxo:
    1. Executar predições (lead_score, decile)
    2. Buscar custos da API Meta (1D, 3D, 7D, Total)
    3. Calcular análise UTM por dimensão
    4. Enriquecer com métricas econômicas (CPL, ROAS, Margem, Ação)
    5. Retornar estrutura por período
    """
    global pipeline

    # Verificar pipeline
    if pipeline is None:
        if not initialize_pipeline():
            raise HTTPException(status_code=500, detail="Pipeline não inicializado")

    start_time = time.time()
    request_id = str(uuid.uuid4())

    logger.info(f"📊 Iniciando análise UTM com custos (Request ID: {request_id})")
    logger.info(f"   Leads: {len(request.leads)} | Account: {request.account_id}")

    temp_file = None  # Para limpeza no finally (apenas para caso de lote único)

    try:
        # Configuração
        product_value = request.product_value or BUSINESS_CONFIG['product_value']
        min_roas = request.min_roas or BUSINESS_CONFIG['min_roas']
        conversion_rates = BUSINESS_CONFIG['conversion_rates']

        logger.info(f"   Product Value: R$ {product_value:.2f} | Min ROAS: {min_roas}x")

        # 1. VERIFICAR SE JÁ EXISTEM PREDIÇÕES
        total_leads = len(request.leads)
        logger.info(f"   Total de leads: {total_leads}")

        # Debug: mostrar estrutura do primeiro lead
        if total_leads > 0:
            first_lead = request.leads[0].data
            logger.info(f"   🔍 DEBUG: Chaves do primeiro lead: {list(first_lead.keys())[:10]}...")
            logger.info(f"   🔍 DEBUG: Tem 'lead_score'? {'lead_score' in first_lead}")
            logger.info(f"   🔍 DEBUG: Tem 'decile'? {'decile' in first_lead}")

            # Verificar se leads já têm predições (apenas lead_score é necessário)
            has_predictions = 'lead_score' in first_lead
            logger.info(f"   🔍 DEBUG: has_predictions = {has_predictions}")
        else:
            logger.error("   ❌ ERRO: Nenhum lead recebido!")
            raise HTTPException(status_code=400, detail="Nenhum lead recebido")

        if has_predictions:
            logger.info("✅ Leads já possuem predições existentes, pulando etapa de predição...")

            # Construir DataFrame com predições existentes
            lead_rows = []
            for i, lead in enumerate(request.leads):
                row = lead.data.copy()
                row['_email'] = lead.email
                row['_row_id'] = lead.row_id or str(i)
                lead_rows.append(row)

            result_df = pd.DataFrame(lead_rows)
            result_df['email'] = result_df['_email']
            result_df['row_id'] = result_df['_row_id']
            result_df = result_df.drop(columns=['_email', '_row_id'], errors='ignore')

            # Renomear colunas se necessário para padronizar
            if 'decile' in result_df.columns:
                result_df = result_df.rename(columns={'decile': 'decil'})

            logger.info(f"✅ {len(result_df)} leads carregados com predições existentes")

        else:
            # 1. PREDIÇÕES COM BATCHING INTERNO
            logger.info("🔄 Executando predições...")

            # Processar em lotes se necessário
            BATCH_SIZE = 500
            all_results = []

            if total_leads <= BATCH_SIZE:
                # Processar todos de uma vez
                logger.info("   Processando em lote único")
                lead_rows = []
                for i, lead in enumerate(request.leads):
                    row = lead.data.copy()
                    row['_email'] = lead.email
                    row['_row_id'] = lead.row_id or str(i)
                    lead_rows.append(row)

                df = pd.DataFrame(lead_rows)

                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                    model_df = df.drop(columns=['_email', '_row_id'], errors='ignore')
                    model_df.to_csv(tmp, index=False)
                    temp_file = tmp.name

                result_df = pipeline.run(temp_file, with_predictions=True)

                if result_df is None or len(result_df) == 0:
                    raise HTTPException(status_code=500, detail="Pipeline retornou resultado vazio")

                result_df['email'] = df['_email'].values
                result_df['row_id'] = df['_row_id'].values

                all_results.append(result_df)

            else:
                # Processar em lotes
                num_batches = (total_leads + BATCH_SIZE - 1) // BATCH_SIZE
                logger.info(f"   Processando em {num_batches} lotes de ~{BATCH_SIZE} leads")

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min(start_idx + BATCH_SIZE, total_leads)
                    batch_leads = request.leads[start_idx:end_idx]

                    logger.info(f"   Lote {batch_idx + 1}/{num_batches}: {len(batch_leads)} leads")

                    lead_rows = []
                    for i, lead in enumerate(batch_leads):
                        row = lead.data.copy()
                        row['_email'] = lead.email
                        row['_row_id'] = lead.row_id or str(start_idx + i)
                        lead_rows.append(row)

                    batch_df = pd.DataFrame(lead_rows)

                    batch_temp_file = None
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                            model_df = batch_df.drop(columns=['_email', '_row_id'], errors='ignore')
                            model_df.to_csv(tmp, index=False)
                            batch_temp_file = tmp.name

                        batch_result = pipeline.run(batch_temp_file, with_predictions=True)

                        if batch_result is None or len(batch_result) == 0:
                            logger.warning(f"   Lote {batch_idx + 1} retornou vazio")
                            continue

                        batch_result['email'] = batch_df['_email'].values
                        batch_result['row_id'] = batch_df['_row_id'].values

                        all_results.append(batch_result)

                    finally:
                        if batch_temp_file and os.path.exists(batch_temp_file):
                            os.remove(batch_temp_file)

            # Consolidar resultados
            if not all_results:
                raise HTTPException(status_code=500, detail="Nenhum resultado de predição obtido")

            result_df = pd.concat(all_results, ignore_index=True)
            logger.info(f"✅ Predições concluídas: {len(result_df)} leads consolidados")

        # 2. CALCULAR JANELAS TEMPORAIS (dias completos: 00:00-23:59)
        now = pd.Timestamp.now(tz=None)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Usar até 00:00 de hoje (= fim de ontem 23:59:59)
        cutoff_end = today_start

        # Calcular início para cada período
        period_windows = {
            '1D': cutoff_end - pd.Timedelta(days=1),
            '3D': cutoff_end - pd.Timedelta(days=3),
            '7D': cutoff_end - pd.Timedelta(days=7)
        }

        logger.info(f"📅 Janelas temporais (dias completos):")
        for period, start_date in period_windows.items():
            logger.info(f"   {period}: {start_date.strftime('%Y-%m-%d %H:%M')} até {cutoff_end.strftime('%Y-%m-%d %H:%M')}")

        # 3. BUSCAR CUSTOS DA API META (HIERARQUIA COMPLETA POR PERÍODO)
        logger.info("💰 Buscando hierarquia de custos da API Meta...")
        meta_client = MetaAdsIntegration(
            access_token=META_CONFIG['access_token'],
            api_version=META_CONFIG['api_version']
        )

        # Buscar hierarquia completa para cada período separadamente
        # IMPORTANTE: Meta API retorna dados AGREGADOS do período solicitado
        # Não é possível buscar 7D e filtrar para 1D - os custos são diferentes!
        hierarchy_by_period = {}
        for period_key, start_date in period_windows.items():
            # Converter timestamps para strings no formato YYYY-MM-DD
            since_str = start_date.strftime('%Y-%m-%d')
            until_str = cutoff_end.strftime('%Y-%m-%d')

            logger.info(f"🔍 Buscando hierarquia Meta para {period_key} (de {since_str} até {until_str} exclusivo)...")

            hierarchy_by_period[period_key] = meta_client.get_costs_hierarchy(
                account_id=request.account_id,
                since_date=since_str,
                until_date=until_str
            )

        logger.info(f"✅ Hierarquia obtida para {len(hierarchy_by_period)} períodos")

        # Log detalhado de hierarquia por período
        for period_key, hierarchy in hierarchy_by_period.items():
            total_campaigns = len(hierarchy['campaigns'])
            total_spend = sum(c['spend'] for c in hierarchy['campaigns'].values())
            total_adsets = sum(len(c['adsets']) for c in hierarchy['campaigns'].values())
            total_ads = sum(sum(len(a['ads']) for a in c['adsets'].values()) for c in hierarchy['campaigns'].values())
            logger.info(f"   {period_key}: {total_campaigns} campaigns, {total_adsets} adsets, {total_ads} ads | R$ {total_spend:.2f}")

        # 4. CALCULAR DECIS UMA VEZ NOS LEADS 21D (BASE DE REFERÊNCIA)
        logger.info("📊 Calculando decis nos leads 21D (base de referência estatística)...")

        # Filtrar leads para 21D (3 semanas - janela de referência)
        cutoff_21d_start = cutoff_end - pd.Timedelta(days=21)
        if 'Data' in result_df.columns:
            dates = pd.to_datetime(result_df['Data'])
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)

            result_df_21d = result_df[(dates >= cutoff_21d_start) & (dates < cutoff_end)].copy()
            logger.info(f"   Leads 21D: {len(result_df_21d)} de {len(result_df)} totais")
        else:
            result_df_21d = result_df.copy()
            logger.warning(f"   Coluna 'Data' não encontrada, usando todos os {len(result_df)} leads")

        # Garantir que lead_score é numérico
        result_df_21d['lead_score'] = pd.to_numeric(result_df_21d['lead_score'], errors='coerce')

        # Remover linhas com lead_score inválido
        result_df_21d = result_df_21d[result_df_21d['lead_score'].notna()].copy()

        if len(result_df_21d) >= 10:
            try:
                result_df_21d['decil'] = pd.qcut(
                    result_df_21d['lead_score'],
                    q=10,
                    labels=[f'D{i}' for i in range(1, 11)],
                    duplicates='drop'
                )
                logger.info(f"✅ Decis calculados para {len(result_df_21d)} leads 21D (base de referência)")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao calcular decis: {e}, usando faixas fixas")
                result_df_21d['decil'] = result_df_21d['lead_score'].apply(
                    lambda x: f"D{min(10, max(1, int(float(x) * 10) + 1))}" if x > 0 else "D1"
                )
        else:
            logger.warning(f"⚠️ Menos de 10 leads em 21D, usando faixas fixas")
            result_df_21d['decil'] = result_df_21d['lead_score'].apply(
                lambda x: f"D{min(10, max(1, int(float(x) * 10) + 1))}" if x > 0 else "D1"
            )

        # 5. GERAR ANÁLISE UTM POR PERÍODO E DIMENSÃO
        logger.info("📈 Gerando análise UTM...")

        periods_analysis = {}

        for period_key, hierarchy in hierarchy_by_period.items():
            logger.info(f"   Processando período: {period_key}")

            # Extrair número de dias do período ('1D' → 1, '3D' → 3, etc.)
            if period_key == 'Total':
                # Para Total, calcular diferença real entre datas
                period_days = 30  # Default conservador
            else:
                try:
                    period_days = int(period_key.replace('D', ''))
                except:
                    period_days = 1  # Fallback

            # Usar janela pré-calculada
            cutoff_start = period_windows[period_key]

            # Filtrar do dataset 21D (que já tem decis calculados)
            if 'Data' in result_df_21d.columns:
                dates_21d = pd.to_datetime(result_df_21d['Data'])
                if dates_21d.dt.tz is not None:
                    dates_21d = dates_21d.dt.tz_localize(None)

                # Filtrar: Data >= cutoff_start AND Data < cutoff_end
                # IMPORTANTE: Decis já foram calculados no dataset 21D e são mantidos
                period_df = result_df_21d[(dates_21d >= cutoff_start) & (dates_21d < cutoff_end)].copy()
                logger.info(f"   Leads no período {period_key}: {len(period_df)} (decis de base 21D)")
            else:
                period_df = result_df_21d.copy()
                logger.warning(f"   Coluna 'Data' não encontrada, usando todos os leads 21D")

            # Capturar metadados: timestamps reais dos leads dentro da janela
            if 'Data' in period_df.columns and len(period_df) > 0:
                period_dates = pd.to_datetime(period_df['Data'])
                if period_dates.dt.tz is not None:
                    period_dates = period_dates.dt.tz_localize(None)

                # Timestamps reais dos leads (não a janela)
                period_start = period_dates.min().isoformat()
                period_end = period_dates.max().isoformat()
            else:
                period_start = cutoff_start.isoformat()
                period_end = cutoff_end.isoformat()

            # Contar leads por fonte
            total_leads = len(period_df)
            if 'Source' in period_df.columns:
                meta_leads = (period_df['Source'] == 'facebook-ads').sum()
                google_leads = (period_df['Source'] == 'google-ads').sum()
            else:
                meta_leads = total_leads
                google_leads = 0

            logger.info(f"   📊 Metadados: {period_start} até {period_end} | Total: {total_leads} | Meta: {meta_leads} | Google: {google_leads}")

            # Validar que temos leads e decis (já calculados em 21D)
            if len(period_df) == 0:
                logger.warning(f"   ⚠️ Nenhum lead no período {period_key}")
                continue

            if 'decil' not in period_df.columns:
                logger.error(f"   ❌ ERRO: Coluna 'decil' não encontrada após filtro (bug!)")
                # Fallback emergencial
                period_df['decil'] = 'D5'

            period_analysis = {}

            # Dimensões a analisar (incluindo google_ads como dimensão separada)
            # Removido 'term' - não tem custo no Meta API, análise inútil
            dimensions = ['campaign', 'medium', 'ad', 'google_ads']

            for dimension in dimensions:
                # Tratamento especial para Google Ads
                if dimension == 'google_ads':
                    # Filtrar apenas leads do Google Ads
                    if 'Source' in period_df.columns:
                        google_df = period_df[period_df['Source'] == 'google-ads']

                        if len(google_df) == 0:
                            logger.info(f"   ℹ️ Nenhum lead Google Ads no período {period_key}")
                            period_analysis[dimension] = []
                            continue

                        logger.info(f"   📊 {len(google_df)} leads Google Ads no período {period_key}")

                        # Agrupar por Term (formato: "keyword--campaign_id--ad_id")
                        # Extrair apenas keyword (primeira parte)
                        def extract_keyword_from_term(term_value):
                            """Extrai keyword da primeira parte do Term"""
                            if pd.isna(term_value) or str(term_value).strip() == '':
                                return None
                            # Formato: "keyword--campaign_id--ad_id"
                            parts = str(term_value).split('--')
                            if len(parts) >= 1 and parts[0].strip() != '':
                                return parts[0].strip()  # Keyword
                            return None

                        google_df['keyword'] = google_df['Term'].apply(extract_keyword_from_term)

                        # Filtrar valores genéricos e vazios
                        google_df_filtered = google_df[
                            google_df['keyword'].notna() &
                            ~google_df['keyword'].isin(['fb', 'ig', 'instagram', 'facebook'])
                        ].copy()

                        if len(google_df_filtered) == 0:
                            logger.info(f"   ⚠️ Nenhum keyword válido encontrado para Google Ads")
                            period_analysis[dimension] = []
                            continue

                        grouped = google_df_filtered.groupby('keyword').agg({
                            'lead_score': 'count',
                            'decil': lambda x: (x == 'D10').sum() / len(x) * 100 if len(x) > 0 else 0,
                        }).rename(columns={'lead_score': 'leads', 'decil': 'pct_d10'})

                        # Calcular distribuição de decis
                        for value in grouped.index:
                            value_df = google_df_filtered[google_df_filtered['keyword'] == value]
                            for i in range(1, 11):
                                decile_key = f'D{i}'
                                pct = (value_df['decil'] == decile_key).sum() / len(value_df) * 100 if len(value_df) > 0 else 0
                                grouped.at[value, f'%{decile_key}'] = pct

                        grouped = grouped.reset_index().rename(columns={'keyword': 'value'})

                        # Google Ads não tem custos no Meta API
                        grouped['spend'] = 0.0

                        # Enriquecer com métricas econômicas (todas zeradas/não aplicáveis)
                        enriched = enrich_utm_with_economic_metrics(
                            utm_df=grouped,
                            product_value=product_value,
                            min_roas=min_roas,
                            conversion_rates=conversion_rates,
                            dimension=dimension,
                            period_days=period_days
                        )

                        enriched = enriched.fillna({
                            'leads': 0,
                            'spend': 0.0,
                            'cpl': 0.0,
                            'taxa_proj': 0.0,
                            'receita_proj': 0.0,
                            'margem_contrib': 0.0,
                            'roas_proj': 0.0,
                            'acao': 'N/A - Google Ads',
                            'budget_current': 0.0,
                            'budget_target': 0.0
                        })

                        period_analysis[dimension] = [
                            UTMDimensionMetrics(
                                campaign=None,
                                value=str(row['value']) if pd.notna(row['value']) else '(vazio)',
                                leads=int(row['leads']),
                                spend=float(row['spend']),
                                cpl=float(row['cpl']),
                                taxa_proj=float(row['taxa_proj']),
                                receita_proj=float(row['receita_proj']),
                                margem_contrib=float(row['margem_contrib']),
                                roas_proj=float(row['roas_proj']),
                                acao=str(row['acao']),
                                budget_current=float(row.get('budget_current', 0.0)),
                                budget_target=float(row.get('budget_target', 0.0))
                            )
                            for _, row in enriched.iterrows()
                        ]

                        logger.info(f"✅ Google Ads analisado: {len(period_analysis[dimension])} grupos")
                        continue
                    else:
                        period_analysis[dimension] = []
                        continue

                # Processar dimensões Meta normalmente
                # Mapear para coluna do DataFrame
                utm_col_map = {
                    'campaign': 'Campaign',
                    'medium': 'Medium',
                    'term': 'Term',
                    'ad': 'Content'  # Ad = Criativo = Coluna Content
                }

                utm_col = utm_col_map.get(dimension, 'Campaign')

                # Agrupar por dimensão
                if utm_col not in period_df.columns:
                    logger.warning(f"⚠️  Coluna '{utm_col}' não encontrada, pulando dimensão '{dimension}'")
                    period_analysis[dimension] = []
                    continue

                # Filtrar UTMs vazios e inválidos
                utm_mask = (
                    (period_df[utm_col].notna()) &
                    (period_df[utm_col] != '') &
                    (~period_df[utm_col].astype(str).str.contains('{{', regex=False))  # Placeholders
                )

                # Filtros específicos por dimensão
                if dimension == 'campaign':
                    # Remover bare names (case insensitive)
                    for bare in BARE_CAMPAIGN_NAMES:
                        utm_mask &= ~(period_df[utm_col].astype(str).str.upper() == bare.upper())

                elif dimension == 'medium':
                    # Remover bare names
                    for bare in BARE_MEDIUM_NAMES:
                        utm_mask &= ~(period_df[utm_col].astype(str).str.upper() == bare.upper())

                elif dimension == 'term':
                    # Remover termos genéricos (fb, ig, etc)
                    for generic in GENERIC_TERMS:
                        utm_mask &= ~(period_df[utm_col].astype(str).str.upper() == generic.upper())

                # Aplicar filtro
                period_df_filtered = period_df[utm_mask]

                # Log detalhado de valores filtrados
                filtered_count = len(period_df) - len(period_df_filtered)
                if filtered_count > 0:
                    filtered_df = period_df[~utm_mask]
                    filtered_values = filtered_df[utm_col].value_counts()

                    logger.info(f"   ⚠️ {filtered_count} leads filtrados de '{dimension}' ({period_key}):")
                    for val, count in filtered_values.head(5).items():
                        logger.info(f"      - '{val}': {count} leads (bare name/genérico)")

                if len(period_df_filtered) == 0:
                    logger.warning(f"⚠️ Nenhum lead com '{utm_col}' válido no período, pulando dimensão '{dimension}'")
                    period_analysis[dimension] = []
                    continue

                # CORREÇÃO: Para Campaign, agrupar por Campaign ID (não por UTM completo)
                if dimension == 'campaign':
                    from api.meta_integration import extract_id_from_utm

                    # Extrair Campaign ID de cada UTM
                    period_df_filtered = period_df_filtered.copy()
                    period_df_filtered['campaign_id'] = period_df_filtered[utm_col].apply(extract_id_from_utm)

                    # Remover linhas sem Campaign ID
                    period_df_filtered = period_df_filtered[period_df_filtered['campaign_id'].notna()]

                    # Agrupar por Campaign ID
                    grouped = period_df_filtered.groupby('campaign_id').agg({
                        'lead_score': 'count',
                        'decil': lambda x: (x == 'D10').sum() / len(x) * 100 if len(x) > 0 else 0,
                    }).rename(columns={'lead_score': 'leads', 'decil': 'pct_d10'})

                    # Calcular distribuição de decis
                    for campaign_id in grouped.index:
                        value_df = period_df_filtered[period_df_filtered['campaign_id'] == campaign_id]
                        for i in range(1, 11):
                            decile_key = f'D{i}'
                            pct = (value_df['decil'] == decile_key).sum() / len(value_df) * 100 if len(value_df) > 0 else 0
                            grouped.at[campaign_id, f'%{decile_key}'] = pct

                    # Resetar index e renomear para 'value' (será o Campaign ID)
                    grouped = grouped.reset_index().rename(columns={'campaign_id': 'value'})

                elif dimension == 'medium':
                    # Para adsets, agrupar por (campaign, adset) para separar mesmo nome em campanhas diferentes
                    # Extrair campaign_id da coluna Campaign
                    period_df_filtered = period_df_filtered.copy()

                    # Verificar se temos coluna Campaign (utm_campaign)
                    if 'Campaign' not in period_df_filtered.columns:
                        logger.warning(f"⚠️  Coluna 'Campaign' não encontrada para matchear adsets. Usando apenas nome do adset.")
                        # Fallback: agrupar só por nome do adset (comportamento antigo)
                        grouped = period_df_filtered.groupby(utm_col).agg({
                            'lead_score': 'count',
                            'decil': lambda x: (x == 'D10').sum() / len(x) * 100 if len(x) > 0 else 0,
                        }).rename(columns={'lead_score': 'leads', 'decil': 'pct_d10'})

                        for value in grouped.index:
                            value_df = period_df_filtered[period_df_filtered[utm_col] == value]
                            for i in range(1, 11):
                                decile_key = f'D{i}'
                                pct = (value_df['decil'] == decile_key).sum() / len(value_df) * 100 if len(value_df) > 0 else 0
                                grouped.at[value, f'%{decile_key}'] = pct

                        grouped = grouped.reset_index().rename(columns={utm_col: 'value'})
                        grouped['campaign_name'] = None  # Sem campaign info
                    else:
                        period_df_filtered['campaign_id'] = period_df_filtered['Campaign'].apply(extract_id_from_utm)

                        # Criar mapeamento (campaign_id, adset_name) → adset_id da hierarquia
                        campaign_adset_to_info = {}
                        for campaign_id, campaign_data in hierarchy['campaigns'].items():
                            for adset_id, adset_data in campaign_data['adsets'].items():
                                key = (campaign_id, adset_data['name'])
                                campaign_adset_to_info[key] = {
                                    'adset_id': adset_id,
                                    'campaign_name': campaign_data['name']
                                }

                        # Matchear cada lead para adset_id específico
                        def match_adset(row):
                            campaign_id = row['campaign_id']
                            adset_name = row[utm_col]
                            if pd.isna(campaign_id) or pd.isna(adset_name):
                                return None, None
                            key = (str(campaign_id), str(adset_name))
                            info = campaign_adset_to_info.get(key)
                            if info:
                                return info['adset_id'], info['campaign_name']
                            return None, None

                        period_df_filtered[['adset_id', 'campaign_name']] = period_df_filtered.apply(
                            match_adset, axis=1, result_type='expand'
                        )

                        # Remover linhas sem match
                        before_match = len(period_df_filtered)
                        period_df_filtered = period_df_filtered[period_df_filtered['adset_id'].notna()]
                        after_match = len(period_df_filtered)
                        if before_match > after_match:
                            logger.info(f"   ⚠️  {before_match - after_match} leads sem match campaign+adset (removidos)")

                        # Agrupar por adset_id (já é único por campanha)
                        grouped = period_df_filtered.groupby('adset_id').agg({
                            'lead_score': 'count',
                            'campaign_name': 'first',  # Pegar o nome da campanha
                            'decil': lambda x: (x == 'D10').sum() / len(x) * 100 if len(x) > 0 else 0,
                        }).rename(columns={'lead_score': 'leads', 'decil': 'pct_d10'})

                        # Calcular distribuição de decis
                        for adset_id in grouped.index:
                            value_df = period_df_filtered[period_df_filtered['adset_id'] == adset_id]
                            for i in range(1, 11):
                                decile_key = f'D{i}'
                                pct = (value_df['decil'] == decile_key).sum() / len(value_df) * 100 if len(value_df) > 0 else 0
                                grouped.at[adset_id, f'%{decile_key}'] = pct

                        # Resetar index e renomear
                        grouped = grouped.reset_index().rename(columns={'adset_id': 'value'})

                elif dimension == 'ad':
                    # Para ads, agrupar por nome e extrair campaign/adset das colunas Campaign/Medium
                    period_df_filtered = period_df_filtered.copy()

                    # Extrair campaign e adset names das colunas UTM
                    if 'Campaign' in period_df_filtered.columns and 'Medium' in period_df_filtered.columns:
                        # Campaign: extrair ID (formato "nome|ID")
                        period_df_filtered['campaign_id'] = period_df_filtered['Campaign'].apply(extract_id_from_utm)

                        # Medium: já contém NOME do adset (não ID), usar diretamente
                        period_df_filtered['adset_name_from_utm'] = period_df_filtered['Medium']

                        # Mapear campaign_id para nome
                        campaign_id_to_name = {
                            campaign_id: campaign_data['name']
                            for campaign_id, campaign_data in hierarchy['campaigns'].items()
                        }

                        def get_campaign_name(row):
                            """Obtém nome da campanha por ID, com fallback para nome do UTM"""
                            campaign_id = row['campaign_id']
                            if pd.notna(campaign_id):
                                # Tentar buscar na hierarquia primeiro
                                name = campaign_id_to_name.get(str(campaign_id))
                                if name:
                                    return name
                            # Fallback: extrair nome do UTM Campaign (remover data e ID)
                            campaign_utm = row.get('Campaign')
                            if pd.notna(campaign_utm):
                                import re
                                # Remover campaign ID do final (|números)
                                clean = re.sub(r'\|\d{18}$', '', str(campaign_utm))
                                # Remover data do final (| YYYY-MM-DD)
                                clean = re.sub(r'\|\s*\d{4}-\d{2}-\d{2}$', '', clean)
                                return clean.strip()
                            return None

                        period_df_filtered['campaign_name'] = period_df_filtered.apply(get_campaign_name, axis=1)

                        # adset_name inicial vem da coluna Medium
                        period_df_filtered['adset_name'] = period_df_filtered['adset_name_from_utm']

                        # CORREÇÃO: Detectar e corrigir UTMs genéricas usando hierarquia Meta
                        GENERIC_UTMS = {'paid', 'dgen', 'facebook', 'instagram', 'meta', 'fb', 'ig', 'cpc'}

                        def correct_generic_adset(row):
                            """Corrige adset_name genérico buscando na hierarquia Meta por nome do ad"""
                            try:
                                adset_name = row['adset_name'] if 'adset_name' in row else None
                                ad_name = row[utm_col] if utm_col in row else None

                                # Se adset_name não é genérico ou está vazio, retornar como está
                                if pd.isna(adset_name) or str(adset_name).lower() not in GENERIC_UTMS:
                                    return adset_name

                                # Se ad_name está vazio, não podemos corrigir
                                if pd.isna(ad_name) or str(ad_name).strip() == '':
                                    logger.debug(f"      ⚠️ Ad sem nome: adset_name='{adset_name}' (será filtrado)")
                                    return None  # Será filtrado

                                # Buscar ad_name na hierarquia para encontrar adset correto
                                ad_name_lower = str(ad_name).lower()
                                for campaign_id, campaign_data in hierarchy['campaigns'].items():
                                    if not isinstance(campaign_data, dict) or 'adsets' not in campaign_data:
                                        continue
                                    for adset_id, adset_data in campaign_data['adsets'].items():
                                        if not isinstance(adset_data, dict) or 'ads' not in adset_data:
                                            continue
                                        for ad_id, ad_data in adset_data['ads'].items():
                                            if not isinstance(ad_data, dict) or 'name' not in ad_data:
                                                continue
                                            # Match por nome (case insensitive)
                                            if ad_data['name'].lower() == ad_name_lower:
                                                logger.info(f"      ✓ Corrigido '{adset_name}' → '{adset_data.get('name', 'Unknown')}' (ad: {str(ad_name)[:40]}...)")
                                                return adset_data.get('name')

                                # Não encontrou na hierarquia, filtrar
                                logger.info(f"      🗑️ UTM genérica sem match: adset='{adset_name}', ad='{str(ad_name)[:40]}...'")
                                return None
                            except Exception as e:
                                logger.error(f"      ❌ Erro em correct_generic_adset: {str(e)}")
                                return row.get('adset_name') if hasattr(row, 'get') else None

                        # Aplicar correção
                        logger.info(f"   🔍 Verificando UTMs genéricas em {len(period_df_filtered)} ads...")
                        period_df_filtered['adset_name'] = period_df_filtered.apply(correct_generic_adset, axis=1)

                        # Filtrar ads com adset_name None (genéricos sem match)
                        before_filter = len(period_df_filtered)
                        period_df_filtered = period_df_filtered[period_df_filtered['adset_name'].notna()].copy()
                        after_filter = len(period_df_filtered)
                        if before_filter > after_filter:
                            removed = before_filter - after_filter
                            logger.info(f"   🗑️ Removidos {removed} ads com UTMs genéricas (sem match na hierarquia)")
                    else:
                        logger.warning(f"   ⚠️ Colunas Campaign ou Medium não encontradas para ads")
                        period_df_filtered['campaign_name'] = None
                        period_df_filtered['adset_name'] = None

                    # Agrupar por nome do ad (coluna Content)
                    grouped = period_df_filtered.groupby(utm_col).agg({
                        'lead_score': 'count',
                        'campaign_name': 'first',  # Pegar primeira ocorrência
                        'adset_name': 'first',
                        'decil': lambda x: (x == 'D10').sum() / len(x) * 100 if len(x) > 0 else 0,
                    }).rename(columns={'lead_score': 'leads', 'decil': 'pct_d10'})

                    # Calcular distribuição de decis
                    for value in grouped.index:
                        value_df = period_df_filtered[period_df_filtered[utm_col] == value]
                        for i in range(1, 11):
                            decile_key = f'D{i}'
                            pct = (value_df['decil'] == decile_key).sum() / len(value_df) * 100 if len(value_df) > 0 else 0
                            grouped.at[value, f'%{decile_key}'] = pct

                    # Resetar index e renomear
                    grouped = grouped.reset_index().rename(columns={utm_col: 'value'})

                else:
                    # Para outras dimensões, agrupar normalmente pelo UTM
                    grouped = period_df_filtered.groupby(utm_col).agg({
                        'lead_score': 'count',  # Número de leads
                        'decil': lambda x: (x == 'D10').sum() / len(x) * 100 if len(x) > 0 else 0,  # %D10
                    }).rename(columns={'lead_score': 'leads', 'decil': 'pct_d10'})

                    # Calcular distribuição de decis para cada valor
                    for value in grouped.index:
                        value_df = period_df_filtered[period_df_filtered[utm_col] == value]
                        for i in range(1, 11):
                            decile_key = f'D{i}'
                            pct = (value_df['decil'] == decile_key).sum() / len(value_df) * 100 if len(value_df) > 0 else 0
                            grouped.at[value, f'%{decile_key}'] = pct

                    grouped = grouped.reset_index().rename(columns={utm_col: 'value'})

                # Adicionar custos usando hierarquia (evita duplicação)
                grouped = enrich_utm_with_hierarchy(
                    utm_analysis_df=grouped,
                    hierarchy=hierarchy,
                    dimension=dimension
                )

                # FILTRO: Para Campaign, remover linhas com spend=0 (indicam Campaign IDs inválidos)
                if dimension == 'campaign':
                    before_filter = len(grouped)
                    grouped = grouped[grouped['spend'] > 0].copy()
                    after_filter = len(grouped)
                    if before_filter > after_filter:
                        removed = before_filter - after_filter
                        logger.info(f"   🗑️ Removidas {removed} campanhas com spend=0 (IDs inválidos)")

                    # Substituir Campaign ID pelo nome legível
                    id_to_name = {
                        campaign_id: campaign_data['name']
                        for campaign_id, campaign_data in hierarchy['campaigns'].items()
                    }

                    grouped['value'] = grouped['value'].apply(
                        lambda x: id_to_name.get(str(x), str(x))
                    )
                    logger.info(f"   ✏️ IDs substituídos por nomes de campanha para exibição")

                elif dimension == 'medium':
                    # Filtrar adsets com spend=0
                    before_filter = len(grouped)
                    grouped = grouped[grouped['spend'] > 0].copy()
                    after_filter = len(grouped)
                    if before_filter > after_filter:
                        removed = before_filter - after_filter
                        logger.info(f"   🗑️ Removidos {removed} adsets com spend=0 (IDs inválidos)")

                    # Substituir Adset ID pelo nome legível
                    id_to_info = {}
                    for campaign_id, campaign_data in hierarchy['campaigns'].items():
                        for adset_id, adset_data in campaign_data['adsets'].items():
                            id_to_info[adset_id] = adset_data['name']

                    grouped['value'] = grouped['value'].apply(
                        lambda x: id_to_info.get(str(x), str(x))
                    )
                    logger.info(f"   ✏️ IDs substituídos por nomes de adset para exibição")

                elif dimension == 'ad':
                    # Filtrar ads com spend=0
                    before_filter = len(grouped)
                    grouped = grouped[grouped['spend'] > 0].copy()
                    after_filter = len(grouped)
                    if before_filter > after_filter:
                        removed = before_filter - after_filter
                        logger.info(f"   🗑️ Removidos {removed} ads com spend=0")

                    # Para ads, 'value' já é o nome (não ID), então não precisa substituir
                    logger.info(f"   ✅ {len(grouped)} ads com custo > 0")

                # Enriquecer com métricas econômicas
                # Para campaigns e adsets (medium), usar budget info para condicionar ação
                if dimension == 'campaign':
                    budget_control_col = 'has_campaign_budget'
                elif dimension == 'medium':
                    budget_control_col = 'has_adset_budget'
                else:
                    budget_control_col = None

                enriched = enrich_utm_with_economic_metrics(
                    utm_df=grouped,
                    product_value=product_value,
                    min_roas=min_roas,
                    conversion_rates=conversion_rates,
                    dimension=dimension,
                    budget_control_col=budget_control_col,
                    period_days=period_days
                )

                # Tratar NaN em métricas calculadas (podem surgir de divisões por zero)
                enriched = enriched.fillna({
                    'leads': 0,
                    'spend': 0.0,
                    'cpl': 0.0,
                    'taxa_proj': 0.0,
                    'receita_proj': 0.0,
                    'margem_contrib': 0.0,
                    'roas_proj': 0.0,
                    'acao': ''
                })

                # Converter para lista de dicts
                metrics_list = []
                for _, row in enriched.iterrows():
                    # Para adsets, incluir campaign_name
                    # Para ads, incluir campaign_name e adset_name
                    campaign_value = None
                    adset_value = None

                    if dimension == 'medium':
                        campaign_value = row.get('campaign_name')
                    elif dimension == 'ad':
                        campaign_value = row.get('campaign_name')
                        adset_value = row.get('adset_name')

                    metrics_list.append(UTMDimensionMetrics(
                        campaign=campaign_value,
                        adset=adset_value,
                        value=str(row['value']) if pd.notna(row['value']) else '(vazio)',
                        leads=int(row['leads']),
                        spend=float(row['spend']),
                        cpl=float(row['cpl']),
                        taxa_proj=float(row['taxa_proj']),
                        receita_proj=float(row['receita_proj']),
                        margem_contrib=float(row['margem_contrib']),
                        roas_proj=float(row['roas_proj']),
                        acao=row['acao'],
                        budget_current=float(row.get('budget_current', 0.0)),
                        budget_target=float(row.get('budget_target', 0.0))
                    ))

                period_analysis[dimension] = metrics_list

            # Adicionar metadados do período
            period_analysis['period_start'] = period_start
            period_analysis['period_end'] = period_end
            period_analysis['total_leads'] = total_leads
            period_analysis['meta_leads'] = meta_leads
            period_analysis['google_leads'] = google_leads

            periods_analysis[period_key] = UTMPeriodAnalysis(**period_analysis)

        processing_time = time.time() - start_time

        logger.info(f"✅ Análise concluída em {processing_time:.2f}s")

        return UTMAnalysisResponse(
            request_id=request_id,
            periods=periods_analysis,
            config={
                'product_value': product_value,
                'min_roas': min_roas
            },
            processing_time_seconds=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        import traceback
        logger.error(f"❌ Erro na análise UTM: {str(e)}")
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn

    # Inicializar pipeline antes de iniciar o servidor
    print("Inicializando pipeline...")
    if initialize_pipeline():
        print("Pipeline inicializado com sucesso!")
    else:
        print("AVISO: Pipeline não foi inicializado. Será inicializado na primeira requisição.")

    # Iniciar o servidor
    print("Iniciando servidor na porta 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)