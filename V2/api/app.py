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
from api.economic_metrics import enrich_utm_with_economic_metrics, calculate_tier

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

# === ANÁLISE UTM COM CUSTOS ===

class UTMAnalysisRequest(BaseModel):
    """Request para análise UTM com custos"""
    leads: List[LeadData] = Field(..., min_items=1)  # Sem limite máximo - batching interno
    account_id: str = Field(..., description="ID da conta Meta Ads (ex: act_123456)")
    product_value: Optional[float] = Field(default=None, description="Valor do produto (padrão: config)")
    min_roas: Optional[float] = Field(default=None, description="ROAS mínimo (padrão: 2.0)")

class UTMDimensionMetrics(BaseModel):
    """Métricas de uma dimensão UTM"""
    value: str
    leads: int
    spend: float
    cpl: float
    pct_d10: float
    taxa_proj: float
    roas_proj: float
    cpl_max: float
    margem: float
    tier: str
    acao: str

class UTMPeriodAnalysis(BaseModel):
    """Análise UTM para um período"""
    campaign: List[UTMDimensionMetrics]
    medium: List[UTMDimensionMetrics]
    term: List[UTMDimensionMetrics]
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

        # 3. BUSCAR CUSTOS DA API META (HIERARQUIA COMPLETA)
        logger.info("💰 Buscando hierarquia de custos da API Meta...")
        meta_client = MetaAdsIntegration(
            access_token=META_CONFIG['access_token'],
            api_version=META_CONFIG['api_version']
        )

        # Buscar hierarquia completa para cada período
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

        # 4. GERAR ANÁLISE UTM POR PERÍODO E DIMENSÃO
        logger.info("📈 Gerando análise UTM...")

        periods_analysis = {}

        for period_key, hierarchy in hierarchy_by_period.items():
            logger.info(f"   Processando período: {period_key}")

            # Usar janela pré-calculada
            cutoff_start = period_windows[period_key]

            # Filtrar leads dentro da janela: [start, end)
            if 'Data' in result_df.columns:
                dates = pd.to_datetime(result_df['Data'])
                if dates.dt.tz is not None:
                    dates = dates.dt.tz_localize(None)

                # Filtrar: Data >= cutoff_start AND Data < cutoff_end
                period_df = result_df[(dates >= cutoff_start) & (dates < cutoff_end)].copy()
                logger.info(f"   Leads no período {period_key}: {len(period_df)} de {len(result_df)} totais")
            else:
                period_df = result_df.copy()
                logger.warning(f"   Coluna 'Data' não encontrada, usando todos os {len(result_df)} leads")

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

            # Calcular decis para os leads deste período
            # Garantir que lead_score é numérico
            period_df['lead_score'] = pd.to_numeric(period_df['lead_score'], errors='coerce')

            # Remover linhas com lead_score inválido (NaN, None, string vazia)
            period_df = period_df[period_df['lead_score'].notna()].copy()

            if len(period_df) == 0:
                logger.warning(f"   ⚠️ Nenhum lead com score válido no período {period_key}")
                continue

            if len(period_df) >= 10 and 'lead_score' in period_df.columns:
                try:
                    period_df['decil'] = pd.qcut(
                        period_df['lead_score'],
                        q=10,
                        labels=[f'D{i}' for i in range(1, 11)],
                        duplicates='drop'
                    )
                    logger.info(f"   ✅ Decis calculados para período {period_key}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Erro ao calcular decis: {e}, usando faixas fixas")
                    # Fallback: usar faixas fixas (agora seguro, lead_score é numérico)
                    period_df['decil'] = period_df['lead_score'].apply(
                        lambda x: f"D{min(10, max(1, int(float(x) * 10) + 1))}" if x > 0 else "D1"
                    )
            else:
                # Poucos leads, usar faixas fixas
                period_df['decil'] = period_df['lead_score'].apply(
                    lambda x: f"D{min(10, max(1, int(float(x) * 10) + 1))}" if x > 0 else "D1"
                )
                logger.warning(f"   ⚠️ Menos de 10 leads no período, usando faixas fixas")

            period_analysis = {}

            # Dimensões a analisar (incluindo google_ads como dimensão separada)
            dimensions = ['campaign', 'medium', 'term', 'ad', 'google_ads']

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

                        # Agrupar por Campaign (ou Content se Campaign for genérico)
                        # Filtrar valores genéricos
                        google_df_filtered = google_df[
                            ~google_df['Campaign'].isin(['devlf', 'DEVLF']) &
                            google_df['Campaign'].notna() &
                            (google_df['Campaign'] != '')
                        ]

                        if len(google_df_filtered) == 0:
                            # Se todos são genéricos, agrupar por Content
                            utm_col = 'Content'
                            google_df_filtered = google_df[google_df[utm_col].notna()]
                        else:
                            utm_col = 'Campaign'

                        if len(google_df_filtered) == 0:
                            period_analysis[dimension] = []
                            continue

                        grouped = google_df_filtered.groupby(utm_col).agg({
                            'lead_score': 'count',
                            'decil': lambda x: (x == 'D10').sum() / len(x) * 100 if len(x) > 0 else 0,
                        }).rename(columns={'lead_score': 'leads', 'decil': 'pct_d10'})

                        # Calcular distribuição de decis
                        for value in grouped.index:
                            value_df = google_df_filtered[google_df_filtered[utm_col] == value]
                            for i in range(1, 11):
                                decile_key = f'D{i}'
                                pct = (value_df['decil'] == decile_key).sum() / len(value_df) * 100 if len(value_df) > 0 else 0
                                grouped.at[value, f'%{decile_key}'] = pct

                        grouped = grouped.reset_index().rename(columns={utm_col: 'value'})

                        # Google Ads não tem custos no Meta API
                        grouped['spend'] = 0.0

                        # Enriquecer com métricas econômicas (todas zeradas/não aplicáveis)
                        enriched = enrich_utm_with_economic_metrics(
                            utm_df=grouped,
                            product_value=product_value,
                            min_roas=min_roas,
                            conversion_rates=conversion_rates,
                            dimension=dimension
                        )

                        enriched['tier'] = enriched['roas_proj'].apply(lambda x: calculate_tier(x, min_roas))
                        enriched = enriched.fillna({
                            'leads': 0,
                            'spend': 0.0,
                            'cpl': 0.0,
                            'pct_d10': 0.0,
                            'taxa_proj': 0.0,
                            'roas_proj': 0.0,
                            'cpl_max': 0.0,
                            'margem': 0.0,
                            'tier': 'N/A',
                            'acao': 'N/A - Google Ads'
                        })

                        period_analysis[dimension] = [
                            UTMDimensionMetrics(
                                value=str(row['value']) if pd.notna(row['value']) else '(vazio)',
                                leads=int(row['leads']),
                                spend=float(row['spend']),
                                cpl=float(row['cpl']),
                                pct_d10=float(row['pct_d10']),
                                taxa_proj=float(row['taxa_proj']),
                                roas_proj=float(row['roas_proj']),
                                cpl_max=float(row['cpl_max']),
                                margem=float(row['margem']),
                                tier=str(row['tier']),
                                acao=str(row['acao'])
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

                # Enriquecer com métricas econômicas
                enriched = enrich_utm_with_economic_metrics(
                    utm_df=grouped,
                    product_value=product_value,
                    min_roas=min_roas,
                    conversion_rates=conversion_rates,
                    dimension=dimension
                )

                # Adicionar tier
                enriched['tier'] = enriched['roas_proj'].apply(lambda x: calculate_tier(x, min_roas))

                # Tratar NaN em métricas calculadas (podem surgir de divisões por zero)
                enriched = enriched.fillna({
                    'leads': 0,
                    'spend': 0.0,
                    'cpl': 0.0,
                    'pct_d10': 0.0,
                    'taxa_proj': 0.0,
                    'roas_proj': 0.0,
                    'cpl_max': 0.0,
                    'margem': 0.0,
                    'tier': 'D',
                    'acao': ''
                })

                # Converter para lista de dicts
                metrics_list = []
                for _, row in enriched.iterrows():
                    metrics_list.append(UTMDimensionMetrics(
                        value=str(row['value']) if pd.notna(row['value']) else '(vazio)',
                        leads=int(row['leads']),
                        spend=float(row['spend']),
                        cpl=float(row['cpl']),
                        pct_d10=float(row['pct_d10']),
                        taxa_proj=float(row['taxa_proj']),
                        roas_proj=float(row['roas_proj']),
                        cpl_max=float(row['cpl_max']),
                        margem=float(row['margem']),
                        tier=row['tier'],
                        acao=row['acao']
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
        logger.error(f"❌ Erro na análise UTM: {str(e)}")
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