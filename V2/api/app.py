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
from src.pipeline import LeadScoringPipeline

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
    leads: List[LeadData] = Field(..., min_items=1, max_items=500)
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

class PredictionResult(BaseModel):
    """Resultado de uma predição"""
    lead_score: float
    decile: str  # D1-D10 formato original
    decile_numeric: int  # 1-10 para facilitar ordenação
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

def convert_decile_to_numeric(decile_str: str) -> int:
    """Converte D1-D10 para 1-10"""
    try:
        return int(decile_str.replace('D', ''))
    except:
        return 5

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
            decile_str = row['decil']
            decile_num = convert_decile_to_numeric(decile_str)

            # Recuperar metadados do lead original
            original_lead = request.leads[i] if i < len(request.leads) else None
            email = original_lead.email if original_lead else None
            row_id = original_lead.row_id if original_lead else str(i)

            predictions.append(PredictionResult(
                lead_score=lead_score,
                decile=decile_str,
                decile_numeric=decile_num,
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
                "decile": row['decil'],  # D1-D10
                "decile_numeric": convert_decile_to_numeric(row['decil']),  # 1-10
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