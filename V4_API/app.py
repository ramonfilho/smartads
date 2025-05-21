# app.py (ajuste do caminho do projeto)
import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import io
import time
from datetime import datetime

# Ajustar para o caminho correto considerando que estamos na pasta V4_API
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Importar a classe GMM_Wrapper para garantir que esteja disponível durante o carregamento
from src.modeling.gmm_wrapper import GMM_Wrapper
import builtins
builtins.GMM_Wrapper = GMM_Wrapper  # Adicionar ao namespace global para desserialização

# Importar a pipeline de inferência
from inference_v4.inference_pipeline import process_inference, load_parameters

# Definição dos limites de decis
DECILE_LIMITS = {
    1: 0.002510760401721664,   # Limite para decil 1
    4: 0.004405286343612335,   # Limite para decil 4
    6: 0.0067861020629750276,  # Limite para decil 6
    7: 0.008585293019783502,   # Limite para decil 7
    8: 0.0123334977799704,     # Limite para decil 8
    9: 0.014276002719238613,   # Limite para decil 9
    10: 0.022128556375131718,  # Limite para decil 10
}

# Carregar parâmetros no início para evitar carregamento repetido
params = None

# Classes para validação de dados de entrada/saída
class LeadPredictionRequest(BaseModel):
    data: Dict[str, Any]

class LeadPredictionResponse(BaseModel):
    prediction_probability: float
    decile: int

# Inicializar a aplicação FastAPI
app = FastAPI(
    title="Smart Ads API",
    description="API para predição de qualidade de leads baseado em respostas de pesquisa",
    version="1.0.0"
)

# Função para determinar o decil com base na probabilidade
def determine_decile(probability):
    """
    Determina o decil com base na probabilidade predita.
    
    Args:
        probability (float): Probabilidade predita pelo modelo
        
    Returns:
        int: Número do decil (1-10)
    """
    # Ordenar os decis por probabilidade para facilitar a busca
    sorted_limits = sorted([(decile, limit) for decile, limit in DECILE_LIMITS.items()], 
                          key=lambda x: x[1])
    
    # Para facilitar a lógica, adicionamos a probabilidade 0 como limite inferior
    if probability < sorted_limits[0][1]:
        return 1
    
    # Verificar em qual faixa a probabilidade se encaixa
    for i in range(len(sorted_limits)-1):
        current_decile, current_limit = sorted_limits[i]
        next_decile, next_limit = sorted_limits[i+1]
        
        if probability >= current_limit and probability < next_limit:
            # Para os decis intermediários, usamos interpolação linear
            if next_decile - current_decile > 1:
                # Posição relativa entre os limites conhecidos
                relative_pos = (probability - current_limit) / (next_limit - current_limit)
                # Interpolação linear
                fractional_decile = current_decile + relative_pos * (next_decile - current_decile)
                return int(round(fractional_decile))
            else:
                return next_decile
    
    # Se a probabilidade for maior que todos os limites, é o decil 10
    return 10

@app.get("/")
def read_root():
    return {"message": "Smart Ads API está online", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=LeadPredictionResponse)
async def predict_single(request: LeadPredictionRequest):
    global params
    
    # Carregar parâmetros se ainda não foram carregados
    if params is None:
        params = load_parameters()
        if params is None:
            raise HTTPException(status_code=500, detail="Falha ao carregar parâmetros do modelo")
    
    # Criar DataFrame a partir dos dados de entrada
    input_data = {k: [v] for k, v in request.data.items()}
    df = pd.DataFrame(input_data)
    
    # Executar inferência
    result_df = process_inference(df, params=params, return_proba=True)
    
    if result_df is None or len(result_df) == 0:
        raise HTTPException(status_code=500, detail="Falha no processamento da inferência")
    
    # Extrair resultados
    probability = float(result_df['prediction_probability'].iloc[0])
    
    # Determinar o decil
    decile = determine_decile(probability)
    
    return {
        "prediction_probability": probability,
        "decile": decile
    }

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    global params
    
    # Carregar parâmetros se ainda não foram carregados
    if params is None:
        params = load_parameters()
        if params is None:
            raise HTTPException(status_code=500, detail="Falha ao carregar parâmetros do modelo")
    
    # Ler CSV do upload
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler arquivo CSV: {str(e)}")
    
    # Executar inferência
    start_time = time.time()
    result_df = process_inference(df, params=params, return_proba=True)
    end_time = time.time()
    
    if result_df is None:
        raise HTTPException(status_code=500, detail="Falha no processamento da inferência")
    
    # Adicionar coluna de decil
    result_df['decile'] = result_df['prediction_probability'].apply(determine_decile)
    
    # Preparar resposta (incluindo apenas as colunas relevantes)
    response_df = result_df[['prediction_probability', 'prediction_class', 'decile']]
    
    # Converter para lista de dicionários para a resposta JSON
    predictions = response_df.to_dict(orient='records')
    
    return {
        "predictions": predictions,
        "processing_time": end_time - start_time,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    # Carregar parâmetros durante a inicialização
    print("Carregando parâmetros do modelo...")
    params = load_parameters()
    
    # Iniciar o servidor
    uvicorn.run(app, host="0.0.0.0", port=8080)