"""
Serviço de API para o pipeline de inferência GMM.
Otimizado para execução em contêineres no Cloud Run.
"""

import os
import sys
import json
import pandas as pd
import logging
from flask import Flask, request, jsonify

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gmm_inference_service')

# Garantir que o caminho do projeto esteja no sys.path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

# Importar a classe do pipeline
from src.inference.gmm_inference_pipeline import GMMInferencePipeline

# Inicializar o pipeline globalmente para reutilização
inference_pipeline = None

# Carregar o pipeline na inicialização (melhor para Cloud Run)
def initialize_pipeline():
    """Inicializa o pipeline na inicialização da aplicação."""
    global inference_pipeline
    try:
        logger.info("Inicializando pipeline de inferência...")
        inference_pipeline = GMMInferencePipeline()
        logger.info("Pipeline inicializado com sucesso")
        return True
    except Exception as e:
        logger.error(f"ERRO ao inicializar pipeline: {e}")
        return False

app = Flask(__name__)

# Inicializar o pipeline na inicialização do serviço
# Esta linha substitui o decorador before_first_request
initialize_pipeline()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar saúde do serviço."""
    global inference_pipeline
    
    if inference_pipeline is None:
        status = "unhealthy: pipeline não inicializado"
        try:
            logger.info("Tentando reinicializar o pipeline...")
            if initialize_pipeline():
                status = "recovered"
            else:
                status = "failed to recover"
        except Exception as e:
            status = f"unhealthy: {str(e)}"
    else:
        status = "healthy"
    
    response = {
        "status": status,
        "model_loaded": inference_pipeline is not None
    }
    
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para fazer predições com o modelo GMM.
    
    Espera um corpo de requisição JSON com um array de registros (cada registro é um objeto).
    Retorna um array com as predições, probabilidades e decis.
    """
    global inference_pipeline
    
    # Verificar se o pipeline está inicializado
    if inference_pipeline is None:
        logger.warning("Pipeline não inicializado. Tentando inicializar...")
        if not initialize_pipeline():
            return jsonify({
                "error": "Falha ao inicializar o pipeline"
            }), 500
    
    try:
        # Registrar solicitação recebida
        logger.info(f"Recebida solicitação de predição")
        
        # Obter dados da requisição
        input_data = request.get_json()
        
        # Verificar se temos dados
        if not input_data or 'records' not in input_data:
            logger.warning("Formato de entrada inválido")
            return jsonify({
                "error": "Formato de entrada inválido. Esperado: {'records': [{...}, {...}]}"
            }), 400
        
        # Converter para DataFrame
        records = input_data['records']
        if not records:
            logger.warning("Nenhum registro fornecido")
            return jsonify({
                "error": "Nenhum registro fornecido"
            }), 400
        
        # Criar DataFrame
        try:
            df = pd.DataFrame(records)
            logger.info(f"Processando {len(df)} registros")
        except Exception as e:
            logger.error(f"Falha ao converter registros para DataFrame: {str(e)}")
            return jsonify({
                "error": f"Falha ao converter registros para DataFrame: {str(e)}"
            }), 400
        
        # Fazer predições
        logger.info("Executando predições...")
        prediction_result = inference_pipeline.predict(df)
        logger.info("Predições concluídas com sucesso")
        
        # Converter resultados para formato JSON
        results_df = prediction_result['results']
        metadata = prediction_result['metadata']
        
        # Formatar resultados
        formatted_results = []
        for i in range(len(results_df)):
            formatted_results.append({
                "prediction": int(results_df.iloc[i]['prediction']),
                "probability": float(results_df.iloc[i]['probability']),
                "decile": int(results_df.iloc[i]['decile'])
            })
        
        # Construir resposta
        response = {
            "results": formatted_results,
            "metadata": metadata
        }
        
        logger.info(f"Retornando {len(formatted_results)} resultados")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erro ao processar a requisição: {str(e)}")
        return jsonify({
            "error": f"Erro ao processar a requisição: {str(e)}"
        }), 500

@app.route('/reload', methods=['POST'])
def reload_model():
    """Endpoint para recarregar o modelo."""
    logger.info("Solicitação para recarregar o modelo")
    
    if initialize_pipeline():
        return jsonify({
            "status": "success",
            "message": "Modelo recarregado com sucesso"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Falha ao recarregar modelo"
        }), 500

def start_service(host='0.0.0.0', port=8080):
    """Inicia o serviço de API."""
    # Cloud Run espera que o serviço escute na porta definida pela variável PORT
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    # Porta definida pela variável de ambiente PORT (padrão do Cloud Run)
    port = int(os.environ.get('PORT', 8080))
    start_service(port=port)