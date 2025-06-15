# test_prediction_pipeline.py
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/ramonmoreira/desktop/smart_ads')

from smart_ads_pipeline.pipelines.prediction_pipeline import PredictionPipeline

def test_prediction_pipeline():
    """Testa o pipeline de predição"""
    
    print("=== Teste do PredictionPipeline ===\n")
    
    # Usar parâmetros do teste anterior
    params_path = '/tmp/smart_ads_alignment_test/pipeline_params.joblib'
    
    print(f"📁 Carregando parâmetros de: {params_path}")
    
    try:
        # Inicializar pipeline
        pipeline = PredictionPipeline(params_path=params_path)
        
        # Mostrar resumo
        summary = pipeline.get_summary()
        print(f"\n📊 Resumo do Pipeline:")
        print(f"   - Features: {summary['n_features']}")
        print(f"   - Modelo carregado: {summary['has_model']}")
        print(f"   - Componentes: {', '.join(summary['components'])}")
        
        # Criar dados de teste (simulando novos surveys)
        test_data = pd.DataFrame({
            'marca_temporal': pd.date_range('2024-01-01', periods=10),
            'cual_es_tu_e_mail': ['test@example.com'] * 10,
            'cual_es_tu_genero': ['Hombre', 'Mujer'] * 5,
            'cual_es_tu_edad': ['25 años a 34 años'] * 10,
            'cual_es_tu_pais': ['México', 'Colombia', 'Brasil'] * 3 + ['Argentina'],
            'utm_source': ['google', 'facebook'] * 5,
            'utm_medium': ['cpc', 'social'] * 5,
            'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida_que_oportunidades_se_abriran_para_ti': 
                ['Mejores oportunidades laborales'] * 10,
            'que_esperas_aprender_en_el_evento_cero_a_ingles_fluido': 
                ['Hablar con fluidez'] * 10,
            'dejame_un_mensaje': ['Gracias por la oportunidad'] * 10,
            # Adicionar outras colunas conforme necessário
        })
        
        print(f"\n🔍 Validando dados de entrada...")
        validation = pipeline.validate_input(test_data)
        print(f"   Válido: {validation['is_valid']}")
        if validation['warnings']:
            print(f"   Avisos: {validation['warnings'][0]}")
        
        print(f"\n🚀 Aplicando transformações...")
        result = pipeline.predict(test_data)
        
        print(f"\n✅ Transformações aplicadas com sucesso!")
        print(f"   Shape original: {test_data.shape}")
        print(f"   Shape final: {result.shape}")
        
        # Mostrar algumas features
        print(f"\n📋 Exemplos de features processadas:")
        feature_names = pipeline.get_feature_names()
        for i, feat in enumerate(feature_names[:5]):
            print(f"   {i+1}. {feat}")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction_pipeline()
    print("\n✅ Teste concluído!")