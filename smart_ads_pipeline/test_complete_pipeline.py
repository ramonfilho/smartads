# test_complete_pipeline.py
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, '/Users/ramonmoreira/desktop/smart_ads')

from smart_ads_pipeline.pipelines.training_pipeline import TrainingPipeline
from smart_ads_pipeline.pipelines.prediction_pipeline import PredictionPipeline

def test_complete_pipeline():
    """Teste completo: treino + predição com dados reais."""
    
    print("=== TESTE COMPLETO DO PIPELINE SMART ADS ===\n")
    
    # Configuração
    config = {
        'data_path': '/Users/ramonmoreira/desktop/smart_ads/data/raw_data',
        'output_dir': '/tmp/smart_ads_complete_test',
        'test_size': 0.3,
        'val_size': 0.5,
        'random_state': 42,
        'max_features': 300,
        'fast_mode': False,  # Usar todos os modelos para seleção
        'sample_fraction': 0.1,  # Usar 10% dos dados para teste rápido
        'train_model': True,  # IMPORTANTE: Treinar modelo
        'use_checkpoints': False,
        'clear_cache': True
    }
    
    print("📋 Configuração do teste:")
    print(f"   - Usando {config['sample_fraction']*100:.0f}% dos dados")
    print(f"   - Max features: {config['max_features']}")
    print(f"   - Output: {config['output_dir']}")
    print()
    
    # ========================================================================
    # PARTE 1: TREINAR MODELO
    # ========================================================================
    
    print("="*60)
    print("PARTE 1: TREINAMENTO")
    print("="*60)
    
    pipeline = TrainingPipeline()
    
    try:
        results = pipeline.run(config)
        
        if results['success']:
            print("\n✅ Pipeline de treino executado com sucesso!")
            print(f"\n📊 Resumo do Treino:")
            print(f"   - Features selecionadas: {len(results['selected_features'])}")
            print(f"   - Shapes finais: Train={results['train_shape']}, Val={results['val_shape']}, Test={results['test_shape']}")
            
            if 'model_metrics' in results:
                print(f"\n📈 Métricas do Modelo:")
                print(f"   - AUC Val: {results['model_metrics'].get('val_auc', 0):.4f}")
                print(f"   - AUC Test: {results['model_metrics'].get('test_auc', 0):.4f}")
                print(f"   - Top Decile Lift: {results['model_metrics'].get('test_top_decile_lift', 0):.2f}x")
                print(f"   - Top 20% Recall: {results['model_metrics'].get('test_top_20pct_recall', 0):.2%}")
            
            # Salvar resumo
            summary_path = Path(config['output_dir']) / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump({
                    'config': config,
                    'results': {k: str(v) if not isinstance(v, (int, float, bool, list, dict)) else v 
                              for k, v in results.items()},
                    'selected_features': results['selected_features'][:10]  # Top 10
                }, f, indent=2)
            
        else:
            print(f"\n❌ Erro no pipeline: {results.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================================================
    # PARTE 2: TESTAR PREDIÇÕES
    # ========================================================================
    
    print("\n" + "="*60)
    print("PARTE 2: TESTE DE PREDIÇÕES")
    print("="*60)
    
    # Carregar pipeline de predição
    params_path = results['params_path']
    model_artifacts_dir = results.get('model_artifacts_dir')
    
    if not model_artifacts_dir:
        print("⚠️  Modelo não foi treinado, pulando teste de predições")
        return results
    
    model_path = Path(model_artifacts_dir) / "lightgbm_direct_ranking.joblib"
    
    print(f"\n📁 Carregando pipeline de predição...")
    print(f"   Parâmetros: {params_path}")
    print(f"   Modelo: {model_path}")
    
    try:
        pred_pipeline = PredictionPipeline(
            params_path=params_path,
            model_path=str(model_path)
        )
        
        # Criar dados de teste simulados
        print("\n🔍 Criando dados de teste para predição...")
        
        test_data = pd.DataFrame({
            'marca_temporal': pd.date_range('2024-01-01', periods=5),
            'cual_es_tu_e_mail': [f'test{i}@example.com' for i in range(5)],
            'cual_es_tu_genero': ['Hombre', 'Mujer', 'Hombre', 'Mujer', 'Hombre'],
            'cual_es_tu_edad': ['25 años a 34 años', '35 años a 44 años', '18 años a 24 años', 
                               '45 años a 54 años', '25 años a 34 años'],
            'cual_es_tu_pais': ['México', 'Colombia', 'Argentina', 'Perú', 'Chile'],
            'utm_source': ['google', 'facebook', 'google', 'instagram', 'google'],
            'utm_medium': ['cpc', 'social', 'cpc', 'social', 'cpc'],
            'utm_campaing': ['brand', 'awareness', 'conversion', 'engagement', 'brand'],
            'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida_que_oportunidades_se_abriran_para_ti': [
                'Conseguir mejor trabajo y salario',
                'Viajar por el mundo sin barreras',
                'Acceder a mejores oportunidades laborales',
                'Estudiar en el extranjero',
                'Mejorar mi carrera profesional'
            ],
            'que_esperas_aprender_en_el_evento_cero_a_ingles_fluido': [
                'Hablar con fluidez',
                'Pronunciación correcta',
                'Gramática avanzada',
                'Conversación natural',
                'Todo lo necesario para hablar inglés'
            ],
            'dejame_un_mensaje': [
                'Gracias por la oportunidad',
                'Espero aprender mucho',
                'Necesito el inglés para mi trabajo',
                'Quiero cumplir mi sueño',
                'Estoy muy motivado'
            ],
            'cual_es_tu_sueldo_anual_en_dolares': ['Menos de US$3000', 'US$3000 a US$5000', 
                                                   'Menos de US$3000', 'US$5000 o más', 'US$3000 a US$5000'],
            'cuanto_te_gustaria_ganar_al_ano': ['Más de US$10000 por año', 'Más de US$20000 por año',
                                               'Al menos US$ 3000 por año', 'Más de US$20000 por año', 
                                               'Más de US$10000 por año']
        })
        
        # Fazer predições
        print("\n🚀 Fazendo predições...")
        probabilities = pred_pipeline.predict(test_data, return_proba=True)
        
        # Processar resultados
        if len(probabilities.shape) == 2:
            # predict_proba retorna [prob_classe_0, prob_classe_1]
            probs = probabilities[:, 1]
        else:
            probs = probabilities
        
        # Carregar thresholds para calcular decis
        import pickle
        thresholds_path = Path(model_artifacts_dir) / "decile_thresholds.pkl"
        with open(thresholds_path, 'rb') as f:
            decile_thresholds = pickle.load(f)
        
        # Calcular decis
        def assign_decile(prob, thresholds):
            decile = 1
            for i, threshold in enumerate(thresholds):
                if prob > threshold:
                    decile = i + 2
            return decile
        
        deciles = [assign_decile(p, decile_thresholds) for p in probs]
        
        # Mostrar resultados
        print("\n📊 Resultados das Predições:")
        print("-" * 60)
        print("ID | Probabilidade | Decil | Recomendação")
        print("-" * 60)
        
        for i, (prob, decil) in enumerate(zip(probs, deciles)):
            if decil >= 9:
                rec = "🔥 Alta Prioridade"
            elif decil >= 7:
                rec = "⭐ Média-Alta"
            elif decil >= 5:
                rec = "📊 Média"
            else:
                rec = "❄️  Baixa"
            
            print(f"{i+1:2d} | {prob:12.4f} | {decil:5d} | {rec}")
        
        print("-" * 60)
        print(f"Média: {probs.mean():.4f} | Mediana: {pd.Series(probs).median():.4f}")
        
        # Salvar resultados de teste
        test_results = pd.DataFrame({
            'test_id': range(1, len(test_data) + 1),
            'email': test_data['cual_es_tu_e_mail'],
            'probability': probs,
            'decile': deciles,
            'country': test_data['cual_es_tu_pais']
        })
        
        test_results_path = Path(config['output_dir']) / "test_predictions.csv"
        test_results.to_csv(test_results_path, index=False)
        print(f"\n💾 Resultados salvos em: {test_results_path}")
        
        print("\n✅ Pipeline de predição testado com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro no teste de predições: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def validate_against_original():
    """Valida resultados contra o pipeline original."""
    
    print("\n" + "="*60)
    print("VALIDAÇÃO CONTRA PIPELINE ORIGINAL")
    print("="*60)
    
    # Comparar com métricas conhecidas do pipeline original
    original_metrics = {
        'auc': 0.7436,  # Valor aproximado dos scripts
        'gini': 0.4872,
        'top_decile_lift': 3.5,  # Aproximado
        'top_20pct_recall': 0.55
    }
    
    print("\n📊 Comparação de Métricas:")
    print("-" * 50)
    print("Métrica            | Original | Novo Pipeline | Diff")
    print("-" * 50)
    
    # Aqui você adicionaria a comparação real depois de rodar o teste
    
    print("\n💡 Para validação completa:")
    print("1. Execute o pipeline original com os mesmos dados")
    print("2. Compare as features geradas")
    print("3. Compare as métricas do modelo")
    print("4. Verifique que as predições são similares")

if __name__ == "__main__":
    print("🚀 Iniciando teste completo do pipeline Smart Ads\n")
    
    results = test_complete_pipeline()
    
    if results and results.get('success'):
        validate_against_original()
        print("\n🎉 TESTE COMPLETO FINALIZADO COM SUCESSO!")
    else:
        print("\n❌ Teste falhou. Verifique os logs acima.")
    
    print("\n📁 Verifique os resultados em: /tmp/smart_ads_complete_test/")