#!/usr/bin/env python
"""
Pipeline completa de inferência para o projeto Smart Ads.
"""

import os
import sys
import traceback

print("SCRIPT INICIADO!")
print(f"Diretório de trabalho atual: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    # Adicionar diretório raiz ao path
    project_root = "/Users/ramonmoreira/desktop/smart_ads"
    sys.path.insert(0, project_root)
    print(f"Diretório raiz adicionado: {project_root}")
    
    # Verificar módulos
    module_paths = {
        "script2": os.path.join(project_root, "inference/modules/script2_module.py"),
        "script3": os.path.join(project_root, "inference/modules/script3_module.py"),
        "script4": os.path.join(project_root, "inference/modules/script4_module.py"),
        "gmm": os.path.join(project_root, "inference/modules/gmm_module.py")
    }
    
    for name, path in module_paths.items():
        if os.path.exists(path):
            print(f"Módulo {name} encontrado: {path}")
        else:
            print(f"AVISO: Módulo {name} não encontrado: {path}")
    
    # Importações básicas
    import pandas as pd
    import joblib
    from datetime import datetime
    import argparse
    print("Importações básicas bem-sucedidas")
    
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from transformers import AutoTokenizer, AutoModel
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Baixando recursos NLTK necessários...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    # Importar módulos da pipeline com melhor tratamento de erros
    try:
        from inference.modules.script2_module import apply_script2_transformations
        print("Módulo script2_module importado com sucesso")
    except Exception as e:
        print(f"ERRO AO IMPORTAR script2_module: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)
    
    script3_module_imported = False
    try:
        from inference.modules.script3_module import apply_script3_transformations
        script3_module_imported = True
        print("Módulo script3_module importado com sucesso")
    except Exception as e:
        print(f"AVISO: Não foi possível importar script3_module: {str(e)}")
        print(traceback.format_exc())
    
    script4_module_imported = False
    try:
        from inference.modules.script4_module import apply_script4_transformations
        script4_module_imported = True
        print("Módulo script4_module importado com sucesso")
    except Exception as e:
        print(f"AVISO: Não foi possível importar script4_module: {str(e)}")
    
    gmm_module_imported = False
    try:
        from inference.modules.gmm_module import apply_gmm_and_predict
        gmm_module_imported = True
        print("Módulo gmm_module importado com sucesso")
    except Exception as e:
        print(f"AVISO: Não foi possível importar gmm_module: {str(e)}")
    
    # No arquivo inference_pipeline.py, modifique a função run_full_pipeline

    def run_full_pipeline(input_path, output_path, params_dir, until_step=4):
        """
        Executa a pipeline completa de inferência ou até uma etapa específica.
        """
        print(f"Iniciando pipeline com: {input_path}")
        print(f"Etapas solicitadas: até {until_step}")
        
        # Carregar dados de entrada
        try:
            input_data = pd.read_csv(input_path)
            print(f"Dados carregados: {input_data.shape}")
        except Exception as e:
            print(f"Erro ao carregar dados: {str(e)}")
            sys.exit(1)
        
        # Variável para armazenar o último resultado processado
        latest_result = input_data
        
        # Etapa 1: Pré-processamento básico (script 2)
        if until_step >= 1:
            print("\n--- Etapa 1: Pré-processamento básico ---")
            params_path = os.path.join(params_dir, "02_all_preprocessing_params.joblib")
            if not os.path.exists(params_path):
                print(f"ERRO: Arquivo de parâmetros não encontrado: {params_path}")
                sys.exit(1)
            
            latest_result = apply_script2_transformations(input_data, params_path)
            
            # Se solicitado apenas até a etapa 1, salvar e retornar
            if until_step == 1:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                latest_result.to_csv(output_path, index=False)
                print(f"Pipeline finalizada na etapa 1. Resultado salvo em: {output_path}")
                return latest_result
        
        # Etapa 2: Processamento de texto avançado (script 3)
        step2_complete = False
        if until_step >= 2 and script3_module_imported:
            print("\n--- Etapa 2: Processamento de texto avançado ---")
            try:
                print(f"Chamando apply_script3_transformations()...")
                params_path = os.path.join(params_dir, "dummy.joblib")  # Caminho base apenas
                latest_result = apply_script3_transformations(latest_result, params_path)
                step2_complete = True
                print(f"Transformações da etapa 2 concluídas com sucesso")
                
                # Se solicitado apenas até a etapa 2, salvar e retornar
                if until_step == 2:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    latest_result.to_csv(output_path, index=False)
                    print(f"Pipeline finalizada na etapa 2. Resultado salvo em: {output_path}")
                    return latest_result
            except Exception as e:
                print(f"ERRO durante a etapa 2: {str(e)}")
                print(traceback.format_exc())
                # Continuar com o último resultado válido
        
        # Etapa 3: Features de motivação profissional (script 4)
        step3_complete = False
        if until_step >= 3 and script4_module_imported:
            print("\n--- Etapa 3: Features de motivação profissional ---")
            try:
                params_path = os.path.join(params_dir, "04_motivation_features_params.joblib")
                latest_result = apply_script4_transformations(latest_result, params_path)
                step3_complete = True
                
                if until_step == 3:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    latest_result.to_csv(output_path, index=False)
                    print(f"Pipeline finalizada na etapa 3. Resultado salvo em: {output_path}")
                    return latest_result
            except Exception as e:
                print(f"ERRO durante a etapa 3: {str(e)}")
                print(traceback.format_exc())
                # Continuar com o último resultado válido
        
        # Etapa 4: Aplicação de GMM e modelos por cluster (scripts 8/10)
        step4_complete = False
        if until_step >= 4 and gmm_module_imported:
            print("\n--- Etapa 4: Aplicação de GMM e predição ---")
            try:
                models_dir = os.path.join(params_dir, "models")
                latest_result = apply_gmm_and_predict(latest_result, models_dir)
                step4_complete = True
            except Exception as e:
                print(f"ERRO durante a etapa 4: {str(e)}")
                print(traceback.format_exc())
                # Continuar com o último resultado válido
        
        # Salvar o resultado final
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        latest_result.to_csv(output_path, index=False)
        
        # Mensagem de conclusão apropriada
        highest_step = 0
        if step4_complete:
            highest_step = 4
        elif step3_complete:
            highest_step = 3
        elif step2_complete:
            highest_step = 2
        else:
            highest_step = 1
        
        print(f"\n=== Pipeline de inferência concluída até a etapa {highest_step} ===")
        print(f"Resultado salvo em: {output_path}")
        return latest_result
    
    def main():
        """Função principal."""
        parser = argparse.ArgumentParser(description="Pipeline completa de inferência")
        parser.add_argument("--input", type=str, default="/Users/ramonmoreira/desktop/smart_ads/data/01_split/test.csv",
                          help="Caminho para dados de entrada")
        parser.add_argument("--output", type=str, help="Caminho para salvar resultado")
        parser.add_argument("--params-dir", type=str, default="/Users/ramonmoreira/desktop/smart_ads/inference/params",
                          help="Diretório com parâmetros salvos")
        parser.add_argument("--until-step", type=int, choices=[1, 2, 3, 4], default=4,
                          help="Executar até qual etapa (1-4)")
        
        args = parser.parse_args()
        print(f"Argumentos: {args}")
        
        # Definir caminho de saída padrão se não especificado
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "/Users/ramonmoreira/desktop/smart_ads/inference/output"
            os.makedirs(output_dir, exist_ok=True)
            
            if args.until_step < 4:
                args.output = os.path.join(output_dir, f"inference_step{args.until_step}_{timestamp}.csv")
            else:
                args.output = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        
        # Executar pipeline
        run_full_pipeline(
            input_path=args.input,
            output_path=args.output,
            params_dir=args.params_dir,
            until_step=args.until_step
        )
        
        print(f"\n=== Pipeline de inferência concluída com sucesso! ===")
        print(f"Resultado salvo em: {args.output}")
    
    if __name__ == "__main__":
        print("Chamando função main()...")
        main()
        print("Função main() concluída.")

except Exception as e:
    print(f"ERRO GERAL: {str(e)}")
    print(traceback.format_exc())