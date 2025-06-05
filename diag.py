import pandas as pd
from unified_pipeline import unified_data_pipeline

print("=== TESTE DE PRESERVAÇÃO DE COLUNAS ===\n")

# Execute com poucos dados
results = unified_data_pipeline(
    test_mode=True,
    max_samples=500,  # Mais amostras para evitar erro de split
    apply_feature_selection=False,
    use_checkpoints=False,
    clear_cache=True
)

if results and 'params' in results:
    params = results['params']
    
    # Verificar se colunas foram preservadas
    if 'feature_engineering' in params:
        fe_params = params['feature_engineering']
        if 'preserved_columns' in fe_params:
            preserved = fe_params['preserved_columns']
            print(f"\n✅ SUCESSO! {len(preserved)} colunas preservadas:")
            for col in preserved.keys():
                print(f"   - {col}")
        else:
            print("\n❌ ERRO: Nenhuma coluna foi preservada!")
    
    # Verificar se as colunas ainda existem no DataFrame final
    train_df = results['train']
    print(f"\nColunas importantes no dataset final:")
    for col in ['cual_es_tu_profesion', 'como_te_llamas', 'cual_es_tu_instagram']:
        exists = col in train_df.columns
        print(f"   {col}: {'✅ Existe' if exists else '❌ Removida'}")