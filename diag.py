from unified_pipeline import unified_data_pipeline
import time

print("=== TESTE DO PIPELINE UNIFICADO ===\n")

start_time = time.time()

# Execute com poucos dados para teste rápido
results = unified_data_pipeline(
    test_mode=True,
    max_samples=500,
    apply_feature_selection=False,
    use_checkpoints=False,
    clear_cache=True
)

end_time = time.time()

if results:
    print("\n=== RESULTADOS ===")
    print(f"Train shape: {results['train'].shape}")
    print(f"Val shape: {results['validation'].shape}")
    print(f"Test shape: {results['test'].shape}")
    print(f"\nTempo total: {(end_time - start_time)/60:.1f} minutos")
    
    # Verificar consistência
    train_cols = set(results['train'].columns)
    val_cols = set(results['validation'].columns)
    test_cols = set(results['test'].columns)
    
    if train_cols == val_cols == test_cols:
        print("\n✅ SUCESSO: Todos os datasets têm as mesmas colunas!")
    else:
        print("\n❌ ERRO: Inconsistência nas colunas!")