#Perguntar sobre a estrutura geral da pipeline. Comparar com outras pipelines já vistas. Prós e contras e oportunidades de melhoria que não vão "quebrar" o pipeline. Considere o tipo de problema que estamos tratando e a natureza dos dados.

#SOBRE NOMES DIFERENTES DE FEATURES AO LONGO DO SCRIPT:
#Eu tive um problema na pipeline de inferência passada. A pipeline de treino com os códigos separados produzia 1050 features, porém, usando diferentes convenções de normalização. 
#Após passar por um script de feature selection, foram selecionadas 300 features para o modelo ser treinado. 
#Porém, na pipeline de inferência, após executar todos os passos, estavam faltando 95 features que o modelo esperava. 
#Ao realizar um diagnóstico de quais features o modelo esperava e estavam faltando, detectamos que havia uma inconsistência. Ele esperava features com um nome X, e estava recebendo com um nome Y. Às vezes as features até existiam, mas com um nome diferente do que ele esperava.

#SCRIPT 01
#COLUNA DE E-MAIL REMOVIDA NA FUNÇÃO PREPARE FINAL DATASET PODE ESTAR IMPACTANDO NEGATIVAMENTE NA CRIAÇÃO DE FEATURES POSTERIORMENTE?

#SCRIPT 02
#COMO A COLUNA DE PROFISSÕES ESTÁ SENDO TRATADA? 

#ATUALIZE A FUNÇÃO APPLY PREPROCESSING PIPELINE, PASSO 6 do script 2, para detectar as colunas de texto usando uma lógica de verificação que identifique pelo conteúdo as features se ela é textual ou não. 
#Alguma variação no conteúdo da resposta somado a uma verificação de tipo na resposta pode ajudar a identificar as colunas de texto.
#Atualize a função identify text columns do texto 3 para usar as mesmas colunas de texto identificadas pelo script 02.
#Detecte se algum outro lugar possui alguma lógica isolada para identificar colunas de texto e unifique com essa mesma lógica.

#SOBRE TOPIC MODELIND (LDA):
#Esse topic modeling gera features que passam pela análise de feature importance? Qual a relevância delas? Se for algo não muito relevante, eliminar, pois é mais um conjunto de parâmetros / modelo para subir e garantir consistência entre treino e produção, e isso fica mais difícil de manter. 
#Existe uma aplicação dupla de LDA no script 3?

#Retreinar o modelo e criar pipeline de inferência.
#-- Chamar LLM (langchain?) para determinar o vocabulário de palavras a serem usadas para criar as features de texto.