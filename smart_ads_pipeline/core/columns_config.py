# smart_ads_pipeline/config/columns_config.py

"""
Configuração centralizada de colunas para todo o pipeline.
"""

INFERENCE_COLUMNS = [
    # Dados de UTM
    'data',
    'e_mail',
    'utm_campaing',
    'utm_source',
    'utm_medium',
    'utm_content',
    'utm_term',
    'gclid',
    
    # Dados da pesquisa
    'marca_temporal',
    'como_te_llamas',
    'cual_es_tu_genero',
    'cual_es_tu_edad',
    'cual_es_tu_pais',
    'cual_es_tu_e_mail',
    'cual_es_tu_telefono',
    'cual_es_tu_instagram',
    'hace_quanto_tiempo_me_conoces',
    'cual_es_tu_disponibilidad_de_tiempo_para_estudiar_ingles',
    'cuando_hables_ingles_con_fluidez_que_cambiara_en_tu_vida_que_oportunidades_se_abriran_para_ti',
    'cual_es_tu_profesion',
    'cual_es_tu_sueldo_anual_en_dolares',
    'cuanto_te_gustaria_ganar_al_ano',
    'crees_que_aprender_ingles_te_acercaria_mas_al_salario_que_mencionaste_anteriormente',
    'crees_que_aprender_ingles_puede_ayudarte_en_el_trabajo_o_en_tu_vida_diaria',
    'que_esperas_aprender_en_el_evento_cero_a_ingles_fluido',
    'dejame_un_mensaje',
    
    # Features novas do L22
    'cuales_son_tus_principales_razones_para_aprender_ingles',
    'has_comprado_algun_curso_para_aprender_ingles_antes',
    
    # Qualidade
    'qualidade_nome',
    'qualidade_numero',
]

# Colunas críticas que DEVEM existir
CRITICAL_COLUMNS = [
    'marca_temporal',
    'cual_es_tu_e_mail',
    'e_mail',
    'cual_es_tu_genero',
    'cual_es_tu_edad',
    'cual_es_tu_pais'
]

# Colunas temporárias que devem ser removidas
TEMPORARY_COLUMNS = [
    'email_norm',
    'email',
    'campanhas',
    'class',
    'lancamento',
    'lancamento_utm',
    'prediction',
    'qualidade',
    'teste',
    'unnamed_8',
    'unnamed_9',
    'unnamed_10',
    'unnamed_11',
    'unnamed_12',
    'utm_campaign2',
    'utm_campaign_correcta',
    'utm_campaign_vini'
]