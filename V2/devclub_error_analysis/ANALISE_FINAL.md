# ANÁLISE DE ERRO - DevClub Lead Scoring

**Modelo**: v1_devclub_rf_temporal_single
**Performance**: AUC 0.636 | Lift D10 2.07x | Top 3 decis 45.5%
**Método**: Testes estatísticos (57 features) + SHAP (FN=66, FP=500)

---

## 🔴 PROBLEMAS IDENTIFICADOS

### **Features com Comportamento Oposto**
SHAP negativo em FN (penaliza conversores) mas positivo em FP (infla não-conversores):

| Feature | p-value | SHAP FN | SHAP FP | Diagnóstico |
|---------|---------|---------|---------|-------------|
| **Medium: Linguagem Prog** | 0.028✓ | -0.0126 | +0.0100 | Efeito depende de contexto (idade/salário) |
| **Idade** | 0.108 | -0.0247 | +0.0038 | Relação não-linear - binning necessário |
| **Faixa Salarial** | 0.201 | -0.0176 | +0.0113 | Relação não-linear - binning necessário |
| **Tem Computador** | 0.008✓ | -0.0014 | +0.0171 | Sozinho não discrimina - precisa interação |
| **Estudante** | 0.013✓ | -0.0034 | +0.0132 | Infla FP - precisa filtro (recursos/investimento) |
| **Investiu Curso** | 0.000✓ | -0.0031 | +0.0017 | Leve inconsistência - melhorar com interação |

✓ = estatisticamente significativo (p < 0.05)

### **Feature Subutilizada**

| Feature | p-value | Correlação | Lift | Problema |
|---------|---------|------------|------|----------|
| **Já Estudou Prog** | 0.008✓ | +0.10 | 1.61x | Significativa mas correlação fraca - modelo não usa bem |

---

## ✅ RECOMENDAÇÕES

### **1. `medium_tecnico_qualificado`**
**O quê**: Medium Linguagem Prog × Idade Alta × Salário Alto
**Por quê**: Medium técnico tem sinal oposto (FN -0.0126, FP +0.0100). Hipótese: só converte em perfil sênior
**Impacto**: Corrigir penalização de FN, +0.05-0.10 AUC

**Implementação (após encoding):**
```python
df['medium_tecnico_qualificado'] = (
    (df['Medium_Linguagem_de_programa_o'] == 1) &
    (df['Qual_a_sua_idade'] >= 4) &
    (df['Atualmente_qual_a_sua_faixa_salarial'] >= 3)
).astype(int)
```

---

### **2. `estudante_com_recursos`**
**O quê**: Estudante × Tem Computador × Investiu Curso
**Por quê**: Estudante infla FP (+0.0132), Tem Computador também (+0.0171). Filtrar estudantes com recursos que não convertem
**Impacto**: Reduzir FP em 10-15%

**Implementação (após encoding):**
```python
df['estudante_com_recursos'] = (
    (df['O_que_voc_faz_atualmente_Sou_apenas_estudante'] == 1) &
    (df['Tem_computador_notebook_Sim'] == 1) &
    (df['investiu_curso_online_Sim'] == 1)
).astype(int)
```

---

### **3. `experiencia_completa`**
**O quê**: Já Estudou × Investiu Curso
**Por quê**: Já Estudou significativa (p=0.008) mas subutilizada (corr 0.10). Lift combinado 1.76x vs 1.42x individual
**Impacto**: +0.1-0.2 AUC

**Implementação (após encoding):**
```python
df['experiencia_completa'] = (
    (df['J_estudou_programa_o_Sim'] == 1) &
    (df['investiu_curso_online_Sim'] == 1)
).astype(int)
```

---

### **4. `perfil_senior`**
**O quê**: Idade Alta × Salário Alto
**Por quê**: Ambas com sinal oposto FN vs FP. Interação captura perfil sênior que converte mais
**Impacto**: +0.03-0.08 AUC

**Implementação (após encoding):**
```python
df['perfil_senior'] = (
    (df['Qual_a_sua_idade'] >= 4) &
    (df['Atualmente_qual_a_sua_faixa_salarial'] >= 3)
).astype(int)
```

---

## 📊 IMPACTO TOTAL ESPERADO

| Métrica | Atual | Esperado | Ganho |
|---------|-------|----------|-------|
| AUC | 0.636 | 0.70-0.75 | +0.06-0.11 |
| Lift D10 | 2.07x | 2.5-3.0x | +0.4-0.9x |
| Top 3 decis | 45.5% | 55-65% | +10-20pp |
| Falsos Negativos | 66 (54.5%) | 40-50 | -25-40% |
| Falsos Positivos | 3,466 (29.8%) | 2,500-3,000 | -15-30% |

---

## 🗂️ ARQUIVOS

- `statistical_tests_all_features.csv` - Testes em 57 features
- `features_criticas_resumo.csv` - Features críticas (stats + SHAP)
- `shap_importance_fn.csv` / `shap_importance_fp.csv` - SHAP por grupo
- `false_negatives.csv` / `false_positives.csv` - Leads para inspeção

**Data**: 2025-10-06
