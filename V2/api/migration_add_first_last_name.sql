-- Migração: Adicionar campos first_name e last_name na tabela leads_capi
-- Data: 2025-11-14
-- Objetivo: Melhorar Event Quality Score da Meta CAPI

-- Adicionar colunas
ALTER TABLE leads_capi
ADD COLUMN IF NOT EXISTS first_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS last_name VARCHAR(255);

-- Comentários nas colunas (PostgreSQL)
COMMENT ON COLUMN leads_capi.first_name IS 'Primeiro nome do lead (para CAPI)';
COMMENT ON COLUMN leads_capi.last_name IS 'Sobrenome do lead (para CAPI)';

-- Popular campos para registros existentes (split do campo name)
-- Exemplo: "João Silva" -> first_name="João", last_name="Silva"
UPDATE leads_capi
SET
    first_name = SPLIT_PART(name, ' ', 1),
    last_name = CASE
        WHEN LENGTH(name) - LENGTH(REPLACE(name, ' ', '')) > 0
        THEN SUBSTRING(name FROM POSITION(' ' IN name) + 1)
        ELSE NULL
    END
WHERE first_name IS NULL
  AND name IS NOT NULL;

-- Verificar resultado
SELECT
    COUNT(*) as total_leads,
    COUNT(first_name) as com_primeiro_nome,
    COUNT(last_name) as com_sobrenome,
    COUNT(CASE WHEN first_name IS NULL AND name IS NOT NULL THEN 1 END) as sem_primeiro_nome
FROM leads_capi;
