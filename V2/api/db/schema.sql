-- Schema para banco de dados CAPI
-- Armazena dados FBP/FBC dos leads para envio via Conversions API

CREATE TABLE IF NOT EXISTS leads_capi (
    id SERIAL PRIMARY KEY,

    -- Identificação do lead
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    phone VARCHAR(50),

    -- Dados CAPI (Meta)
    fbp VARCHAR(255),
    fbc VARCHAR(255),
    event_id VARCHAR(255) UNIQUE,

    -- Dados de tracking
    user_agent TEXT,
    client_ip VARCHAR(50),
    event_source_url TEXT,

    -- UTMs
    utm_source VARCHAR(255),
    utm_medium VARCHAR(255),
    utm_campaign VARCHAR(255),
    utm_term VARCHAR(255),
    utm_content VARCHAR(255),

    -- Outros
    tem_comp VARCHAR(50),

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Índices para queries rápidas
    CONSTRAINT email_event_unique UNIQUE(email, event_id)
);

-- Índices para performance
CREATE INDEX idx_email ON leads_capi(email);
CREATE INDEX idx_event_id ON leads_capi(event_id);
CREATE INDEX idx_created_at ON leads_capi(created_at);

-- View para leads recentes (útil para debug)
CREATE OR REPLACE VIEW leads_capi_recent AS
SELECT
    email,
    fbp,
    fbc,
    event_id,
    created_at
FROM leads_capi
WHERE created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Comentários
COMMENT ON TABLE leads_capi IS 'Dados de leads capturados para envio via Meta Conversions API';
COMMENT ON COLUMN leads_capi.fbp IS 'Facebook Browser ID (_fbp cookie)';
COMMENT ON COLUMN leads_capi.fbc IS 'Facebook Click ID (_fbc cookie)';
COMMENT ON COLUMN leads_capi.event_id IS 'ID único do evento para deduplicação Pixel + CAPI';
