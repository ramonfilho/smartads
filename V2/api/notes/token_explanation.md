⏺ 🔐 Arquitetura de Autenticação Meta Ads

  Componentes e Suas Funções

  ---
  1️⃣ App da Meta (Facebook App)

  O que é?

  Um "aplicativo" registrado no Facebook Developers que representa seu sistema Smart Ads.

  O que faz?

  - Identifica quem está fazendo as requisições à Meta Ads API
  - Define quais permissões pode solicitar (ads_read, read_insights, etc.)
  - Gerencia níveis de acesso (Standard ou Advanced)
  - Controla rate limits (quantas chamadas por hora)

  Onde está?

  - https://developers.facebook.com/apps
  - Cada app tem um App ID (ex: 123456789012345)

  Analogia:

  É como um "cadastro de empresa" - antes de fazer negócios, você precisa estar registrado.

  ---
  2️⃣ System User (Usuário do Sistema)

  O que é?

  Um "usuário robô" criado dentro do seu Business Manager para representar aplicações automatizadas.

  O que faz?

  - Representa sua aplicação (Smart Ads API) de forma persistente
  - Não depende de usuário humano estar logado
  - Pode ter permissões específicas em contas de anúncios
  - Gera tokens de longa duração (60 dias ou mais)

  Por que precisa?

  Porque tokens de usuários normais:
  - ❌ Expiram em 1-2 horas
  - ❌ Dependem de pessoa estar logada
  - ❌ Param de funcionar se pessoa mudar senha

  System User:
  - ✅ Tokens duram 60+ dias
  - ✅ Funcionam 24/7
  - ✅ Independentes de logins humanos

  Onde está?

  - Seu Business Manager > Business Settings > Users > System Users

  Analogia:

  É como uma "conta de serviço" ou "service account" - um usuário técnico para automação.

  ---
  3️⃣ Access Token (Token de Acesso)

  O que é?

  Uma "chave secreta" que prova que você tem permissão para acessar dados.

  O que faz?

  - Autentica cada requisição à API do Meta
  - Carrega informações sobre:
    - Qual App está fazendo a chamada
    - Qual System User gerou o token
    - Quais permissões tem (ads_read, etc.)
    - Quais contas pode acessar
  - Expira após 60 dias (para System User)

  Como é usado?

  # meta_integration.py
  response = requests.get(
      f"https://graph.facebook.com/v18.0/{account_id}/insights",
      params={
          'access_token': 'EAAS9hlWC7lkBPqBZAD...',  # ← Token aqui
          'level': 'campaign',
          'fields': 'spend'
      }
  )

  Formato:

  EAAS9hlWC7lkBPqBZADSlJOZAgXWAfHF8WmGzmZBiZBtbZA8b35asUvx3O6ELvFa...
  └─┬─┘
    └── Prefixo que identifica tipo de token

  Analogia:

  É como um "crachá temporário" - permite entrar em áreas restritas por tempo limitado.

  ---
  🔄 Como Tudo se Conecta

  ┌────────────────────────────────────────────────────────────┐
  │  1️⃣ VOCÊ cria um APP no Facebook Developers               │
  │     https://developers.facebook.com/apps                   │
  │                                                             │
  │     App ID: 123456789012345                                │
  │     App Name: "Smart Ads Integration"                      │
  │     Permissões solicitadas: ads_read, read_insights       │
  └────────────────────┬───────────────────────────────────────┘
                       │
                       │ App criado
                       ▼
  ┌────────────────────────────────────────────────────────────┐
  │  2️⃣ Você cria SYSTEM USER no seu Business Manager         │
  │     Business Settings > System Users > Add                 │
  │                                                             │
  │     System User Name: "smart-ads-api-user"                 │
  │     Role: Admin (no SEU Business Manager)                  │
  └────────────────────┬───────────────────────────────────────┘
                       │
                       │ System User criado
                       ▼
  ┌────────────────────────────────────────────────────────────┐
  │  3️⃣ Você GERA TOKEN no System User                        │
  │     System Users > smart-ads-api-user > Generate Token     │
  │                                                             │
  │     Seleciona:                                              │
  │     - App: "Smart Ads Integration"                         │
  │     - Permissões: ads_read, read_insights                  │
  │                                                             │
  │     Token gerado: EAAS9hlWC7lk...                          │
  │     Expira em: 60 dias                                     │
  └────────────────────┬───────────────────────────────────────┘
                       │
                       │ Token copiado
                       ▼
  ┌────────────────────────────────────────────────────────────┐
  │  4️⃣ CLIENTE adiciona você como PARTNER                    │
  │     Business Settings > Partners > Add                     │
  │                                                             │
  │     Partner Business ID: (seu Business ID)                 │
  │     Permissão: Analyst na conta act_123456789              │
  └────────────────────┬───────────────────────────────────────┘
                       │
                       │ Permissão concedida
                       ▼
  ┌────────────────────────────────────────────────────────────┐
  │  5️⃣ Você ATRIBUI acesso do System User à conta cliente   │
  │     System Users > Assign Assets > Ad Accounts             │
  │                                                             │
  │     Seleciona: act_123456789 (conta do cliente)            │
  │     Permissão: Analyst (já definida pelo cliente)          │
  └────────────────────┬───────────────────────────────────────┘
                       │
                       │ Tudo configurado
                       ▼
  ┌────────────────────────────────────────────────────────────┐
  │  6️⃣ Smart Ads API FAZ REQUISIÇÕES                         │
  │                                                             │
  │  GET https://graph.facebook.com/v18.0/                     │
  │      act_123456789/insights                                │
  │      ?access_token=EAAS9hlWC7lk...                        │
  │      &level=campaign                                       │
  │      &fields=spend                                         │
  │                                                             │
  │  Meta Ads API valida:                                      │
  │  ✅ Token válido? (não expirou)                           │
  │  ✅ App permitido? (smart-ads-integration)                │
  │  ✅ System User tem acesso? (analyst na conta)            │
  │  ✅ Permissão correta? (ads_read)                         │
  │                                                             │
  │  Se tudo OK → Retorna dados                                │
  └────────────────────────────────────────────────────────────┘

  ---
  🔍 Detalhamento do Token

  Estrutura Interna (o que o token "carrega")

  Quando você gera um token, ele contém (codificado):

  {
    "app_id": "123456789012345",
    "user_id": "system_user_789",
    "scopes": ["ads_read", "read_insights"],
    "issued_at": "2025-10-13T10:00:00Z",
    "expires_at": "2025-12-12T10:00:00Z",
    "granted_accounts": ["act_123456789", "act_987654321"]
  }

  Processo de Validação (o que Meta faz ao receber)

  # Quando você faz:
  requests.get(
      "https://graph.facebook.com/v18.0/act_123456789/insights",
      params={'access_token': 'EAAS9hlWC7lk...'}
  )

  # Meta Ads API faz internamente:
  1. Decodifica token
  2. Verifica se não expirou (expires_at > now)
  3. Verifica se app_id existe e está ativo
  4. Verifica se user_id tem permissão em act_123456789
  5. Verifica se 'ads_read' está nos scopes
  6. Se TUDO OK → retorna dados
     Se ALGO falha → erro 401 Unauthorized

  ---
  📍 Onde Cada Coisa Está no Código

  1. Token é usado aqui:

  api/meta_config.py (linha 8):
  META_CONFIG = {
      "access_token": "EAAS9hlWC7lkBPqBZAD...",  # ← Token aqui
      "api_version": "v18.0",
  }

  2. Token é passado para cada requisição:

  api/app.py (linha 491-493):
  meta_client = MetaAdsIntegration(
      access_token=META_CONFIG['access_token'],  # ← Token passado
      api_version=META_CONFIG['api_version']
  )

  3. Token é usado na requisição HTTP:

  api/meta_integration.py (linha 51-56):
  params = {
      'access_token': self.access_token,  # ← Token enviado
      'level': level,
      'fields': ','.join(fields),
      'time_range': f'{{"since":"{since}","until":"{until}"}}',
  }

  response = requests.get(url, params=params)  # ← Requisição com token

  ---
  🔒 Segurança do Token

  O que o token PODE fazer:

  - ✅ Ler dados de campanhas (campaign_name, spend, etc.)
  - ✅ Ler insights (impressões, cliques)
  - ✅ Apenas nas contas onde foi concedido acesso

  O que o token NÃO PODE fazer:

  - ❌ Criar/editar/deletar campanhas (só leitura)
  - ❌ Ver métodos de pagamento
  - ❌ Acessar outras contas não autorizadas
  - ❌ Mudar configurações do Business Manager

  Se o token vazar:

  1. Alguém pode ler seus dados de campanha
  2. Mas não pode modificar nada
  3. Você pode revogar o token a qualquer momento
  4. Gerar um novo em 30 segundos

  ---
  ✅ Checklist de Dúvidas - Marque o que Entendeu

  Conceitos Básicos

  - O que é o App da Meta e para que serve
  - O que é um System User e por que preciso dele
  - O que é um Access Token e como funciona
  - Diferença entre token de usuário normal vs System User

  Fluxo de Configuração

  - Como criar um App no Facebook Developers
  - Como criar um System User no meu Business Manager
  - Como gerar um token a partir do System User
  - Como cliente me adiciona como Partner
  - Como atribuir acesso do System User à conta do cliente

  Uso no Código

  - Onde o token é armazenado no código (meta_config.py)
  - Como o token é usado nas requisições (meta_integration.py)
  - Onde o account_id é definido (apps-script-code.js)
  - Como a requisição HTTP é feita para Meta Ads API

  Segurança e Permissões

  - Quais permissões o token tem (ads_read, read_insights)
  - O que o token PODE fazer (ler dados)
  - O que o token NÃO PODE fazer (modificar, billing)
  - Como revogar/renovar token se necessário

  Fluxo de Dados Completo

  - Apps Script envia account_id para API
  - API usa token para chamar Meta Ads API
  - Meta valida token e retorna dados
  - API processa e retorna para Apps Script
  - Apps Script cria abas com análise

  Troubleshooting

  - O que fazer se token expirar (60 dias)
  - Como mudar para conta real do cliente
  - Como verificar se token está funcionando
  - Onde ver logs de erros de autenticação

  ---
  🎯 Resumo Ultra-Simplificado

  App da Meta     = "Registro de empresa" (quem você é)
  System User     = "Conta robô" (para automação 24/7)
  Access Token    = "Crachá temporário" (prova de autorização)

  Fluxo:
  1. Criar App → Gera App ID
  2. Criar System User → Pode gerar tokens
  3. Gerar Token com App + System User → EAAS9hlWC7lk...
  4. Usar Token nas requisições → Authorization
  5. Meta valida e retorna dados → Sucesso!

  ---