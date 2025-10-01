# 📋 Instruções: Configurar Apps Script no Google Sheets

## 🎯 O que vamos fazer

Configurar o Google Sheets para chamar automaticamente sua API de Lead Scoring no Cloud Run usando OAuth2 (sem expor credenciais).

---

## 📝 Passo a Passo

### 1️⃣ **Abrir o Apps Script**

1. Abra sua planilha no Google Sheets
2. Clique em **Extensões** → **Apps Script**
3. Uma nova aba será aberta com o editor de código

### 2️⃣ **Copiar o código**

1. Apague todo o código existente no editor
2. Abra o arquivo `apps-script-code.js` (que acabei de criar)
3. Copie TODO o conteúdo
4. Cole no editor do Apps Script
5. **Salve** o projeto (Ctrl+S ou ícone de disquete)
6. Dê um nome para o projeto, ex: "Smart Ads - Lead Scoring"

### 3️⃣ **Configurar permissões do OAuth2**

1. No editor do Apps Script, clique no **ícone de engrenagem** ⚙️ (Configurações do projeto)
2. Role até "Escopos do OAuth2"
3. Adicione manualmente estes escopos clicando em "+ Adicionar escopo":
   ```
   https://www.googleapis.com/auth/spreadsheets
   https://www.googleapis.com/auth/script.external_request
   ```

### 4️⃣ **Testar a conexão**

1. No editor, selecione a função `testConnection` no menu dropdown
2. Clique em **Executar** (▶️)
3. Uma janela de autorização aparecerá:
   - Clique em **Revisar permissões**
   - Escolha sua conta do Google
   - Clique em **Avançado** → **Ir para [nome do projeto] (não seguro)**
   - Clique em **Permitir**

4. Aguarde a execução
5. Se tudo estiver OK, verá um alerta: **"Conexão OK!"**

### 5️⃣ **Testar as predições**

1. Certifique-se que sua planilha tem:
   - **Linha 1:** Cabeçalhos (ex: Nome Completo, E-mail, Telefone, etc.)
   - **Linhas 2+:** Dados dos leads

2. No editor, selecione a função `getPredictions`
3. Clique em **Executar** (▶️)
4. Aguarde o processamento
5. Volte para sua planilha → Verá novas colunas:
   - `lead_score`: Probabilidade de conversão (0-1)
   - `decile`: Decil do lead (D1-D10)

### 6️⃣ **Configurar execução automática (opcional)**

Se quiser que rode automaticamente a cada 6 horas:

1. No editor, selecione a função `createTimeDrivenTrigger`
2. Clique em **Executar** (▶️)
3. Autorize se solicitado
4. Pronto! Agora roda automaticamente

**Para remover o agendamento:**
- Execute a função `removeTimeDrivenTrigger`

### 7️⃣ **Menu customizado**

Após salvar o código e recarregar a planilha, verá um novo menu **"Smart Ads"** com opções:
- Buscar Predições
- Configurar Agendamento
- Remover Agendamento

---

## 🔒 Segurança

✅ **O que está protegido:**
- API no Cloud Run é **privada** (não aceita requisições públicas)
- Apenas a Service Account `smart-ads-451319@appspot.gserviceaccount.com` pode acessar
- Apps Script usa OAuth2 automático do Google (sem credenciais no código)

✅ **Para o cliente:**
- Você vai copiar o mesmo código para o Sheets dele
- Quando ele executar pela primeira vez, vai autorizar o script
- O script roda no contexto da conta dele, mas usa a Service Account configurada no GCP
- Ele **não vê** credenciais no código

---

## 🐛 Troubleshooting

### Erro: "API retornou erro 403"
**Causa:** Service Account sem permissão
**Solução:** Já configuramos isso, mas confirme:
```bash
gcloud run services add-iam-policy-binding smart-ads-api \
  --region=us-central1 \
  --member="serviceAccount:smart-ads-451319@appspot.gserviceaccount.com" \
  --role="roles/run.invoker"
```

### Erro: "Planilha vazia ou só tem cabeçalho"
**Causa:** Sem dados na planilha
**Solução:** Adicione pelo menos 1 linha de dados (além do cabeçalho)

### Erro de autorização
**Causa:** Escopos OAuth2 não configurados
**Solução:** Repita o passo 3️⃣

---

## 📊 Estrutura da Planilha

**Colunas necessárias (exemplos):**
- Nome Completo
- E-mail
- Telefone
- Data de Nascimento
- CPF
- Origem do Lead
- ... (todos os campos que o modelo espera)

**Colunas criadas automaticamente:**
- `lead_score`: Score de 0 a 1 (ex: 0.8523)
- `decile`: D1 (melhor) até D10 (pior)

---

## 🚀 Próximos passos

1. Teste tudo na sua planilha
2. Quando estiver funcionando, repita o processo no Sheets do cliente
3. Para cada cliente novo, basta copiar o código (mesma Service Account funciona para todos)

---

## 📞 Suporte

Se algo der errado, verifique os logs:
1. No Apps Script, vá em **Execuções** (ícone de relógio ⏱️)
2. Clique na execução que falhou
3. Veja os logs detalhados

Ou abra **Visualizar** → **Registros** no editor do Apps Script.