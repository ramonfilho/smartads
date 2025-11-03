/**
 * ========================================
 * SMART ADS - LEAD SCORING ML AUTOMATION
 * ========================================
 *
 * Sistema automatizado de predições ML e análise UTM
 * Execução diária à meia-noite (00:00) com análises 1D, 3D, 7D
 */

// =============================================================================
// CONFIGURAÇÕES
// =============================================================================

const API_URL = 'https://smart-ads-api-12955519745.us-central1.run.app';
const SERVICE_ACCOUNT_EMAIL = 'smart-ads-451319@appspot.gserviceaccount.com';
const META_ACCOUNT_ID = 'act_188005769808959';  // Los Angeles Producciones LTDA (PRODUÇÃO)

// =============================================================================
// MENU
// =============================================================================

function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu('Smart Ads')
    .addItem('Ativar ML', 'activateML')
    .addSeparator()
    .addItem('Testar Conexão', 'testConnection')
    .addToUi();
}

// =============================================================================
// FUNÇÃO PRINCIPAL: ATIVAR ML
// =============================================================================

/**
 * Ativa sistema ML:
 * 1. Verifica e completa predições dos últimos 7 dias
 * 2. Cria trigger diário para 08:00
 * 3. Executa primeira atualização imediatamente
 */
function activateML() {
  try {
    Logger.log('🚀 Ativando Smart Ads ML...');

    const ui = SpreadsheetApp.getUi();

    // Etapa 1: Completar predições dos últimos 7 dias
    Logger.log('📊 Verificando predições dos últimos 7 dias...');
    const missingBlocks = checkMissingPredictions7D();

    if (missingBlocks.length > 0) {
      Logger.log(`⚠️ Encontrados ${missingBlocks.length} blocos de 24h sem predições`);

      for (let i = 0; i < missingBlocks.length; i++) {
        const block = missingBlocks[i];
        Logger.log(`🔄 Gerando predições ${i+1}/${missingBlocks.length}: ${block.start.toLocaleDateString()}`);
        generatePredictionsFor24hBlock(block.start, block.end);
      }

      Logger.log('✅ Todas as predições dos últimos 7 dias foram geradas');
    } else {
      Logger.log('✅ Todos os últimos 7 dias já possuem predições');
    }

    // Etapa 2: Criar trigger diário às 08:00
    Logger.log('⏰ Configurando execução diária às 08:00...');
    removeDailyTrigger();  // Remove trigger antigo se existir
    createDailyTrigger();

    // Etapa 3: Executar primeira atualização
    Logger.log('🔄 Executando primeira atualização...');
    updateUTMAnalysis();
    updateModelInfoIfChanged();

    Logger.log('✅ Smart Ads ML ativado com sucesso!');

    ui.alert(
      'ML Ativado',
      'Smart Ads ML foi ativado com sucesso!\n\n' +
      '✅ Predições dos últimos 7 dias: OK\n' +
      '✅ Execução diária à meia-noite: Configurada\n' +
      '✅ Análises UTM: Atualizadas\n\n' +
      'O sistema irá rodar automaticamente todos os dias à 00:00 (meia-noite).',
      ui.ButtonSet.OK
    );

  } catch (error) {
    Logger.log(`❌ Erro ao ativar ML: ${error.message}`);
    Logger.log(error.stack);

    SpreadsheetApp.getUi().alert(
      'Erro ao Ativar ML',
      `Não foi possível ativar o sistema:\n${error.message}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );
  }
}

// =============================================================================
// EXECUÇÃO DIÁRIA AUTOMÁTICA (Trigger 00:00)
// =============================================================================

/**
 * Executado diariamente à meia-noite via trigger
 * 1. Gera predições do dia anterior (ontem 00:00 → hoje 00:00)
 * 2. Atualiza análises UTM (1D, 3D, 7D)
 * 3. Atualiza "Info do Modelo" se metadados mudaram
 */
function executeDailyMLUpdate() {
  try {
    Logger.log('🌙 Executando atualização diária ML - ' + new Date().toISOString());

    // Etapa 1: Gerar predições do dia anterior (00:00 → 00:00)
    const now = new Date();
    const yesterday00 = new Date(now);
    yesterday00.setDate(yesterday00.getDate() - 1);
    yesterday00.setHours(0, 0, 0, 0);

    const today00 = new Date(now);
    today00.setHours(0, 0, 0, 0);

    Logger.log(`📅 Gerando predições: ${yesterday00.toLocaleString()} → ${today00.toLocaleString()}`);
    generatePredictionsFor24hBlock(yesterday00, today00);

    // Etapa 2: Atualizar análises UTM
    Logger.log('📊 Atualizando análises UTM...');
    updateUTMAnalysis();

    // Etapa 3: Enviar batch CAPI para leads D10
    Logger.log('📤 Enviando batch CAPI para leads D10...');
    sendCapiBatchForD10Leads(yesterday00, today00);

    // Etapa 4: Atualizar Info do Modelo se necessário
    updateModelInfoIfChanged();

    Logger.log('✅ Atualização diária concluída com sucesso');

  } catch (error) {
    Logger.log(`❌ Erro na atualização diária: ${error.message}`);
    Logger.log(error.stack);

    // Enviar email de erro (opcional)
    const email = Session.getEffectiveUser().getEmail();
    MailApp.sendEmail({
      to: email,
      subject: '❌ Erro Smart Ads ML - Atualização Diária',
      body: `Erro na execução diária de ${new Date().toLocaleString()}:\n\n${error.message}\n\n${error.stack}`
    });
  }
}

// =============================================================================
// FUNÇÕES AUXILIARES: PREDIÇÕES
// =============================================================================

/**
 * Verifica se há blocos de 24h sem predições nos últimos 7 dias
 * Retorna array de blocos faltantes: [{start: Date, end: Date}, ...]
 */
function checkMissingPredictions7D() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('[LF] Pesquisa');
  if (!sheet) throw new Error('Aba "[LF] Pesquisa" não encontrada');

  const values = sheet.getDataRange().getValues();
  if (values.length <= 1) return [];

  const headers = values[0];
  const dataColIndex = headers.indexOf('Data');
  const scoreColIndex = headers.indexOf('lead_score');

  if (dataColIndex === -1) {
    Logger.log('⚠️ Coluna "Data" não encontrada, não é possível verificar predições faltantes');
    return [];
  }

  // Criar blocos de 24h dos últimos 7 dias (excluindo hoje)
  const blocks = [];
  const now = new Date();
  const today8am = new Date(now);
  today8am.setHours(8, 0, 0, 0);

  for (let i = 1; i <= 7; i++) {
    const blockStart = new Date(today8am);
    blockStart.setDate(blockStart.getDate() - i);

    const blockEnd = new Date(blockStart);
    blockEnd.setDate(blockEnd.getDate() + 1);

    blocks.push({ start: blockStart, end: blockEnd });
  }

  // Verificar quais blocos têm leads sem predição
  const missingBlocks = [];

  for (const block of blocks) {
    let hasLeadsWithoutScore = false;

    for (let i = 1; i < values.length; i++) {
      const row = values[i];
      const leadDate = new Date(row[dataColIndex]);
      const hasScore = scoreColIndex !== -1 && row[scoreColIndex];

      // Se lead está no bloco e não tem score
      if (leadDate >= block.start && leadDate < block.end && !hasScore) {
        hasLeadsWithoutScore = true;
        break;
      }
    }

    if (hasLeadsWithoutScore) {
      missingBlocks.push(block);
    }
  }

  return missingBlocks;
}

/**
 * Gera predições para leads em um bloco de 24 horas
 */
function generatePredictionsFor24hBlock(startDate, endDate) {
  Logger.log(`🔄 Gerando predições: ${startDate.toLocaleString()} → ${endDate.toLocaleString()}`);

  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('[LF] Pesquisa');
  if (!sheet) throw new Error('Aba "[LF] Pesquisa" não encontrada');

  const values = sheet.getDataRange().getValues();
  if (values.length <= 1) {
    Logger.log('⚠️ Nenhum dado na planilha');
    return;
  }

  const headers = values[0];
  const dataColIndex = headers.indexOf('Data');
  const scoreColIndex = headers.indexOf('lead_score');

  // Coletar leads do período sem predição
  const leads = [];
  for (let i = 1; i < values.length; i++) {
    const row = values[i];
    const leadDate = new Date(row[dataColIndex]);
    const hasScore = scoreColIndex !== -1 && row[scoreColIndex];

    // Lead está no período e não tem score
    if (leadDate >= startDate && leadDate < endDate && !hasScore) {
      const leadData = {};
      headers.forEach((header, index) => {
        leadData[header] = row[index];
      });

      const emailValue = row[headers.indexOf('E-mail')];
      const email = emailValue ? String(emailValue) : null;

      leads.push({
        data: leadData,
        email: email,
        row_id: (i + 1).toString()
      });
    }
  }

  if (leads.length === 0) {
    Logger.log(`✅ Nenhum lead sem predição no período`);
    return;
  }

  Logger.log(`📊 Processando ${leads.length} leads do período`);

  // Processar em lotes de 600
  const MAX_BATCH_SIZE = 600;
  const batches = [];
  for (let i = 0; i < leads.length; i += MAX_BATCH_SIZE) {
    batches.push(leads.slice(i, i + MAX_BATCH_SIZE));
  }

  Logger.log(`📦 Dividindo em ${batches.length} lotes`);

  let allPredictions = [];

  for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
    const batch = batches[batchIndex];
    Logger.log(`📤 Enviando lote ${batchIndex + 1}/${batches.length} (${batch.length} leads)`);

    const payload = JSON.stringify({ leads: batch });
    const options = {
      method: 'post',
      contentType: 'application/json',
      payload: payload,
      muteHttpExceptions: true
    };

    const response = UrlFetchApp.fetch(`${API_URL}/predict/batch`, options);
    const responseCode = response.getResponseCode();

    if (responseCode !== 200) {
      throw new Error(`API retornou erro ${responseCode}: ${response.getContentText()}`);
    }

    const result = JSON.parse(response.getContentText());
    allPredictions = allPredictions.concat(result.predictions);

    Logger.log(`✅ Lote ${batchIndex + 1} processado: ${result.predictions.length} predições`);

    // Delay entre lotes
    if (batchIndex < batches.length - 1) {
      Utilities.sleep(1000);
    }
  }

  // Escrever predições na planilha
  Logger.log(`💾 Escrevendo ${allPredictions.length} predições na planilha...`);

  if (scoreColIndex === -1) {
    // Adicionar coluna se não existe
    sheet.getRange(1, headers.length + 1).setValue('lead_score');
  }

  const scoreCol = scoreColIndex !== -1 ? scoreColIndex + 1 : headers.length + 1;

  for (const pred of allPredictions) {
    const rowNum = parseInt(pred.row_id);
    sheet.getRange(rowNum, scoreCol).setValue(pred.lead_score);
  }

  SpreadsheetApp.flush();
  Logger.log(`✅ Predições escritas com sucesso`);
}

// =============================================================================
// FUNÇÕES AUXILIARES: ANÁLISE UTM
// =============================================================================

/**
 * Atualiza análises UTM (1D, 3D, 7D) com custos do Meta Ads
 */
function updateUTMAnalysis() {
  try {
    Logger.log('📊 Atualizando análises UTM...');

    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('[LF] Pesquisa');
    if (!sheet) throw new Error('Aba "[LF] Pesquisa" não encontrada');

    // Ler dados da planilha
    const values = sheet.getDataRange().getValues();
    if (values.length <= 1) {
      Logger.log('⚠️ Nenhum dado na planilha');
      return;
    }

    const headers = values[0];

    // Preparar leads para análise
    const leads = [];
    for (let i = 1; i < values.length; i++) {
      const row = values[i];
      const leadData = {};

      headers.forEach((header, index) => {
        leadData[header] = row[index];
      });

      // Formato esperado pela API: {data: {...}}
      leads.push({
        data: leadData
      });
    }

    Logger.log(`📋 Enviando ${leads.length} leads para análise...`);

    // Chamar API de análise UTM
    const payload = JSON.stringify({
      leads: leads,
      account_id: META_ACCOUNT_ID
    });

    const options = {
      method: 'post',
      contentType: 'application/json',
      payload: payload,
      muteHttpExceptions: true
    };

    const response = UrlFetchApp.fetch(`${API_URL}/analyze_utms_with_costs`, options);
    const responseCode = response.getResponseCode();

    if (responseCode !== 200) {
      throw new Error(`API retornou erro: ${responseCode} - ${response.getContentText()}`);
    }

    const result = JSON.parse(response.getContentText());

    Logger.log(`✅ Análise recebida: ${result.processing_time_seconds}s`);
    Logger.log(`   Períodos: ${Object.keys(result.periods).join(', ')}`);

    // Criar abas para períodos 1D, 3D, 7D (sem Total)
    const periods = ['1D', '3D', '7D'];

    // IMPORTANTE: Processar cada aba separadamente com tratamento de erro individual
    // Se uma aba falhar, as outras ainda serão criadas
    for (const period of periods) {
      if (result.periods[period]) {
        try {
          Logger.log(`📝 Processando aba ${period}...`);
          writeAnalysisSheet(period, result.periods[period], result.config);
          Logger.log(`✅ Aba ${period} criada com sucesso`);
        } catch (periodError) {
          Logger.log(`❌ Erro ao criar aba ${period}: ${periodError.message}`);
          // Não throw - continuar processando outras abas
        }
      }
    }

    Logger.log('✅ Análises UTM atualizadas');

  } catch (error) {
    Logger.log(`❌ Erro ao atualizar análises UTM: ${error.message}`);
    throw error;
  }
}

/**
 * Atualiza aba "Info do Modelo" apenas se metadados mudaram
 */
function updateModelInfoIfChanged() {
  try {
    Logger.log('📊 Verificando atualização da Info do Modelo...');

    // Buscar metadados atuais da API
    const response = UrlFetchApp.fetch(`${API_URL}/model/info`, {
      method: 'get',
      muteHttpExceptions: true
    });

    if (response.getResponseCode() !== 200) {
      Logger.log('⚠️ Não foi possível obter informações do modelo');
      return;
    }

    const modelInfo = JSON.parse(response.getContentText());
    const currentModelName = modelInfo.model_info.model_name;
    const currentTrainedAt = modelInfo.model_info.trained_at;

    // Verificar se aba existe e tem metadados salvos
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    let infoSheet = ss.getSheetByName('Info do Modelo');

    if (!infoSheet) {
      // Aba não existe, criar
      Logger.log('📋 Aba "Info do Modelo" não existe, criando...');
      writeModelInfoSheet(modelInfo);

      // Salvar metadados na aba (hidden row)
      infoSheet = ss.getSheetByName('Info do Modelo');
      infoSheet.getRange('Z1').setValue(currentModelName);
      infoSheet.getRange('Z2').setValue(currentTrainedAt);
      infoSheet.hideRows(1, 1);

      Logger.log('✅ Aba "Info do Modelo" criada');
      return;
    }

    // Verificar se metadados mudaram
    const savedModelName = infoSheet.getRange('Z1').getValue();
    const savedTrainedAt = infoSheet.getRange('Z2').getValue();

    if (savedModelName === currentModelName && savedTrainedAt === currentTrainedAt) {
      Logger.log('✅ Metadados do modelo não mudaram, aba não precisa atualização');
      return;
    }

    // Metadados mudaram, recriar aba
    Logger.log(`🔄 Metadados mudaram: ${savedModelName} → ${currentModelName}`);
    writeModelInfoSheet(modelInfo);

    // Atualizar metadados salvos
    infoSheet = ss.getSheetByName('Info do Modelo');
    infoSheet.getRange('Z1').setValue(currentModelName);
    infoSheet.getRange('Z2').setValue(currentTrainedAt);

    Logger.log('✅ Aba "Info do Modelo" atualizada');

  } catch (error) {
    Logger.log(`⚠️ Erro ao verificar Info do Modelo: ${error.message}`);
    // Não lançar erro, apenas logar
  }
}

// =============================================================================
// FUNÇÕES AUXILIARES: TRIGGERS
// =============================================================================

/**
 * Cria trigger diário para executar à meia-noite (00:00)
 */
function createDailyTrigger() {
  ScriptApp.newTrigger('executeDailyMLUpdate')
    .timeBased()
    .atHour(0)  // Meia-noite (00:00)
    .everyDays(1)
    .create();

  Logger.log('✅ Trigger diário criado para 00:00 (meia-noite)');
}

/**
 * Remove trigger diário existente
 */
function removeDailyTrigger() {
  const triggers = ScriptApp.getProjectTriggers();

  for (const trigger of triggers) {
    if (trigger.getHandlerFunction() === 'executeDailyMLUpdate') {
      ScriptApp.deleteTrigger(trigger);
      Logger.log('🗑️ Trigger diário removido');
    }
  }
}

// =============================================================================
// FUNÇÕES AUXILIARES: VISUALIZAÇÃO
// =============================================================================

/**
 * Escreve aba de análise UTM para um período
 */
function writeAnalysisSheet(period, periodData, config) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheetName = `Análise UTM - ${period}`;

  // Deletar aba se já existir (com tratamento robusto)
  try {
    let sheet = ss.getSheetByName(sheetName);
    if (sheet) {
      Logger.log(`🗑️ Deletando aba existente: ${sheetName}`);
      ss.deleteSheet(sheet);
      SpreadsheetApp.flush();  // Garantir que deleção foi aplicada
      Utilities.sleep(500);     // Pequeno delay para evitar conflito
    }
  } catch (deleteError) {
    Logger.log(`⚠️ Erro ao deletar aba ${sheetName}: ${deleteError.message}`);
    // Continuar mesmo se não conseguir deletar
  }

  // Criar nova aba
  const sheet = ss.insertSheet(sheetName);
  Logger.log(`📝 Criando aba: ${sheetName}`);

  // =============================================================================
  // SEÇÃO DE METADADOS DO PERÍODO
  // =============================================================================
  let headerRow = 1;

  // Linha 1: Período analisado
  if (periodData.period_start && periodData.period_end) {
    const periodStart = new Date(periodData.period_start);
    const periodEnd = new Date(periodData.period_end);

    // Formatar datas no formato brasileiro
    const formatDate = (date) => {
      const day = String(date.getDate()).padStart(2, '0');
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const year = date.getFullYear();
      const hours = String(date.getHours()).padStart(2, '0');
      const minutes = String(date.getMinutes()).padStart(2, '0');
      return `${day}/${month}/${year} ${hours}:${minutes}`;
    };

    const periodCell = sheet.getRange(headerRow, 1, 1, 12);
    periodCell.merge();
    periodCell.setValue(`📅 Período: ${formatDate(periodStart)} até ${formatDate(periodEnd)}`);
    periodCell.setFontWeight('bold');
    periodCell.setFontSize(11);
    periodCell.setBackground('#E8F0FE');
    periodCell.setHorizontalAlignment('center');
    headerRow++;
  }

  // Linha 2: Contadores de leads
  if (periodData.total_leads !== undefined) {
    const metaLeads = periodData.meta_leads || 0;
    const googleLeads = periodData.google_leads || 0;
    const totalLeads = periodData.total_leads || 0;

    const countersCell = sheet.getRange(headerRow, 1, 1, 12);
    countersCell.merge();
    countersCell.setValue(`📊 Leads analisados: ${totalLeads} (Meta: ${metaLeads}, Google: ${googleLeads})`);
    countersCell.setFontWeight('bold');
    countersCell.setFontSize(10);
    countersCell.setBackground('#F1F3F4');
    countersCell.setHorizontalAlignment('center');
    headerRow++;
  }

  // Linha 3: Espaço em branco
  headerRow++;

  // =============================================================================
  // CABEÇALHOS DA TABELA
  // =============================================================================
  const headers = [
    'Campaign', 'Adset', 'Ad', 'Leads', 'Gasto (R$)', 'CPL (R$)',
    'Taxa Proj. (%)', 'Receita Proj. (R$)', 'Margem Contrib (R$)', 'ROAS Proj.',
    'Orç. Atual (R$)', 'Orç. Alvo (R$)', 'Ação'
  ];

  sheet.getRange(headerRow, 1, 1, headers.length).setValues([headers]);

  // Formatação do cabeçalho
  const headerRange = sheet.getRange(headerRow, 1, 1, headers.length);
  headerRange.setFontWeight('bold');
  headerRange.setBackground('#4285F4');
  headerRange.setFontColor('#FFFFFF');
  headerRange.setHorizontalAlignment('center');

  let currentRow = headerRow + 1;

  // =============================================================================
  // OTIMIZAÇÃO: Coletar todos os dados primeiro, depois escrever em LOTE
  // =============================================================================

  const allRowsData = [];        // Dados das células
  const rowBackgrounds = [];     // Cores de fundo por linha
  const acaoFormatting = [];     // Formatação especial da coluna Ação

  // Dimensões (ordem: campaign, medium, ad, google_ads)
  const dimensions = ['campaign', 'medium', 'ad', 'google_ads'];

  for (const dimension of dimensions) {
    const metrics = periodData[dimension];

    if (!metrics || metrics.length === 0) {
      continue;
    }

    // Adicionar título destacado para Google Ads
    if (dimension === 'google_ads' && metrics.length > 0) {
      // Linha vazia antes do título
      allRowsData.push(Array(13).fill(''));
      rowBackgrounds.push(Array(13).fill('#FFFFFF'));
      acaoFormatting.push(null);

      // Título Google Ads (será mesclado depois)
      allRowsData.push(['🔍 GOOGLE ADS (sem custos Meta - plataforma diferente)', ...Array(12).fill('')]);
      rowBackgrounds.push(Array(13).fill('#FFF3E0'));
      acaoFormatting.push(null);
    }

    for (const metric of metrics) {
      // Montar row baseado na dimensão
      let row;
      let backgroundColor;  // Cor de fundo por seção

      if (dimension === 'campaign') {
        row = [
          metric.value,           // Campaign
          '',                     // Adset (vazio)
          '',                     // Ad (vazio)
          metric.leads, metric.spend, metric.cpl,
          metric.taxa_proj * 100, metric.receita_proj, metric.margem_contrib, metric.roas_proj,
          metric.budget_current, metric.budget_target,
          metric.acao
        ];
        backgroundColor = '#E8F5E9';  // Verde claro para campaigns
      } else if (dimension === 'medium') {
        row = [
          metric.campaign || '',  // Campaign
          metric.value,           // Adset
          '',                     // Ad (vazio)
          metric.leads, metric.spend, metric.cpl,
          metric.taxa_proj * 100, metric.receita_proj, metric.margem_contrib, metric.roas_proj,
          metric.budget_current, metric.budget_target,
          metric.acao
        ];
        backgroundColor = '#FFF3E0';  // Laranja claro para adsets
      } else if (dimension === 'ad') {
        row = [
          metric.campaign || '',  // Campaign
          metric.adset || '',     // Adset
          metric.value,           // Ad
          metric.leads, metric.spend, metric.cpl,
          metric.taxa_proj * 100, metric.receita_proj, metric.margem_contrib, metric.roas_proj,
          metric.budget_current, metric.budget_target,
          metric.acao
        ];
        backgroundColor = '#E3F2FD';  // Azul claro para ads
      } else { // google_ads
        row = [
          '',                     // Campaign (vazio)
          '',                     // Adset (vazio)
          metric.value,           // Keyword
          metric.leads, metric.spend, metric.cpl,
          metric.taxa_proj * 100, metric.receita_proj, metric.margem_contrib, metric.roas_proj,
          metric.budget_current, metric.budget_target,
          metric.acao
        ];
        backgroundColor = '#F3E5F5';  // Roxo claro para Google Ads
      }

      allRowsData.push(row);
      rowBackgrounds.push(Array(13).fill(backgroundColor));

      // Determinar formatação da coluna Ação
      let acaoColor = null;
      if (metric.acao === 'ABO' || metric.acao === 'Manter' || metric.acao === 'CBO - Manter' || metric.acao.includes('Aguardar dados')) {
        acaoColor = { bg: '#E0E0E0', fg: '#666666' };  // Cinza neutro
      } else if (metric.acao === 'CBO - Pausar / Alterar' || metric.acao.includes('Pausar')) {
        acaoColor = { bg: '#EA4335', fg: '#FFFFFF' };  // Vermelho para pausar
      } else if (metric.acao.includes('Aumentar')) {
        const match = metric.acao.match(/Aumentar (\d+)/);
        if (match && parseInt(match[1]) > 30) {
          acaoColor = { bg: '#34A853', fg: '#FFFFFF' };
        } else {
          acaoColor = { bg: '#FBBC04', fg: '#000000' };
        }
      } else if (metric.acao.includes('Reduzir') || metric.acao === 'Remover') {
        acaoColor = { bg: '#EA4335', fg: '#FFFFFF' };
      } else {
        acaoColor = { bg: '#E0E0E0', fg: '#666666' };
      }
      acaoFormatting.push(acaoColor);
    }

    // Linha vazia de separação entre dimensões
    allRowsData.push(Array(13).fill(''));
    rowBackgrounds.push(Array(13).fill('#FFFFFF'));
    acaoFormatting.push(null);
  }

  // Escrever TODOS os dados de uma vez (MUITO mais rápido!)
  if (allRowsData.length > 0) {
    const dataRange = sheet.getRange(currentRow, 1, allRowsData.length, 13);
    dataRange.setValues(allRowsData);
    Logger.log(`✅ Escreveu ${allRowsData.length} linhas em lote`);

    SpreadsheetApp.flush();  // Forçar aplicação

    // Aplicar formatações em lote
    dataRange.setBackgrounds(rowBackgrounds);

    // Aplicar formatação especial da coluna Ação
    for (let i = 0; i < acaoFormatting.length; i++) {
      const fmt = acaoFormatting[i];
      if (fmt) {
        const acaoCell = sheet.getRange(currentRow + i, 13);
        acaoCell.setBackground(fmt.bg);
        acaoCell.setFontColor(fmt.fg);
        acaoCell.setFontWeight('bold');
      }
    }

    currentRow += allRowsData.length;
    SpreadsheetApp.flush();  // Forçar aplicação de formatação
  }

  // Formatar colunas numéricas EM LOTE (muito mais rápido!)
  const lastRow = currentRow - 1;
  const firstDataRow = headerRow + 1;
  if (lastRow >= firstDataRow) {
    const numDataRows = lastRow - firstDataRow + 1;

    // Formato moeda: Gasto, CPL, Receita Proj, Margem Contrib, Orç. Atual, Orç. Alvo
    sheet.getRange(firstDataRow, 5, numDataRows, 1).setNumberFormat('R$ #,##0.00');  // Gasto
    sheet.getRange(firstDataRow, 6, numDataRows, 1).setNumberFormat('R$ #,##0.00');  // CPL
    sheet.getRange(firstDataRow, 8, numDataRows, 1).setNumberFormat('R$ #,##0.00');  // Receita Proj
    sheet.getRange(firstDataRow, 9, numDataRows, 1).setNumberFormat('R$ #,##0.00');  // Margem Contrib
    sheet.getRange(firstDataRow, 11, numDataRows, 1).setNumberFormat('R$ #,##0.00'); // Orç. Atual
    sheet.getRange(firstDataRow, 12, numDataRows, 1).setNumberFormat('R$ #,##0.00'); // Orç. Alvo

    // Percentual: Taxa Proj
    sheet.getRange(firstDataRow, 7, numDataRows, 1).setNumberFormat('0.00"%"');  // Taxa Proj

    // ROAS
    sheet.getRange(firstDataRow, 10, numDataRows, 1).setNumberFormat('0.00"x"');  // ROAS Proj

    SpreadsheetApp.flush();  // Forçar aplicação dos formatos numéricos

    // Destacar Margem Contrib (coluna 9) com cores - EM LOTE
    const margemValues = sheet.getRange(firstDataRow, 9, numDataRows, 1).getValues();
    const margemBackgrounds = [];
    const margemFontWeights = [];

    for (let i = 0; i < margemValues.length; i++) {
      const margemValue = margemValues[i][0];
      if (margemValue > 0) {
        margemBackgrounds.push(['#D4EDDA']);  // Verde claro (lucrativa)
        margemFontWeights.push(['bold']);
      } else if (margemValue < 0) {
        margemBackgrounds.push(['#F8D7DA']);  // Vermelho claro (prejuízo)
        margemFontWeights.push(['bold']);
      } else {
        margemBackgrounds.push(['#FFFFFF']);  // Branco (neutro)
        margemFontWeights.push(['normal']);
      }
    }

    sheet.getRange(firstDataRow, 9, numDataRows, 1).setBackgrounds(margemBackgrounds);
    sheet.getRange(firstDataRow, 9, numDataRows, 1).setFontWeights(margemFontWeights);

    SpreadsheetApp.flush();  // Forçar aplicação da formatação de margem
  }

  // Ajustar largura das colunas
  for (let i = 1; i <= headers.length; i++) {
    sheet.autoResizeColumn(i);
  }

  // Adicionar nota com configuração
  sheet.getRange(lastRow + 2, 1).setValue(`Configuração: Product Value = R$ ${config.product_value.toFixed(2)} | ROAS Mínimo de Segurança = 2.5x | CAP Variação Máxima = 80%`);
  sheet.getRange(lastRow + 2, 1).setFontStyle('italic');
  sheet.getRange(lastRow + 2, 1).setFontColor('#666666');

  Logger.log(`✅ Aba ${sheetName} criada com ${lastRow - 1} registros`);
}

/**
 * Escreve aba "Info do Modelo" com metadados e feature importances
 */
function writeModelInfoSheet(modelInfo) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheetName = 'Info do Modelo';

  // Deletar aba se já existir
  let sheet = ss.getSheetByName(sheetName);
  if (sheet) {
    ss.deleteSheet(sheet);
  }

  // Criar nova aba
  sheet = ss.insertSheet(sheetName);

  Logger.log('📊 Criando aba: Info do Modelo');

  let currentRow = 1;

  // === SEÇÃO 1: INFORMAÇÕES DO MODELO ===
  sheet.getRange(currentRow, 1).setValue('📋 INFORMAÇÕES DO MODELO');
  sheet.getRange(currentRow, 1).setFontWeight('bold');
  sheet.getRange(currentRow, 1).setFontSize(14);
  sheet.getRange(currentRow, 1).setBackground('#4285F4');
  sheet.getRange(currentRow, 1).setFontColor('#FFFFFF');
  currentRow += 2;

  const modelInfo_data = modelInfo.model_info || {};
  const infoRows = [
    ['Nome do Modelo:', modelInfo_data.model_name || 'N/A'],
    ['Tipo:', modelInfo_data.model_type || 'N/A'],
    ['Biblioteca:', `${modelInfo_data.library || 'N/A'} ${modelInfo_data.library_version || ''}`],
    ['Data de Treinamento:', modelInfo_data.trained_at ? new Date(modelInfo_data.trained_at).toLocaleString('pt-BR') : 'N/A'],
    ['Split:', modelInfo_data.split_type || 'N/A']
  ];

  for (const [label, value] of infoRows) {
    sheet.getRange(currentRow, 1).setValue(label);
    sheet.getRange(currentRow, 1).setFontWeight('bold');
    sheet.getRange(currentRow, 2).setValue(value);
    currentRow++;
  }

  currentRow += 2;

  // === SEÇÃO 2: DADOS DE TREINAMENTO ===
  sheet.getRange(currentRow, 1).setValue('📊 DADOS DE TREINAMENTO');
  sheet.getRange(currentRow, 1).setFontWeight('bold');
  sheet.getRange(currentRow, 1).setFontSize(14);
  sheet.getRange(currentRow, 1).setBackground('#34A853');
  sheet.getRange(currentRow, 1).setFontColor('#FFFFFF');
  currentRow += 2;

  const trainingData = modelInfo.training_data || {};
  const temporalSplit = trainingData.temporal_split || {};
  const targetDist = trainingData.target_distribution || {};

  const trainingRows = [
    ['Total de Registros:', trainingData.total_records || 'N/A'],
    ['Registros de Treino:', trainingData.training_records || 'N/A'],
    ['Registros de Teste:', trainingData.test_records || 'N/A'],
    ['Número de Features:', trainingData.features_count || 'N/A'],
    ['Período:', `${temporalSplit.period_start || 'N/A'} a ${temporalSplit.period_end || 'N/A'}`],
    ['Data de Corte:', temporalSplit.cut_date || 'N/A'],
    ['Taxa de Conversão (Treino):', targetDist.training_positive_rate ? (targetDist.training_positive_rate * 100).toFixed(2) + '%' : 'N/A'],
    ['Taxa de Conversão (Teste):', targetDist.test_positive_rate ? (targetDist.test_positive_rate * 100).toFixed(2) + '%' : 'N/A']
  ];

  for (const [label, value] of trainingRows) {
    sheet.getRange(currentRow, 1).setValue(label);
    sheet.getRange(currentRow, 1).setFontWeight('bold');
    sheet.getRange(currentRow, 2).setValue(value);
    currentRow++;
  }

  currentRow += 2;

  // === SEÇÃO 3: MÉTRICAS DE PERFORMANCE ===
  sheet.getRange(currentRow, 1).setValue('🎯 MÉTRICAS DE PERFORMANCE');
  sheet.getRange(currentRow, 1).setFontWeight('bold');
  sheet.getRange(currentRow, 1).setFontSize(14);
  sheet.getRange(currentRow, 1).setBackground('#FBBC04');
  sheet.getRange(currentRow, 1).setFontColor('#000000');
  currentRow += 2;

  const performance = modelInfo.performance_metrics || {};
  const perfRows = [
    ['AUC:', performance.auc ? performance.auc.toFixed(4) : 'N/A'],
    ['Lift Máximo:', performance.lift_maximum ? performance.lift_maximum.toFixed(2) + 'x' : 'N/A'],
    ['Concentração Top 3 Decis:', performance.top3_decil_concentration ? performance.top3_decil_concentration.toFixed(2) + '%' : 'N/A'],
    ['Concentração Top 5 Decis:', performance.top5_decil_concentration ? performance.top5_decil_concentration.toFixed(2) + '%' : 'N/A'],
    ['Monotonia:', performance.monotonia_percentage ? performance.monotonia_percentage.toFixed(1) + '%' : 'N/A']
  ];

  for (const [label, value] of perfRows) {
    sheet.getRange(currentRow, 1).setValue(label);
    sheet.getRange(currentRow, 1).setFontWeight('bold');
    sheet.getRange(currentRow, 2).setValue(value);
    currentRow++;
  }

  currentRow += 2;

  // === SEÇÃO 4: ANÁLISE POR DECIL ===
  sheet.getRange(currentRow, 1).setValue('📈 ANÁLISE POR DECIL');
  sheet.getRange(currentRow, 1).setFontWeight('bold');
  sheet.getRange(currentRow, 1).setFontSize(14);
  sheet.getRange(currentRow, 1).setBackground('#EA4335');
  sheet.getRange(currentRow, 1).setFontColor('#FFFFFF');
  currentRow += 2;

  const decilHeaders = ['Decil', 'Leads', 'Conversões', 'Taxa Conv.', '% Total Conv.', 'Lift'];
  sheet.getRange(currentRow, 1, 1, decilHeaders.length).setValues([decilHeaders]);
  sheet.getRange(currentRow, 1, 1, decilHeaders.length).setFontWeight('bold');
  sheet.getRange(currentRow, 1, 1, decilHeaders.length).setBackground('#666666');
  sheet.getRange(currentRow, 1, 1, decilHeaders.length).setFontColor('#FFFFFF');
  currentRow++;

  const decilAnalysis = modelInfo.decil_analysis || {};
  for (let i = 1; i <= 10; i++) {
    const decilKey = `decil_${i}`;
    const decilData = decilAnalysis[decilKey] || {};

    const row = [
      `D${i}`,
      decilData.total_leads || 0,
      decilData.conversions || 0,
      decilData.conversion_rate ? (decilData.conversion_rate * 100).toFixed(2) + '%' : '0.00%',
      decilData.pct_total_conversions ? decilData.pct_total_conversions.toFixed(2) + '%' : '0.00%',
      decilData.lift ? decilData.lift.toFixed(2) + 'x' : '0.00x'
    ];

    sheet.getRange(currentRow, 1, 1, row.length).setValues([row]);
    currentRow++;
  }

  currentRow += 2;

  // === SEÇÃO 5: TOP 20 FEATURE IMPORTANCES ===
  sheet.getRange(currentRow, 1).setValue('🔍 TOP 20 FEATURES MAIS IMPORTANTES');
  sheet.getRange(currentRow, 1).setFontWeight('bold');
  sheet.getRange(currentRow, 1).setFontSize(14);
  sheet.getRange(currentRow, 1).setBackground('#9C27B0');
  sheet.getRange(currentRow, 1).setFontColor('#FFFFFF');
  currentRow += 2;

  const featureHeaders = ['Rank', 'Feature', 'Importância'];
  sheet.getRange(currentRow, 1, 1, featureHeaders.length).setValues([featureHeaders]);
  sheet.getRange(currentRow, 1, 1, featureHeaders.length).setFontWeight('bold');
  sheet.getRange(currentRow, 1, 1, featureHeaders.length).setBackground('#666666');
  sheet.getRange(currentRow, 1, 1, featureHeaders.length).setFontColor('#FFFFFF');
  currentRow++;

  const featureImportances = modelInfo.feature_importances || [];
  for (let i = 0; i < featureImportances.length; i++) {
    const feature = featureImportances[i];
    const row = [
      i + 1,
      feature.feature || 'N/A',
      feature.importance ? (feature.importance * 100).toFixed(2) + '%' : '0.00%'
    ];

    sheet.getRange(currentRow, 1, 1, row.length).setValues([row]);
    currentRow++;
  }

  // Ajustar largura das colunas
  for (let i = 1; i <= 6; i++) {
    sheet.autoResizeColumn(i);
  }

  Logger.log('✅ Aba "Info do Modelo" criada com sucesso');
}

// =============================================================================
// FUNÇÕES DE DEBUG
// =============================================================================

/**
 * Testa conexão com a API
 */
function testConnection() {
  try {
    Logger.log('🔍 Testando conexão com API...');

    const response = UrlFetchApp.fetch(`${API_URL}/health`);
    const result = JSON.parse(response.getContentText());

    Logger.log('✅ Conexão bem-sucedida!');
    Logger.log(`Status: ${result.status}`);
    Logger.log(`Pipeline: ${result.pipeline_status}`);
    Logger.log(`Modelo: ${result.model_loaded}`);
    Logger.log(`Versão: ${result.version}`);

    SpreadsheetApp.getUi().alert(
      'Conexão OK',
      `API está funcionando!\n\n` +
      `Status: ${result.status}\n` +
      `Pipeline: ${result.pipeline_status}\n` +
      `Modelo Carregado: ${result.model_loaded}\n` +
      `Versão: ${result.version}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );

  } catch (error) {
    Logger.log(`❌ Erro ao testar conexão: ${error.message}`);

    SpreadsheetApp.getUi().alert(
      'Erro de Conexão',
      `Não foi possível conectar à API:\n${error.message}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );
  }
}

// =============================================================================
// CAPI: ENVIO DE BATCH PARA LEADS D10
// =============================================================================

/**
 * Envia leads D10 do período para API processar batch CAPI
 * Chamado diariamente após classificação ML
 */
function sendCapiBatchForD10Leads(startDate, endDate) {
  try {
    Logger.log(`📤 Enviando batch CAPI: ${startDate.toLocaleString()} → ${endDate.toLocaleString()}`);

    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('[LF] Pesquisa');
    if (!sheet) {
      Logger.log('⚠️ Aba "[LF] Pesquisa" não encontrada');
      return;
    }

    const values = sheet.getDataRange().getValues();
    if (values.length <= 1) {
      Logger.log('⚠️ Nenhum dado na planilha');
      return;
    }

    const headers = values[0];
    const dataColIndex = headers.indexOf('Data');
    const emailColIndex = headers.indexOf('E-mail');
    const phoneColIndex = headers.indexOf('Telefone');
    const scoreColIndex = headers.indexOf('lead_score');
    const decilColIndex = headers.indexOf('decil');

    // Coletar leads D10 do período
    const leadsD10 = [];
    for (let i = 1; i < values.length; i++) {
      const row = values[i];
      const leadDate = new Date(row[dataColIndex]);
      const decil = row[decilColIndex];
      const leadScore = row[scoreColIndex];
      const email = row[emailColIndex];

      // Lead está no período e é D10
      if (leadDate >= startDate && leadDate < endDate && decil === 'D10') {
        leadsD10.push({
          email: email,
          phone: row[phoneColIndex],
          lead_score: leadScore,
          decil: decil,
          data: Utilities.formatDate(leadDate, Session.getScriptTimeZone(), "yyyy-MM-dd'T'HH:mm:ss")
        });
      }
    }

    if (leadsD10.length === 0) {
      Logger.log('✅ Nenhum lead D10 no período');
      return;
    }

    Logger.log(`📊 ${leadsD10.length} leads D10 encontrados, enviando para API...`);

    // Enviar para API
    const API_URL = 'https://smart-ads-api-12955519745.us-central1.run.app';
    const payload = {
      leads_d10: leadsD10
    };

    const options = {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify(payload),
      muteHttpExceptions: true
    };

    const response = UrlFetchApp.fetch(`${API_URL}/capi/process_daily_batch`, options);
    const responseCode = response.getResponseCode();
    const responseBody = response.getContentText();

    if (responseCode === 200) {
      const result = JSON.parse(responseBody);
      Logger.log(`✅ Batch CAPI enviado: ${result.success}/${result.total} eventos com sucesso`);
      Logger.log(`   Leads com dados CAPI: ${result.leads_with_capi_data}`);
    } else {
      Logger.log(`❌ Erro no batch CAPI: ${responseCode} - ${responseBody}`);
    }

  } catch (error) {
    Logger.log(`❌ Erro ao enviar batch CAPI: ${error.message}`);
    Logger.log(error.stack);
  }
}
