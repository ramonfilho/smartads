const API_URL = 'https://smart-ads-api-12955519745.us-central1.run.app/predict/batch';
const SERVICE_ACCOUNT_EMAIL = 'smart-ads-451319@appspot.gserviceaccount.com';

function getPredictions() {
  try {
    Logger.log('🚀 Iniciando busca de predições...');

    const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = spreadsheet.getSheetByName('[LF] Pesquisa');

    if (!sheet) {
      throw new Error('Aba "[LF] Pesquisa" não encontrada!');
    }

    const lastRow = sheet.getMaxRows();
    const lastCol = sheet.getMaxColumns();
    const dataRange = sheet.getRange(1, 1, lastRow, lastCol);
    const values = dataRange.getValues();

    const nonEmptyValues = values.filter((row, index) => {
      if (index === 0) return true;
      return row.some(cell => cell !== null && cell !== undefined && cell !== '');
    });

    Logger.log(`📊 Linhas totais na planilha: ${lastRow}, após filtrar vazias: ${nonEmptyValues.length}`);

    if (nonEmptyValues.length <= 1) {
      throw new Error('Planilha vazia ou só tem cabeçalho');
    }

    const headers = nonEmptyValues[0];
    Logger.log(`📋 Encontrados ${headers.length} campos: ${headers.join(', ')}`);

    const leadScoreColCheck = headers.indexOf('lead_score');
    const dataColIndex = headers.indexOf('Data');

    // Calcular timestamp de 24 horas atrás
    const now = new Date();
    const twentyFourHoursAgo = new Date(now.getTime() - (24 * 60 * 60 * 1000));

    const leads = [];
    for (let i = 1; i < nonEmptyValues.length; i++) {
      const row = nonEmptyValues[i];

      // Pular se lead já tem score
      if (leadScoreColCheck !== -1 && row[leadScoreColCheck]) {
        continue;
      }

      // Pular se lead não é das últimas 24 horas
      if (dataColIndex !== -1 && row[dataColIndex]) {
        const leadDate = new Date(row[dataColIndex]);
        if (leadDate < twentyFourHoursAgo) {
          continue;
        }
      }

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

    if (leads.length === 0) {
      Logger.log('✅ Nenhum lead novo para processar nas últimas 24h');
      SpreadsheetApp.getUi().alert(
        'Sem leads novos',
        'Nenhum lead novo das últimas 24 horas para processar.',
        SpreadsheetApp.getUi().ButtonSet.OK
      );
      return;
    }

    Logger.log(`📊 Processando ${leads.length} leads das últimas 24h sem predições`);

    const MAX_BATCH_SIZE = 600;
    const numBatches = Math.ceil(leads.length / MAX_BATCH_SIZE);
    const idealBatchSize = Math.ceil(leads.length / numBatches);

    Logger.log(`🔄 Dividindo ${leads.length} leads em ${numBatches} lotes equilibrados (~${idealBatchSize} leads cada)`);

    const batches = [];
    for (let i = 0; i < leads.length; i += idealBatchSize) {
      batches.push(leads.slice(i, i + idealBatchSize));
    }

    const batchSizes = batches.map(b => b.length).join(', ');
    Logger.log(`📦 Tamanhos dos lotes: [${batchSizes}]`);

    let allPredictions = [];
    let totalProcessingTime = 0;

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      Logger.log(`📦 Processando lote ${batchIndex + 1}/${batches.length} (${batch.length} leads)...`);

      const options = {
        method: 'post',
        contentType: 'application/json',
        payload: JSON.stringify({
          leads: batch
        }),
        muteHttpExceptions: true
      };

      const response = UrlFetchApp.fetch(API_URL, options);
      const statusCode = response.getResponseCode();

      if (statusCode !== 200) {
        throw new Error(`API retornou erro ${statusCode} no lote ${batchIndex + 1}: ${response.getContentText()}`);
      }

      const result = JSON.parse(response.getContentText());
      allPredictions = allPredictions.concat(result.predictions);
      totalProcessingTime += result.processing_time_seconds;

      Logger.log(`✅ Lote ${batchIndex + 1} concluído em ${result.processing_time_seconds}s`);

      if (batchIndex < batches.length - 1) {
        Utilities.sleep(500);
      }
    }

    Logger.log(`✅ Total: ${allPredictions.length} predições em ${totalProcessingTime.toFixed(2)}s`);

    let leadScoreCol = headers.indexOf('lead_score');
    let decileCol = headers.indexOf('decile');
    let timestampCol = headers.indexOf('data_processamento');

    if (leadScoreCol === -1) {
      leadScoreCol = headers.length;
      sheet.getRange(1, leadScoreCol + 1).setValue('lead_score');
      headers.push('lead_score');
    }

    if (decileCol === -1) {
      decileCol = headers.length;
      sheet.getRange(1, decileCol + 1).setValue('decile');
      headers.push('decile');
    }

    if (timestampCol === -1) {
      timestampCol = headers.length;
      sheet.getRange(1, timestampCol + 1).setValue('data_processamento');
      headers.push('data_processamento');
    }

    Logger.log('📝 Escrevendo predições na planilha...');

    const currentTimestamp = new Date();

    allPredictions.forEach(pred => {
      const rowNum = parseInt(pred.row_id);
      try {
        const scoreCell = sheet.getRange(rowNum, leadScoreCol + 1);
        scoreCell.setValue(pred.lead_score);
        scoreCell.setNumberFormat('0.0000');

        sheet.getRange(rowNum, decileCol + 1).setValue(pred.decile);
        sheet.getRange(rowNum, timestampCol + 1).setValue(currentTimestamp);
      } catch (e) {
        Logger.log(`⚠️ Erro ao escrever linha ${rowNum}: ${e.message}`);
      }
    });

    Logger.log('✅ Predições escritas com sucesso');

    const minScore = Math.min(...allPredictions.map(p => p.lead_score));
    const maxScore = Math.max(...allPredictions.map(p => p.lead_score));

    SpreadsheetApp.getUi().alert(
      'Sucesso!',
      `${allPredictions.length} leads processados em ${totalProcessingTime.toFixed(2)}s\n` +
      `Processados em ${batches.length} lote(s)\n\n` +
      `Scores: ${minScore.toFixed(3)} - ${maxScore.toFixed(3)}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );

  } catch (error) {
    Logger.log(`❌ Erro: ${error.message}`);
    Logger.log(error.stack);

    SpreadsheetApp.getUi().alert(
      'Erro',
      `Falha ao buscar predições:\n${error.message}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );

    throw error;
  }
}

function createDailyTrigger() {
  // Remove triggers existentes da função getPredictions
  const triggers = ScriptApp.getProjectTriggers();
  triggers.forEach(trigger => {
    if (trigger.getHandlerFunction() === 'getPredictions') {
      ScriptApp.deleteTrigger(trigger);
    }
  });

  // Criar novo trigger diário às 8h da manhã
  ScriptApp.newTrigger('getPredictions')
    .timeBased()
    .atHour(8)
    .everyDays(1)
    .create();

  Logger.log('✅ Trigger criado: getPredictions rodará todo dia às 8h');

  SpreadsheetApp.getUi().alert(
    'Agendamento configurado!',
    'A função getPredictions será executada automaticamente todo dia às 8h da manhã.\n\n' +
    'Apenas leads das últimas 24 horas sem predições serão processados.',
    SpreadsheetApp.getUi().ButtonSet.OK
  );
}

// Manter função antiga para compatibilidade (agora cria trigger diário)
function createTimeDrivenTrigger() {
  createDailyTrigger();
}

function removeTimeDrivenTrigger() {
  const triggers = ScriptApp.getProjectTriggers();
  triggers.forEach(trigger => {
    if (trigger.getHandlerFunction() === 'getPredictions') {
      ScriptApp.deleteTrigger(trigger);
    }
  });

  Logger.log('✅ Triggers removidos');

  SpreadsheetApp.getUi().alert(
    'Agendamento removido!',
    'A execução automática foi desativada.',
    SpreadsheetApp.getUi().ButtonSet.OK
  );
}

function clearPredictions() {
  try {
    const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = spreadsheet.getSheetByName('[LF] Pesquisa');

    if (!sheet) {
      throw new Error('Aba "[LF] Pesquisa" não encontrada!');
    }
    const headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];

    const leadScoreCol = headers.indexOf('lead_score');
    const decileCol = headers.indexOf('decile');

    if (leadScoreCol === -1 && decileCol === -1) {
      SpreadsheetApp.getUi().alert(
        'Nenhuma predição encontrada',
        'As colunas lead_score e decile não existem na planilha.',
        SpreadsheetApp.getUi().ButtonSet.OK
      );
      return;
    }

    const ui = SpreadsheetApp.getUi();
    const response = ui.alert(
      'Confirmar limpeza',
      'Deseja realmente limpar TODAS as predições (lead_score e decile)?',
      ui.ButtonSet.YES_NO
    );

    if (response !== ui.Button.YES) {
      Logger.log('Limpeza cancelada pelo usuário');
      return;
    }

    const lastRow = sheet.getLastRow();

    if (leadScoreCol !== -1) {
      sheet.getRange(2, leadScoreCol + 1, lastRow - 1, 1).clearContent();
      Logger.log(`✅ Coluna lead_score limpa (${lastRow - 1} linhas)`);
    }

    if (decileCol !== -1) {
      sheet.getRange(2, decileCol + 1, lastRow - 1, 1).clearContent();
      Logger.log(`✅ Coluna decile limpa (${lastRow - 1} linhas)`);
    }

    SpreadsheetApp.getUi().alert(
      'Predições limpas!',
      `${lastRow - 1} linhas foram limpas com sucesso.`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );

  } catch (error) {
    Logger.log(`❌ Erro ao limpar predições: ${error.message}`);
    SpreadsheetApp.getUi().alert(
      'Erro',
      `Falha ao limpar predições:\n${error.message}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );
  }
}

function generateUTMAnalysis() {
  try {
    const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = spreadsheet.getSheetByName('[LF] Pesquisa');

    if (!sheet) {
      throw new Error('Aba "[LF] Pesquisa" não encontrada!');
    }
    const dataRange = sheet.getDataRange();
    const values = dataRange.getValues();
    const headers = values[0];

    const leadScoreCol = headers.indexOf('lead_score');
    const decileCol = headers.indexOf('decile');
    const timestampCol = headers.indexOf('data_processamento');

    if (leadScoreCol === -1 || decileCol === -1) {
      SpreadsheetApp.getUi().alert(
        'Erro',
        'Execute "Buscar Predições" primeiro para gerar lead_score e decile.',
        SpreadsheetApp.getUi().ButtonSet.OK
      );
      return;
    }

    if (timestampCol === -1) {
      SpreadsheetApp.getUi().alert(
        'Erro',
        'Coluna data_processamento não encontrada. Execute "Buscar Predições" novamente.',
        SpreadsheetApp.getUi().ButtonSet.OK
      );
      return;
    }

    const utmColumns = {
      'Campaign': headers.indexOf('Campaign'),
      'Medium': headers.indexOf('Medium'),
      'Content': headers.indexOf('Content'),
      'Term': headers.indexOf('Term')
    };

    const weights = {
      'D10': 10.0, 'D9': 3.1, 'D8': 1.4, 'D7': 1.4, 'D6': 0.7,
      'D5': -0.1, 'D4': -0.9, 'D3': -1.8, 'D2': -7.5, 'D1': -7.5
    };

    function calculateQualityScore(leads) {
      let score = 0;
      const total = leads.length;

      for (const [decil, weight] of Object.entries(weights)) {
        const count = leads.filter(l => l.decile === decil).length;
        const pct = (count / total) * 100;
        score += pct * weight;
      }

      return score;
    }

    // Encontrar o timestamp mais recente (última execução)
    let latestTimestamp = null;
    for (let i = 1; i < values.length; i++) {
      const timestamp = values[i][timestampCol];
      if (timestamp && (!latestTimestamp || timestamp > latestTimestamp)) {
        latestTimestamp = timestamp;
      }
    }

    if (!latestTimestamp) {
      SpreadsheetApp.getUi().alert(
        'Erro',
        'Nenhum timestamp encontrado. Execute "Buscar Predições" primeiro.',
        SpreadsheetApp.getUi().ButtonSet.OK
      );
      return;
    }

    // Filtrar apenas leads processados na última execução
    const allLeads = [];
    for (let i = 1; i < values.length; i++) {
      const row = values[i];
      const rowTimestamp = row[timestampCol];

      // Comparar timestamps (mesma data/hora)
      if (row[leadScoreCol] && row[decileCol] && rowTimestamp &&
          rowTimestamp.getTime() === latestTimestamp.getTime()) {
        allLeads.push({
          campaign: row[utmColumns['Campaign']],
          medium: row[utmColumns['Medium']],
          content: row[utmColumns['Content']],
          term: row[utmColumns['Term']],
          lead_score: row[leadScoreCol],
          decile: row[decileCol]
        });
      }
    }

    if (allLeads.length === 0) {
      SpreadsheetApp.getUi().alert(
        'Erro',
        'Nenhum lead encontrado com o timestamp da última execução.',
        SpreadsheetApp.getUi().ButtonSet.OK
      );
      return;
    }

    Logger.log(`📊 Analisando ${allLeads.length} leads da última execução (${latestTimestamp})`);

    const utmAnalysis = {};

    for (const [dimension, colIndex] of Object.entries(utmColumns)) {
      if (colIndex === -1) continue;

      const groups = {};

      allLeads.forEach(lead => {
        const key = lead[dimension.toLowerCase()];
        if (!key || key === '') return;

        if (!groups[key]) groups[key] = [];
        groups[key].push(lead);
      });

      const results = [];

      for (const [value, leads] of Object.entries(groups)) {
        const total = leads.length;
        const pctD10 = (leads.filter(l => l.decile === 'D10').length / total) * 100;
        const pctD8_10 = (leads.filter(l => ['D8','D9','D10'].includes(l.decile)).length / total) * 100;
        const qualityScore = calculateQualityScore(leads);

        let tier, acao;
        if (qualityScore > 200) {
          tier = 'S'; acao = '🔥 Aumentar +50%';
        } else if (qualityScore > 150) {
          tier = 'A'; acao = '✅ Aumentar +30%';
        } else if (qualityScore > 100) {
          tier = 'B'; acao = '✅ Aumentar +15%';
        } else if (qualityScore > 50) {
          tier = 'C'; acao = '⚠️  Manter';
        } else if (qualityScore > 0) {
          tier = 'D'; acao = '⚠️  Reduzir -20%';
        } else {
          tier = 'F'; acao = '❌ Reduzir -50%';
        }

        results.push({
          value: value,
          total: total,
          pctD10: pctD10,
          pctD8_10: pctD8_10,
          qualityScore: qualityScore,
          tier: tier,
          acao: acao
        });
      }

      results.sort((a, b) => b.qualityScore - a.qualityScore);
      utmAnalysis[dimension] = results;
    }

    const ss = SpreadsheetApp.getActiveSpreadsheet();
    let analysisSheet = ss.getSheetByName('Análise UTM');

    if (analysisSheet) {
      analysisSheet.clear();
    } else {
      analysisSheet = ss.insertSheet('Análise UTM');
    }

    let currentRow = 1;

    // Título principal com cor de fundo
    const titleRange = analysisSheet.getRange(currentRow, 1, 1, 7);
    titleRange.merge()
      .setValue(`ANÁLISE DE PERFORMANCE POR UTM`)
      .setFontWeight('bold')
      .setFontSize(16)
      .setBackground('#4a86e8')
      .setFontColor('#ffffff')
      .setHorizontalAlignment('center')
      .setVerticalAlignment('middle');
    analysisSheet.setRowHeight(currentRow, 40);
    currentRow++;

    // Subtítulo com info
    const subtitleRange = analysisSheet.getRange(currentRow, 1, 1, 7);
    subtitleRange.merge()
      .setValue(`${allLeads.length} leads processados em ${latestTimestamp.toLocaleString('pt-BR')}`)
      .setFontSize(10)
      .setFontColor('#666666')
      .setHorizontalAlignment('center');
    currentRow += 2;

    for (const [dimension, results] of Object.entries(utmAnalysis)) {
      // Cabeçalho da dimensão
      const dimHeader = analysisSheet.getRange(currentRow, 1, 1, 7);
      dimHeader.merge()
        .setValue(`${dimension.toUpperCase()} (${results.length} categorias)`)
        .setFontWeight('bold')
        .setFontSize(12)
        .setBackground('#f3f3f3')
        .setHorizontalAlignment('left');
      analysisSheet.setRowHeight(currentRow, 30);
      currentRow++;

      // Cabeçalhos das colunas
      const headerRow = ['Valor', 'Total Leads', '%D10', '%D8-D10', 'Quality Score', 'Tier', 'Ação'];
      const headerRange = analysisSheet.getRange(currentRow, 1, 1, 7);
      headerRange.setValues([headerRow])
        .setFontWeight('bold')
        .setBackground('#434343')
        .setFontColor('#ffffff')
        .setHorizontalAlignment('center')
        .setVerticalAlignment('middle');
      analysisSheet.setRowHeight(currentRow, 25);
      currentRow++;

      // Preparar dados em batch
      const batchData = [];
      const tierSRows = [];
      const tierARows = [];
      const tierBRows = [];
      const tierFRows = [];

      results.forEach((r, index) => {
        batchData.push([
          r.value, r.total, r.pctD10.toFixed(1) + '%', r.pctD8_10.toFixed(1) + '%',
          r.qualityScore.toFixed(1), r.tier, r.acao
        ]);

        const rowNum = currentRow + index;
        if (r.tier === 'S') {
          tierSRows.push(rowNum);
        } else if (r.tier === 'A') {
          tierARows.push(rowNum);
        } else if (r.tier === 'B') {
          tierBRows.push(rowNum);
        } else if (r.tier === 'F') {
          tierFRows.push(rowNum);
        }
      });

      // Escrever tudo de uma vez
      if (batchData.length > 0) {
        const dataRange = analysisSheet.getRange(currentRow, 1, batchData.length, 7);
        dataRange.setValues(batchData)
          .setVerticalAlignment('middle');

        // Alinhar colunas numéricas à direita
        analysisSheet.getRange(currentRow, 2, batchData.length, 1).setHorizontalAlignment('right');
        analysisSheet.getRange(currentRow, 3, batchData.length, 3).setHorizontalAlignment('center');
        analysisSheet.getRange(currentRow, 6, batchData.length, 1).setHorizontalAlignment('center');

        // Aplicar cores por tier
        tierSRows.forEach(row => {
          analysisSheet.getRange(row, 1, 1, 7).setBackground('#b7e1cd').setFontWeight('bold');
        });
        tierARows.forEach(row => {
          analysisSheet.getRange(row, 1, 1, 7).setBackground('#d9ead3');
        });
        tierBRows.forEach(row => {
          analysisSheet.getRange(row, 1, 1, 7).setBackground('#fff2cc');
        });
        tierFRows.forEach(row => {
          analysisSheet.getRange(row, 1, 1, 7).setBackground('#f4cccc');
        });

        // Adicionar bordas
        dataRange.setBorder(true, true, true, true, true, true, '#cccccc', SpreadsheetApp.BorderStyle.SOLID);

        currentRow += batchData.length;
      }

      currentRow += 2;
    }

    // Auto-resize todas as colunas
    for (let col = 1; col <= 7; col++) {
      analysisSheet.autoResizeColumn(col);
    }

    // Congelar primeira linha (título)
    analysisSheet.setFrozenRows(1);

    SpreadsheetApp.getUi().alert(
      'Análise Concluída!',
      `Aba "Análise UTM" criada com sucesso.\n\nAnalisados ${allLeads.length} leads da última execução.\n\nDimensões: Campaign, Medium, Content, Term`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );

  } catch (error) {
    Logger.log(`Erro ao gerar análise UTM: ${error.message}`);
    SpreadsheetApp.getUi().alert(
      'Erro',
      `Falha ao gerar análise UTM:\n${error.message}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );
  }
}

function getPredictionsSinceLastRun() {
  try {
    Logger.log('🚀 Iniciando busca de predições desde a última execução...');

    const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = spreadsheet.getSheetByName('[LF] Pesquisa');

    if (!sheet) {
      throw new Error('Aba "[LF] Pesquisa" não encontrada!');
    }

    const lastRow = sheet.getMaxRows();
    const lastCol = sheet.getMaxColumns();
    const dataRange = sheet.getRange(1, 1, lastRow, lastCol);
    const values = dataRange.getValues();

    const nonEmptyValues = values.filter((row, index) => {
      if (index === 0) return true;
      return row.some(cell => cell !== null && cell !== undefined && cell !== '');
    });

    Logger.log(`📊 Linhas totais na planilha: ${lastRow}, após filtrar vazias: ${nonEmptyValues.length}`);

    if (nonEmptyValues.length <= 1) {
      throw new Error('Planilha vazia ou só tem cabeçalho');
    }

    const headers = nonEmptyValues[0];
    Logger.log(`📋 Encontrados ${headers.length} campos: ${headers.join(', ')}`);

    const leadScoreColCheck = headers.indexOf('lead_score');
    const dataColIndex = headers.indexOf('Data');
    const dataProcessamentoColIndex = headers.indexOf('data_processamento');

    // Encontrar a última data de LEAD que foi processado (tem score)
    let lastProcessedLeadDate = null;
    if (leadScoreColCheck !== -1 && dataColIndex !== -1) {
      for (let i = 1; i < nonEmptyValues.length; i++) {
        const hasScore = nonEmptyValues[i][leadScoreColCheck];
        const leadDate = nonEmptyValues[i][dataColIndex];

        // Se tem score, pegar a data do lead
        if (hasScore && leadDate) {
          if (!lastProcessedLeadDate || leadDate > lastProcessedLeadDate) {
            lastProcessedLeadDate = leadDate;
          }
        }
      }
    }

    if (lastProcessedLeadDate) {
      Logger.log(`📅 Último lead processado tinha data: ${lastProcessedLeadDate}`);
    } else {
      Logger.log(`📅 Nenhuma execução anterior encontrada - processando todos os leads sem score`);
    }

    const leads = [];
    for (let i = 1; i < nonEmptyValues.length; i++) {
      const row = nonEmptyValues[i];

      // Pular se lead já tem score
      if (leadScoreColCheck !== -1 && row[leadScoreColCheck]) {
        continue;
      }

      // Se há última execução, pegar apenas leads com data >= último processado
      // Isso garante que leads da mesma data que não foram processados sejam incluídos
      if (lastProcessedLeadDate && dataColIndex !== -1 && row[dataColIndex]) {
        const leadDate = new Date(row[dataColIndex]);
        if (leadDate < lastProcessedLeadDate) {
          continue;
        }
        // Leads com data >= lastProcessedLeadDate são processados (se não tiverem score)
      }

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

    if (leads.length === 0) {
      Logger.log('✅ Nenhum lead novo para processar desde a última execução');
      SpreadsheetApp.getUi().alert(
        'Sem leads novos',
        'Nenhum lead novo desde a última predição para processar.',
        SpreadsheetApp.getUi().ButtonSet.OK
      );
      return;
    }

    const timeDescription = lastProcessedLeadDate
      ? `desde ${lastProcessedLeadDate.toLocaleString('pt-BR')}`
      : 'sem predições';
    Logger.log(`📊 Processando ${leads.length} leads ${timeDescription}`);

    const MAX_BATCH_SIZE = 600;
    const numBatches = Math.ceil(leads.length / MAX_BATCH_SIZE);
    const idealBatchSize = Math.ceil(leads.length / numBatches);

    Logger.log(`🔄 Dividindo ${leads.length} leads em ${numBatches} lotes equilibrados (~${idealBatchSize} leads cada)`);

    const batches = [];
    for (let i = 0; i < leads.length; i += idealBatchSize) {
      batches.push(leads.slice(i, i + idealBatchSize));
    }

    const batchSizes = batches.map(b => b.length).join(', ');
    Logger.log(`📦 Tamanhos dos lotes: [${batchSizes}]`);

    let allPredictions = [];
    let totalProcessingTime = 0;

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      Logger.log(`📦 Processando lote ${batchIndex + 1}/${batches.length} (${batch.length} leads)...`);

      const options = {
        method: 'post',
        contentType: 'application/json',
        payload: JSON.stringify({
          leads: batch
        }),
        muteHttpExceptions: true
      };

      const response = UrlFetchApp.fetch(API_URL, options);
      const statusCode = response.getResponseCode();

      if (statusCode !== 200) {
        throw new Error(`API retornou erro ${statusCode} no lote ${batchIndex + 1}: ${response.getContentText()}`);
      }

      const result = JSON.parse(response.getContentText());
      allPredictions = allPredictions.concat(result.predictions);
      totalProcessingTime += result.processing_time_seconds;

      Logger.log(`✅ Lote ${batchIndex + 1} concluído em ${result.processing_time_seconds}s`);

      if (batchIndex < batches.length - 1) {
        Utilities.sleep(500);
      }
    }

    Logger.log(`✅ Total: ${allPredictions.length} predições em ${totalProcessingTime.toFixed(2)}s`);

    let leadScoreCol = headers.indexOf('lead_score');
    let decileCol = headers.indexOf('decile');
    let timestampCol = headers.indexOf('data_processamento');

    if (leadScoreCol === -1) {
      leadScoreCol = headers.length;
      sheet.getRange(1, leadScoreCol + 1).setValue('lead_score');
      headers.push('lead_score');
    }

    if (decileCol === -1) {
      decileCol = headers.length;
      sheet.getRange(1, decileCol + 1).setValue('decile');
      headers.push('decile');
    }

    if (timestampCol === -1) {
      timestampCol = headers.length;
      sheet.getRange(1, timestampCol + 1).setValue('data_processamento');
      headers.push('data_processamento');
    }

    Logger.log('📝 Escrevendo predições na planilha...');

    const currentTimestamp = new Date();

    allPredictions.forEach(pred => {
      const rowNum = parseInt(pred.row_id);
      try {
        const scoreCell = sheet.getRange(rowNum, leadScoreCol + 1);
        scoreCell.setValue(pred.lead_score);
        scoreCell.setNumberFormat('0.0000');

        sheet.getRange(rowNum, decileCol + 1).setValue(pred.decile);
        sheet.getRange(rowNum, timestampCol + 1).setValue(currentTimestamp);
      } catch (e) {
        Logger.log(`⚠️ Erro ao escrever linha ${rowNum}: ${e.message}`);
      }
    });

    Logger.log('✅ Predições escritas com sucesso');

    const minScore = Math.min(...allPredictions.map(p => p.lead_score));
    const maxScore = Math.max(...allPredictions.map(p => p.lead_score));

    SpreadsheetApp.getUi().alert(
      'Sucesso!',
      `${allPredictions.length} leads processados em ${totalProcessingTime.toFixed(2)}s\n` +
      `Processados em ${batches.length} lote(s)\n` +
      (lastProcessedLeadDate ? `Desde: ${lastProcessedLeadDate.toLocaleString('pt-BR')}\n` : 'Primeira execução\n') +
      `\nScores: ${minScore.toFixed(3)} - ${maxScore.toFixed(3)}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );

  } catch (error) {
    Logger.log(`❌ Erro: ${error.message}`);
    Logger.log(error.stack);

    SpreadsheetApp.getUi().alert(
      'Erro',
      `Falha ao buscar predições:\n${error.message}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );

    throw error;
  }
}

function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu('Smart Ads')
    .addItem('Buscar Predições (últimas 24h)', 'getPredictions')
    .addItem('Buscar Predições (desde última execução)', 'getPredictionsSinceLastRun')
    .addSeparator()
    .addItem('Gerar Análise UTM', 'generateUTMAnalysis')
    .addItem('💰 Análise UTM com Custos', 'generateUTMAnalysisWithCosts')
    .addSeparator()
    .addItem('Limpar Predições', 'clearPredictions')
    .addSeparator()
    .addItem('Configurar Agendamento (todo dia às 8h)', 'createDailyTrigger')
    .addItem('Remover Agendamento', 'removeTimeDrivenTrigger')
    .addToUi();
}

function testConnection() {
  try {
    const healthUrl = 'https://smart-ads-api-12955519745.us-central1.run.app/health';

    const options = {
      method: 'get',
      muteHttpExceptions: true
    };

    const response = UrlFetchApp.fetch(healthUrl, options);
    const result = JSON.parse(response.getContentText());

    Logger.log('✅ Conexão OK!');
    Logger.log(JSON.stringify(result, null, 2));

    SpreadsheetApp.getUi().alert(
      'Conexão OK!',
      `Status: ${result.status}\nPipeline: ${result.pipeline_status}\nVersão: ${result.version}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );

  } catch (error) {
    Logger.log(`❌ Erro de conexão: ${error.message}`);

    SpreadsheetApp.getUi().alert(
      'Erro de Conexão',
      `Não foi possível conectar à API:\n${error.message}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );
  }
}

// ============================================================================
// ANÁLISE UTM COM CUSTOS DO META ADS
// ============================================================================

const META_ACCOUNT_ID = 'act_1948313086122284';  // Conta SANDBOX smart_ads
const ANALYSIS_API_URL = 'https://smart-ads-api-12955519745.us-central1.run.app/analyze_utms_with_costs';

function generateUTMAnalysisWithCosts() {
  try {
    Logger.log('🚀 Iniciando análise UTM com custos...');

    // IMPORTANTE: Sempre usar a aba de leads, não a aba ativa
    const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = spreadsheet.getSheetByName('[LF] Pesquisa');

    if (!sheet) {
      throw new Error('Aba "[LF] Pesquisa" não encontrada!');
    }

    Logger.log('📋 Usando aba: [LF] Pesquisa');

    // Ler dados da planilha
    const lastRow = sheet.getMaxRows();
    const lastCol = sheet.getMaxColumns();
    const dataRange = sheet.getRange(1, 1, lastRow, lastCol);
    const values = dataRange.getValues();

    const nonEmptyValues = values.filter((row, index) => {
      if (index === 0) return true;
      return row.some(cell => cell !== null && cell !== undefined && cell !== '');
    });

    if (nonEmptyValues.length <= 1) {
      throw new Error('Planilha vazia ou só tem cabeçalho');
    }

    const headers = nonEmptyValues[0];
    Logger.log(`📋 Campos: ${headers.length}`);

    const leadScoreColIndex = headers.indexOf('lead_score');
    const decileColIndex = headers.indexOf('decile');
    const dataColIndex = headers.indexOf('Data');

    // ETAPA 1: Gerar predições para leads dos últimos 7 dias sem score
    Logger.log('🔍 ETAPA 1: Verificando leads sem predição (últimos 7 dias)...');

    const now = new Date();
    const sevenDaysAgo = new Date(now.getTime() - (7 * 24 * 60 * 60 * 1000));

    const leadsWithoutPrediction = [];

    for (let i = 1; i < nonEmptyValues.length; i++) {
      const row = nonEmptyValues[i];

      // Verificar se é dos últimos 7 dias
      const isRecent = dataColIndex !== -1 && row[dataColIndex] &&
                       new Date(row[dataColIndex]) >= sevenDaysAgo;

      // Verificar se não tem score
      const hasNoScore = leadScoreColIndex === -1 || !row[leadScoreColIndex];

      if (isRecent && hasNoScore) {
        const leadData = {};
        headers.forEach((header, index) => {
          leadData[header] = row[index];
        });

        const emailValue = row[headers.indexOf('E-mail')];
        const email = emailValue ? String(emailValue) : null;

        leadsWithoutPrediction.push({
          data: leadData,
          email: email,
          row_id: (i + 1).toString(),
          sheetRowIndex: i + 1  // Para escrever de volta
        });
      }
    }

    Logger.log(`📊 Leads sem predição (últimos 7D): ${leadsWithoutPrediction.length}`);

    // Gerar predições com batching inteligente
    if (leadsWithoutPrediction.length > 0) {
      Logger.log('🔄 Gerando predições...');

      if (leadsWithoutPrediction.length <= 600) {
        // Enviar todos de uma vez
        Logger.log(`   Enviando ${leadsWithoutPrediction.length} leads em lote único`);

        const payload = { leads: leadsWithoutPrediction };
        const options = {
          method: 'post',
          contentType: 'application/json',
          payload: JSON.stringify(payload),
          muteHttpExceptions: true
        };

        const response = UrlFetchApp.fetch(API_URL, options);
        const responseCode = response.getResponseCode();

        if (responseCode !== 200) {
          throw new Error(`Erro ao gerar predições: ${responseCode} - ${response.getContentText()}`);
        }

        const result = JSON.parse(response.getContentText());

        // Escrever predições na planilha
        result.predictions.forEach(pred => {
          const rowIndex = parseInt(pred.row_id);
          if (leadScoreColIndex !== -1) {
            sheet.getRange(rowIndex, leadScoreColIndex + 1).setValue(pred.lead_score);
          }
          if (decileColIndex !== -1) {
            sheet.getRange(rowIndex, decileColIndex + 1).setValue(pred.decile);
          }
        });

        Logger.log(`✅ ${result.predictions.length} predições escritas`);

      } else {
        // Dividir em lotes iguais maiores possíveis próximos de 600
        const numLotes = Math.ceil(leadsWithoutPrediction.length / 600);
        const tamanhoLote = Math.ceil(leadsWithoutPrediction.length / numLotes);

        Logger.log(`   Dividindo ${leadsWithoutPrediction.length} leads em ${numLotes} lotes de ${tamanhoLote}`);

        for (let batchIndex = 0; batchIndex < numLotes; batchIndex++) {
          const start = batchIndex * tamanhoLote;
          const end = Math.min(start + tamanhoLote, leadsWithoutPrediction.length);
          const batchLeads = leadsWithoutPrediction.slice(start, end);

          Logger.log(`   Lote ${batchIndex + 1}/${numLotes}: ${batchLeads.length} leads`);

          const payload = { leads: batchLeads };
          const options = {
            method: 'post',
            contentType: 'application/json',
            payload: JSON.stringify(payload),
            muteHttpExceptions: true
          };

          const response = UrlFetchApp.fetch(API_URL, options);
          const responseCode = response.getResponseCode();

          if (responseCode !== 200) {
            throw new Error(`Erro no lote ${batchIndex + 1}: ${responseCode} - ${response.getContentText()}`);
          }

          const result = JSON.parse(response.getContentText());

          // Escrever predições na planilha
          result.predictions.forEach(pred => {
            const rowIndex = parseInt(pred.row_id);
            if (leadScoreColIndex !== -1) {
              sheet.getRange(rowIndex, leadScoreColIndex + 1).setValue(pred.lead_score);
            }
            if (decileColIndex !== -1) {
              sheet.getRange(rowIndex, decileColIndex + 1).setValue(pred.decile);
            }
          });

          Logger.log(`   ✅ Lote ${batchIndex + 1} concluído (${result.predictions.length} predições)`);

          // Delay entre lotes
          if (batchIndex < numLotes - 1) {
            Utilities.sleep(1000);
          }
        }
      }

      Logger.log('✅ Predições concluídas!');
    }

    // ETAPA 2: Análise UTM com custos (usando TODOS os leads com predição)
    Logger.log('📈 ETAPA 2: Gerando análise UTM com custos...');

    // Recarregar dados (predições podem ter sido adicionadas)
    const updatedValues = sheet.getRange(1, 1, lastRow, lastCol).getValues();
    const updatedNonEmpty = updatedValues.filter((row, index) => {
      if (index === 0) return true;
      return row.some(cell => cell !== null && cell !== undefined && cell !== '');
    });

    // Coletar TODOS os leads com predição
    const leadsWithPrediction = [];

    for (let i = 1; i < updatedNonEmpty.length; i++) {
      const row = updatedNonEmpty[i];

      // Verificar se tem score
      const hasScore = leadScoreColIndex !== -1 && row[leadScoreColIndex];

      if (hasScore) {
        const leadData = {};
        headers.forEach((header, index) => {
          leadData[header] = row[index];
        });

        const emailValue = row[headers.indexOf('E-mail')];
        const email = emailValue ? String(emailValue) : null;

        leadsWithPrediction.push({
          data: leadData,
          email: email,
          row_id: (i + 1).toString()
        });
      }
    }

    Logger.log(`📊 Total de leads com predição: ${leadsWithPrediction.length}`);

    if (leadsWithPrediction.length === 0) {
      throw new Error('Nenhum lead com predição encontrado');
    }

    // Chamar API de análise (sem limite - batching interno na API)
    Logger.log('🔄 Chamando API de análise UTM...');

    const payload = {
      leads: leadsWithPrediction,
      account_id: META_ACCOUNT_ID,
      product_value: null,  // Usar padrão da config (R$ 2.027,38)
      min_roas: null  // Usar padrão (2.0x)
    };

    const options = {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify(payload),
      muteHttpExceptions: true
    };

    const response = UrlFetchApp.fetch(ANALYSIS_API_URL, options);
    const responseCode = response.getResponseCode();

    if (responseCode !== 200) {
      throw new Error(`API retornou erro: ${responseCode} - ${response.getContentText()}`);
    }

    const result = JSON.parse(response.getContentText());

    Logger.log(`✅ Análise recebida: ${result.processing_time_seconds}s`);
    Logger.log(`   Períodos: ${Object.keys(result.periods).join(', ')}`);
    Logger.log(`   Config: Product Value = R$ ${result.config.product_value}, ROAS Min = ${result.config.min_roas}x`);

    // Criar abas para cada período
    const periods = ['1D', '3D', '7D', 'Total'];

    for (const period of periods) {
      if (result.periods[period]) {
        writeAnalysisSheet(period, result.periods[period], result.config);
      }
    }

    // Buscar informações do modelo e criar aba
    Logger.log('📊 Buscando informações do modelo...');
    try {
      const modelInfoResponse = UrlFetchApp.fetch(`${API_URL}/model/info`, {
        method: 'get',
        muteHttpExceptions: true
      });

      if (modelInfoResponse.getResponseCode() === 200) {
        const modelInfo = JSON.parse(modelInfoResponse.getContentText());
        writeModelInfoSheet(modelInfo);
        Logger.log('✅ Aba "Info do Modelo" criada/atualizada');
      } else {
        Logger.log('⚠️ Não foi possível obter informações do modelo');
      }
    } catch (error) {
      Logger.log(`⚠️ Erro ao buscar info do modelo: ${error.message}`);
    }

    Logger.log('✅ Análise UTM com custos concluída!');

    SpreadsheetApp.getUi().alert(
      'Análise Concluída',
      `Análise UTM com custos gerada com sucesso!\n\n` +
      `Abas criadas: ${periods.join(', ')}\n` +
      `Tempo de processamento: ${result.processing_time_seconds}s\n\n` +
      `Configuração:\n` +
      `• Product Value: R$ ${result.config.product_value.toFixed(2)}\n` +
      `• ROAS Mínimo: ${result.config.min_roas}x`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );

  } catch (error) {
    Logger.log(`❌ Erro na análise UTM: ${error.message}`);
    Logger.log(error.stack);

    SpreadsheetApp.getUi().alert(
      'Erro na Análise UTM',
      `Não foi possível gerar análise:\n${error.message}`,
      SpreadsheetApp.getUi().ButtonSet.OK
    );
  }
}

function writeAnalysisSheet(period, periodData, config) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheetName = `Análise UTM - ${period}`;

  // Deletar aba se já existir
  let sheet = ss.getSheetByName(sheetName);
  if (sheet) {
    ss.deleteSheet(sheet);
  }

  // Criar nova aba
  sheet = ss.insertSheet(sheetName);

  Logger.log(`📝 Criando aba: ${sheetName}`);

  // Cabeçalhos
  const headers = [
    'Dimensão', 'Valor', 'Leads', 'Gasto (R$)', 'CPL (R$)',
    '%D10', 'Taxa Proj. (%)', 'ROAS Proj.',
    'CPL Máx (R$)', 'Margem (%)', 'Tier', 'Ação'
  ];

  sheet.getRange(1, 1, 1, headers.length).setValues([headers]);

  // Formatação do cabeçalho
  const headerRange = sheet.getRange(1, 1, 1, headers.length);
  headerRange.setFontWeight('bold');
  headerRange.setBackground('#4285F4');
  headerRange.setFontColor('#FFFFFF');
  headerRange.setHorizontalAlignment('center');

  // Dimensões
  const dimensions = ['campaign', 'medium', 'term', 'ad'];
  const dimensionLabels = {
    'campaign': 'Campaign',
    'medium': 'Medium',
    'term': 'Term',
    'ad': 'Ad'
  };

  let currentRow = 2;

  for (const dimension of dimensions) {
    const metrics = periodData[dimension];

    if (!metrics || metrics.length === 0) {
      continue;
    }

    for (const metric of metrics) {
      const row = [
        dimensionLabels[dimension],
        metric.value,
        metric.leads,
        metric.spend,
        metric.cpl,
        metric.pct_d10,
        metric.taxa_proj * 100,  // Converter para %
        metric.roas_proj,
        metric.cpl_max,
        metric.margem,
        metric.tier,
        metric.acao
      ];

      sheet.getRange(currentRow, 1, 1, row.length).setValues([row]);

      // Formatação condicional da margem
      const margemCell = sheet.getRange(currentRow, 11);  // Coluna Margem

      if (metric.margem > 50) {
        margemCell.setBackground('#34A853');  // Verde
        margemCell.setFontColor('#FFFFFF');
      } else if (metric.margem >= 0) {
        margemCell.setBackground('#FBBC04');  // Amarelo
        margemCell.setFontColor('#000000');
      } else {
        margemCell.setBackground('#EA4335');  // Vermelho
        margemCell.setFontColor('#FFFFFF');
      }

      currentRow++;
    }
  }

  // Formatar colunas numéricas
  const lastRow = currentRow - 1;
  if (lastRow >= 2) {
    // Gasto, CPL, CPL Máx (formato moeda)
    sheet.getRange(2, 4, lastRow - 1, 1).setNumberFormat('R$ #,##0.00');
    sheet.getRange(2, 5, lastRow - 1, 1).setNumberFormat('R$ #,##0.00');
    sheet.getRange(2, 10, lastRow - 1, 1).setNumberFormat('R$ #,##0.00');

    // Percentuais
    sheet.getRange(2, 6, lastRow - 1, 1).setNumberFormat('0.00"%"');
    sheet.getRange(2, 7, lastRow - 1, 1).setNumberFormat('0.00"%"');
    sheet.getRange(2, 8, lastRow - 1, 1).setNumberFormat('0.00"%"');
    sheet.getRange(2, 11, lastRow - 1, 1).setNumberFormat('0.00"%"');

    // ROAS
    sheet.getRange(2, 9, lastRow - 1, 1).setNumberFormat('0.00"x"');
  }

  // Ajustar largura das colunas
  for (let i = 1; i <= headers.length; i++) {
    sheet.autoResizeColumn(i);
  }

  // Adicionar nota com configuração
  sheet.getRange(lastRow + 2, 1).setValue(`Configuração: Product Value = R$ ${config.product_value.toFixed(2)} | ROAS Mínimo = ${config.min_roas}x`);
  sheet.getRange(lastRow + 2, 1).setFontStyle('italic');
  sheet.getRange(lastRow + 2, 1).setFontColor('#666666');

  Logger.log(`✅ Aba ${sheetName} criada com ${lastRow - 1} registros`);
}

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
