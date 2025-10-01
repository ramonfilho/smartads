const API_URL = 'https://smart-ads-api-12955519745.us-central1.run.app/predict/batch';
const SERVICE_ACCOUNT_EMAIL = 'smart-ads-451319@appspot.gserviceaccount.com';

function getPredictions() {
  try {
    Logger.log('🚀 Iniciando busca de predições...');

    const sheet = SpreadsheetApp.getActiveSheet();

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

    const leads = [];
    for (let i = 1; i < nonEmptyValues.length; i++) {
      const row = nonEmptyValues[i];

      // Pular se lead já tem score
      if (leadScoreColCheck !== -1 && row[leadScoreColCheck]) {
        continue;
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

    Logger.log(`📊 Processando ${leads.length} leads`);

    const MAX_BATCH_SIZE = 500;
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

function createTimeDrivenTrigger() {
  const triggers = ScriptApp.getProjectTriggers();
  triggers.forEach(trigger => {
    if (trigger.getHandlerFunction() === 'getPredictions') {
      ScriptApp.deleteTrigger(trigger);
    }
  });

  ScriptApp.newTrigger('getPredictions')
    .timeBased()
    .everyHours(6)
    .create();

  Logger.log('✅ Trigger criado: getPredictions rodará a cada 6 horas');

  SpreadsheetApp.getUi().alert(
    'Agendamento configurado!',
    'A função getPredictions será executada automaticamente a cada 6 horas.',
    SpreadsheetApp.getUi().ButtonSet.OK
  );
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
    const sheet = SpreadsheetApp.getActiveSheet();
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
    const sheet = SpreadsheetApp.getActiveSheet();
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

function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu('Smart Ads')
    .addItem('Buscar Predições', 'getPredictions')
    .addItem('Gerar Análise UTM', 'generateUTMAnalysis')
    .addItem('Limpar Predições', 'clearPredictions')
    .addSeparator()
    .addItem('Configurar Agendamento (a cada 6h)', 'createTimeDrivenTrigger')
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
