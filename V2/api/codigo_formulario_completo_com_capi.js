/**
 * ============================================================================
 * CÓDIGO COMPLETO DO FORMULÁRIO COM CAPI INTEGRADO
 * ============================================================================
 *
 * Para usar:
 * 1. Substitua o código JavaScript existente da página por este arquivo completo
 * 2. Teste preenchendo o formulário
 * 3. Verifique logs no Console (Cmd + Option + I no Mac)
 */

// ============================================================================
// FUNÇÕES CAPI (NOVAS - ADICIONADAS)
// ============================================================================

function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return null;
}

function generateEventID() {
  return `lead_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

async function sendToCapiAPI(name, email, phone, hasComputer, utm, fbp, fbc, eventID, userAgent, eventSourceUrl) {
  const payload = {
    name: name,
    email: email,
    phone: phone,
    tem_comp: hasComputer,
    fbp: fbp,
    fbc: fbc,
    event_id: eventID,
    user_agent: userAgent,
    event_source_url: eventSourceUrl,
    utm_source: utm.utm_source || null,
    utm_medium: utm.utm_medium || null,
    utm_campaign: utm.utm_campaign || null,
    utm_term: utm.utm_term || null,
    utm_content: utm.utm_content || null
  };

  try {
    const response = await fetch('https://smart-ads-api-12955519745.us-central1.run.app/webhook/lead_capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result = await response.json();
    console.log('✅ CAPI enviado:', result);
    return result;
  } catch (error) {
    console.error('❌ Erro CAPI:', error);
    return null;
  }
}

// ============================================================================
// CÓDIGO ORIGINAL (SEM ALTERAÇÕES)
// ============================================================================

document.querySelectorAll('.bottom').forEach(button => {
  button.addEventListener('click', function() {
    button.classList.remove('ripple-animate');
    void button.offsetWidth;
    button.classList.add('ripple-animate');
  });
});

document.addEventListener("DOMContentLoaded", function() {
  const phoneInput = document.querySelector("#phone-input");
  let iti;

  if (phoneInput) {
    iti = window.intlTelInput(phoneInput, {
      initialCountry: "br",
      separateDialCode: true,
      utilsScript: "https://cdnjs.cloudflare.com/ajax/libs/intl-tel-input/17.0.13/js/utils.js"
    });

    function setMaskAndPlaceholder() {
      const d = iti.getSelectedCountryData();
      if (d && d.iso2) {
        const iso = d.iso2;
        let maskPattern;
        if (iso === 'br') {
          maskPattern = '(00) 00000-0000';
          phoneInput.placeholder = "(11) 96123-4567";
        } else if (window.intlTelInputUtils) {
          const ex = intlTelInputUtils.getExampleNumber(iso, true, intlTelInputUtils.numberFormat.NATIONAL);
          if (ex) {
            maskPattern = ex.replace(/\d/g, '9');
            phoneInput.placeholder = ex;
          }
        }
        if (maskPattern && typeof $ !== 'undefined' && $.fn.mask) {
          $(phoneInput).mask(maskPattern);
        }
      } else {
        phoneInput.placeholder = "(DDD) WhatsApp / Celular";
        if (typeof $ !== 'undefined' && $.fn.unmask) {
          $(phoneInput).unmask();
        }
      }
    }

    setMaskAndPlaceholder();
    phoneInput.addEventListener('countrychange', setMaskAndPlaceholder);
  }

  const form = document.getElementById("email-form");
  const submitButton = document.getElementById("email-form_submit");
  const fullnameInput = document.getElementById("fullname");
  const emailInput = document.getElementById("email");
  const radioSim = document.getElementById("field_144SIM");
  const radioNao = document.getElementById("field_144Não");

  if (form) {
    form.onsubmit = function(e) {
      e.preventDefault();
      return false;
    };
  }

  function getUTMParameters() {
    const urlParams = new URLSearchParams(window.location.search);
    return {
      utm_source: urlParams.get("utm_source") || "",
      utm_medium: urlParams.get("utm_medium") || "",
      utm_campaign: urlParams.get("utm_campaign") || "",
      utm_term: urlParams.get("utm_term") || "",
      utm_content: urlParams.get("utm_content") || ""
    };
  }

  function sendToSellFlux(name, email, phone, hasComputer, utm) {
    const baseUrl = "https://webhook.sellflux.app/v2/webhook/custom/fa992dd333629168fd067e62ff1b830f";
    const now = new Date();
    const dataHora = now.toLocaleString('pt-BR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });

    const params = new URLSearchParams();
    if (utm.utm_source) params.append("utm_source", utm.utm_source);
    if (utm.utm_medium) params.append("utm_medium", utm.utm_medium);
    if (utm.utm_campaign) params.append("utm_campaign", utm.utm_campaign);
    if (utm.utm_term) params.append("utm_term", utm.utm_term);
    if (utm.utm_content) params.append("utm_content", utm.utm_content);
    if (hasComputer) params.append("tem_comp", hasComputer);
    params.append("data", dataHora);

    const webhookUrl = baseUrl + "?" + params.toString();
    const payload = {
      name: name,
      email: email,
      phone: phone,
      tem_comp: hasComputer,
      utm_source: utm.utm_source,
      utm_medium: utm.utm_medium,
      utm_campaign: utm.utm_campaign,
      utm_term: utm.utm_term,
      utm_content: utm.utm_content,
      data: dataHora
    };

    return fetch(webhookUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return res.json();
      })
      .then(data => {
        console.log("✅ SellFlux - Dados enviados com sucesso:", data);
        return true;
      })
      .catch(err => {
        console.error("❌ Erro ao enviar para SellFlux:", err);
        throw err;
      });
  }

  if (submitButton) {
    submitButton.addEventListener("click", async function(event) {
      event.preventDefault();

      if (!fullnameInput || !fullnameInput.value.trim() || !emailInput || !emailInput.value.trim() || !phoneInput || !phoneInput.value.trim() || !((radioSim && radioSim.checked) || (radioNao && radioNao.checked))) {
        alert("Por favor, preencha todos os campos obrigatórios");
        return;
      }

      const fullname = fullnameInput.value.trim();
      const email = emailInput.value.trim();
      const phone = iti ? iti.getNumber() : phoneInput.value.replace(/\D/g, "");
      const hasComputer = (radioSim && radioSim.checked) ? "Sim" : "Não";
      const utm = getUTMParameters();

      // ============================================================================
      // NOVO: CAPTURAR E ENVIAR DADOS CAPI
      // ============================================================================
      const fbp = getCookie('_fbp');
      const fbc = getCookie('_fbc');
      const eventID = generateEventID();
      const userAgent = navigator.userAgent;
      const eventSourceUrl = window.location.href;

      console.log('📊 CAPI - FBP:', fbp || '❌ ausente', '| FBC:', fbc || '⚠️ ausente (normal)');

      // Enviar para CAPI API (não bloqueia o fluxo)
      sendToCapiAPI(fullname, email, phone, hasComputer, utm, fbp, fbc, eventID, userAgent, eventSourceUrl);
      // ============================================================================

      const originalText = submitButton.value;
      submitButton.value = "Enviando...";
      submitButton.disabled = true;

      try {
        await sendToSellFlux(fullname, email, phone, hasComputer, utm);

        let redirectURL = "https://lp.devclub.com.br/parabens-psq-devf/";
        const params = new URLSearchParams({
          nome: fullname,
          email: email,
          telefone: phone,
          computador: hasComputer,
          utm_source: utm.utm_source,
          utm_medium: utm.utm_medium,
          utm_campaign: utm.utm_campaign,
          utm_term: utm.utm_term,
          utm_content: utm.utm_content
        });

        redirectURL = redirectURL + "?" + params.toString();

        setTimeout(() => {
          window.location.href = redirectURL;
        }, 500);
      } catch (err) {
        console.error("Erro no processamento:", err);
        alert("Ocorreu um erro ao processar seu cadastro. Por favor, tente novamente.");
        submitButton.disabled = false;
        submitButton.value = originalText;
      }
    });
  }
});
