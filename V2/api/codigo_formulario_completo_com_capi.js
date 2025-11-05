/**
 * Captura cookie Meta (_fbp ou _fbc)
 */
function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return null;
}

/**
 * Gera ID único para evento (deduplicação)
 */
function generateEventID() {
  return `lead_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Envia dados CAPI para nossa API
 */
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

// ========================================================================
// FUNÇÃO ACTIVECAMPAIGN (MANTIDA)
// ========================================================================

function submitToActiveCampaign(formData) {
  return new Promise((resolve, reject) => {
    const iframeId = "hidden-submit-iframe";
    let iframe = document.getElementById(iframeId);

    if (!iframe) {
      iframe = document.createElement("iframe");
      iframe.id = iframeId;
      iframe.name = iframeId;
      iframe.style.display = "none";
      document.body.appendChild(iframe);
    }

    const tempForm = document.createElement("form");
    tempForm.method = "POST";
    tempForm.action = "https://rodolfomori.activehosted.com/proc.php";
    tempForm.target = iframeId;

    for (const [key, value] of formData.entries()) {
      const input = document.createElement("input");
      input.type = "hidden";
      input.name = key;
      input.value = value;
      tempForm.appendChild(input);
    }

    document.body.appendChild(tempForm);

    const timeoutId = setTimeout(() => {
      resolve();
    }, 3000);

    iframe.onload = function() {
      clearTimeout(timeoutId);
      resolve();
    };

    tempForm.submit();

    setTimeout(() => {
      if (document.body.contains(tempForm)) {
        document.body.removeChild(tempForm);
      }
    }, 100);
  });
}

// ========================================================================
// CÓDIGO ORIGINAL DA PÁGINA (MANTIDO)
// ========================================================================

// Efeito ripple nos botões
document.querySelectorAll('.bottom').forEach(button => {
  button.addEventListener('click', function() {
    button.classList.remove('ripple-animate');
    void button.offsetWidth;
    button.classList.add('ripple-animate');
  });
});

// Inicialização do formulário
document.addEventListener("DOMContentLoaded", function() {
  const phoneInput = document.querySelector("#phone-input");
  let iti;

  // Configurar máscara de telefone internacional
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

  const form = document.getElementById("cadastro") || document.getElementById("email-form");
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

  function addToDataLayer(leadData) {
    if (typeof dataLayer !== 'undefined') {
      dataLayer.push({
        event: 'cadastro',
        ...leadData
      });
    }
  }

  function saveFormData() {
    // Implementação de cache se necessário
    return true;
  }

  // Event listener do botão de envio
  if (submitButton) {
    submitButton.addEventListener("click", function(event) {
      event.preventDefault();

      // Validação de campos obrigatórios
      const hasComputerSelected = (radioSim && radioSim.checked) || (radioNao && radioNao.checked);

      if ((!fullnameInput || !fullnameInput.value.trim()) ||
          (!emailInput || !emailInput.value.trim()) ||
          (!phoneInput || !phoneInput.value.trim()) ||
          !hasComputerSelected) {
        alert("Por favor, preencha todos os campos obrigatórios");
        return;
      }

      // Captura de dados do formulário
      const cachedData = saveFormData();
      const phoneFormatted = iti ? iti.getNumber() : phoneInput.value.replace(/\D/g, "");
      const fullname = fullnameInput.value.trim();
      const email = emailInput.value.trim();
      const phone = phoneFormatted;
      const hasComputer = radioSim && radioSim.checked ? "SIM" : "Não";
      const utmParams = getUTMParameters();

      // DataLayer GTM
      const leadData = {
        user_name: fullname,
        user_email: email,
        user_phone: phone,
        has_computer: hasComputer,
        form_id: 'cadastro',
        form_name: 'Formulário de Cadastro',
        timestamp: new Date().toISOString(),
        page_url: window.location.href,
        page_title: document.title,
        utm_source: utmParams.utm_source || null,
        utm_medium: utmParams.utm_medium || null,
        utm_campaign: utmParams.utm_campaign || null,
        utm_term: utmParams.utm_term || null,
        utm_content: utmParams.utm_content || null
      };

      addToDataLayer(leadData);

      // ========================================================================
      // CAPTURA DE DADOS CAPI (NOVO)
      // ========================================================================
      const fbp = getCookie('_fbp');
      const fbc = getCookie('_fbc');
      const eventID = generateEventID();
      const userAgent = navigator.userAgent;
      const eventSourceUrl = window.location.href;

      console.log('📊 CAPI - FBP:', fbp || '❌ ausente', '| FBC:', fbc || '⚠️ ausente (normal se não clicou em anúncio)');

      // Enviar para CAPI API (não bloqueia o fluxo)
      sendToCapiAPI(fullname, email, phone, hasComputer, utmParams, fbp, fbc, eventID, userAgent, eventSourceUrl);
      // ========================================================================

      // Preparar dados para ActiveCampaign
      const formData = new FormData();
      formData.append("u", "359");
      formData.append("f", "359");
      formData.append("s", "");
      formData.append("c", "0");
      formData.append("m", "0");
      formData.append("act", "sub");
      formData.append("v", "2");
      formData.append("or", "a84f9ed3437e39229a15e731cae61176");
      formData.append("fullname", fullname);
      formData.append("email", email);
      formData.append("phone", phone);
      formData.append("field[144]", hasComputer);
      formData.append("field[10]", utmParams.utm_source || "");
      formData.append("field[11]", utmParams.utm_medium || "");
      formData.append("field[13]", utmParams.utm_campaign || "");
      formData.append("field[12]", utmParams.utm_term || "");
      formData.append("field[14]", utmParams.utm_content || "");

      // Desabilitar botão durante envio
      const originalButtonText = submitButton.value || submitButton.textContent.trim();
      if (submitButton.tagName.toLowerCase() === 'input') {
        submitButton.value = "Enviando...";
      } else {
        submitButton.textContent = "Enviando...";
      }
      submitButton.disabled = true;

      // Enviar para ActiveCampaign
      submitToActiveCampaign(formData)
        .then(() => {
          // Redirecionar para página de obrigado
          let redirectURL = "https://lp5.rodolfomori.com.br/parabens-psq-devf/";
          const redirectParams = new URLSearchParams();
          redirectParams.append("nome", fullname);
          redirectParams.append("email", email);
          redirectParams.append("telefone", phone);
          redirectParams.append("computador", hasComputer);

          Object.keys(utmParams).forEach(key => {
            if (utmParams[key]) {
              redirectParams.append(key, utmParams[key]);
            }
          });

          redirectURL = `${redirectURL}?${redirectParams.toString()}`;
          window.location.href = redirectURL;
        })
        .catch(error => {
          alert("Ocorreu um erro ao processar seu cadastro. Por favor, tente novamente.");
          submitButton.disabled = false;
          if (submitButton.tagName.toLowerCase() === 'input') {
            submitButton.value = originalButtonText;
          } else {
            submitButton.textContent = originalButtonText;
          }
        });
    });
  }
});
