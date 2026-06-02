const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json());

// ══════════════════════════════════════════
// CONFIGURACIÓN — rellenar con datos de Meta
// ══════════════════════════════════════════
const VERIFY_TOKEN = "sofia_clinica_2024";        // tú lo inventas, puede ser cualquier texto
const WHATSAPP_TOKEN = process.env.WHATSAPP_TOKEN; // viene de Meta (empieza con EAA...)
const PHONE_NUMBER_ID = process.env.PHONE_NUMBER_ID; // número largo de Meta
const HUGGINGFACE_URL = "https://kevinmarce19-mediasistente-clinica.hf.space"; // tu Space

// ══════════════════════════════════════════
// WEBHOOK — verificación de Meta
// ══════════════════════════════════════════
app.get("/webhook", (req, res) => {
  const mode = req.query["hub.mode"];
  const token = req.query["hub.verify_token"];
  const challenge = req.query["hub.challenge"];

  if (mode === "subscribe" && token === VERIFY_TOKEN) {
    console.log("✅ Webhook verificado por Meta");
    res.status(200).send(challenge);
  } else {
    res.sendStatus(403);
  }
});

// ══════════════════════════════════════════
// WEBHOOK — recibir mensajes de WhatsApp
// ══════════════════════════════════════════
app.post("/webhook", async (req, res) => {
  res.sendStatus(200); // responder rápido a Meta

  try {
    const entry = req.body?.entry?.[0];
    const changes = entry?.changes?.[0];
    const value = changes?.value;
    const message = value?.messages?.[0];

    if (!message || message.type !== "text") return;

    const from = message.from;           // número del paciente
    const texto = message.text.body;     // lo que escribió

    console.log(`📩 Mensaje de ${from}: ${texto}`);

    // Llamar a Sofia en HuggingFace
    const respuesta = await preguntarASofia(texto);

    // Enviar respuesta por WhatsApp
    await enviarMensaje(from, respuesta);

  } catch (error) {
    console.error("❌ Error:", error.message);
  }
});

// ══════════════════════════════════════════
// FUNCIÓN — preguntar a Sofia en HuggingFace
// ══════════════════════════════════════════
async function preguntarASofia(mensaje) {
  try {
    const response = await axios.post(
      `${HUGGINGFACE_URL}/run/predict`,
      {
        data: [mensaje, []]
      },
      { timeout: 30000 }
    );

    const respuesta = response.data?.data?.[0];
    return respuesta || "Lo siento, en este momento no puedo responder. Llámanos al (01) 611-5000.";

  } catch (error) {
    console.error("❌ Error HuggingFace:", error.message);
    return "Lo siento, en este momento no puedo responder. Llámanos al (01) 611-5000.";
  }
}

// ══════════════════════════════════════════
// FUNCIÓN — enviar mensaje por WhatsApp
// ══════════════════════════════════════════
async function enviarMensaje(to, texto) {
  try {
    await axios.post(
      `https://graph.facebook.com/v18.0/${PHONE_NUMBER_ID}/messages`,
      {
        messaging_product: "whatsapp",
        to: to,
        type: "text",
        text: { body: texto }
      },
      {
        headers: {
          Authorization: `Bearer ${WHATSAPP_TOKEN}`,
          "Content-Type": "application/json"
        }
      }
    );
    console.log(`✅ Respuesta enviada a ${to}`);
  } catch (error) {
    console.error("❌ Error enviando mensaje:", error.response?.data || error.message);
  }
}

// ══════════════════════════════════════════
// INICIAR SERVIDOR
// ══════════════════════════════════════════
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Sofia WhatsApp Bot corriendo en puerto ${PORT}`);
});
