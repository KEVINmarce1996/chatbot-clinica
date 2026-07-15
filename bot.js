const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json());

// ══════════════════════════════════════════
// CONFIGURACIÓN — rellenar con datos de Meta
// ══════════════════════════════════════════
const VERIFY_TOKEN = "sofia_clinica_2024";          // tú lo inventas, puede ser cualquier texto
const WHATSAPP_TOKEN = process.env.WHATSAPP_TOKEN;   // viene de Meta (empieza con EAA...)
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
  res.sendStatus(200); // responder rápido a Meta, siempre antes de procesar

  try {
    const entry = req.body?.entry?.[0];
    const changes = entry?.changes?.[0];
    const value = changes?.value;
    const message = value?.messages?.[0];

    if (!message || message.type !== "text") return;

    const from = message.from;        // número del paciente
    const texto = message.text.body;  // lo que escribió

    console.log(`📩 Mensaje de ${from}: ${texto}`);

    const respuesta = await preguntarASofia(texto);
    await enviarMensaje(from, respuesta);

  } catch (error) {
    console.error("❌ Error procesando mensaje:", error.message);
  }
});

// ══════════════════════════════════════════
// FUNCIÓN — preguntar a Sofia en HuggingFace (Gradio 5)
// ══════════════════════════════════════════
// Gradio 5 funciona en DOS pasos:
//  1) POST /call/preguntar   → devuelve un event_id
//  2) GET  /call/preguntar/EVENT_ID → devuelve la respuesta en formato SSE
async function preguntarASofia(mensaje) {
  try {
    // PASO 1 — enviar la pregunta
    const envio = await axios.post(
      `${HUGGINGFACE_URL}/call/preguntar`,
      { data: [mensaje] },
      { timeout: 30000, headers: { "Content-Type": "application/json" } }
    );

    const eventId = envio.data?.event_id;
    if (!eventId) throw new Error("No se recibió event_id de HuggingFace");

    // PASO 2 — leer el resultado (Server-Sent Events)
    const resultado = await axios.get(
      `${HUGGINGFACE_URL}/call/preguntar/${eventId}`,
      { timeout: 30000, responseType: "text" }
    );

    // La respuesta llega como texto tipo:
    // event: complete
    // data: ["texto de la respuesta"]
    const texto = resultado.data;
    const match = texto.match(/data:\s*(\[.*\])/);

    if (match) {
      const parsed = JSON.parse(match[1]);
      return parsed[0] || "Lo siento, en este momento no puedo responder. Llámanos al (01) 611-5000.";
    }

    return "Lo siento, en este momento no puedo responder. Llámanos al (01) 611-5000.";

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
      `https://graph.facebook.com/v21.0/${PHONE_NUMBER_ID}/messages`,
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
