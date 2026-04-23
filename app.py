import os
import shutil
import gradio as gr
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_community.embeddings import FakeEmbeddings

# ══════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DOCS_FILE    = "clinica_docs.txt"
PERSIST_DIR  = "./chroma_clinica"

# ══════════════════════════════════════════
# VECTORIZACIÓN — usa FastEmbed (sin torch, liviano)
# ══════════════════════════════════════════
def build_vectorstore():
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    print("⏳ Indexando...")
    loader   = TextLoader(DOCS_FILE, encoding="utf-8")
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)
    embedder = FakeEmbeddings(size=384)
    vs       = Chroma.from_documents(chunks, embedder, persist_directory=PERSIST_DIR)
    print(f"✅ {len(chunks)} fragmentos indexados.")
    return vs

# ══════════════════════════════════════════
# RESPUESTA
# ══════════════════════════════════════════
def responder(mensaje, historial, vectorstore):
    resultados = vectorstore.similarity_search(mensaje, k=6)
    contexto   = "\n\n".join([r.page_content for r in resultados])

    historial_texto = ""
    for h in (historial or []):
        rol = "Paciente" if h["role"] == "user" else "Sofia"
        historial_texto += f"{rol}: {h['content']}\n"

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        streaming=True,
    )

    prompt = f"""Eres Sofia, asistente virtual de la Clínica San Martín en Lima, Perú.

REGLAS ESTRICTAS:
1. Responde SOLO con información del CONTEXTO. Nunca inventes datos.
2. Si la respuesta está en el contexto, respóndela de forma breve y directa.
3. Si NO está en el contexto, responde exactamente: "Para esa consulta contáctanos: WhatsApp 987-654-321 o al (01) 611-5000."
4. NUNCA diagnostiques enfermedades ni recetes medicamentos.
5. Responde en español, de forma breve, cálida y directa.
6. Solo di "Hola soy Sofia" en el primer mensaje. Después ve directo a la respuesta.

CONTEXTO:
{contexto}

CONVERSACIÓN PREVIA:
{historial_texto}
PACIENTE: {mensaje}
SOFIA:"""

    respuesta = ""
    for chunk in llm.stream(prompt):
        respuesta += chunk.content
        yield respuesta

# ══════════════════════════════════════════
# CONSTRUIR APP
# ══════════════════════════════════════════
vs = build_vectorstore()

BOTONES_CONSULTAS = [
    "¿Cuáles son los horarios?",
    "¿Atienden de noche o domingos?",
    "¿Cuánto cuesta una consulta?",
    "¿Aceptan EsSalud o Rímac?",
    "¿Cómo agendo una cita?",
    "¿Tienen teleconsulta?",
]

BOTONES_EXAMENES = [
    "¿Cómo me preparo para análisis de sangre?",
    "¿Cómo me preparo para ecografía?",
    "¿Cómo me preparo para mamografía?",
    "¿Cómo me preparo para colonoscopía?",
    "¿Cuánto cuesta una ecografía?",
    "¿Cuánto cuesta una tomografía?",
]

def chat_fn(mensaje, historial):
    yield from responder(mensaje, historial, vs)

with gr.Blocks(title="Clínica San Martín — Sofia IA") as demo:

    gr.HTML("""
    <div style="background:#1a5276;padding:16px 20px;border-radius:10px;margin-bottom:12px;
                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:28px;">🏥</span>
            <div>
                <div style="color:white;font-size:17px;font-weight:600;">Clínica San Martín — Lima</div>
                <div style="color:#aed6f1;font-size:12px;">Sofia · Asistente Virtual IA · Disponible 24/7</div>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="background:#27ae60;color:white;font-size:11px;padding:3px 10px;border-radius:20px;display:inline-block;margin-bottom:4px;">● EN LÍNEA</div>
            <div style="color:#aed6f1;font-size:11px;">📱 WhatsApp: 987-654-321</div>
        </div>
    </div>
    """)

    chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": (
            "¡Hola! Soy **Sofia** 👋, asistente virtual de la Clínica San Martín.\n\n"
            "Puedo ayudarte con horarios, precios, citas y preparación para exámenes.\n\n"
            "**Usa los botones de abajo o escribe tu consulta:**"
        )}],
        height=350,
        show_label=False,
        type="messages",
        avatar_images=(None, "https://img.icons8.com/color/48/nurse-female.png"),
    )

    gr.HTML("<div style='font-size:12px;color:#555;margin:10px 0 5px;font-weight:600;'>📋 Consultas frecuentes:</div>")
    with gr.Row():
        b1 = gr.Button(BOTONES_CONSULTAS[0], size="sm", variant="secondary")
        b2 = gr.Button(BOTONES_CONSULTAS[1], size="sm", variant="secondary")
        b3 = gr.Button(BOTONES_CONSULTAS[2], size="sm", variant="secondary")
    with gr.Row():
        b4 = gr.Button(BOTONES_CONSULTAS[3], size="sm", variant="secondary")
        b5 = gr.Button(BOTONES_CONSULTAS[4], size="sm", variant="secondary")
        b6 = gr.Button(BOTONES_CONSULTAS[5], size="sm", variant="secondary")

    gr.HTML("<div style='font-size:12px;color:#555;margin:10px 0 5px;font-weight:600;'>🔬 Exámenes y preparación:</div>")
    with gr.Row():
        b7  = gr.Button(BOTONES_EXAMENES[0], size="sm", variant="secondary")
        b8  = gr.Button(BOTONES_EXAMENES[1], size="sm", variant="secondary")
        b9  = gr.Button(BOTONES_EXAMENES[2], size="sm", variant="secondary")
    with gr.Row():
        b10 = gr.Button(BOTONES_EXAMENES[3], size="sm", variant="secondary")
        b11 = gr.Button(BOTONES_EXAMENES[4], size="sm", variant="secondary")
        b12 = gr.Button(BOTONES_EXAMENES[5], size="sm", variant="secondary")

    with gr.Row():
        txt = gr.Textbox(
            placeholder="Escribe tu consulta aquí...",
            show_label=False,
            scale=5,
            container=False,
        )
        btn_send  = gr.Button("Enviar ➤", variant="primary", scale=1)
        btn_clear = gr.Button("🗑", scale=0, min_width=44)

    gr.HTML("""
    <div style="text-align:center;margin-top:10px;font-size:11px;color:#999;padding:8px;
                background:#f8f9fa;border-radius:6px;">
        ⚠️ Sofia brinda información general. No reemplaza la consulta médica.
        · Emergencias 24h: <strong>(01) 611-5001</strong>
    </div>
    """)

    def enviar(pregunta, historial):
        if not pregunta.strip():
            return historial, ""
        historial = list(historial or [])
        historial.append({"role": "user", "content": pregunta})
        historial.append({"role": "assistant", "content": ""})
        return historial, ""

    def stream_resp(historial):
        if not historial or historial[-1]["role"] != "assistant":
            yield historial
            return
        pregunta = historial[-2]["content"]
        for parcial in responder(pregunta, historial[:-2], vs):
            historial[-1]["content"] = parcial
            yield historial

    def limpiar():
        return [{"role": "assistant", "content": "¡Hola! Soy **Sofia**. ¿En qué te puedo ayudar?"}]

    txt.submit(enviar, [txt, chatbot], [chatbot, txt]).then(stream_resp, chatbot, chatbot)
    btn_send.click(enviar, [txt, chatbot], [chatbot, txt]).then(stream_resp, chatbot, chatbot)
    btn_clear.click(limpiar, outputs=chatbot)

    def hacer_click(pregunta, historial):
        historial = list(historial or [])
        historial.append({"role": "user", "content": pregunta})
        historial.append({"role": "assistant", "content": ""})
        return historial

    for btn, preg in [
        (b1, BOTONES_CONSULTAS[0]), (b2, BOTONES_CONSULTAS[1]),
        (b3, BOTONES_CONSULTAS[2]), (b4, BOTONES_CONSULTAS[3]),
        (b5, BOTONES_CONSULTAS[4]), (b6, BOTONES_CONSULTAS[5]),
        (b7, BOTONES_EXAMENES[0]),  (b8, BOTONES_EXAMENES[1]),
        (b9, BOTONES_EXAMENES[2]),  (b10, BOTONES_EXAMENES[3]),
        (b11, BOTONES_EXAMENES[4]), (b12, BOTONES_EXAMENES[5]),
    ]:
        btn.click(
            fn=lambda h, p=preg: hacer_click(p, h),
            inputs=chatbot,
            outputs=chatbot,
        ).then(stream_resp, chatbot, chatbot)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
