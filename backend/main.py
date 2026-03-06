from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from backend.agent import agent_app, get_system_prompt
from backend.pdf_export import PdfExportRequest, generate_conversation_pdf
import re
import json
import tempfile
import os
import io
from backend.asr.asr_whisper_acc import LiteASRTranscriptionService

# Inicialização global do serviço de ASR (lite-whisper)
try:
    print("[ASR] Inicializando serviço Whisper...")
    asr_service = LiteASRTranscriptionService()
except Exception as e:
    print(f"[ASR] Erro ao carregar ASR: {e}")
    asr_service = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

async def process_agent_message(message: str, history: List[Dict[str, str]]):
    messages = [SystemMessage(content=get_system_prompt())]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=message))

    # Thread ID estático para demo
    config = {"configurable": {"thread_id": "demo_session"}}

    # Invocação do agente React - iterativamente chama ferramentas até ter a resposta final
    print(f"\n[INVOKING AGENT] user_message: '{message}'")

    # Processando em stream para logar passo a passo das ferramentas
    final_message = ""
    for event in agent_app.stream({"messages": messages}, config=config):
        for key, value in event.items():
            if key == "agent":
                agent_msg = value["messages"][-1]
                if hasattr(agent_msg, "tool_calls") and agent_msg.tool_calls:
                    for tool in agent_msg.tool_calls:
                        print(f" -> [AGENT TOOL CALL] {tool['name']}({tool['args']})")
                else:
                    if isinstance(agent_msg.content, list):
                        final_message = " ".join([c.get("text", "") for c in agent_msg.content if isinstance(c, dict) and "text" in c])
                    else:
                        final_message = agent_msg.content
                    print(f" <- [AGENT RESPONSE] {str(final_message)[:300]}...")
            elif key == "tools":
                tool_msg = value["messages"][-1]
                print(f" -> [TOOL FINISHED] Result length: {len(tool_msg.content)}")

    # Buscando se o Plotly JSON foi gerado puxando da memória em /backend/tools.py
    # Isso evita problemas de Regex na saída final do LLM e bugs de alucinação de prompt cheio.
    from backend.tools import get_and_clear_last_chart
    chart_json = None
    chart_string = get_and_clear_last_chart()

    if chart_string:
        try:
            print(" -> [FRONTEND CACHE] Encontrado objeto Plotly de visualização na variável em memória.")
            chart_json = json.loads(chart_string)
        except Exception as e:
            print(f"Erro analisando JSON do plotly: {e}")

    # Extrair sugestões da resposta do agente
    clean_response = final_message.strip()
    suggestions = []
    match = re.search(r'\[SUGESTOES\](.*?)\[/SUGESTOES\]', clean_response, re.DOTALL)
    if match:
        suggestions = [s.strip() for s in match.group(1).split('|') if s.strip()]
        clean_response = clean_response[:match.start()].strip()

    return {
        "response": clean_response,
        "chartData": chart_json,
        "suggestions": suggestions
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return await process_agent_message(request.message, request.history)

@app.post("/chat/audio")
async def chat_audio_endpoint(audio_file: UploadFile = File(...), history: str = Form(...)):
    history_list = json.loads(history)
    
    if asr_service is None:
        return {"response": "Erro: Serviço de transcrição de áudio não disponível.", "chartData": None, "transcription": ""}
        
    # Salvar áudio num tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        content = await audio_file.read()
        temp_audio.write(content)
        temp_audio_path = temp_audio.name
        
    try:
        print(f"\n[ASR] Transcrevendo áudio recebido ({len(content)} bytes)...")
        result = asr_service.transcribe_file(temp_audio_path)
        transcribed_text = result.text.strip()
        print(f"[ASR] Transcrição concluída: '{transcribed_text}'")
        
        # Filtro de silêncios/alucinações comuns do Whisper large-v3-turbo em PT-BR
        silence_hallucinations = ["e aí", "e ai", "e", "é", "ah", "hum", "...", "e aí?", "."]
        
        if not transcribed_text or transcribed_text.lower() in silence_hallucinations or len(transcribed_text) < 2:
            return {
                "response": "Desculpe, não consegui te ouvir direito. Seu áudio ficou vazio ou muito baixo. Poderia verificar seu microfone e tentar de novo?", 
                "chartData": None, 
                "transcription": ""
            }
            
        agent_response = await process_agent_message(transcribed_text, history_list)
        agent_response["transcription"] = transcribed_text
        return agent_response
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.post("/chat/export-pdf")
async def export_pdf(request: PdfExportRequest):
    # Reconstruir mensagens para o agente gerar insights
    insights_messages = [SystemMessage(content=get_system_prompt())]
    for msg in request.history:
        if msg["role"] == "user":
            insights_messages.append(HumanMessage(content=msg["content"]))
        else:
            insights_messages.append(AIMessage(content=msg["content"]))

    insights_messages.append(HumanMessage(
        content=(
            "Voce esta gerando um relatorio PDF profissional para o usuario. "
            "Com base em TODA a nossa conversa acima, gere um relatorio completo e bem estruturado seguindo EXATAMENTE esta estrutura:\n\n"
            "1. RESUMO EXECUTIVO\n"
            "Um paragrafo curto (3-5 linhas) resumindo o que foi analisado nesta conversa, "
            "o periodo dos dados, as plataformas e categorias discutidas.\n\n"
            "2. PRINCIPAIS DADOS E METRICAS\n"
            "Liste TODOS os numeros e dados quantitativos mencionados na conversa: "
            "faturamentos, unidades vendidas, precos medios, percentuais de crescimento, "
            "rankings. Use bullet points com os valores exatos que foram discutidos. "
            "Nao omita nenhum dado numerico relevante.\n\n"
            "3. ANALISE E INSIGHTS\n"
            "Apresente as analises e conclusoes discutidas: tendencias identificadas, "
            "comparacoes entre plataformas/categorias, pontos fortes e fracos, "
            "padroes de comportamento observados nos dados.\n\n"
            "4. DESTAQUES\n"
            "Liste os 3-5 pontos mais importantes da conversa que merecem atencao especial. "
            "Seja objetivo e direto.\n\n"
            "5. RECOMENDACOES\n"
            "Com base nos dados discutidos, sugira acoes estrategicas e proximos passos. "
            "Seja especifico e pratico.\n\n"
            "REGRAS: Nao use ferramentas. Nao invente dados que nao foram discutidos. "
            "Use apenas informacoes que apareceram na conversa. Seja detalhado e completo. "
            "Nao inclua tags [SUGESTOES]. Escreva em portugues claro e profissional."
        )
    ))

    print("\n[PDF EXPORT] Gerando insights via agente...")
    config = {"configurable": {"thread_id": "demo_pdf_export"}}
    result = agent_app.invoke({"messages": insights_messages}, config=config)

    final_msg = result["messages"][-1]
    if isinstance(final_msg.content, list):
        insights_text = " ".join(
            c.get("text", "") for c in final_msg.content
            if isinstance(c, dict) and "text" in c
        )
    else:
        insights_text = final_msg.content

    print(f"[PDF EXPORT] Insights gerados ({len(insights_text)} chars). Gerando PDF...")

    pdf_bytes = generate_conversation_pdf(
        history=request.history,
        chart_images=request.chart_images,
        insights_text=insights_text,
        title=request.conversation_title,
    )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="neotrust_conversa.pdf"'},
    )
