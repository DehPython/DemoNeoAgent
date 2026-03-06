from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from backend.agent import agent_app, get_system_prompt
import re
import json

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

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    messages = [SystemMessage(content=get_system_prompt())]
    for msg in request.history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=request.message))
    
    # Thread ID estático para demo
    config = {"configurable": {"thread_id": "demo_session"}} 
    
    # Invocação do agente React - iterativamente chama ferramentas até ter a resposta final
    print(f"\n[INVOKING AGENT] user_message: '{request.message}'")
    
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
        
    return {
        "response": final_message.strip(),
        "chartData": chart_json
    }
