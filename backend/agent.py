import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from backend.tools import TOOLS
from langchain_core.messages import SystemMessage

load_dotenv()

llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    temperature=0,
    max_tokens=4096,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
)

from datetime import datetime

def get_system_prompt():
    agora = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read()
        
    return prompt_content.format(agora=agora)

from langgraph.checkpoint.memory import MemorySaver

agent_app = create_react_agent(
    model=llm, 
    tools=TOOLS
)
