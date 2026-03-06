# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeoTrust is an AI-powered e-commerce analytics chat assistant. Users interact via a floating chat widget that queries mock 2025 market data (Mercado Livre, Amazon, Shopee) across three categories (Eletrônicos, Moda, Casa e Decoração). The agent responds in Brazilian Portuguese.

## Commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run backend (FastAPI on port 8000)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run frontend (serve static files, then open browser)
python -m http.server 8001 --directory frontend
```

## Architecture

```
Frontend (index.html)  ──POST /chat──>  FastAPI (main.py)  ──>  LangGraph ReactAgent (agent.py)
                                                                      │
                                                                      ├─ get_market_data()  → filters data/market_data.json, returns aggregated JSON
                                                                      └─ plot_chart()       → generates Plotly figure, stores JSON in global _last_generated_chart
                                                                            │
                                                          main.py calls get_and_clear_last_chart()
                                                          and sends chartData alongside response
```

**Key design decisions:**
- **In-memory chart cache** (`tools._last_generated_chart`): Plotly JSON (~44KB) is stored in a module-level global and retrieved by `main.py` after agent execution, avoiding embedding large JSON in LLM context (reduces token cost and hallucination).
- **Static thread ID** (`demo_session`): Single-user demo, no session management.
- **LLM**: Google Gemini 2.5 Flash via `langchain_google_genai`, temperature=0.
- **Agent pattern**: LangGraph `create_react_agent` — the LLM iteratively calls tools until producing a final text response.
- **Frontend**: Single vanilla HTML file with embedded CSS/JS, uses Plotly.js from CDN.

## AGENTS.md Rules (Summary)

The `AGENTS.md` file defines execution rules in Portuguese. Key constraints:
- **Autonomy**: Execute by default. Only ask if blocked or high-risk.
- **Triage**: L1 (<=5 lines, 1 file) execute silently; L2 (2-5 files) plan then execute; L3 (critical) design then execute.
- **L3 triggers**: auth, payments, public API contracts, migrations, concurrency, prod deletes, external integrations, security/crypto.
- **Hard limits**: No `any`/dynamic types, no empty catches, no dead code, no generic folders (`utils/`, `helpers/`, `common/`), no commented-out code (except TODO+issue).
- **Definition of Done**: Acceptance criteria verified, all gates green (fmt/lint/type/tests = 0 errors), no dead code, no vague TODOs.
- **Stop-loss**: After 2 failures, stop, document root cause hypothesis, rank 2-3 options, execute best.
