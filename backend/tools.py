import json
import os
import pandas as pd
import plotly.express as px
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool

# Global memory to skip placing 44kb of JSON in LLM context
_last_generated_chart = None

def get_and_clear_last_chart():
    global _last_generated_chart
    val = _last_generated_chart
    _last_generated_chart = None
    return val


DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "market_data.json")

def load_data() -> pd.DataFrame:
    """Carrega os dados mockados no formato de DataFrame."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['data'] = pd.to_datetime(df['data'])
    return df

@tool
def get_market_data(
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    plataforma: Optional[str] = None, 
    categoria: Optional[str] = None
) -> str:
    """
    Busca e filtra os dados brutos de mercado da NeoTrust.
    
    Args:
        start_date (str, optional): Data inicial no formato YYYY-MM-DD.
        end_date (str, optional): Data final no formato YYYY-MM-DD.
        plataforma (str, optional): Filtrar por plataforma (ex: "Mercado Livre", "Amazon", "Shopee").
        categoria (str, optional): Filtrar por categoria (ex: "Eletrônicos", "Moda", "Casa e Decoração").
        
    Returns:
        str: Um resumo JSON em formato string com as métricas agregadas do período filtrado.
    """
    df = load_data()
    
    if start_date:
        df = df[df['data'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['data'] <= pd.to_datetime(end_date)]
    if plataforma:
        df = df[df['plataforma'].str.lower() == plataforma.lower()]
    if categoria:
        df = df[df['categoria'].str.lower() == categoria.lower()]
        
    if df.empty:
        return "Nenhum dado encontrado para os filtros especificados."
        
    # Agrupamento geral para facilitar a leitura do LLM
    agg_df = df.groupby(['plataforma', 'categoria']).agg({
        'unidades_vendidas': 'sum',
        'faturamento': 'sum',
        'preco_medio': 'mean'
    }).reset_index()
    
    # Adicionando totais globais no topo
    total_faturamento = df['faturamento'].sum()
    total_unidades = df['unidades_vendidas'].sum()
    
    resumo = {
        "periodo": f"{start_date or 'Indefinido'} até {end_date or 'Indefinido'}",
        "filtros": {"plataforma": plataforma, "categoria": categoria},
        "total_global": {
            "faturamento": round(total_faturamento, 2),
            "unidades_vendidas": int(total_unidades)
        },
        "detalhamento": agg_df.to_dict(orient="records")
    }
    
    return json.dumps(resumo, ensure_ascii=False)

@tool
def plot_chart(
    tipo_grafico: str,
    eixo_x: str,
    eixo_y: str,
    agrupamento: Optional[str] = None,
    titulo: str = "Gráfico de Mercado",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    Gera as configurações de um gráfico Plotly baseado nos dados.
    Esta ferramenta deve ser usada quando o usuário pedir explicitamente para "ver", "plotar" ou "mostrar" um gráfico da evolução das métricas.
    
    Args:
        tipo_grafico (str): O tipo de gráfico ('linha' ou 'barra')
        eixo_x (str): A coluna para o eixo X (opções: 'data', 'plataforma', 'categoria')
        eixo_y (str): A variável para exibir no eixo Y (opções: 'faturamento', 'unidades_vendidas', 'preco_medio')
        agrupamento (str, optional): Coluna para separar linhas/barras por cores (opções: 'plataforma', 'categoria')
        titulo (str): Título principal do gráfico
        start_date (str, optional): Filtrar data de início (YYYY-MM-DD)
        end_date (str, optional): Filtrar data de fim (YYYY-MM-DD)
        
    Returns:
        str: Uma string contendo uma marcação especial [PLOTLY_JSON] seguida do JSON de configuração do gráfico, para que o frontend o renderize.
    """
    df = load_data()
    
    if start_date:
        df = df[df['data'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['data'] <= pd.to_datetime(end_date)]
        
    if df.empty:
        return "Erro: Não há dados para as datas selecionadas para gerar o gráfico."

    # Se o eixo X for "data", precisamos agrupar para não plotar milhões de pontos brutos empilhados
    groupby_cols = [eixo_x]
    if agrupamento and agrupamento != eixo_x:
        groupby_cols.append(agrupamento)
        
    # Agrega os dados
    if eixo_y in ['faturamento', 'unidades_vendidas']:
        df_plot = df.groupby(groupby_cols)[eixo_y].sum().reset_index()
    else: # preco_medio
        df_plot = df.groupby(groupby_cols)[eixo_y].mean().reset_index()
        
    # Garante que as datas fiquem bonitinhas se for temporal
    if eixo_x == 'data':
        df_plot = df_plot.sort_values(by='data')

    # Cria a figura com Plotly Express
    fig = None
    color_col = agrupamento if agrupamento in df_plot.columns else None
    
    if tipo_grafico.lower() in ['linha', 'line']:
        fig = px.line(df_plot, x=eixo_x, y=eixo_y, color=color_col, title=titulo, markers=True)
    else:
        # Default para barras
        fig = px.bar(df_plot, x=eixo_x, y=eixo_y, color=color_col, barmode="group", title=titulo)
        
    # Melhorando o layout padrão visual para ficar limpo
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Converte o gráfico para JSON amigável do plotly.js
    global _last_generated_chart
    _last_generated_chart = fig.to_json()
    
    # Retornamos com uma anotação estrita para o Agente não repetir a chamada
    return "SUCESSO: O gráfico foi gerado com êxito e enviado ao frontend. NÃO chame a ferramenta de plotagem novamente. Responda ao usuário com uma menção curta dizendo que o gráfico está na tela e encerre sua fala."

# Lista de tools disponíveis
TOOLS = [get_market_data, plot_chart]
