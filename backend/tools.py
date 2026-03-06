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

def _auto_select_chart_type(eixo_x: str, eixo_y: str, agrupamento: Optional[str], n_categories: int) -> str:
    """Seleciona automaticamente o melhor tipo de gráfico com base nos dados."""
    if eixo_x == "data":
        return "area" if agrupamento else "linha"
    if n_categories <= 4 and not agrupamento:
        return "pizza"
    if agrupamento:
        return "barra_empilhada" if eixo_y == "faturamento" else "barra_agrupada"
    return "barra"


_NEOTRUST_COLORS = [
    "#6a64f7", "#f7647a", "#36c9c6", "#f7a864", "#8b5cf6",
    "#10b981", "#f43f5e", "#3b82f6", "#eab308", "#64748b",
]


def _format_value(val, eixo_y: str) -> str:
    """Formata valores para exibição em tooltips e anotações."""
    if eixo_y == "faturamento":
        if val >= 1_000_000:
            return f"R$ {val/1_000_000:.1f}M"
        if val >= 1_000:
            return f"R$ {val/1_000:.1f}K"
        return f"R$ {val:,.2f}"
    if eixo_y == "unidades_vendidas":
        if val >= 1_000:
            return f"{val/1_000:.1f}K un."
        return f"{int(val)} un."
    return f"R$ {val:,.2f}"


@tool
def plot_chart(
    eixo_x: str,
    eixo_y: str,
    agrupamento: Optional[str] = None,
    tipo_grafico: Optional[str] = None,
    titulo: str = "Gráfico de Mercado",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """
    Gera um gráfico Plotly profissional baseado nos dados de mercado.
    O tipo de gráfico é escolhido automaticamente se não for especificado.

    Args:
        eixo_x (str): Coluna para o eixo X ('data', 'plataforma', 'categoria')
        eixo_y (str): Métrica para o eixo Y ('faturamento', 'unidades_vendidas', 'preco_medio')
        agrupamento (str, optional): Separar por cores ('plataforma', 'categoria')
        tipo_grafico (str, optional): Tipo do gráfico. Se omitido, será escolhido automaticamente o melhor para os dados. Opções: 'linha', 'area', 'barra', 'barra_empilhada', 'barra_agrupada', 'pizza', 'treemap'
        titulo (str): Título principal do gráfico
        start_date (str, optional): Filtrar data de início (YYYY-MM-DD)
        end_date (str, optional): Filtrar data de fim (YYYY-MM-DD)

    Returns:
        str: Mensagem de sucesso indicando que o gráfico foi gerado.
    """
    df = load_data()

    if start_date:
        df = df[df['data'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['data'] <= pd.to_datetime(end_date)]

    if df.empty:
        return "Erro: Não há dados para as datas selecionadas para gerar o gráfico."

    # Agregar dados
    groupby_cols = [eixo_x]
    if agrupamento and agrupamento != eixo_x:
        groupby_cols.append(agrupamento)

    if eixo_y in ['faturamento', 'unidades_vendidas']:
        df_plot = df.groupby(groupby_cols)[eixo_y].sum().reset_index()
    else:
        df_plot = df.groupby(groupby_cols)[eixo_y].mean().reset_index()

    if eixo_x == 'data':
        df_plot = df_plot.sort_values(by='data')
        df_plot['data_fmt'] = df_plot['data'].dt.strftime('%d/%m/%Y')

    n_categories = df_plot[eixo_x].nunique()

    # Seleção automática do melhor tipo de gráfico
    if not tipo_grafico:
        tipo_grafico = _auto_select_chart_type(eixo_x, eixo_y, agrupamento, n_categories)

    tipo = tipo_grafico.lower().strip()

    label_map = {
        'data': 'Data',
        'data_fmt': 'Data',
        'plataforma': 'Plataforma',
        'categoria': 'Categoria',
        'faturamento': 'Faturamento (R$)',
        'unidades_vendidas': 'Unidades Vendidas',
        'preco_medio': 'Preço Médio (R$)',
    }
    for col in df_plot.columns:
        if col not in label_map:
            label_map[col] = col.replace('_', ' ').title()

    x_col = 'data_fmt' if eixo_x == 'data' and 'data_fmt' in df_plot.columns else eixo_x
    color_col = agrupamento if agrupamento and agrupamento in df_plot.columns else None

    # Tooltip customizado
    hover_template_y = "R$ %{y:,.2f}" if eixo_y in ['faturamento', 'preco_medio'] else "%{y:,.0f}"

    fig = None

    if tipo in ['pizza', 'pie']:
        df_pie = df_plot.groupby(eixo_x)[eixo_y].sum().reset_index() if color_col else df_plot
        fig = px.pie(
            df_pie, names=eixo_x, values=eixo_y, title=titulo,
            labels=label_map, color_discrete_sequence=_NEOTRUST_COLORS,
            hole=0.4,
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=12,
            hovertemplate="<b>%{label}</b><br>" + label_map.get(eixo_y, eixo_y) + ": %{value:,.2f}<br>Participação: %{percent}<extra></extra>",
        )

    elif tipo == 'treemap':
        path_cols = [eixo_x]
        if color_col:
            path_cols = [color_col, eixo_x] if color_col != eixo_x else [eixo_x]
        fig = px.treemap(
            df_plot, path=path_cols, values=eixo_y, title=titulo,
            labels=label_map, color_discrete_sequence=_NEOTRUST_COLORS,
        )
        fig.update_traces(
            textinfo='label+value+percent root',
            hovertemplate="<b>%{label}</b><br>Valor: %{value:,.2f}<br>Participação: %{percentRoot:.1%}<extra></extra>",
        )

    elif tipo in ['area', 'área']:
        fig = px.area(
            df_plot, x=x_col, y=eixo_y, color=color_col, title=titulo,
            labels=label_map, color_discrete_sequence=_NEOTRUST_COLORS,
            markers=True,
        )
        fig.update_traces(
            line=dict(width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>" + label_map.get(eixo_y, eixo_y) + ": " + hover_template_y + "<extra></extra>",
        )

    elif tipo in ['linha', 'line']:
        fig = px.line(
            df_plot, x=x_col, y=eixo_y, color=color_col, title=titulo,
            labels=label_map, color_discrete_sequence=_NEOTRUST_COLORS,
            markers=True,
        )
        fig.update_traces(
            line=dict(width=2.5),
            marker=dict(size=6, line=dict(width=1, color='white')),
            hovertemplate="<b>%{x}</b><br>" + label_map.get(eixo_y, eixo_y) + ": " + hover_template_y + "<extra></extra>",
        )

    elif tipo in ['barra_empilhada', 'stacked']:
        fig = px.bar(
            df_plot, x=x_col, y=eixo_y, color=color_col, barmode='stack',
            title=titulo, labels=label_map, color_discrete_sequence=_NEOTRUST_COLORS,
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" + label_map.get(eixo_y, eixo_y) + ": " + hover_template_y + "<extra></extra>",
        )

    elif tipo in ['barra_agrupada', 'grouped']:
        fig = px.bar(
            df_plot, x=x_col, y=eixo_y, color=color_col, barmode='group',
            title=titulo, labels=label_map, color_discrete_sequence=_NEOTRUST_COLORS,
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" + label_map.get(eixo_y, eixo_y) + ": " + hover_template_y + "<extra></extra>",
        )

    else:
        b_mode = "group" if color_col and color_col != eixo_x else "relative"
        fig = px.bar(
            df_plot, x=x_col, y=eixo_y, color=color_col, barmode=b_mode,
            title=titulo, labels=label_map, color_discrete_sequence=_NEOTRUST_COLORS,
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" + label_map.get(eixo_y, eixo_y) + ": " + hover_template_y + "<extra></extra>",
        )

    # Anotação de valor máximo em gráficos de linha/área/barra temporais
    if tipo in ['linha', 'line', 'area', 'área'] and eixo_x == 'data':
        if not color_col:
            idx_max = df_plot[eixo_y].idxmax()
            row_max = df_plot.loc[idx_max]
            fig.add_annotation(
                x=row_max[x_col],
                y=row_max[eixo_y],
                text=f"Pico: {_format_value(row_max[eixo_y], eixo_y)}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="#6a64f7",
                font=dict(size=11, color="#6a64f7", family="Arial"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#6a64f7",
                borderwidth=1,
                borderpad=4,
            )

    # Layout profissional
    if tipo not in ['pizza', 'pie', 'treemap']:
        fig.update_xaxes(type='category')

        y_axis_fmt = dict()
        if eixo_y in ['faturamento', 'preco_medio']:
            y_axis_fmt = dict(tickprefix="R$ ", tickformat=",.", separatethousands=True)
        elif eixo_y == 'unidades_vendidas':
            y_axis_fmt = dict(tickformat=",.", separatethousands=True)

        fig.update_yaxes(**y_axis_fmt, gridcolor="rgba(0,0,0,0.06)", gridwidth=1)
        fig.update_xaxes(showgrid=False, tickfont=dict(size=11))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=80, b=40),
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color="#2e323c"),
        title=dict(
            font=dict(size=16, color="#1a1a2e"),
            x=0.01, xanchor="left",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, Helvetica, Arial, sans-serif",
            bordercolor="#e5e5ea",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    global _last_generated_chart
    _last_generated_chart = fig.to_json()

    return "SUCESSO: O gráfico foi gerado com êxito e enviado ao frontend. NÃO chame a ferramenta de plotagem novamente. Responda ao usuário com uma menção curta dizendo que o gráfico está na tela e encerre sua fala."

# Lista de tools disponíveis
TOOLS = [get_market_data, plot_chart]
