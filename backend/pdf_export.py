import base64
import tempfile
import os
from datetime import datetime
from typing import Any, List, Dict

from fpdf import FPDF
from pydantic import BaseModel


class PdfExportRequest(BaseModel):
    history: List[Dict[str, str]]
    chart_images: List[Dict[str, Any]] = []
    conversation_title: str = "Conversa NeoTrust"


class NeoTrustPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(106, 100, 247)
            self.cell(0, 8, "NeoTrust AI", 0, 0, "L")
            self.set_text_color(150, 150, 150)
            self.set_font("Helvetica", "", 8)
            self.cell(0, 8, "Relatorio de Conversa", 0, 1, "R")
            self.set_draw_color(229, 229, 234)
            self.line(10, 18, 200, 18)
            self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Pagina {self.page_no()}/{{nb}}", 0, 0, "C")


def _safe_text(text: str) -> str:
    """Encode text to Latin-1 safely for fpdf2 built-in fonts."""
    return text.encode("latin-1", "replace").decode("latin-1")


def _clean_markdown(text: str) -> str:
    """Strip basic markdown markers for clean PDF text."""
    return text.replace("**", "").replace("*", "")


def _embed_chart_image(pdf: FPDF, base64_data: str) -> None:
    """Decode base64 PNG and embed it in the PDF."""
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]

    img_bytes = base64.b64decode(base64_data)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        tmp.write(img_bytes)
        tmp.close()
        pdf.image(tmp.name, x=15, w=180)
        pdf.ln(8)
    finally:
        os.remove(tmp.name)


def _strip_suggestions(text: str) -> str:
    """Remove tags [SUGESTOES]...[/SUGESTOES] do texto."""
    import re
    return re.sub(r'\[SUGESTOES\].*?\[/SUGESTOES\]', '', text, flags=re.DOTALL).strip()


def generate_conversation_pdf(
    history: List[Dict[str, str]],
    chart_images: List[Dict[str, Any]],
    insights_text: str,
    title: str,
) -> bytes:
    pdf = NeoTrustPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── COVER PAGE ──
    pdf.add_page()
    pdf.ln(40)

    # Logo area
    pdf.set_fill_color(106, 100, 247)
    pdf.rect(60, 50, 90, 50, "F")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(60, 65)
    pdf.cell(90, 20, "NeoTrust", 0, 0, "C")

    pdf.ln(60)
    pdf.set_text_color(26, 26, 46)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, _safe_text(title), 0, 1, "C")

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(139, 139, 158)
    pdf.cell(0, 10, "Relatorio de Analise com IA", 0, 1, "C")
    pdf.cell(
        0, 8, datetime.now().strftime("%d/%m/%Y as %H:%M"), 0, 1, "C"
    )
    n_charts = len(chart_images)
    if n_charts:
        pdf.cell(0, 8, f"{n_charts} graficos incluidos", 0, 1, "C")

    # ── REPORT SECTION ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 12, "Relatorio de Analise", 0, 1, "L")
    pdf.ln(2)

    pdf.set_draw_color(106, 100, 247)
    pdf.set_line_width(1)
    pdf.line(10, pdf.get_y(), 80, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(8)

    clean_insights = _strip_suggestions(insights_text)

    # Render each line with section header detection
    section_headers = [
        "RESUMO EXECUTIVO", "PRINCIPAIS DADOS", "METRICAS",
        "ANALISE E INSIGHTS", "ANALISE", "INSIGHTS",
        "DESTAQUES", "RECOMENDACOES", "RECOMENDAÇÕES",
    ]

    for line in clean_insights.split("\n"):
        stripped = line.strip()
        if not stripped:
            pdf.ln(4)
            continue

        # Check if line is a section header (e.g. "1. RESUMO EXECUTIVO" or "**RESUMO EXECUTIVO**")
        clean_line = stripped.lstrip("#0123456789.-) ").replace("**", "").replace("*", "").strip()
        is_header = any(h in clean_line.upper() for h in section_headers)

        if is_header:
            pdf.ln(6)
            pdf.set_fill_color(240, 239, 254)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(106, 100, 247)
            safe_header = _safe_text(clean_line)
            pdf.cell(0, 10, f"  {safe_header}", 0, 1, "L", fill=True)
            pdf.ln(4)
        elif stripped.lstrip("# ").startswith(("-", "•")):
            # Bullet point
            content = stripped.lstrip("# -•").strip()
            content = content.replace("**", "").replace("*", "")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(46, 50, 60)
            x_start = pdf.get_x()
            pdf.set_x(x_start + 6)
            pdf.set_text_color(106, 100, 247)
            pdf.cell(5, 6, "-", 0, 0, "L")
            pdf.set_text_color(46, 50, 60)
            pdf.multi_cell(0, 6, _safe_text(content), 0, "L")
            pdf.ln(2)
        else:
            # Regular paragraph
            content = stripped.lstrip("# ").replace("**", "").replace("*", "").strip()
            if content:
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(46, 50, 60)
                pdf.multi_cell(0, 6, _safe_text(content), 0, "L")
                pdf.ln(2)

    # ── CHARTS ──
    if chart_images:
        pdf.ln(6)
        pdf.set_fill_color(240, 239, 254)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(106, 100, 247)
        pdf.cell(0, 10, "  GRAFICOS DA ANALISE", 0, 1, "L", fill=True)
        pdf.ln(6)

        for ci in chart_images:
            _embed_chart_image(pdf, ci["image_base64"])
            pdf.ln(6)

    return pdf.output()
