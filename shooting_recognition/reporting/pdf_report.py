from pathlib import Path
from typing import Optional

from fpdf import FPDF

from config.settings import OUTPUT_FOLDER
from database.models import get_session, get_session_shots


class _ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Shooting Session Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(80, 80, 80)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def export_pdf(session_id: int, target_image_path: Optional[str] = None) -> Path:
    session = get_session(session_id)
    shots = get_session_shots(session_id)
    total_score = sum(s["score"] for s in shots)
    max_score = len(shots) * 10

    pdf = _ReportPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(30, 30, 30)

    # Session metadata
    pdf.cell(0, 8, f"Player: {session['player_name']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Started: {(session['started_at'] or '')[:19]}", new_x="LMARGIN", new_y="NEXT")
    if session.get("finished_at"):
        pdf.cell(0, 8, f"Finished: {session['finished_at'][:19]}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Target image (optional)
    if target_image_path and Path(target_image_path).exists():
        pdf.image(target_image_path, x=10, w=80)
        pdf.ln(4)

    # Shots table
    col_widths = [20, 22, 28, 42]
    headers = ["Shot #", "Score", "Cumulative", "Timestamp"]

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 230, 230)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", size=10)
    cumulative = 0
    for idx, shot in enumerate(shots):
        cumulative += shot["score"]
        ts = (shot["timestamp"] or "")[:19]
        row = [str(shot["shot_number"]), str(shot["score"]), str(cumulative), ts]
        fill = idx % 2 == 0
        pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
        for i, val in enumerate(row):
            pdf.cell(col_widths[i], 7, val, border=1, fill=fill)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 80, 200)
    pdf.cell(0, 10, f"Total Score: {total_score} / {max_score}", new_x="LMARGIN", new_y="NEXT")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    safe_name = session["player_name"].replace(" ", "_")
    path = OUTPUT_FOLDER / f"session_{session_id}_{safe_name}.pdf"
    pdf.output(str(path))
    return path
