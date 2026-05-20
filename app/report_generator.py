from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer

from reportlab.lib.styles import getSampleStyleSheet

import pandas as pd


def generate_pdf(df):

    pdf_file = "AI_Report.pdf"

    doc = SimpleDocTemplate(
        pdf_file
    )

    styles = getSampleStyleSheet()

    elements = []

    # Title
    title = Paragraph(

        "InsightGPT AI Report",

        styles['Title']
    )

    elements.append(title)

    elements.append(
        Spacer(1, 20)
    )

    # Dataset info
    rows = df.shape[0]

    cols = df.shape[1]

    info = Paragraph(

        f"""
        Total Rows: {rows}
        <br/>
        Total Columns: {cols}
        """,

        styles['BodyText']
    )

    elements.append(info)

    elements.append(
        Spacer(1, 20)
    )

    # Missing values
    missing = df.isnull().sum().sum()

    missing_text = Paragraph(

        f"""
        Missing Values: {missing}
        """,

        styles['BodyText']
    )

    elements.append(missing_text)

    elements.append(
        Spacer(1, 20)
    )

    # Statistical summary
    summary = df.describe().to_string()

    summary_text = Paragraph(

        f"""
        Statistical Summary:
        <br/><br/>
        {summary}
        """,

        styles['BodyText']
    )

    elements.append(summary_text)

    # Build PDF
    doc.build(elements)

    return pdf_file