from PyPDF2 import PdfReader

def pdf_to_text(file):
    file = PdfReader(file)
    text = ""
    for page in file.pages:
        text += page.extract_text()
    return text