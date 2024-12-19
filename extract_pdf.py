import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

if __name__ == "__main__":
    pdf_path = "motorcycle_service_manual.pdf"  # Change this path to your PDF
    text = extract_text_from_pdf(pdf_path)
    with open("extracted_text.txt", "w") as file:
        file.write(text)
    print("Text extraction complete!")
