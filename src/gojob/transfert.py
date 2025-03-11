import os
import fitz # PyMuPDF
import pypandoc

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def convert_text_to_markdown(text):
    """
    Converts plain text to Markdown.

    :param text: Plain text to convert.
    :return: Markdown formatted text.
    """
    # Use pypandoc to convert text to Markdown
    markdown = pypandoc.convert_text(text, 'md', format='markdown')
    return markdown

def pdf_to_markdown(pdf_path, output_path):
    """
    Converts a PDF file to a Markdown file.

    :param pdf_path: Path to the PDF file.
    :param output_path: Path to save the Markdown file.
     """
     # Check if the PDF file exists
    if not os.path.exists(pdf_path):
      raise FileNotFoundError(f"No such file: '{pdf_path}'")

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Convert text to Markdown
    markdown = convert_text_to_markdown(text)

    # Save Markdown to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

# Download and install Pandoc if not already installed
pypandoc.download_pandoc()

# Example usage
pdf_path = "/Users/yanjin/training/ai/gojob/knowledge/CV_Mohan.pdf"
output_path = "/Users/yanjin/training/ai/gojob/knowledge/CV_Mohan.md"
pdf_to_markdown(pdf_path, output_path)