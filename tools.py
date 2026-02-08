import pymupdf4llm

def read_cv(file_path: str) -> str:
    """
    Reads the uploaded CV PDF and converts it to Markdown text.
    Use this tool to extract information from the resume.
    
    Args:
        file_path (str): The full path to the PDF file.
    """
    try:
        # Converts PDF layout (columns) to Markdown text
        md_text = pymupdf4llm.to_markdown(file_path)
        return md_text
    except Exception as e:
        return f"Failed to read PDF: {str(e)}"