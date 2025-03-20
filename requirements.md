# Project Dependencies

## Core Dependencies
- streamlit>=1.24.0
- crewai>=0.11.0
- crewai-tools>=0.0.10
- pydantic>=2.0.0
- python-dotenv>=1.0.0
- fpdf>=2.7.0
- PyPDF2>=3.0.0
- fitz>=0.0.1.dev2
- pypandoc>=1.11
- sentence-transformers>=2.2.2
- qdrant-client>=1.7.0

## Development Dependencies
- black>=23.3.0
- flake8>=6.0.0
- pytest>=7.3.1
- mypy>=1.3.0

## Optional Dependencies
- groq>=0.3.0
- litellm>=1.0.0

## System Requirements
- Python 3.8 or higher
- Pandoc (for PDF to Markdown conversion)
- Git

## Environment Variables
Create a `.env` file in the root directory with the following variables:
```
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
```

## Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Pandoc (if not already installed):
   - Windows: Download from https://github.com/jgm/pandoc/releases
   - Linux: `sudo apt-get install pandoc`
   - macOS: `brew install pandoc`