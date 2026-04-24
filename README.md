# Dorm-Net

Dorm-Net is an offline-first AI tutor for engineering study. It combines:

- Ollama for fully local LLM inference
- ChromaDB plus `all-MiniLM-L6-v2` for RAG
- PyMuPDF for PDF ingestion
- Tesseract plus OpenCV for optional OCR of handwritten notes

After setup, normal tutoring flows run without external APIs.

## Folder Structure

```text
dorm/
├── app.py
├── main_app.py
├── requirements.txt
├── dorm_net_db/
└── modules/
    ├── brain_module.py
    ├── persona_module.py
    ├── tutor_controller.py
    ├── ui_components.py
    └── vision_module.py
```

## Core Features

- Offline Ollama-only tutor engine
- Selectable personas:
  - Software Engineering Tutor
  - Mechanical Engineering Tutor
  - Electrical Engineering Tutor
  - Math Tutor
  - Explain Like I'm 12
- Adaptive explanation depth
- Step-by-step mode toggle
- Session memory using recent turns
- Grounded answers from uploaded PDFs
- Debug mode with retrieved sources and scores
- Quiz generation from retrieved material
- Concept breakdown mode
- Error diagnosis mode
- Lightweight note generation mode

## Setup

### 1. Create and activate a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Install Ollama from the official installer for your OS, then start it:

```bash
ollama serve
```

### 4. Pull a local model

Recommended for 8GB RAM:

```bash
ollama pull mistral:latest
```

Alternative lighter model:

```bash
ollama pull phi3:latest
```

If you want coding-heavy behavior, you can also use:

```bash
ollama pull qwen2.5-coder:7b
```

### 5. Optional OCR setup

Install Tesseract OCR and, if needed, set:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Or on Linux/macOS:

```bash
export TESSERACT_CMD=/usr/bin/tesseract
```

## Run Instructions

### Streamlit UI

```bash
streamlit run main_app.py
```

### CLI

Interactive mode:

```bash
python app.py
```

Single question:

```bash
python app.py --question "Explain Kirchhoff's current law"
```

Index PDFs first:

```bash
python app.py --ingest path/to/book.pdf
```

Use a different persona or mode:

```bash
python app.py --persona electrical --mode concept_breakdown --question "Explain RC charging"
```

## Notes on Performance

- Keep one Ollama model loaded at a time on 8GB RAM systems.
- `mistral:latest` is the default.
- Smaller models like `phi3:latest` may feel more responsive on low-memory setups.
- ChromaDB storage is persistent in `dorm_net_db/`.
- PDF ingestion is page-by-page to reduce memory spikes.

## Environment Variables

Optional:

- `OLLAMA_URL`
- `DORM_NET_DB_PATH`
- `TESSERACT_CMD`

No cloud API keys are required.
