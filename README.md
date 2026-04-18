# Dorm-Net 🎓

**An Offline-First AI Peer Tutor for AASTU Students**

Dorm-Net is an innovative, local-first artificial intelligence assistant designed specifically for AASTU students to continue learning and revising course materials even when internet connectivity is unreliable or completely unavailable. 

By leveraging cutting-edge open-source AI models and local vector search, Dorm-Net provides a seamless educational experience right from your dorm room.

## ✨ Features

- **Offline-First Architecture**: Runs completely locally using [Ollama](https://ollama.com/), meaning no internet connection is required once installed.
- **RAG (Retrieval-Augmented Generation)**: Uses a local ChromaDB instance to search through uploaded course materials (PDFs) and provide highly accurate, context-aware answers.
- **Handwritten Notes Digitization**: Incorporates a built-in OCR (Optical Character Recognition) pipeline powered by Tesseract to scan, read, and understand handwritten notes.
- **Custom Personas**: Meet your virtual peer tutor! The system orchestrates responses prioritizing academic context and structured learning.
- **User-Friendly Interface**: Built with [Streamlit](https://streamlit.io/), offering a clean, responsive, and easy-to-use web UI.

## 🏗️ Architecture

- **Frontend UI & Controller**: `main_app.py` & Streamlit.
- **Offline Brain / RAG Server**: `modules/brain_module.py` (Local ChromaDB + Sentence Transformers).
- **Vision Pipeline**: `modules/vision_module.py` (OpenCV, MediaPipe & PyTesseract).
- **LLM Engine**: `modules/persona_module.py` (Ollama running `llama3.2:3b` or similar local models).
- **Batch Ingestion Script**: `ingest_notes.py`.

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.10+** (Tested on Windows 10/11 & Ubuntu 22.04)
2.  **Ollama**: Install from [ollama.com](https://ollama.com)
3.  **Tesseract OCR** (For reading handwritten notes)
    *   **Windows**: Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and make sure to add it to your system PATH or configure `TESSERACT_CMD` in `.env`.
    *   **Linux**: `sudo apt-get install tesseract-ocr`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/dorm-net.git
    cd dorm-net
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # For Windows:
    venv\Scripts\activate
    # For Linux/Mac:
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull the offline Language Model:**
    Make sure Ollama is running (`ollama serve`), then pull the model you intend to use.
    ```bash
    ollama pull llama3.2:3b
    ```

### Running the App

1.  Start the Streamlit application:
    ```bash
    streamlit run main_app.py
    ```
2.  Open your browser and navigate to the provided localhost URL (usually `http://localhost:8501`).

## ⚙️ Configuration

Dorm-Net relies on a few configuration options which can be set in an `.env` file or environment variables:

- `DORM_NET_DB_PATH`: Path to the local Chroma DB directory (default: `./dorm_net_db`).
- `OLLAMA_URL`: Local URL for the Ollama inference server (default: `http://localhost:11434`).
- `TESSERACT_CMD`: Absolute path to `tesseract.exe` (only needed on Windows if not added to PATH).

## 📄 How It Works

1.  **Upload Documents**: Via the Streamlit sidebar, upload your PDF course materials or textbooks. Dorm-Net will split, chunk, embed, and store them securely in its local vector database.
2.  **Scan Handwritten Notes**: Use the OCR panel to upload images of your handwritten notes for text extraction and real-time comprehension.
3.  **Ask Questions**: Submit complex queries to the tutor chat. The `RAGManager` pulls the most relevant sources from your textbooks while the `PersonaManager` orchestrates an educational, step-by-step guidance response from the local LLM.

## 🤝 Contributing

Contributions are welcome! If you're a student at AASTU or just passionate about building better offline ed-tech tools, feel free to open a pull request or submit issues for new features (e.g., more robust OCR models, better math LaTeX rendering, support for more languages).
