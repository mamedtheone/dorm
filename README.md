# 🎓 Dorm-Net

### Offline-First AI Peer Tutor for AASTU Students

Dorm-Net is a **local-first AI assistant** built for students at Addis Ababa Science and Technology University (AASTU). It allows continuous studying, revision, and concept clarification **without internet access**, using local large language models and intelligent document retrieval.

---

## ✨ Features

### 🔌 Offline-First by Design

Runs entirely on your machine using **Ollama**, ensuring:

* No internet dependency
* Full privacy
* Fast, low-latency responses

### 📚 Retrieval-Augmented Generation (RAG)

* Upload textbooks and PDFs
* Automatically index them into a local vector database (ChromaDB)
* Get answers grounded in your own study materials

### ✍️ Handwritten Notes OCR

* Upload images of your notes
* Extract and process text using **Tesseract + OpenCV**
* Supports preprocessing (denoising, deskewing, contrast enhancement)

### 🧠 AASTU Tutor Persona ("Kebede")

* Friendly senior-student-style explanations
* Uses structured reasoning (Chain-of-Thought)
* Focused on clarity and exam understanding

### ⚡ Async Processing

* Smooth Streamlit UI
* Non-blocking operations during:

  * PDF ingestion
  * Model inference

---

## 🏗️ Architecture Overview

**Frontend**

* Streamlit (custom dark mode UI)

**Core Intelligence (RAG System)**

* ChromaDB (vector storage)
* sentence-transformers (CPU-friendly embeddings)

**Vision Pipeline**

* PyTesseract
* OpenCV preprocessing pipeline

**LLM Engine**

* Ollama (local model orchestration)
* Example model: `llama3.2:3b`

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python 3.10+
* Ollama (installed and running)
* Tesseract OCR

#### Install Tesseract

**Windows:**
Install via UB-Mannheim build

**Linux:**

```bash
sudo apt-get install tesseract-ocr
```

---

### ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/dorm-net.git
cd dorm-net
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 🤖 Set Up the Model

Start Ollama:

```bash
ollama serve
```

Pull the required model:

```bash
ollama pull llama3.2:3b
```

---

### ▶️ Run the Application

```bash
streamlit run main_app.py
```

---

## 📂 Project Structure

```
dorm-net/
│
├── main_app.py              # Entry point (Streamlit app)
│
├── modules/
│   ├── brain_module.py      # RAG system & vector DB
│   ├── vision_module.py     # OCR + preprocessing pipeline
│   ├── persona_module.py    # Prompt engineering & LLM calls
│   ├── ui_components.py     # UI elements & styling
│
└── requirements.txt
```

---

## 👥 Team

Built with collaboration and late-night debugging sessions by:

* mamedtheone
* Haregeweyn Tewabe
* Natidev
* Wassie Tesfaye
* yaredmihretthe1st

---

## 🤝 Contributing

Dorm-Net is built **for students, by students**.

Ways to contribute:

* Improve OCR accuracy
* Optimize embedding performance
* Enhance UI/UX
* Add new study features

Feel free to:

* Open issues
* Submit pull requests

---

## 🌱 Vision

Dorm-Net aims to become a **fully offline academic companion**, especially valuable in environments with limited internet access. The goal is simple:

> Make learning uninterrupted, personal, and powerful.

---

## 📜 License

Add your license here (MIT recommended)

---

## 💡 Future Ideas

* Voice input/output 🎙️
* Multi-language support (Amharic + English)
* Quiz generation from PDFs
* Spaced repetition system
* Mobile version

---

Built with curiosity, caffeine, and a refusal to depend on WiFi ☕🚫📶
