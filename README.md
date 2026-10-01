# 📄 ChatPDF Intelligence — Chat with PDFs using Google Gemini & FAISS

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.44%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C.svg?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20%7C%202.0-8E75C4.svg?logo=google-gemini&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, multi-document conversational AI application powered by **Google Gemini** (`gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-1.5-pro`), **LangChain LCEL**, and **FAISS Vector Database**. Upload single or multiple PDF documents, automatically index and extract knowledge, and ask questions with precise page citations and context highlights.

---

## 🌟 Key Features

- **Multi-Document PDF Ingestion**: Upload multiple PDFs simultaneously; extracts text page-by-page while preserving exact document and page metadata.
- **Accurate Document Citations**: Every answer provides an expandable source breakdown citing the exact document name, page number, and matched chunk text.
- **Frontier Google Gemini & OpenRouter Support**: Seamlessly switch between native **Google Gemini** (`gemini-3.8-flash`, `gemini-3.5-flash`) and **OpenRouter** multi-model LLMs (`meta-llama`, `deepseek`, `openai`, `qwen`).
- **High-Performance Vector Search**: Uses Google's `models/gemini-embedding-001` (3072-dim embeddings) paired with in-memory **FAISS** similarity indexing.
- **Modern LCEL Architecture**: Modern LangChain Expression Language (`prompt | llm | StrOutputParser()`) — clean, future-proof, with zero deprecation warnings.
- **Interactive Chat Interface**: Multi-turn conversational flow with custom glassmorphism styling, metrics dashboard, quick prompt shortcuts, and chat clearing.
- **Dual API Key Management**: Configure Gemini and OpenRouter keys via `.env` or interactively in the Streamlit sidebar.

---

## 🏛 Architecture Overview

```mermaid
flowchart TD
    A[📄 Uploaded PDF Files] --> B[PyPDF Document Extractor]
    B --> C[Page & Document Metadata Tagging]
    C --> D[Recursive Character Text Splitter]
    D --> E[Google text-embedding-004]
    E --> F[(FAISS Vector Database)]
    
    G[👤 User Question] --> H[Vector Similarity Search Top-K]
    F --> H
    H --> I[Context Assembly & Prompt Formatting]
    I --> J[Gemini 1.5 / 2.0 LLM]
    J --> K[💬 Detailed Response + Exact Citations]
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/ruturajg4/ChatPDF.git
cd ChatPDF
```

### 2. Create and Activate a Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Your Gemini API Key

1. Obtain a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy the template configuration file:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and add your key:
   ```env
   GOOGLE_API_KEY="your_actual_gemini_api_key_here"
   ```
*(Alternatively, you can input your API key directly into the sidebar in the web application).*

### 5. Launch the Application

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## ⚙️ Configuration & Parameters

| Parameter | Default | Options | Description |
|---|---|---|---|
| **Gemini Model** | `gemini-1.5-flash` | `gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-1.5-pro` | LLM used for comprehension and answer synthesis |
| **Embedding Model** | `models/text-embedding-004` | `models/text-embedding-004`, `models/embedding-001` | Generates 768-dimensional semantic embeddings |
| **Temperature** | `0.2` | `0.0` - `1.0` | Lower values ensure factual, grounded document answers |
| **Chunk Size** | `1000` | `300` - `3000` | Character length for each split chunk |
| **Chunk Overlap** | `200` | `50` - `500` | Overlapping characters between consecutive chunks |
| **Retrieved Chunks ($k$)** | `4` | `2` - `8` | Number of most relevant passages sent to Gemini |

---

## 📁 Repository Structure

```
ChatPDF/
├── app.py              # Main Streamlit web application & LCEL RAG pipeline
├── requirements.txt    # Pinned, tested Python dependencies
├── .env.example        # Environment variable template for API keys
├── .gitignore          # Safeguards API keys, caches, and virtual environments
├── LICENSE             # MIT Open Source License
└── README.md           # Comprehensive project documentation
```

---

## 🔒 Security Best Practices

- **Never Commit Secrets**: The `.gitignore` file is pre-configured to ignore `.env`, `.venv`, and cached FAISS indices.
- **Local FAISS Storage**: Vector indices are computed and kept in memory for your session; indices are not transmitted to third-party database servers.

---

## 🤝 Contributing

Contributions, feedback, and pull requests are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

**Developed with ❤️ by [ruturajg4](https://github.com/ruturajg4)**
