# Local RAG Knowledge Engine

A fully **offline** Retrieval-Augmented Generation system that runs entirely on your machine — no API keys, no cloud calls.

Optimised for a **6 GB RTX 3050** (works on any NVIDIA GPU with ≥ 4 GB VRAM, or CPU-only for the LLM via Ollama).

## Architecture

```
┌────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌───────────────┐
│  Document   │───▶│  Chunk & Embed   │───▶│   ChromaDB   │───▶│  Retrieve top │
│  .txt/.pdf  │    │  (RecursiveChar  │    │  (embedded)  │    │  K chunks     │
└────────────┘    │   + MiniLM-L6)   │    └──────────────┘    └──────┬────────┘
                   └──────────────────┘                               │
                                                                      ▼
                   ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
                   │   Final answer   │◀───│   Augmented   │◀───│  User query  │
                   │   (grounded)     │    │   prompt      │    │              │
                   └──────────────────┘    │  + Ollama LLM │    └──────────────┘
                                           └──────────────┘
```

## Stack

| Component       | Technology              | Resource Usage              |
|----------------|-------------------------|-----------------------------|
| Embedding      | `all-MiniLM-L6-v2`     | CPU (or ~80 MB VRAM)        |
| Generator      | `qwen2.5-coder:3b`     | ~1.9–3 GB VRAM via Ollama   |
| Vector DB      | ChromaDB (embedded)     | Disk only                   |
| Orchestration  | LangChain               | Python                      |

## Prerequisites

1. **Python 3.10+** (verified on Python 3.12)
2. **Git**
3. **Ollama** — install from [ollama.com](https://ollama.com), then pull the generative model:
   ```bash
   ollama pull qwen2.5-coder:3b
   ```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Prajwal-Acharyaa/local_rag_system.git
   cd local_rag_system
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

Or install packages manually:

```bash
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface langchain-chroma langchain-ollama chromadb sentence-transformers pypdf
```

## Usage

```bash
# Make sure Ollama is running (it starts automatically on install,
# or run `ollama serve` in another terminal).

python app.py
```

1. Paste the path to any `.txt` or `.pdf` file.
2. Ask questions about the document.
3. Type `new` to load a different file, or `exit` to quit.

## Configuration

Edit the constants at the top of `app.py`:

| Variable        | Default              | Description                              |
|----------------|----------------------|------------------------------------------|
| `CHUNK_SIZE`   | 800                  | Characters per text chunk                |
| `CHUNK_OVERLAP`| 200                  | Overlap between consecutive chunks       |
| `TOP_K`        | 5                    | Number of chunks retrieved per query     |
| `LLM_MODEL`    | `qwen2.5-coder:3b`  | Ollama model name                        |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address            |
