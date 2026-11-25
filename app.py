"""
Local RAG Knowledge Engine
===========================
A fully local Retrieval-Augmented Generation system optimised for a 6 GB
RTX 3050. No API keys, no cloud calls — everything runs on your machine.

Stack
-----
- Embedding  : all-MiniLM-L6-v2  (HuggingFace, CPU / tiny VRAM)
- Generator  : qwen2.5-coder:3b  (served locally by Ollama, ~2-3 GB VRAM)
- Vector DB  : ChromaDB           (embedded, zero-server)
- Orchestrator: LangChain

Usage
-----
    python app.py
    → paste a .txt or .pdf path → ask questions → type 'exit' to quit.
"""

# ── stdlib ──────────────────────────────────────────────────────────────
import os
import sys
import hashlib
import textwrap
import warnings
from pathlib import Path

# Ensure UTF-8 output, ANSI colour support on Windows, and clean terminal UI
if os.name == "nt":
    os.system("")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

# ── 3rd-party ───────────────────────────────────────────────────────────
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Configuration ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5-coder:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 200       # overlap between consecutive chunks
TOP_K = 5                 # number of chunks retrieved per query

CHROMA_DIR = Path(__file__).parent / "chroma_store"

# ── Terminal colours (optional, degrades gracefully) ────────────────────
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

def _banner() -> None:
    """Print a startup banner."""
    w = 62
    lines = [
        "LOCAL  RAG  KNOWLEDGE  ENGINE".center(w),
        f"  Embedding : {EMBEDDING_MODEL} (HuggingFace)".ljust(w),
        f"  Generator : {LLM_MODEL} (Ollama)".ljust(w),
        "  Vector DB : ChromaDB (embedded)".ljust(w),
    ]
    border_top = f"╔{'═' * w}╗"
    border_bot = f"╚{'═' * w}╝"
    body = "\n".join(f"║{line}║" for line in lines)
    print(f"\n{CYAN}{BOLD}{border_top}\n{body}\n{border_bot}{RESET}\n")


def _file_hash(path: str) -> str:
    """Return an MD5 hex-digest for a file (used as a collection name)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _resolve_path(raw: str) -> Path:
    """Normalise and validate the user-supplied file path."""
    # Strip surrounding quotes (drag-and-drop often adds them)
    cleaned = raw.strip().strip("\"'")
    p = Path(cleaned).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() not in (".txt", ".pdf"):
        raise ValueError(f"Unsupported file type '{p.suffix}'. Only .txt and .pdf are accepted.")
    return p


# ════════════════════════════════════════════════════════════════════════
#  CORE RAG COMPONENTS
# ════════════════════════════════════════════════════════════════════════

def load_document(path: Path) -> list:
    """Load a .txt or .pdf file and return a list of LangChain Documents."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    return loader.load()


def chunk_documents(docs: list) -> list:
    """Split documents into semantically meaningful chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_vectorstore(chunks: list, embeddings, collection_name: str) -> Chroma:
    """
    Embed chunks and persist them into a ChromaDB collection.

    Each ingested file gets its own collection (keyed by content hash)
    so re-ingesting the same file is instantaneous.
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def get_retriever(vectorstore: Chroma):
    """Return a similarity-search retriever over the vector store."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def build_rag_chain(retriever, llm):
    """
    Assemble the full RAG chain:
        query → retrieve chunks → augment prompt → generate answer
    """
    system_prompt = textwrap.dedent("""\
        You are a precise, helpful assistant. Answer the user's question
        using ONLY the context provided below.  If the context does not
        contain enough information, say "I don't have enough information
        in the provided document to answer that."

        Do NOT make up facts.  Cite specific details from the context
        where possible.

        CONTEXT:
        {context}
    """)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    def _format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Chunk {i+1}] {doc.page_content}" for i, doc in enumerate(docs)
        )

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ════════════════════════════════════════════════════════════════════════
#  INITIALISATION
# ════════════════════════════════════════════════════════════════════════

def init_embeddings():
    """Load the sentence-transformer embedding model."""
    print(f"{DIM}  ⏳ Loading embedding model ({EMBEDDING_MODEL})...{RESET}", end=" ", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"{GREEN}✔{RESET}")
    return embeddings


def init_llm():
    """Connect to the Ollama-hosted generative model."""
    print(f"{DIM}  ⏳ Connecting to Ollama ({LLM_MODEL})...{RESET}", end=" ", flush=True)
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,          # low temperature → factual, grounded
    )
    # Quick health-check: a tiny invoke to verify the model is reachable
    try:
        llm.invoke("hi")
    except Exception as exc:
        print(f"\n{RED}✘ Cannot reach Ollama at {OLLAMA_BASE_URL}.{RESET}")
        print(f"  Make sure Ollama is running and the model is pulled:")
        print(f"    ollama pull {LLM_MODEL}")
        print(f"  Error: {exc}")
        sys.exit(1)
    print(f"{GREEN}✔{RESET}")
    return llm


# ════════════════════════════════════════════════════════════════════════
#  INGEST PIPELINE
# ════════════════════════════════════════════════════════════════════════

def ingest(filepath: Path, embeddings) -> Chroma | None:
    """
    Full ingest pipeline:
        file → load → chunk → embed → store in ChromaDB
    Returns the populated Chroma vectorstore.
    """
    collection_name = "doc_" + _file_hash(str(filepath))[:16]

    # Check if this document was already ingested
    try:
        existing = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        if existing._collection.count() > 0:
            n = existing._collection.count()
            print(f"  {GREEN}✔ Document already ingested ({n} chunks cached).{RESET}")
            return existing
    except Exception:
        pass  # collection doesn't exist yet — proceed to ingest

    # 1. Load
    print(f"  {DIM}📄 Loading document...{RESET}", end=" ", flush=True)
    docs = load_document(filepath)
    print(f"{GREEN}✔{RESET}  ({len(docs)} page(s))")

    # 2. Chunk
    print(f"  {DIM}✂️  Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...{RESET}", end=" ", flush=True)
    chunks = chunk_documents(docs)
    print(f"{GREEN}✔{RESET}  ({len(chunks)} chunks)")

    if not chunks:
        print(f"  {RED}✘ No extractable text found in the document.{RESET}\n")
        return None

    # 3. Embed + Store
    print(f"  {DIM}🧠 Embedding & storing in ChromaDB...{RESET}", end=" ", flush=True)
    vectorstore = build_vectorstore(chunks, embeddings, collection_name)
    print(f"{GREEN}✔{RESET}")

    return vectorstore


# ════════════════════════════════════════════════════════════════════════
#  INTERACTIVE Q&A LOOP
# ════════════════════════════════════════════════════════════════════════

def qa_loop(chain) -> None:
    """Run a continuous question-and-answer session in the terminal."""
    print(
        f"\n{CYAN}{'─' * 60}{RESET}"
        f"\n  Ready!  Ask questions about the document."
        f"\n  Type {YELLOW}'exit'{RESET} to quit or {YELLOW}'new'{RESET} to load another file."
        f"\n{CYAN}{'─' * 60}{RESET}\n"
    )
    while True:
        try:
            question = input(f"{BOLD}❓ You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print(f"{DIM}Goodbye!{RESET}")
            break
        if question.lower() == "new":
            return  # signal the outer loop to ask for a new file

        print(f"{DIM}  ⏳ Thinking...{RESET}", flush=True)
        try:
            answer = chain.invoke(question)
        except Exception as exc:
            print(f"{RED}  ✘ Error generating answer: {exc}{RESET}")
            continue

        print(f"\n{GREEN}{BOLD}🤖 Answer:{RESET}\n{answer}\n")

    sys.exit(0)


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    _banner()

    # ── 1. Initialise models (one-time) ─────────────────────────────────
    embeddings = init_embeddings()
    llm = init_llm()
    print()

    # ── 2. Document ingest + Q&A loop ───────────────────────────────────
    while True:
        # Prompt for a file
        print(f"{BOLD}📂 Enter the path to a .txt or .pdf file:{RESET}")
        try:
            raw_path = input(f"   > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if raw_path.lower() == "exit":
            print(f"{DIM}Goodbye!{RESET}")
            break

        # Validate
        try:
            filepath = _resolve_path(raw_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{RED}  ✘ {exc}{RESET}\n")
            continue

        print(f"  {DIM}File: {filepath}{RESET}")

        # Ingest
        vectorstore = ingest(filepath, embeddings)
        if vectorstore is None:
            continue

        # Build chain
        retriever = get_retriever(vectorstore)
        chain = build_rag_chain(retriever, llm)

        # Q&A (returns here if user types 'new')
        qa_loop(chain)


if __name__ == "__main__":
    main()
