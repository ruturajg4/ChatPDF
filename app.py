import os
import io
import time
from typing import List, Dict, Any, Tuple
import streamlit as st
from dotenv import load_dotenv

# PDF Reading
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# LangChain & Google GenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="ChatPDF - AI Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Gradient Hero Title */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Badge & Chip Styles */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.8rem;
    }

    .badge-ready {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
    }

    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    /* Chat bubble enhancements */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }

    /* Source Reference Expander Box */
    .source-box {
        background: rgba(15, 23, 42, 0.5);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.88rem;
    }

    /* Quick Prompt Button */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# Helper: Extract Documents with Page Metadata
def extract_pdf_documents(pdf_files: List[Any]) -> Tuple[List[Document], List[Dict[str, Any]]]:
    documents: List[Document] = []
    metadata_list: List[Dict[str, Any]] = []

    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            total_pages = len(reader.pages)
            file_char_count = 0
            file_name = pdf.name

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if page_text:
                    file_char_count += len(page_text)
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file_name,
                                "page": page_num,
                                "total_pages": total_pages
                            }
                        )
                    )

            metadata_list.append({
                "filename": file_name,
                "pages": total_pages,
                "char_count": file_char_count,
                "size_kb": round(len(pdf.getvalue()) / 1024, 1) if hasattr(pdf, "getvalue") else 0
            })
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")

    return documents, metadata_list


# Helper: Chunk Documents
def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


# Helper: Build Vector Store
def create_vector_store(chunks: List[Document], api_key: str, embedding_model: str = "models/text-embedding-004"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=api_key
    )
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    try:
        vector_store.save_local("faiss_index")
    except Exception:
        pass
    return vector_store


# Helper: Conversational Q&A Chain (LCEL)
def answer_user_question(user_question: str, vector_store, api_key: str, model_name: str, temperature: float = 0.2, top_k: int = 4):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(user_question)

    if not retrieved_docs:
        return "No relevant information could be found in the uploaded documents for your query.", []

    formatted_context = "\n\n---\n\n".join(
        [f"[Document: {doc.metadata.get('source', 'Unknown')} | Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
         for doc in retrieved_docs]
    )

    system_prompt = (
        "You are an expert document assistant powered by Google Gemini.\n"
        "Your task is to answer questions thoroughly, accurately, and strictly based on the provided context.\n"
        "Guidelines:\n"
        "1. Provide clear, well-structured, detailed answers with markdown formatting (bullet points, bold highlights, tables if applicable).\n"
        "2. If the answer cannot be determined strictly from the provided context, state clearly:\n"
        "   'The requested answer is not available in the provided documents.'\n"
        "3. Explicitly cite the document name and page number when referencing key facts.\n"
        "4. Never hallucinate or extrapolate beyond what is documented.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": formatted_context,
        "question": user_question
    })

    return answer, retrieved_docs


# ==========================================
# Sidebar: Setup, Model & File Management
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Gemini Configuration")

    # API Key Management
    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        help="Get a free API key at https://aistudio.google.com/app/apikey"
    )

    effective_api_key = api_key_input.strip() if api_key_input else env_api_key.strip()

    if effective_api_key:
        st.markdown(
            '<div class="status-badge badge-ready">● API Key Configured</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-badge badge-warning">▲ API Key Required</div>',
            unsafe_allow_html=True
        )
        st.caption("👉 [Get a free key from Google AI Studio](https://aistudio.google.com/app/apikey)")

    st.markdown("---")
    st.markdown("### 🧠 Model Parameters")
    
    selected_model = st.selectbox(
        "Gemini Model",
        options=["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
        index=0,
        help="gemini-1.5-flash is ultra-fast & highly accurate. gemini-1.5-pro provides deep reasoning."
    )

    selected_embedding = st.selectbox(
        "Embedding Model",
        options=["models/text-embedding-004", "models/embedding-001"],
        index=0,
        help="Google's standard text-embedding-004 model generates 768-dim embeddings."
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    with st.expander("🛠 Advanced Chunk Settings"):
        chunk_size = st.slider("Chunk Size", 300, 3000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        retrieval_k = st.slider("Retrieved Chunks (k)", 2, 8, 4, 1)

    st.markdown("---")
    st.markdown("### 📂 Document Uploader")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or multiple PDF documents to analyze."
    )

    process_btn = st.button("🚀 Process & Index Documents", use_container_width=True, type="primary")

    if process_btn:
        if not effective_api_key:
            st.error("Please enter a valid Google Gemini API Key first!")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Extracting pages & building vector database..."):
                start_time = time.time()
                docs, meta = extract_pdf_documents(uploaded_files)

                if not docs:
                    st.error("No readable text could be extracted from the uploaded PDFs. Please check if the files contain selectable text.")
                else:
                    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
                    try:
                        vector_store = create_vector_store(chunks, effective_api_key, selected_embedding)
                        st.session_state.vector_store = vector_store
                        st.session_state.doc_metadata = meta
                        st.session_state.total_chunks = len(chunks)
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                        elapsed = round(time.time() - start_time, 2)
                        st.success(f"Indexed {len(chunks)} chunks across {len(meta)} files in {elapsed}s!")
                    except Exception as err:
                        st.error(f"Vector store creation failed: {str(err)}")

    if st.session_state.vector_store is not None:
        st.markdown("---")
        if st.button("🧹 Clear Conversation History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ==========================================
# Main Content Area
# ==========================================
st.markdown('<div class="hero-title">ChatPDF Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Chat with multiple PDF documents using Google Gemini & FAISS Vector Search</div>',
    unsafe_allow_html=True
)

# Metric Summary Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    files_count = len(st.session_state.processed_files)
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-value">{files_count}</div>
        <div class="metric-label">Documents Loaded</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_pages = sum([m.get("pages", 0) for m in st.session_state.doc_metadata])
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-value">{total_pages}</div>
        <div class="metric-label">Total Pages</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-value">{st.session_state.total_chunks}</div>
        <div class="metric-label">Vector Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-value" style="font-size: 1.15rem; line-height: 2rem;">{selected_model}</div>
        <div class="metric-label">Active Engine</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab_chat, tab_docs, tab_info = st.tabs(["💬 Interactive Chat", "📄 Document Insights", "ℹ️ Architecture & Setup"])

with tab_chat:
    # Notice banner if no document processed yet
    if st.session_state.vector_store is None:
        st.info("👋 **Welcome to ChatPDF!** Upload one or more PDF files in the sidebar and click **'Process & Index Documents'** to start asking questions.")
        
        # Starter prompt suggestions (preview)
        st.markdown("#### Sample Questions you can ask once indexed:")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.button("📌 Summarize the key findings of the documents", disabled=True, use_container_width=True)
            st.button("🔍 What are the methodology and primary results?", disabled=True, use_container_width=True)
        with p_col2:
            st.button("📋 List all action items and recommendations", disabled=True, use_container_width=True)
            st.button("⚠️ Are there any risks or limitations highlighted?", disabled=True, use_container_width=True)
    else:
        # Prompt quick buttons if there are already files indexed
        if not st.session_state.messages:
            st.markdown("##### 💡 Quick Start Prompts:")
            qcol1, qcol2, qcol3 = st.columns(3)
            quick_query = None
            with qcol1:
                if st.button("📝 Summarize Document", use_container_width=True):
                    quick_query = "Provide a comprehensive summary of the main points in the uploaded document."
            with qcol2:
                if st.button("🎯 Key Takeaways", use_container_width=True):
                    quick_query = "List the top 5 key takeaways and insights with citations."
            with qcol3:
                if st.button("📋 Action Items & Next Steps", use_container_width=True):
                    quick_query = "Extract all recommended action items, next steps, or conclusions."

            if quick_query:
                st.session_state.messages.append({"role": "user", "content": quick_query})
                with st.spinner("Analyzing document with Gemini..."):
                    try:
                        ans, sources = answer_user_question(
                            quick_query,
                            st.session_state.vector_store,
                            effective_api_key,
                            selected_model,
                            temperature,
                            retrieval_k
                        )
                        st.session_state.messages.append({"role": "assistant", "content": ans, "sources": sources})
                    except Exception as err:
                        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {str(err)}", "sources": []})
                st.rerun()

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 View Retrieved Document Context & Citations"):
                    for idx, src in enumerate(msg["sources"], start=1):
                        meta = src.metadata
                        st.markdown(f"**Chunk #{idx}** — *{meta.get('source', 'Document')}* (Page {meta.get('page', '?')})")
                        st.markdown(f'<div class="source-box">{src.page_content}</div>', unsafe_allow_html=True)

    # Chat Input Box
    user_query = st.chat_input("Ask any question regarding the uploaded PDF documents...")

    if user_query:
        if not effective_api_key:
            st.error("Please configure your Google Gemini API Key in the sidebar.")
        elif st.session_state.vector_store is None:
            # Fallback check if faiss_index exists on disk
            if os.path.exists("faiss_index"):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model=selected_embedding,
                        google_api_key=effective_api_key
                    )
                    st.session_state.vector_store = FAISS.load_local(
                        "faiss_index",
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception:
                    st.warning("Please upload and process your PDF documents in the sidebar first.")
            else:
                st.warning("Please upload and process your PDF documents in the sidebar first.")

        if st.session_state.vector_store is not None and effective_api_key:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner(f"Searching index and generating answer with {selected_model}..."):
                    try:
                        reply, sources = answer_user_question(
                            user_query,
                            st.session_state.vector_store,
                            effective_api_key,
                            selected_model,
                            temperature,
                            retrieval_k
                        )
                        st.markdown(reply)
                        if sources:
                            with st.expander("📚 View Retrieved Document Context & Citations"):
                                for idx, src in enumerate(sources, start=1):
                                    meta = src.metadata
                                    st.markdown(f"**Chunk #{idx}** — *{meta.get('source', 'Document')}* (Page {meta.get('page', '?')})")
                                    st.markdown(f'<div class="source-box">{src.page_content}</div>', unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reply,
                            "sources": sources
                        })
                    except Exception as err:
                        error_msg = f"⚠️ Gemini API Error: {str(err)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": []
                        })

with tab_docs:
    st.markdown("### 📊 Document Metadata & Chunking Analytics")
    if not st.session_state.doc_metadata:
        st.info("No documents have been indexed in this session. Upload PDFs in the sidebar to view statistics.")
    else:
        for idx, doc in enumerate(st.session_state.doc_metadata, start=1):
            with st.container():
                st.markdown(f"""
                <div class="glass-card">
                    <h4 style="margin: 0 0 8px 0; color: #6366f1;">📄 {doc['filename']}</h4>
                    <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">
                        <strong>Pages:</strong> {doc['pages']} &nbsp;|&nbsp; 
                        <strong>Characters:</strong> {doc['char_count']:,} &nbsp;|&nbsp; 
                        <strong>File Size:</strong> {doc['size_kb']} KB
                    </p>
                </div>
                """, unsafe_allow_html=True)

with tab_info:
    st.markdown("### 🏛 Architecture & Processing Pipeline")
    st.markdown("""
    ```
    ┌────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
    │  Uploaded PDFs ├──────►│ PyPDF Text Extraction  ├──────►│ Recursive Text Split  │
    └────────────────┘       │ (Preserves Page Meta)  │       │ (Chunk Size: 1000)     │
                             └────────────────────────┘       └───────────┬────────────┘
                                                                          │
    ┌────────────────┐       ┌────────────────────────┐                   ▼
    │ Gemini Response│◄──────┤ Context & Prompt Chain ├───────┌────────────────────────┐
    │ (with Citations)│      │ (ChatGoogleGenAI)      │       │ Google Embeddings      │
    └────────────────┘       └───────────▲────────────┘       │ (text-embedding-004)   │
                                         │                    └───────────┬────────────┘
                                         │                                │
                             ┌───────────┴────────────┐                   ▼
                             │ Similarity Search (Top-k)│◄─────┌────────────────────────┐
                             │ (Cosine / L2 Distance) │       │   FAISS Vector Index   │
                             └────────────────────────┘       └────────────────────────┘
    ```
    """)
    st.markdown("""
    #### 🚀 Key Features:
    - **Multi-Document Support**: Upload and analyze multiple PDFs simultaneously.
    - **Exact Page Citations**: Each retrieved passage includes the document title and page number.
    - **Modern LCEL Architecture**: Built using LangChain Expression Language for low latency and zero deprecation warnings.
    - **Frontier Gemini Models**: Supports `gemini-1.5-flash`, `gemini-2.0-flash`, and `gemini-1.5-pro`.
    - **Interactive Multi-Turn Chat**: Natural conversational flow with chat memory and quick prompt shortcuts.
    """)