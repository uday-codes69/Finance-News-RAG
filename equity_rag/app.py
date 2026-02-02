from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import time


import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Optional: verify it's loaded
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="News Research Tool", page_icon="📈", layout="wide")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">📈 News Research AI Assistant</h1>
    <p style="text-align:center; color: gray;">
    Ask grounded questions on Finance and equity research documents and reports
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

#Sidebar
st.sidebar.title("News Article Urls")

urls = []

for i in range(1):
    url = st.sidebar.text_input(
        f"URL {i+1}",
        placeholder="https://www.moneycontrol.com/"
    )
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
if process_url_clicked:
    loader = WebBaseLoader(urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    #split data 
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n','.'],
        chunk_size = 1000
    )
    main_placeholder.text("Data Loading...Started...✅✅✅")
    time.sleep(2)
    docs = text_splitter.split_documents(data)
   
   
# --------------------------------------------------
# Load Vector DB (Cached)
# --------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory="./equity_db", embedding_function=embeddings)
    return vectordb


vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --------------------------------------------------
# Load ChatGroq LLM
# --------------------------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=1.0)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: list[Document]) -> list[str]:
    return sorted(set(d.metadata.get("source", "Unknown") for d in docs))


# --------------------------------------------------
# UI Layout
# --------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_area(
        "🔍 Ask your equity research question",
        placeholder="What are the key risks highlighted in the equity research report?",
        height=120,
    )

with col2:
    st.markdown(
        """
        **Example questions**
        
        1)  What is an Equity Research Report?
        2) Describe recent news of money control?
        3) Budget of 2026?
         
        """
    )

# --------------------------------------------------
# Run RAG
# --------------------------------------------------
if st.button("🚀 Get Answer", use_container_width=True) and query:
    with st.spinner("Retrieving documents and generating answer..."):
        docs = retriever.invoke(query)
        context = format_docs(docs)

        prompt = f"""
You are an equity research assistant.

Rules:
- Answer ONLY from the provided context and get some information from the internet
- Be concise and factual and clear
- If the answer is not present, say "Information not available in the documents.
- If user give spelling mismatch then handle it using llm"

Context:
{context}

Question:
{query}

Answer:
"""

        response = llm.invoke(prompt)

    # --------------------------------------------------
    # Answer Section
    # --------------------------------------------------
    st.divider()
    st.subheader("📌 Answer")
    st.markdown(
        f"""
    <div style="
        background-color:#f9f9f9;
        color:#111111;
        padding:16px;
        border-radius:8px;
        font-size:16px;
        line-height:1.6;
    ">
    {response.content}
    </div>
    """,
        unsafe_allow_html=True,
    )

    SOURCE_HOVER = {
    "https://www.moneycontrol.com/":
        "Moneycontrol – Indian markets, stocks, and financial news",
    }

    # --------------------------------------------------
    # Sources Section
    # --------------------------------------------------
    st.subheader("🔗 Sources")
    for src in extract_sources(docs):
        st.markdown(f"- {src}")
