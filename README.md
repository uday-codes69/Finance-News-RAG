# 📈 News Research AI Assistant (RAG-Based)

A **Retrieval-Augmented Generation (RAG)** application built using **LangChain**, **Streamlit**, and **Groq LLM** that enables users to ask **grounded financial and equity research questions** from real-time news articles such as *Moneycontrol* and other finance websites.

This project demonstrates **end-to-end GenAI system design**, including document ingestion, chunking, embeddings, vector search, and LLM-based answering with source attribution.

---

## 🚀 Demo Screenshot
<img width="1591" height="759" alt="Screenshot 2026-02-02 at 4 18 26 PM" src="https://github.com/user-attachments/assets/84fc238f-570c-4689-85b1-4f0890a28711" />


> 📌 *The UI allows users to input financial news URLs, ask questions, and receive grounded answers with sources.*

---

## ✨ Key Features

- 🌐 Ingest **live finance news articles via URLs**
- 🔍 Semantic search using **Chroma Vector Database**
- 🧠 Context-aware answers using **Groq (LLaMA 3.1)**
- 📊 Grounded responses from retrieved documents only
- 🔗 Transparent source attribution
- ⚡ Fast local embeddings with **Ollama**
- 🧩 Handles spelling mismatch using LLM reasoning

---

## 🧠 High-Level Architecture
User Query
↓
Chroma Vector Store (Similarity Search)
↓
Relevant Document Chunks
↓
Groq LLM (LLaMA-3.1-8B-Instant)
↓
Grounded Answer + Sources

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|-----------|
| Frontend | Streamlit |
| LLM | Groq – LLaMA-3.1-8B-Instant |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector Store | Chroma |
| Document Loader | WebBaseLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Language | Python |

---

## 📂 Project Structure
Equity_Research_RAG/
│
├── app.py                  # Streamlit application
├── equity_db/              # Chroma vector database (auto-created)
├── data/
│   └── nividia.txt         # Financial knowledge base text
├── assets/
│   └── app_ui.png          # Screenshot for README
├── .env                    # GROQ_API_KEY
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/uday-codes69/Equity_Research_RAG.git
cd Equity_Research_RAG


## ⚙️ Setup Instructions

### STEP 2: Create Virtual Environment
Create a Python virtual environment to isolate project dependencies.

```bash
python -m venv .venv


STEP 3: Activate Virtual Environment

Activate the virtual environment before installing dependencies.

For macOS/Linux
source .venv/bin/activate

For Windows
.venv\Scripts\activate

STEP 4: Install Project Dependencies

Install all required libraries using the requirements file.

uv install -r requirements.txt

In app.py
Run

Streamlit run app.py
