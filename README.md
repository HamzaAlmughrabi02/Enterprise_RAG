# 💼 TechTrack Smart AI Assistant (Enterprise RAG Pipeline)

An advanced, production-ready **Retrieval-Augmented Generation (RAG)** system designed to analyze and query corporate financial compliance documents and internal policies with **zero hallucinations**.

---

## 🚀 Features
- **Accurate Financial Cross-Referencing:** Powered by `Llama-3.1-8b-instant` via Groq API with low temperature configuration for maximum numerical precision.
- **Local Vector Storage:** Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and `ChromaDB` to chunk, embed, and store sensitive corporate data locally.
- **Anti-Hallucination Guard:** Strict system prompting constraints combined with an interactive Streamlit UI source-verification expander to audit exact text chunks.

---

## 🛠️ Tech Stack
- **Framework:** LangChain
- **LLM Provider:** Groq Cloud API (`Llama 3.1`)
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace Transformers
- **Frontend UI:** Streamlit
- **Language:** Python

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Enterprise_RAG.git](https://github.com/YOUR_USERNAME/Enterprise_RAG.git)
   cd Enterprise_RAG
