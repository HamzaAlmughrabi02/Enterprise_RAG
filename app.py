import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- 1. STREAMLIT CONFIGURATION 
st.set_page_config(page_title="TechTrack AI Assistant", page_icon="💼", layout="centered")

# --- 2. SET YOUR GROQ API KEY HERE ---
os.environ["GROQ_API_KEY"] = "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"

DB_DIR = "vector_db"

@st.cache_resource
def load_rag_system():
    """Loads the vector database and initializes the LLM only once."""
    print("--> [RAG LOG] Loading the initialized Vector Database...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Fetches top 3 matching chunks
    
    print("--> [RAG LOG] Connecting to Groq LLM (Llama-3)...")
    # Low temperature (0.1) ensures maximum accuracy for financial and compliance data
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
    return retriever, llm

retriever = None
llm = None
prompt = None

try:
    retriever, llm = load_rag_system()
    
    # Strict prompt to prevent AI from inventing (Hallucinating) financial figures
    system_prompt = (
        "You are an expert financial and corporate compliance assistant. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer or if it's not explicitly found in the context, say exactly: "
        "'I am sorry, but this information is not available in the company's uploaded documents.' "
        "Do not make up figures or facts.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])

except Exception as e:
    st.error(f"Configuration Error: Please ensure your GROQ_API_KEY is correct. Details: {e}")

# --- 3. STREAMLIT USER INTERFACE (UI) ---
st.title("💼 TechTrack Smart AI Assistant")
st.subheader("Corporate Policy & Financial Analytics Navigator")
st.write("Ask anything about the company's net profits, revenues, regional branch performance, or internal HR policies.")

# User Input Box
user_question = st.text_input("💬 Enter your question here (e.g., What was the net profit for 2025?):")

if user_question:
    if retriever is not None and llm is not None and prompt is not None:
        with st.spinner("⏳ Searching corporate documents and analyzing financial data..."):
            try:
                docs = retriever.invoke(user_question)
                if not docs:
                    st.warning("No relevant documents were found for this question.")

                context = "\n\n".join(
                    f"Chunk #{i+1}: {doc.page_content}" for i, doc in enumerate(docs)
                )
                messages = prompt.format_messages(query=user_question, context=context)
                response_message = llm.invoke(messages)
                answer_text = getattr(response_message, "text", str(response_message))

                # Display Answer
                st.markdown("### 🤖 Answer:")
                st.info(answer_text)

                # Show Sources
                with st.expander("🔍 View Verified Source Text Chunks (Anti-Hallucination Guard):"):
                    for i, doc in enumerate(docs):
                        st.write(f"**Chunk #{i+1}:** {doc.page_content}")
                        st.write("---")
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
    else:
        st.error("RAG system is not initialized properly. Please check the configuration errors above.")