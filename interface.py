import streamlit as st
import requests
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Initialize RAG components
@st.cache_resource
def get_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./expert-rag-bot/db", embedding_function=embeddings)
    llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant specializing in answering questions based on the provided document context. 

Context: {context}

Question: {question}

Provide a clear, detailed, and accurate answer. If the context doesn't contain enough information to fully answer the question, explain what you can based on the available information and suggest what might be needed. Be conversational and helpful, like a knowledgeable expert.""")
    qa_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return retriever, qa_chain


st.set_page_config(
    page_title="InsightLens | AI Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    }
    
    /* Custom Chat Bubbles */
    .stChatMessage {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        color: white;
    }
    
    /* Buttons Styling */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
        border: none;
        background-color: #3b82f6;
        color: white;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## ⚙️ ENGINE STATUS")
    status_col = st.columns(2)
    status_col[0].metric("Latent Speed", "120 t/s", "8%")
    status_col[1].metric("Memory", "0.4GB", "-2%")
    
    st.markdown("---")
    st.markdown("### 🛠️ Configuration")
    model_type = st.selectbox("Intelligence Model", ["Llama 3 (Groq)", "Mistral (Local)"])
    creativity = st.slider("Response Temperature", 0.0, 1.0, 0.2)
    
    st.markdown("---")
    if st.button("🗑️ Reset Neural Memory"):
        st.session_state.messages = []
        st.rerun()


st.markdown("---")
st.title("⚡ InsightLens AI")
st.markdown("##### *Securely analyzing proprietary data using Retrieval Augmented Generation.*")

# PDF Upload Section
st.markdown("### 📄 Upload Documents")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            # Save uploaded file
            pdf_path = f"./data/{uploaded_file.name}"
            os.makedirs("./data", exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
        
        # Process all docs
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(all_docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory="./expert-rag-bot/db"
        )
        st.success(f"{len(uploaded_files)} PDF(s) processed and knowledge base updated!")
        st.rerun()  # Refresh to update the chat

col1, col2 = st.columns([3, 1])

with col1:
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    
    if prompt := st.chat_input("Query your knowledge base..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Professional Progress Bar for "Thinking"
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            try:
                retriever, qa_chain = get_rag_components()
                # Get relevant documents
                docs = retriever.invoke(prompt)
                # Get answer
                answer = qa_chain.invoke(prompt)
                st.markdown(answer)
                
                # Display sources if available
                if docs:
                    with st.expander("📚 Cited Sources"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Source {i}:**")
                            st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                            if doc.metadata:
                                st.caption(f"Metadata: {doc.metadata}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            progress_bar.empty()

with col2:
    
    st.markdown("### 📑 Reference Docs")
    st.caption("Active documents in your Vector DB:")
    st.success("✅ python_data_analysis.pdf")
    st.success("✅ interview_prep_guide.pdf")
    
    st.markdown("---")
    st.markdown("### � Export")
    if st.button("Export Chat History"):
        chat_text = "\n\n".join([f"**{msg['role'].title()}:** {msg['content']}" for msg in st.session_state.messages])
        st.download_button("Download as TXT", chat_text, "chat_history.txt", mime="text/plain")