⚡ InsightLens | AI Document Intelligence
InsightLens is a Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents in real-time. It combines high-speed inference with a professional dashboard for secure, proprietary data analysis.

🌟 Features

Intelligent RAG Pipeline: Uses RecursiveCharacterTextSplitter to chunk documents and Chroma for vector storage.



Fast Inference: Integrated with Groq using the llama-3.1-8b-instant model for rapid response times.



Local Embeddings: Leverages HuggingFaceEmbeddings with the all-MiniLM-L6-v2 model for private data processing.


Dynamic UI: A custom Streamlit interface featuring engine status metrics, document uploaders, and source citations.


Multi-Interface Support: Run the project via a Web UI (interface.py), a CLI (query.py), or as an API (main.py).

📂 Project Structure

interface.py: The main Streamlit dashboard with custom CSS and sidebar controls.


ingest.py: Script to process PDF files into the vector database located in ./db.


main.py: A FastAPI implementation for serving the assistant as an API endpoint.


app.py: A lightweight, single-file version of the RAG assistant.



query.py: A command-line script for testing the RAG chain quickly.

🚀 Getting Started
1. Prerequisites
Install the required Python packages:

Bash

pip install -r reqirements.txt.txt
2. Environment Setup
Create a .env file in the root directory and add your Groq API key:

Code snippet

GROQ_API_KEY=your_actual_key_here
3. Usage
To run the Web Dashboard:

Bash

streamlit run interface.py
To ingest documents manually:

Bash

python ingest.py
🛠️ Tech Stack

Frontend: Streamlit 


Orchestration: LangChain 



LLM: Groq (Llama 3.1) 



Vector Database: ChromaDB 



Embeddings: HuggingFace
