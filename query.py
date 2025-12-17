import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def ask_bot(question):
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    
    vector_db = Chroma(persist_directory="./db", embedding_function=embeddings)

    
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=os.getenv("GROQ_API_KEY"), 
        model_name="llama-3.1-8b-instant"
    )

    
    retriever = vector_db.as_retriever()
    
    prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant specializing in answering questions based on the provided document context.

Context: {context}

Question: {question}

Provide a clear, detailed, and accurate answer. If the context doesn't contain enough information to fully answer the question, explain what you can based on the available information and suggest what might be needed. Be conversational and helpful, like a knowledgeable expert.""")
    
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    
    response = chain.invoke(question)
    print(f"\nAI Answer: {response}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = input("Ask a question about your PDF: ")
    ask_bot(query)