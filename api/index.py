from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="iPsychiatrist API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = []

# Global variables for lazy loading
retrieval_chain = None
vectorstore = None

def initialize_rag():
    """Initialize RAG system with lazy loading"""
    global retrieval_chain, vectorstore
    
    if retrieval_chain is not None:
        return retrieval_chain
    
    try:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_classic.chains import create_retrieval_chain
        
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Load vector store
        vector_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "groq", "vectors")
        if os.path.exists(vector_path):
            vectorstore = FAISS.load_local(
                vector_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # Return a simple response if vectors don't exist
            return None
        
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7
        )
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template("""
You are iPsychiatrist, a compassionate and knowledgeable AI mental health assistant.
Provide empathetic, evidence-based responses using psychiatric literature.

Context from psychiatry:
{context}

User Question: {input}

Provide a clear, empathetic, and professional response.
If unsure, recommend consulting a licensed mental health professional.
""")
        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
    
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        return None

@app.get("/")
async def root():
    """Serve the frontend"""
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "iPsychiatrist API", "status": "running", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "ok", "message": "API is running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages"""
    try:
        chain = initialize_rag()
        
        if chain is None:
            # Fallback response if RAG is not initialized
            from langchain_groq import ChatGroq
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
            
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.7
            )
            
            response = llm.invoke(f"""You are iPsychiatrist, a compassionate AI mental health assistant.
            
User: {message.message}
            
Provide a helpful, empathetic response. If this is a serious mental health concern, recommend consulting a licensed professional.""")
            
            return ChatResponse(
                response=response.content,
                sources=[]
            )
        
        # Use RAG chain
        result = chain.invoke({"input": message.message})
        
        sources = []
        if "context" in result:
            sources = [doc.page_content[:200] + "..." for doc in result["context"][:3]]
        
        return ChatResponse(
            response=result["answer"],
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/api/info")
async def info():
    return {
        "name": "iPsychiatrist",
        "version": "1.0.0",
        "description": "AI-Powered Mental Health Assistant",
        "rag_enabled": vectorstore is not None
    }
