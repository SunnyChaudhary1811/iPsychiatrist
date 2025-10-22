from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Import HTML template
try:
    from api.frontend import HTML_TEMPLATE
except ImportError:
    try:
        from frontend import HTML_TEMPLATE
    except ImportError:
        # Fallback if imports fail
        HTML_TEMPLATE = None

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

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    if HTML_TEMPLATE:
        return HTMLResponse(content=HTML_TEMPLATE, status_code=200)
    
    # Inline fallback HTML
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iPsychiatrist - AI Mental Health Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .chat-container { height: calc(100vh - 200px); overflow-y: auto; scroll-behavior: smooth; }
        .chat-container::-webkit-scrollbar { width: 8px; }
        .chat-container::-webkit-scrollbar-track { background: #2d3748; }
        .chat-container::-webkit-scrollbar-thumb { background: #4a5568; border-radius: 4px; }
        .message { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .typing-indicator span { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #a0aec0; margin: 0 2px; animation: typing 1.4s infinite; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-10px); } }
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .glass-effect { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
    </style>
</head>
<body class="bg-gray-900 text-gray-100">
    <div class="min-h-screen flex flex-col">
        <header class="gradient-bg shadow-lg">
            <div class="max-w-4xl mx-auto px-4 py-6">
                <div class="flex items-center justify-center space-x-3">
                    <div class="text-4xl">🧠</div>
                    <div>
                        <h1 class="text-3xl font-bold text-white">iPsychiatrist</h1>
                        <p class="text-purple-200 text-sm">AI-Powered Mental Health Assistant</p>
                    </div>
                </div>
            </div>
        </header>
        <main class="flex-1 max-w-4xl w-full mx-auto px-4 py-6">
            <div id="chatContainer" class="chat-container space-y-4 mb-4">
                <div class="text-center py-12">
                    <div class="text-6xl mb-4">👋</div>
                    <h2 class="text-2xl font-semibold mb-2">Welcome to iPsychiatrist</h2>
                    <p class="text-gray-400 mb-6">I'm here to help with your mental health questions</p>
                    <div class="space-y-2 text-left max-w-md mx-auto">
                        <p class="text-sm text-gray-500 font-semibold">Try asking:</p>
                        <button onclick="sendSuggestion('What are the symptoms of anxiety?')" class="w-full text-left px-4 py-3 glass-effect rounded-lg hover:bg-gray-700 transition">💭 What are the symptoms of anxiety?</button>
                        <button onclick="sendSuggestion('How can I manage stress better?')" class="w-full text-left px-4 py-3 glass-effect rounded-lg hover:bg-gray-700 transition">🧘 How can I manage stress better?</button>
                        <button onclick="sendSuggestion('What is cognitive behavioral therapy?')" class="w-full text-left px-4 py-3 glass-effect rounded-lg hover:bg-gray-700 transition">💡 What is cognitive behavioral therapy?</button>
                    </div>
                </div>
            </div>
            <div class="sticky bottom-0 bg-gray-900 pt-4 pb-6">
                <div class="flex space-x-2">
                    <input type="text" id="messageInput" placeholder="💬 Type your question here..." class="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500" onkeypress="if(event.key==='Enter')sendMessage()">
                    <button onclick="sendMessage()" id="sendButton" class="px-6 py-3 gradient-bg text-white rounded-lg font-semibold hover:opacity-90 transition disabled:opacity-50">Send</button>
                </div>
                <p class="text-xs text-gray-500 mt-2 text-center">⚠️ Not medical advice. Always consult professionals.</p>
            </div>
        </main>
        <div id="crisisModal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-gray-800 rounded-lg p-6 max-w-md mx-4">
                <h3 class="text-xl font-bold mb-4">🆘 Crisis Help</h3>
                <p class="mb-4">If you're in crisis:</p>
                <ul class="space-y-2 mb-4">
                    <li>🇺🇸 USA: <strong>988</strong></li>
                    <li>🇮🇳 India: <strong>9152987821</strong></li>
                    <li>🌍 <a href="https://findahelpline.com" class="text-purple-400 hover:underline">findahelpline.com</a></li>
                </ul>
                <button onclick="closeModal()" class="w-full py-2 bg-purple-600 rounded-lg hover:bg-purple-700">Close</button>
            </div>
        </div>
        <button onclick="openModal()" class="fixed bottom-24 right-6 bg-red-600 text-white px-4 py-2 rounded-full shadow-lg hover:bg-red-700">🆘 Crisis Help</button>
    </div>
    <script>
        let messageCount = 0;
        function sendSuggestion(text) { document.getElementById('messageInput').value = text; sendMessage(); }
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            input.value = '';
            const sendButton = document.getElementById('sendButton');
            sendButton.disabled = true;
            sendButton.textContent = '⏳';
            if (messageCount === 0) document.getElementById('chatContainer').innerHTML = '';
            messageCount++;
            addMessage(message, 'user');
            const typingId = addTypingIndicator();
            try {
                const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: message }) });
                if (!response.ok) throw new Error('Failed');
                const data = await response.json();
                removeTypingIndicator(typingId);
                addMessage(data.response, 'assistant', data.sources);
            } catch (error) {
                removeTypingIndicator(typingId);
                addMessage('Sorry, I encountered an error. Please try again.', 'assistant', [], true);
            }
            sendButton.disabled = false;
            sendButton.textContent = 'Send';
        }
        function addMessage(text, role, sources = [], isError = false) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role === 'user' ? 'text-right' : 'text-left'}`;
            const bubbleClass = role === 'user' ? 'inline-block bg-purple-600 text-white' : isError ? 'inline-block bg-red-900 text-white' : 'inline-block bg-gray-800 text-gray-100';
            let sourcesHTML = '';
            if (sources && sources.length > 0) {
                sourcesHTML = `<div class="mt-2 text-xs text-gray-400"><details><summary class="cursor-pointer hover:text-gray-300">📚 Sources (${sources.length})</summary><div class="mt-2 space-y-1">${sources.map((s, i) => `<div class="p-2 bg-gray-900 rounded"><strong>Source ${i + 1}:</strong> ${s}</div>`).join('')}</div></details></div>`;
            }
            messageDiv.innerHTML = `<div class="${bubbleClass} px-4 py-3 rounded-lg max-w-2xl shadow-lg"><div class="font-semibold mb-1 text-sm">${role === 'user' ? 'You' : '🧠 iPsychiatrist'}</div><div class="whitespace-pre-wrap">${text}</div>${sourcesHTML}<div class="text-xs opacity-70 mt-2">${new Date().toLocaleTimeString()}</div></div>`;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        function addTypingIndicator() {
            const container = document.getElementById('chatContainer');
            const typingDiv = document.createElement('div');
            const id = 'typing-' + Date.now();
            typingDiv.id = id;
            typingDiv.className = 'message text-left';
            typingDiv.innerHTML = '<div class="inline-block bg-gray-800 px-4 py-3 rounded-lg shadow-lg"><div class="typing-indicator"><span></span><span></span><span></span></div></div>';
            container.appendChild(typingDiv);
            container.scrollTop = container.scrollHeight;
            return id;
        }
        function removeTypingIndicator(id) { const el = document.getElementById(id); if (el) el.remove(); }
        function openModal() { document.getElementById('crisisModal').classList.remove('hidden'); }
        function closeModal() { document.getElementById('crisisModal').classList.add('hidden'); }
        window.onload = () => { document.getElementById('messageInput').focus(); };
    </script>
</body>
</html>
    """, status_code=200)

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
