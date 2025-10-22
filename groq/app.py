import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="iPsychiatrist - AI Mental Health Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- LOAD ENV ---
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "vectors")
PDF_PATH = os.path.join(os.path.dirname(__file__), "New Oxford Textbook of Psychiatry-2161hlm.pdf")

# --- CACHE FUNCTIONS ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def initialize_vector_store():
    embeddings = get_embeddings()

    if os.path.exists(VECTOR_STORE_PATH):
        try:
            return FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"Could not load existing vectors: {e}. Rebuilding...")

    if not os.path.exists(PDF_PATH):
        st.error(f"PDF file not found at: {PDF_PATH}")
        st.stop()

    with st.spinner("📚 Creating vector embeddings from textbook (first 50 pages)..."):
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs[:50])
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)
        st.success("✅ Embeddings created successfully.")
    return vectorstore

# --- CSS THEME ---
st.markdown("""
<style>
    .stApp {
        background: #343541;
        color: #ececf1;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
        max-width: 900px;
        margin: auto;
    }

    .main-header {
        text-align: center;
        padding: 1rem;
        background: #444654;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: #ececf1;
        font-size: 1.7rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #b1b1c0;
        font-size: 0.9rem;
        margin: 0;
    }

    .chat-container {
        background: #343541;
        border-radius: 10px;
        padding: 1rem;
        height: calc(100vh - 280px);
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        scroll-behavior: smooth;
        border: 1px solid #565869;
    }

    .chat-message {
        border-radius: 10px;
        padding: 1.2rem;
        line-height: 1.6;
        font-size: 1rem;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    .user-message {
        background: #3c3d46;
        align-self: flex-end;
        border-left: 3px solid #10a37f;
    }

    .assistant-message {
        background: #444654;
        align-self: flex-start;
        border-left: 3px solid #19c37d;
    }

    .message-time {
        font-size: 0.8rem;
        color: #8e8ea0;
        margin-top: 0.5rem;
        text-align: right;
    }

    .welcome-message {
        text-align: center;
        color: #c5c5d2;
        padding: 4rem 1rem;
    }
    .welcome-message h3 {
        font-size: 1.8rem;
        color: #ececf1;
        margin-bottom: 1rem;
    }
    .welcome-message p {
        color: #a9a9b4;
        margin: 0.4rem 0;
    }

    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #40414f;
        padding: 1rem 1.5rem;
        border-top: 1px solid #565869;
        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.3);
        z-index: 999;
    }

    .stTextInput > div > div > input {
        background: #343541;
        border: 2px solid #565869;
        border-radius: 8px;
        color: #ececf1;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        height: 52px;
        transition: all 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #19c37d;
        box-shadow: 0 0 0 2px rgba(25,195,125,0.2);
    }
    .stTextInput > label { display: none; }

    .stButton > button {
        height: 52px;
        background: #19c37d;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #15b06c;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(25,195,125,0.3);
    }
    .stButton > button:disabled {
        background: #565869;
        opacity: 0.6;
        cursor: not-allowed;
    }

    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: #565869;
        border-radius: 10px;
    }
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #6e6e80;
    }

    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🧠 iPsychiatrist</h1>
    <p>AI-Powered Mental Health Assistant</p>
</div>
""", unsafe_allow_html=True)

# --- INIT ---
try:
    vectorstore = initialize_vector_store()
except Exception as e:
    st.error(f"Error initializing vector store: {e}")
    st.stop()

try:
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )
except Exception as e:
    st.error(f"❌ Error initializing Groq model: {e}")
    st.info("💡 Check available models at https://console.groq.com/docs/models")
    st.stop()

prompt_template = ChatPromptTemplate.from_template("""
You are iPsychiatrist, a compassionate and knowledgeable AI mental health assistant.
Provide empathetic, evidence-based responses using psychiatric literature.

Context from psychiatry:
{context}

User Question: {input}

Provide a clear, empathetic, and professional response.
If unsure, recommend consulting a licensed mental health professional.
""")

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- SESSION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"**💬 Total Chats:** {st.session_state.total_queries}")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()
    st.markdown("---")
    st.caption("⚠️ Not medical advice. Always consult professionals.")
    with st.expander("🆘 Crisis Help"):
        st.markdown("""
        **If you're in crisis:**
        - 🇺🇸 USA: 988
        - 🇮🇳 India: 9152987821
        """)

# --- CHAT UI ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-message">
        <h3>👋 Welcome to iPsychiatrist</h3>
        <p>I'm here to help with your mental health questions.</p>
        <br>
        <p><strong>Try asking:</strong></p>
        <p>💭 "What are the symptoms of anxiety?"</p>
        <p>🧘 "How can I manage stress better?"</p>
        <p>💡 "What is cognitive behavioral therapy?"</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    timestamp = msg.get("timestamp", "")
    css_class = "user-message" if role == "user" else "assistant-message"
    name = "You" if role == "user" else "🧠 iPsychiatrist"

    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div><strong>{name}:</strong></div>
        <div>{content}</div>
        <div class="message-time">{timestamp}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- INPUT AREA ---
st.markdown('<div class="input-container">', unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "message",
        key="user_input",
        placeholder="💬 Type your question here...",
        label_visibility="collapsed"
    )

with col2:
    is_processing = st.session_state.get("is_processing", False)
    send_button = st.button(
        "⏳" if is_processing else "Send",
        use_container_width=True,
        disabled=is_processing
    )

st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIC ---
if send_button and user_input:
    st.session_state.is_processing = True
    timestamp = datetime.now().strftime("%I:%M %p")

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })

    with st.spinner("🤔 Thinking..."):
        try:
            response = retrieval_chain.invoke({"input": user_input})
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['answer'],
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            st.session_state.total_queries += 1
            st.session_state.is_processing = False

            with st.expander("📚 View Sources"):
                for i, doc in enumerate(response["context"], 1):
                    st.caption(f"**Source {i}:**")
                    st.text(doc.page_content[:400] + "...")
                    if i < len(response["context"]):
                        st.markdown("---")
            st.rerun()
        except Exception as e:
            st.session_state.is_processing = False
            st.error(f"❌ Error: {e}")

elif send_button and not user_input:
    st.warning("⚠️ Please enter a question.")
