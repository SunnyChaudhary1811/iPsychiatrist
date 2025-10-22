# 📁 Project Structure

```
iPsychiatrist/
│
├── api/                          # Backend API
│   ├── index.py                  # FastAPI application with RAG
│   └── requirements.txt          # Python dependencies
│
├── public/                       # Frontend
│   └── index.html               # Chat interface (HTML/CSS/JS)
│
├── groq/                         # Streamlit version (optional)
│   ├── app.py                   # Streamlit app (alternative UI)
│   ├── vectors/                 # FAISS vector store
│   │   ├── index.faiss          # Vector embeddings
│   │   └── index.pkl            # Metadata
│   └── *.pdf                    # Source documents (gitignored)
│
├── .env                          # Environment variables (gitignored)
├── .gitignore                   # Git ignore rules
├── .vercelignore                # Vercel ignore rules
├── vercel.json                  # Vercel configuration
├── package.json                 # Project metadata
├── Requirements.txt             # Full Python dependencies
├── README.md                    # Main documentation
├── VERCEL_DEPLOYMENT.md         # Deployment guide
└── PROJECT_STRUCTURE.md         # This file

```

## 📄 File Descriptions

### Core Application Files

**`api/index.py`**
- FastAPI REST API server
- Handles chat requests via `/api/chat` endpoint
- Implements RAG (Retrieval-Augmented Generation)
- Loads FAISS vector store for context retrieval
- Falls back to direct LLM if vectors unavailable

**`public/index.html`**
- Single-page chat application
- Modern UI with Tailwind CSS
- Real-time messaging with typing indicators
- Crisis help resources modal
- Fully responsive design

**`groq/app.py`**
- Alternative Streamlit interface
- Can be used for local development
- Same RAG functionality as FastAPI version
- Run with: `streamlit run groq/app.py`

### Configuration Files

**`vercel.json`**
- Vercel deployment configuration
- Routes API calls to Python backend
- Serves static frontend files
- Environment variable references

**`package.json`**
- Node.js project metadata
- Scripts for development
- Project information

**`.env`** (create this file)
```
GROQ_API_KEY=your_api_key_here
```

**`.gitignore`**
- Excludes sensitive files (.env)
- Ignores virtual environments
- Excludes large PDF files
- Python cache files

**`.vercelignore`**
- Excludes unnecessary files from deployment
- Reduces deployment size
- Keeps only production files

### Data Files

**`groq/vectors/`**
- FAISS vector database
- Pre-computed embeddings from psychiatry textbook
- Used for semantic search in RAG pipeline
- Optional but recommended for better responses

**`groq/*.pdf`**
- Source documents (psychiatry textbooks)
- Gitignored due to size
- Only needed for generating vectors
- Not required for deployment if vectors exist

## 🔄 Data Flow

```
User Input (public/index.html)
    ↓
POST /api/chat (api/index.py)
    ↓
Load Vector Store (groq/vectors/)
    ↓
Semantic Search (FAISS)
    ↓
Retrieve Context
    ↓
LLM Generation (Groq API)
    ↓
Response with Sources
    ↓
Display to User
```

## 🛠️ Technology Stack

### Frontend
- HTML5
- CSS3 (Tailwind CSS via CDN)
- Vanilla JavaScript
- Fetch API for HTTP requests

### Backend
- Python 3.9+
- FastAPI (web framework)
- LangChain (RAG framework)
- FAISS (vector database)
- HuggingFace (embeddings)
- Groq (LLM inference)

### Deployment
- Vercel (hosting)
- Serverless functions
- Static file serving

## 📦 Dependencies

### Production (api/requirements.txt)
- `fastapi` - Web framework
- `python-dotenv` - Environment variables
- `langchain` - RAG framework
- `langchain-groq` - Groq integration
- `langchain-huggingface` - Embeddings
- `sentence-transformers` - Text embeddings
- `faiss-cpu` - Vector search
- `groq` - LLM API client

### Development (Requirements.txt)
- All production dependencies
- Plus: `streamlit`, `uvicorn`, additional tools

## 🚀 Deployment Targets

### Vercel (Recommended)
- Serverless Python functions
- Static file hosting
- Automatic deployments
- Environment variables
- Custom domains

### Alternative: Streamlit Cloud
- For Streamlit version only
- Free tier available
- Easy GitHub integration
- Built-in secrets management

## 🔐 Environment Variables

Required:
- `GROQ_API_KEY` - Your Groq API key

Optional:
- `MODEL_NAME` - LLM model (default: llama-3.3-70b-versatile)
- `TEMPERATURE` - Response creativity (default: 0.7)

## 📝 Notes

- The FastAPI version is optimized for Vercel
- The Streamlit version is for local development
- Both versions share the same vector store
- PDF files are excluded from git (use Git LFS if needed)
- Vectors should be committed for best performance
