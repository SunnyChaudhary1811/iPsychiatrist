# 🧠 iPsychiatrist - AI Mental Health Assistant

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/SunnyChaudhary1811/iPsychiatrist)

An AI-powered mental health assistant that uses RAG (Retrieval-Augmented Generation) to provide evidence-based responses using psychiatric literature.

## 🚀 Live Demo

Visit the deployed app: [Deploy on Vercel](https://vercel.com/new/clone?repository-url=https://github.com/SunnyChaudhary1811/iPsychiatrist)

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes! ⚡
- **[SETUP.md](SETUP.md)** - Complete setup and installation guide
- **[VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md)** - Vercel deployment instructions
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project architecture and file structure

## ✨ Features

- 💬 Real-time chat interface
- 🧠 RAG-powered responses using psychiatric textbook
- 📚 Source citations for transparency
- 🎨 Beautiful, responsive UI
- 🆘 Crisis help resources
- ⚡ Fast API responses

## 🏗️ Architecture

### Frontend
- HTML/CSS/JavaScript with Tailwind CSS
- Real-time chat interface
- Typing indicators and animations

### Backend
- FastAPI REST API
- LangChain for RAG pipeline
- FAISS vector store for document retrieval

## 📖 How It Works

### RAG Pipeline

1. **User Input** - Question submitted via web interface
2. **Document Retrieval** - FAISS searches vector store for relevant context
3. **Context Generation** - Retrieved chunks combined with user query
4. **LLM Processing** - Groq LLM generates response using context
5. **Response Display** - Answer shown with source citations

**Flow:** User Query → Vector Search → Context Retrieval → LLM Generation → Response + Sources

This RAG approach ensures responses are grounded in psychiatric literature, providing accurate and evidence-based information.

---

## 🚀 Deployment

### Deploy to Vercel (Recommended)

1. **Fork/Clone this repository**

2. **Add Environment Variable:**
   - Go to Vercel Dashboard → Settings → Environment Variables
   - Add: `GROQ_API_KEY` = `your_groq_api_key_here`

3. **Deploy:**
   ```bash
   git add .
   git commit -m "Deploy to Vercel"
   git push origin main
   ```

4. **Vercel will auto-deploy!** 🎉

See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) for detailed instructions.

---

## 💻 Local Development

### Run the Web App (FastAPI):
```bash
pip install -r api/requirements.txt
python -m uvicorn api.index:app --reload
```
Open: http://localhost:8000

### Run the Streamlit Version (Alternative):
```bash
cd groq
pip install -r ../Requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
iPsychiatrist/
├── api/
│   ├── index.py              # FastAPI backend
│   └── requirements.txt      # Python dependencies
├── public/
│   └── index.html           # Frontend chat interface
├── groq/
│   ├── app.py               # Streamlit version (alternative)
│   ├── vectors/             # FAISS vector store
│   └── *.pdf                # Psychiatry textbook
├── vercel.json              # Vercel configuration
├── package.json             # Project metadata
└── README.md                # This file
```

---

## 🔧 Configuration

### Environment Variables:
- `GROQ_API_KEY` - Your Groq API key (required)

### Get a Groq API Key:
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Generate an API key
4. Add it to your `.env` file or Vercel environment variables

---

## ⚠️ Important Notes

- **Not Medical Advice:** This is an AI assistant for informational purposes only
- **Crisis Resources:** Built-in crisis helpline information
- **Data Privacy:** No conversation data is stored
- **Rate Limits:** Subject to Groq API rate limits

---

## 🆘 Crisis Resources

If you're in crisis, please contact:
- 🇺🇸 USA: **988** (Suicide & Crisis Lifeline)
- 🇮🇳 India: **9152987821**
- 🌍 International: [findahelpline.com](https://findahelpline.com)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

**Sunny Chaudhary**
- GitHub: [@SunnyChaudhary1811](https://github.com/SunnyChaudhary1811)

---

## 🙏 Acknowledgments

- Groq for fast LLM inference
- LangChain for RAG framework
- HuggingFace for embeddings
- Vercel for hosting
