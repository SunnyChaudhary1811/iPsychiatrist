# 🚀 Deployment Guide

## Quick Deploy to Vercel

Your app is ready to deploy on Vercel with FastAPI backend + HTML/JS frontend.

---

## 🔧 Deployment Steps

### 1. **Add Environment Variable in Vercel**

Before deploying, you MUST add your API key:

1. Go to your Vercel project dashboard
2. Click **Settings** → **Environment Variables**
3. Add:
   - **Name:** `GROQ_API_KEY`
   - **Value:** `your_groq_api_key_here`
   - **Environment:** Production, Preview, Development (select all)
4. Click **Save**

### 2. **Commit and Push Changes**

```bash
git add .
git commit -m "Convert to FastAPI + React for Vercel deployment"
git push origin main
```

### 3. **Vercel Will Auto-Deploy**

- Vercel will automatically detect the changes
- Build will complete successfully
- Your app will be live! 🎉

---

## 🎨 Features

### Frontend (public/index.html):
- ✨ Beautiful dark theme UI
- 💬 Real-time chat interface
- 📱 Fully responsive
- 🎯 Suggested questions
- 🆘 Crisis help modal
- ⌨️ Keyboard shortcuts (Enter to send)
- 🎭 Typing indicators
- 📚 Source citations (when RAG is enabled)

### Backend (api/index.py):
- 🤖 FastAPI REST API
- 🧠 RAG (Retrieval-Augmented Generation) support
- 📖 Uses your psychiatry textbook vectors
- 🔄 Fallback to direct LLM if vectors unavailable
- ⚡ Fast response times
- 🔒 CORS enabled for security

---

## 📊 API Endpoints

- **`GET /`** - Serves the chat interface
- **`POST /api/chat`** - Send messages and get responses
- **`GET /api/info`** - Get API information
- **`GET /health`** - Health check

---

## 🧪 Local Testing

### Test the FastAPI backend:
```bash
cd d:\iPsychiatrist
pip install -r api/requirements.txt
python -m uvicorn api.index:app --reload
```

Then open: http://localhost:8000

### Test the Streamlit app (still works):
```bash
cd groq
streamlit run app.py
```

---

## ⚠️ Important Notes

### Vector Store:
- The app looks for vectors in `groq/vectors/`
- If vectors exist, it uses RAG for better responses
- If not, it falls back to direct LLM queries
- **To enable RAG:** Make sure `groq/vectors/` is committed to git

### Large Files:
- The PDF file (28MB) might be too large for Vercel
- Consider using Git LFS or hosting it separately
- The app works without the PDF if vectors are pre-generated

### Vercel Limits:
- **Function timeout:** 10 seconds (Hobby), 60 seconds (Pro)
- **Function size:** 50MB max
- If you hit limits, consider:
  - Pre-generating vectors
  - Using smaller embedding models
  - Upgrading to Vercel Pro

---

## 🔍 Troubleshooting

### "GROQ_API_KEY not configured" error:
- Make sure you added the environment variable in Vercel
- Redeploy after adding the variable

### "Error initializing RAG" warning:
- This is normal if vectors don't exist
- App will still work using direct LLM
- To fix: Generate vectors locally and commit them

### Build fails:
- Check Vercel build logs
- Ensure all dependencies are in `api/requirements.txt`
- Some packages might not work on Vercel (check compatibility)

### Slow responses:
- First request is slow (cold start)
- Subsequent requests are faster
- Consider Vercel Pro for better performance

---

## 🎯 Next Steps

1. ✅ Commit and push the changes
2. ✅ Add GROQ_API_KEY in Vercel
3. ✅ Wait for deployment
4. ✅ Test your live app!
5. 🎨 Customize the UI if needed
6. 📊 Monitor usage in Vercel dashboard

---

## 🆘 Need Help?

- Check Vercel deployment logs
- Test locally first
- Ensure API key is set
- Check function timeout limits

---

## 🎉 You're All Set!

Your app is now ready to deploy on Vercel. Just commit, push, and watch it go live! 🚀
