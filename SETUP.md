# 🛠️ Setup Guide

## Prerequisites

- Python 3.9 or higher
- Git
- Groq API key ([Get one free](https://console.groq.com))

---

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SunnyChaudhary1811/iPsychiatrist.git
cd iPsychiatrist
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r Requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the Application

**Option A: FastAPI Version (Recommended for Vercel)**
```bash
python -m uvicorn api.index:app --reload
```
Open: http://localhost:8000

**Option B: Streamlit Version (Alternative UI)**
```bash
cd groq
streamlit run app.py
```
Open: http://localhost:8501

---

## Vercel Deployment Setup

### 1. Fork/Clone Repository

Fork this repository to your GitHub account.

### 2. Import to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Vercel will auto-detect the configuration

### 3. Add Environment Variables

In Vercel Dashboard:
1. Go to **Settings** → **Environment Variables**
2. Add:
   - **Key:** `GROQ_API_KEY`
   - **Value:** Your Groq API key
   - **Environments:** Production, Preview, Development (all)

### 4. Deploy

```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

Vercel will automatically deploy your app! 🎉

---

## Vector Store Setup (Optional but Recommended)

For better responses, generate vector embeddings from your documents:

### 1. Add PDF Documents

Place your psychiatry textbooks in the `groq/` directory.

### 2. Generate Vectors

Run the Streamlit app once:
```bash
cd groq
streamlit run app.py
```

The app will automatically create the `vectors/` directory with embeddings.

### 3. Commit Vectors

```bash
git add groq/vectors/
git commit -m "Add vector embeddings"
git push
```

**Note:** If vectors are too large, consider using [Git LFS](https://git-lfs.github.com/).

---

## Troubleshooting

### "GROQ_API_KEY not found"
- Ensure `.env` file exists in root directory
- Check that the key is correctly formatted
- For Vercel, verify environment variable is set in dashboard

### "Module not found" errors
```bash
pip install -r Requirements.txt --upgrade
```

### Vector store not loading
- Ensure `groq/vectors/` directory exists
- Check file permissions
- Regenerate vectors if corrupted

### Vercel deployment fails
- Check build logs in Vercel dashboard
- Ensure all dependencies are in `api/requirements.txt`
- Verify Python version compatibility

### Slow responses
- First request has cold start delay (normal)
- Consider upgrading to Vercel Pro for better performance
- Check Groq API rate limits

---

## Development Tips

### Hot Reload
Both FastAPI and Streamlit support hot reload:
- FastAPI: `--reload` flag (already included)
- Streamlit: Automatic on file save

### Testing API Endpoints
Use the built-in FastAPI docs:
- Open: http://localhost:8000/docs
- Interactive API testing interface

### Debugging
Enable debug mode in `.env`:
```env
DEBUG=True
LOG_LEVEL=DEBUG
```

---

## Next Steps

1. ✅ Complete setup
2. ✅ Test locally
3. ✅ Generate vectors (optional)
4. ✅ Deploy to Vercel
5. 🎨 Customize UI (optional)
6. 📊 Monitor usage

---

## Need Help?

- Check [README.md](README.md) for overview
- See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for architecture
- Read [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) for deployment details
- Open an issue on GitHub

---

Happy coding! 🚀
