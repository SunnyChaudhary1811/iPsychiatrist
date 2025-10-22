# 🚀 iPsychiatrist Deployment Guide

## ⚠️ Important: Vercel Limitation

**Your Streamlit app cannot run on Vercel** because:
- Streamlit requires persistent WebSocket connections
- Vercel serverless functions have a 10-second timeout
- Streamlit needs a long-running process

The current Vercel deployment will show an informational page explaining this.

---

## ✅ Recommended Deployment Platforms

### 1. **Streamlit Cloud** (Easiest & Free)

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub repository: `SunnyChaudhary1811/iPsychiatrist`
5. Set main file path: `groq/app.py`
6. Click "Advanced settings" → "Secrets"
7. Add your environment variable:
   ```toml
   GROQ_API_KEY = "your_api_key_here"
   ```
8. Click "Deploy"

**Pros:**
- Free for public repos
- Built specifically for Streamlit
- Auto-deploys on git push
- Easy secrets management

---

### 2. **Railway** (Recommended for Production)

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect Python
5. Add environment variable:
   - Key: `GROQ_API_KEY`
   - Value: `your_api_key_here`
6. Set start command: `streamlit run groq/app.py --server.port=$PORT --server.address=0.0.0.0`
7. Deploy

**Pros:**
- $5 free credit monthly
- Better performance than free tiers
- Custom domains
- Persistent storage

---

### 3. **Render** (Free Tier Available)

**Steps:**
1. Go to [render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repo
4. Configure:
   - **Name:** iPsychiatrist
   - **Environment:** Python 3
   - **Build Command:** `pip install -r Requirements.txt`
   - **Start Command:** `streamlit run groq/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. Add environment variable:
   - Key: `GROQ_API_KEY`
   - Value: `your_api_key_here`
6. Click "Create Web Service"

**Pros:**
- Free tier available
- Auto-deploy on push
- SSL certificates included

---

### 4. **Hugging Face Spaces** (Free & AI-Focused)

**Steps:**
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Streamlit" as SDK
4. Upload your files or connect GitHub
5. Create a file named `app.py` in root with:
   ```python
   import sys
   sys.path.append('./groq')
   from groq.app import *
   ```
6. Add secret in Space settings:
   - Name: `GROQ_API_KEY`
   - Value: `your_api_key_here`

**Pros:**
- Completely free
- Great for AI/ML apps
- Built-in GPU support (if needed)

---

## 📝 Files Created for Vercel

I've created these files to fix the 404 error on Vercel:

1. **`vercel.json`** - Vercel configuration
2. **`api/index.py`** - Informational endpoint
3. **`api/requirements.txt`** - Minimal dependencies

Now when you visit your Vercel deployment, you'll see a helpful page instead of 404.

---

## 🔄 Next Steps

1. **Commit and push the new files:**
   ```bash
   git add vercel.json api/
   git commit -m "Add Vercel config and deployment info"
   git push
   ```

2. **Choose a deployment platform** from the options above

3. **Deploy your Streamlit app** to the chosen platform

4. **(Optional)** Keep Vercel deployment as an informational landing page

---

## 🧪 Local Testing

To run locally:
```bash
cd groq
streamlit run app.py
```

---

## 🆘 Troubleshooting

### PDF File Path Issue
Your app has a hardcoded path:
```python
PDF_PATH = "D:\\iPsychiatrist\\groq\\New Oxford Textbook of Psychiatry-2161hlm.pdf"
```

For deployment, change this to a relative path:
```python
PDF_PATH = os.path.join(os.path.dirname(__file__), "New Oxford Textbook of Psychiatry-2161hlm.pdf")
```

### Large File Warning
The PDF file (28MB) might cause issues. Consider:
- Using Git LFS for large files
- Hosting the PDF separately (S3, Google Drive)
- Pre-processing and uploading only the vector embeddings

---

## 📚 Additional Resources

- [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [Railway Python Guide](https://docs.railway.app/guides/python)
- [Render Streamlit Guide](https://render.com/docs/deploy-streamlit)

---

**Need help?** Open an issue or check the platform-specific documentation.
