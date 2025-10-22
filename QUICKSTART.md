# ⚡ Quick Start Guide

Get iPsychiatrist running in 5 minutes!

## 🚀 Deploy to Vercel (Fastest)

1. **Click the button:**
   
   [![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/SunnyChaudhary1811/iPsychiatrist)

2. **Add your API key:**
   - In Vercel dashboard: Settings → Environment Variables
   - Add: `GROQ_API_KEY` = `your_key_here`

3. **Done!** Your app is live 🎉

---

## 💻 Run Locally (5 minutes)

### Step 1: Clone
```bash
git clone https://github.com/SunnyChaudhary1811/iPsychiatrist.git
cd iPsychiatrist
```

### Step 2: Install
```bash
pip install -r api/requirements.txt
```

### Step 3: Configure
Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 4: Run
```bash
python -m uvicorn api.index:app --reload
```

### Step 5: Open
Visit: http://localhost:8000

---

## 🔑 Get Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Create API key
4. Copy and use in `.env` or Vercel

---

## 📖 Need More Help?

- **Setup Issues?** → [SETUP.md](SETUP.md)
- **Deployment?** → [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md)
- **Architecture?** → [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **General Info?** → [README.md](README.md)

---

## ✅ What's Next?

After setup:
1. Test the chat interface
2. Try example questions
3. Check crisis resources
4. Customize if needed
5. Share with others!

---

**That's it! You're ready to go! 🎉**
