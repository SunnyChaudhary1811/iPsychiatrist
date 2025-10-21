# Converting iPsychiatrist to React/Next.js

## ✅ Current Setup (Streamlit + JavaScript)
Your app now uses:
- **Streamlit** for backend and UI framework
- **JavaScript** for interactive features (Enter key, auto-scroll, input clearing)
- **Python** for AI/ML processing

## 🚀 Option 1: Keep Streamlit + Enhanced JavaScript (Recommended for now)

**Pros:**
- ✅ Already working
- ✅ Easy to maintain
- ✅ No need to rewrite backend
- ✅ JavaScript handles all interactivity

**Current JavaScript Features:**
- Enter key to send
- Auto-clear input after send
- Auto-scroll to bottom
- Loading states
- All interactive features work!

---

## 🔄 Option 2: Convert to React Frontend + FastAPI Backend

If you want a full React app, here's the architecture:

### **Architecture:**
```
Frontend (React/Next.js)          Backend (FastAPI)
├── Chat UI                       ├── /api/chat endpoint
├── Message handling              ├── LangChain integration
├── State management (Redux)      ├── Vector store
└── Styling (TailwindCSS)         └── AI model calls
```

### **Tech Stack:**
- **Frontend:** Next.js 14 + TypeScript + TailwindCSS + shadcn/ui
- **Backend:** FastAPI + LangChain + Python
- **State:** React Context or Zustand
- **Styling:** TailwindCSS + Framer Motion

### **Steps to Convert:**

#### 1. **Create FastAPI Backend**
```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Your existing LangChain code here
    response = retrieval_chain.invoke({"input": request.message})
    return ChatResponse(
        answer=response['answer'],
        sources=[doc.page_content for doc in response['context']]
    )
```

#### 2. **Create React Frontend**
```bash
npx create-next-app@latest ipsychiatrist-frontend
cd ipsychiatrist-frontend
npm install axios framer-motion
```

```tsx
// app/page.tsx
'use client';

import { useState } from 'react';
import axios from 'axios';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMessage = { role: 'user', content: input };
    setMessages([...messages, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/chat', {
        message: input
      });
      
      const aiMessage = { role: 'assistant', content: response.data.answer };
      setMessages([...messages, userMessage, aiMessage]);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 p-4 text-center">
        <h1 className="text-2xl font-bold text-white">🧠 iPsychiatrist</h1>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`p-4 rounded-lg ${
              msg.role === 'user'
                ? 'bg-blue-600 ml-auto max-w-[80%]'
                : 'bg-gray-700 mr-auto max-w-[80%]'
            }`}
          >
            <p className="text-white">{msg.content}</p>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="bg-gray-800 p-4 border-t border-gray-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about mental health..."
            className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500"
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="bg-green-500 text-white px-6 py-3 rounded-lg hover:bg-green-600 disabled:opacity-50"
          >
            {loading ? '⏳' : '➤'} Send
          </button>
        </div>
      </div>
    </div>
  );
}
```

#### 3. **Run Both Servers**
```bash
# Terminal 1 - Backend
cd D:\iPsychiatrist\groq
uvicorn backend.main:app --reload --port 8000

# Terminal 2 - Frontend
cd ipsychiatrist-frontend
npm run dev
```

---

## 📊 Comparison

| Feature | Streamlit + JS | React + FastAPI |
|---------|---------------|-----------------|
| **Development Speed** | ⚡ Fast | 🐢 Slower |
| **Customization** | 🟡 Limited | ✅ Full control |
| **Performance** | 🟢 Good | ✅ Excellent |
| **Deployment** | ✅ Easy | 🟡 More complex |
| **Learning Curve** | ✅ Easy | 🟡 Moderate |
| **Current State** | ✅ Working! | ❌ Need to build |

---

## 💡 My Recommendation

**Stick with Streamlit + JavaScript for now!**

**Why?**
1. ✅ Your app is already working perfectly
2. ✅ JavaScript handles all interactivity you need
3. ✅ Easier to maintain and update
4. ✅ Faster development
5. ✅ Python backend is already optimized

**When to switch to React:**
- Need mobile app (React Native)
- Need complex animations
- Need offline functionality
- Need to scale to millions of users
- Want to learn React/Next.js

---

## 🎯 Current Status

Your Streamlit app now has:
- ✅ JavaScript for Enter key
- ✅ JavaScript for auto-clear input
- ✅ JavaScript for auto-scroll
- ✅ Loading states
- ✅ Dark theme
- ✅ All interactive features

**It's production-ready as is!** 🚀

---

## 📝 Next Steps (If you want React)

1. Create FastAPI backend (1-2 hours)
2. Create React frontend (2-3 hours)
3. Style with TailwindCSS (1 hour)
4. Add animations (1 hour)
5. Deploy both (1 hour)

**Total:** ~8 hours of work

**Or keep current Streamlit app:** 0 hours, already done! ✅

---

## 🚀 Quick Start React Template

If you decide to go React, I can provide:
- Complete FastAPI backend code
- Complete Next.js frontend code
- Docker setup for deployment
- Vercel deployment config
- All styling and animations

Just let me know! 🎉
