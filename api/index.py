from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import os

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>iPsychiatrist - Deployment Info</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                padding: 20px;
            }
            .container {
                background: white;
                padding: 3rem;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                text-align: center;
            }
            h1 {
                color: #667eea;
                margin-bottom: 1rem;
                font-size: 2.5rem;
            }
            .icon {
                font-size: 4rem;
                margin-bottom: 1rem;
            }
            p {
                color: #555;
                line-height: 1.6;
                margin: 1rem 0;
            }
            .info-box {
                background: #f8f9fa;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1.5rem 0;
                border-left: 4px solid #667eea;
            }
            .warning {
                background: #fff3cd;
                border-left-color: #ffc107;
                color: #856404;
            }
            code {
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
            }
            .steps {
                text-align: left;
                margin: 1.5rem 0;
            }
            .steps li {
                margin: 0.5rem 0;
                padding-left: 0.5rem;
            }
            a {
                color: #667eea;
                text-decoration: none;
                font-weight: 600;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon">🧠</div>
            <h1>iPsychiatrist</h1>
            <p><strong>AI-Powered Mental Health Assistant</strong></p>
            
            <div class="info-box warning">
                <p><strong>⚠️ Streamlit Apps Cannot Run on Vercel Serverless</strong></p>
                <p>Streamlit requires a persistent WebSocket connection and long-running process, which Vercel's serverless functions don't support (10-second timeout).</p>
            </div>
            
            <div class="info-box">
                <p><strong>✅ Recommended Deployment Options:</strong></p>
                <div class="steps">
                    <ol>
                        <li><strong>Streamlit Cloud</strong> (Free & Easy)
                            <br>→ <a href="https://streamlit.io/cloud" target="_blank">streamlit.io/cloud</a>
                        </li>
                        <li><strong>Railway</strong> (Free tier available)
                            <br>→ <a href="https://railway.app" target="_blank">railway.app</a>
                        </li>
                        <li><strong>Render</strong> (Free tier available)
                            <br>→ <a href="https://render.com" target="_blank">render.com</a>
                        </li>
                        <li><strong>Hugging Face Spaces</strong> (Free)
                            <br>→ <a href="https://huggingface.co/spaces" target="_blank">huggingface.co/spaces</a>
                        </li>
                    </ol>
                </div>
            </div>
            
            <div class="info-box">
                <p><strong>📝 Quick Deploy to Streamlit Cloud:</strong></p>
                <div class="steps">
                    <ol>
                        <li>Push your code to GitHub</li>
                        <li>Go to <a href="https://share.streamlit.io" target="_blank">share.streamlit.io</a></li>
                        <li>Connect your GitHub repo</li>
                        <li>Set main file path: <code>groq/app.py</code></li>
                        <li>Add <code>GROQ_API_KEY</code> in secrets</li>
                    </ol>
                </div>
            </div>
            
            <p style="margin-top: 2rem; color: #999; font-size: 0.9rem;">
                For local development, run: <code>streamlit run groq/app.py</code>
            </p>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    return {"status": "ok", "message": "API is running, but Streamlit app requires different hosting"}
