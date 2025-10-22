from http.server import BaseHTTPRequestHandler
import json
import os

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

HTML_CONTENT = """<!DOCTYPE html>
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
        .message { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .glass-effect { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); }
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
                    <button onclick="sendMessage()" id="sendButton" class="px-6 py-3 gradient-bg text-white rounded-lg font-semibold hover:opacity-90 transition">Send</button>
                </div>
                <p class="text-xs text-gray-500 mt-2 text-center">⚠️ Not medical advice. Always consult professionals.</p>
            </div>
        </main>
        <button onclick="alert('🆘 Crisis Help:\\n\\n🇺🇸 USA: 988\\n🇮🇳 India: 9152987821\\n🌍 International: findahelpline.com')" class="fixed bottom-24 right-6 bg-red-600 text-white px-4 py-2 rounded-full shadow-lg hover:bg-red-700">🆘 Crisis Help</button>
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
            addMessage('Typing...', 'assistant');
            try {
                const response = await fetch('/api/chat', { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }, 
                    body: JSON.stringify({ message: message }) 
                });
                const data = await response.json();
                document.getElementById('chatContainer').lastChild.remove();
                addMessage(data.response || 'Sorry, I encountered an error.', 'assistant');
            } catch (error) {
                document.getElementById('chatContainer').lastChild.remove();
                addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
            }
            sendButton.disabled = false;
            sendButton.textContent = 'Send';
        }
        function addMessage(text, role) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role === 'user' ? 'text-right' : 'text-left'}`;
            const bubbleClass = role === 'user' ? 'inline-block bg-purple-600 text-white' : 'inline-block bg-gray-800 text-gray-100';
            messageDiv.innerHTML = `<div class="${bubbleClass} px-4 py-3 rounded-lg max-w-2xl shadow-lg"><div class="font-semibold mb-1 text-sm">${role === 'user' ? 'You' : '🧠 iPsychiatrist'}</div><div>${text}</div><div class="text-xs opacity-70 mt-2">${new Date().toLocaleTimeString()}</div></div>`;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
    </script>
</body>
</html>"""

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "ok", "api_key_set": bool(GROQ_API_KEY)}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            try:
                if not GROQ_API_KEY:
                    response = {"response": "API key not configured. Please set GROQ_API_KEY in Vercel environment variables."}
                else:
                    from groq import Groq
                    client = Groq(api_key=GROQ_API_KEY)
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are iPsychiatrist, a compassionate AI mental health assistant. Provide helpful, empathetic responses."},
                            {"role": "user", "content": data['message']}
                        ],
                        temperature=0.7,
                        max_tokens=1024
                    )
                    response = {"response": completion.choices[0].message.content}
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {"response": f"Error: {str(e)}"}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()
