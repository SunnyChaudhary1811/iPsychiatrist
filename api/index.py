from http.server import BaseHTTPRequestHandler
import json
import os
import time

# Cache buster: 20251022114500
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>iPsychiatrist - Mental Health AI</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
*{font-family:'Inter',sans-serif}
.gradient-bg{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%)}
</style>
</head>
<body class="bg-gray-900 text-white">
<div class="min-h-screen flex flex-col">
<header class="gradient-bg shadow-lg py-6">
<div class="max-w-4xl mx-auto px-4 text-center">
<div class="text-5xl mb-2">🧠</div>
<h1 class="text-4xl font-bold">iPsychiatrist</h1>
<p class="text-purple-200">AI Mental Health Assistant</p>
</div>
</header>
<main class="flex-1 max-w-4xl w-full mx-auto px-4 py-8">
<div id="chat" class="space-y-4 mb-6 h-96 overflow-y-auto">
<div class="text-center py-12">
<div class="text-6xl mb-4">👋</div>
<h2 class="text-2xl font-bold mb-4">Welcome!</h2>
<p class="text-gray-400 mb-6">Ask me anything about mental health</p>
<div class="space-y-2 max-w-md mx-auto">
<button onclick="ask('What are symptoms of anxiety?')" class="w-full text-left px-4 py-3 bg-gray-800 rounded-lg hover:bg-gray-700">💭 What are symptoms of anxiety?</button>
<button onclick="ask('How to manage stress?')" class="w-full text-left px-4 py-3 bg-gray-800 rounded-lg hover:bg-gray-700">🧘 How to manage stress?</button>
<button onclick="ask('What is CBT?')" class="w-full text-left px-4 py-3 bg-gray-800 rounded-lg hover:bg-gray-700">💡 What is CBT?</button>
</div>
</div>
</div>
<div class="flex gap-2">
<input type="text" id="input" placeholder="Type your question..." class="flex-1 px-4 py-3 bg-gray-800 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500" onkeypress="if(event.key==='Enter')send()">
<button onclick="send()" id="btn" class="px-6 py-3 gradient-bg rounded-lg font-semibold hover:opacity-90">Send</button>
</div>
<p class="text-xs text-gray-500 mt-2 text-center">⚠️ Not medical advice</p>
</main>
</div>
<script>
let count=0;
function ask(q){document.getElementById('input').value=q;send()}
async function send(){
const input=document.getElementById('input');
const msg=input.value.trim();
if(!msg)return;
input.value='';
const btn=document.getElementById('btn');
btn.disabled=true;
btn.textContent='⏳';
if(count===0)document.getElementById('chat').innerHTML='';
count++;
add(msg,'user');
add('Typing...','bot');
try{
const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
const data=await res.json();
document.getElementById('chat').lastChild.remove();
add(data.response||'Error occurred','bot');
}catch(e){
document.getElementById('chat').lastChild.remove();
add('Error: '+e.message,'bot');
}
btn.disabled=false;
btn.textContent='Send';
}
function add(text,role){
const chat=document.getElementById('chat');
const div=document.createElement('div');
div.className=role==='user'?'text-right':'text-left';
const bubble=role==='user'?'inline-block bg-purple-600 text-white':'inline-block bg-gray-800';
div.innerHTML=`<div class="${bubble} px-4 py-3 rounded-lg max-w-xl shadow-lg"><div class="font-semibold text-sm mb-1">${role==='user'?'You':'🧠 iPsychiatrist'}</div><div>${text}</div></div>`;
chat.appendChild(div);
chat.scrollTop=chat.scrollHeight;
}
</script>
</body>
</html>"""

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status":"ok","key_set":bool(GROQ_API_KEY)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/chat':
            try:
                length = int(self.headers['Content-Length'])
                data = json.loads(self.rfile.read(length).decode())
                
                if not GROQ_API_KEY:
                    response = {"response": "⚠️ API key not set. Add GROQ_API_KEY in Vercel environment variables."}
                else:
                    from groq import Groq
                    client = Groq(api_key=GROQ_API_KEY)
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are iPsychiatrist, a compassionate AI mental health assistant. Provide helpful, empathetic responses. Keep responses concise."},
                            {"role": "user", "content": data['message']}
                        ],
                        temperature=0.7,
                        max_tokens=500
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
                self.wfile.write(json.dumps({"response": f"Error: {str(e)}"}).encode())
        else:
            self.send_response(404)
            self.end_headers()
