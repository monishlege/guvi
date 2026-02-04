from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi import Request
from pydantic import BaseModel, Field
from typing import Optional
from preprocessing import decode_audio, extract_features
from model import VoiceClassifier
import uvicorn
import sys

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="API to detect AI-generated voices in multiple languages.",
    version="1.0.0"
)

# Initialize the classifier (in a real app, you'd load the model path here)
classifier = VoiceClassifier()

class AudioRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded MP3 audio string")
    language: str = Field(..., description="Language of the audio (Tamil, English, Hindi, Malayalam, Telugu, Kannada)")

class AudioResponse(BaseModel):
    classification: str
    confidence_score: float
    explanation: str
    metadata: Optional[dict] = None

@app.get("/", response_class=HTMLResponse)
async def root(request: Request, format: str | None = None):
    accept = request.headers.get("accept", "")
    if format == "json" or "application/json" in accept:
        return JSONResponse({
            "status": "active",
            "message": "AI Voice Detection System is running",
            "endpoints": {
                "detect": "/detect",
                "docs": "/docs",
                "app": "/app",
                "health": "/health"
            }
        })
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>AI Voice Detection</title>
      <style>
        :root { color-scheme: dark; }
        @keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; min-height: 100vh; padding: 3rem 2rem; background: linear-gradient(135deg, #0b1220 0%, #101826 50%, #0b1220 100%); background-size: 200% 200%; animation: gradientShift 18s ease infinite; display: flex; align-items: center; color: #e2e8f0; }
        .card { max-width: 900px; margin: 0 auto; border-radius: 18px; padding: 2rem; backdrop-filter: blur(14px) saturate(160%); background: rgba(15,23,42,0.55); border: 1px solid rgba(99,102,241,0.35); box-shadow: 0 10px 30px rgba(2,6,23,0.6), 0 0 20px rgba(99,102,241,0.2); }
        h1 { font-size: 2rem; margin-bottom: 1rem; letter-spacing: .02em; color: #000000; }
        label { display: block; margin: .6rem 0 .3rem; font-weight: 700; color: #cbd5e1; }
        input[type=file], select, textarea { width: 100%; padding: .75rem .9rem; border: 1px solid rgba(148,163,184,0.25); border-radius: 14px; background: rgba(2,6,23,0.6); color: #e2e8f0; box-shadow: inset 0 1px 6px rgba(2,6,23,0.3); }
        input[type=file]:focus, select:focus, textarea:focus { outline: none; border-color: #22d3ee; box-shadow: 0 0 0 3px rgba(34,211,238,0.25); }
        button { margin-top: 1rem; padding: .85rem 1.2rem; border: none; border-radius: 14px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: #fff; cursor: pointer; box-shadow: 0 12px 24px rgba(59,130,246,0.35), 0 0 16px rgba(139,92,246,0.35); transition: transform .18s ease, box-shadow .18s ease; }
        button:hover { transform: translateY(-1px); box-shadow: 0 16px 30px rgba(59,130,246,0.5), 0 0 22px rgba(139,92,246,0.45); }
        button:disabled { background: linear-gradient(90deg,#64748b,#475569); cursor: not-allowed; box-shadow: none; }
        .dialog { border-radius: 18px; border: 1px solid rgba(99,102,241,0.35); background: rgba(2,6,23,0.75); box-shadow: 0 10px 28px rgba(2,6,23,0.55), 0 0 20px rgba(99,102,241,0.18); }
        .dialog-header { display: flex; align-items: center; gap: .75rem; padding: .75rem 1rem; border-bottom: 1px solid rgba(148,163,184,0.2); }
        .badge { padding: .35rem .6rem; border-radius: 999px; font-size: .85rem; background: linear-gradient(90deg,#22d3ee,#a78bfa); color: #0b1220; }
        .badge--ai { background: linear-gradient(90deg,#ef4444,#f59e0b); color: #0b1220; }
        .badge--human { background: linear-gradient(90deg,#10b981,#22d3ee); color: #0b1220; }
        .dialog-body { padding: .9rem 1rem; }
        .summary { color: #cbd5e1; margin-bottom: .5rem; }
        pre { background: rgba(2,6,23,0.9); color: #e2e8f0; padding: 1rem; border-radius: 14px; overflow: auto; border: 1px solid rgba(99,102,241,0.35); box-shadow: 0 0 0 1px rgba(99,102,241,0.25) inset, 0 8px 24px rgba(59,130,246,0.25), 0 0 16px rgba(139,92,246,0.2); }
        #copy { margin-left: auto; padding: .6rem .85rem; border: none; border-radius: 12px; background: linear-gradient(90deg, #22d3ee, #a78bfa); color: #0b1220; cursor: pointer; box-shadow: 0 8px 18px rgba(34,211,238,0.35), 0 0 12px rgba(167,139,250,0.3); }
        .row { display: grid; grid-template-columns: 1fr; gap: 1rem; }
        @media (min-width: 640px) { .row { grid-template-columns: 1fr 1fr; } }
        .small { font-size: .95rem; color: #94a3b8; }
        .links a { margin-right: .75rem; color: #93c5fd; text-decoration: none; }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>AI-Generated Voice Detection</h1>
        <div class="links">
          <a href="/docs">API Docs</a>
          <a href="/health" target="_blank">Health JSON</a>
          <a href="/?format=json" target="_blank">Root JSON</a>
        </div>
        <p class="small">Upload audio and select language. The API returns classification, confidence, and explanation.</p>
        <div class="row">
          <div>
            <label for="file">Audio file (WAV/MP3)</label>
            <input id="file" type="file" accept=".wav,.mp3,audio/*">
          </div>
          <div>
            <label for="lang">Language</label>
            <select id="lang">
              <option>English</option>
              <option>Tamil</option>
              <option>Hindi</option>
              <option>Malayalam</option>
              <option>Telugu</option>
              <option>Kannada</option>
            </select>
          </div>
        </div>
        <button id="send">Detect</button>
        <div id="out" style="margin-top:1rem;">
          <div class="dialog">
            <div class="dialog-header">
              <span id="status-badge" class="badge">Ready</span>
              <button id="copy">Copy</button>
            </div>
            <div class="dialog-body">
              <div class="summary" id="summary-text">Awaiting input</div>
              <pre id="result">{ "status": "ready" }</pre>
            </div>
          </div>
        </div>
        <p class="small">Or paste Base64 audio below (optional):</p>
        <textarea id="base64" rows="4" placeholder="Base64 audio string"></textarea>
      </div>
      <script>
        async function toBase64(file) {
          return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = reject;
            reader.readAsDataURL(file);
          });
        }
        async function detect() {
          const btn = document.getElementById('send');
          btn.disabled = true;
          const fileInput = document.getElementById('file');
          const lang = document.getElementById('lang').value;
          let audioB64 = document.getElementById('base64').value.trim();
          try {
            if (!audioB64) {
              const f = fileInput.files[0];
              if (!f) throw new Error('Select a file or paste Base64 audio');
              audioB64 = await toBase64(f);
            }
            const res = await fetch('/detect', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ audio_base64: audioB64, language: lang })
            });
            const txt = await res.text();
            document.getElementById('result').textContent = txt;
          } catch (err) {
            document.getElementById('result').textContent = JSON.stringify({ error: String(err) }, null, 2);
          } finally {
            btn.disabled = false;
          }
        }
        document.getElementById('send').addEventListener('click', detect);
        document.getElementById('copy').addEventListener('click', async () => {
          const txt = document.getElementById('result').textContent;
          try { await navigator.clipboard.writeText(txt); } catch(e) {}
        });
        function renderSummary(obj) {
          const badge = document.getElementById('status-badge');
          const summary = document.getElementById('summary-text');
          badge.className = 'badge';
          if (obj && typeof obj === 'object' && obj.classification) {
            const cls = String(obj.classification);
            const conf = typeof obj.confidence_score === 'number' ? Math.round(obj.confidence_score * 100) / 100 : obj.confidence_score;
            summary.textContent = `${cls} • ${conf}`;
            badge.textContent = cls;
            if (/AI-Generated/i.test(cls)) badge.className = 'badge badge--ai';
            if (/Human/i.test(cls)) badge.className = 'badge badge--human';
          } else {
            summary.textContent = 'Response';
            badge.textContent = 'JSON';
          }
        }
      </script>
    </body>
    </html>
    """

@app.get("/health")
def health_check():
    return {"status": "active", "message": "AI Voice Detection System is running"}

@app.get("/app", response_class=HTMLResponse)
def app_page():
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <meta name="apple-mobile-web-app-capable" content="yes">
      <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
      <title>AI Voice Detection</title>
      <link rel="manifest" href="/manifest.json">
      <style>
        :root { color-scheme: dark; }
        @keyframes floatbg { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; min-height: 100vh; padding: 2.5rem 2rem; background: linear-gradient(135deg, #0b1220 0%, #0f172a 50%, #0b1220 100%); background-size: 200% 200%; animation: floatbg 18s ease-in-out infinite; display: flex; align-items: center; color: #e2e8f0; }
        .card { max-width: 760px; margin: 0 auto; border-radius: 18px; padding: 2rem; backdrop-filter: blur(12px) saturate(160%); background: rgba(15,23,42,0.55); border: 1px solid rgba(99,102,241,0.35); box-shadow: 0 10px 30px rgba(2,6,23,0.6), 0 0 20px rgba(99,102,241,0.2); }
        h1 { font-size: 1.8rem; margin-bottom: 1rem; letter-spacing: .02em; color: #000000; }
        label { display: block; margin: .6rem 0 .3rem; font-weight: 700; color: #cbd5e1; }
        input[type=file], select, textarea { width: 100%; padding: .75rem .9rem; border: 1px solid rgba(148,163,184,0.25); border-radius: 14px; background: rgba(2,6,23,0.6); color: #e2e8f0; box-shadow: inset 0 1px 6px rgba(2,6,23,0.3); }
        input[type=file]:focus, select:focus, textarea:focus { outline: none; border-color: #22d3ee; box-shadow: 0 0 0 3px rgba(34,211,238,0.25); }
        button { margin-top: 1rem; padding: .85rem 1.2rem; border: none; border-radius: 14px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: #fff; cursor: pointer; box-shadow: 0 12px 24px rgba(59,130,246,0.35), 0 0 16px rgba(139,92,246,0.35); transition: transform .18s ease, box-shadow .18s ease; }
        button:hover { transform: translateY(-1px); box-shadow: 0 16px 30px rgba(59,130,246,0.5), 0 0 22px rgba(139,92,246,0.45); }
        button:disabled { background: linear-gradient(90deg,#64748b,#475569); cursor: not-allowed; box-shadow: none; }
        .dialog { border-radius: 18px; border: 1px solid rgba(99,102,241,0.35); background: rgba(2,6,23,0.75); box-shadow: 0 10px 28px rgba(2,6,23,0.55), 0 0 20px rgba(99,102,241,0.18); }
        .dialog-header { display: flex; align-items: center; gap: .75rem; padding: .75rem 1rem; border-bottom: 1px solid rgba(148,163,184,0.2); }
        .badge { padding: .35rem .6rem; border-radius: 999px; font-size: .85rem; background: linear-gradient(90deg,#22d3ee,#a78bfa); color: #0b1220; }
        .badge--ai { background: linear-gradient(90deg,#ef4444,#f59e0b); color: #0b1220; }
        .badge--human { background: linear-gradient(90deg,#10b981,#22d3ee); color: #0b1220; }
        .dialog-body { padding: .9rem 1rem; }
        .summary { color: #cbd5e1; margin-bottom: .5rem; }
        pre { background: rgba(2,6,23,0.9); color: #e2e8f0; padding: 1rem; border-radius: 14px; overflow: auto; border: 1px solid rgba(99,102,241,0.35); box-shadow: 0 0 0 1px rgba(99,102,241,0.25) inset, 0 8px 24px rgba(59,130,246,0.25), 0 0 16px rgba(139,92,246,0.2); }
        #copy { margin-left: auto; padding: .6rem .85rem; border: none; border-radius: 12px; background: linear-gradient(90deg, #22d3ee, #a78bfa); color: #0b1220; cursor: pointer; box-shadow: 0 8px 18px rgba(34,211,238,0.35), 0 0 12px rgba(167,139,250,0.3); }
        .row { display: grid; grid-template-columns: 1fr; gap: 1rem; }
        @media (min-width: 640px) { .row { grid-template-columns: 1fr 1fr; } }
        .small { font-size: .95rem; color: #94a3b8; }
      </style>
      <script>
        if ('serviceWorker' in navigator) {
          window.addEventListener('load', () => {
            navigator.serviceWorker.register('/sw.js').catch(() => {});
          });
        }
      </script>
    </head>
    <body>
      <div class="card">
        <h1>AI-Generated Voice Detection</h1>
        <p class="small">Upload audio and select language. The API returns classification, confidence, and explanation.</p>
        <div class="row">
          <div>
            <label for="file">Audio file (WAV/MP3)</label>
            <input id="file" type="file" accept=".wav,.mp3,audio/*">
          </div>
          <div>
            <label for="lang">Language</label>
            <select id="lang">
              <option>English</option>
              <option>Tamil</option>
              <option>Hindi</option>
              <option>Malayalam</option>
              <option>Telugu</option>
              <option>Kannada</option>
            </select>
          </div>
        </div>
        <button id="send">Detect</button>
        <div id="out" style="margin-top:1rem;">
          <div class="dialog">
            <div class="dialog-header">
              <span id="status-badge" class="badge">Ready</span>
              <button id="copy">Copy</button>
            </div>
            <div class="dialog-body">
              <div class="summary" id="summary-text">Awaiting input</div>
              <pre id="result">{ "status": "ready" }</pre>
            </div>
          </div>
        </div>
        <p class="small">Or paste Base64 audio below (optional):</p>
        <textarea id="base64" rows="4" placeholder="Base64 audio string"></textarea>
      </div>
      <script>
        async function toBase64(file) {
          return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = reject;
            reader.readAsDataURL(file);
          });
        }
        async function detect() {
          const btn = document.getElementById('send');
          btn.disabled = true;
          const fileInput = document.getElementById('file');
          const lang = document.getElementById('lang').value;
          let audioB64 = document.getElementById('base64').value.trim();
          try {
            if (!audioB64) {
              const f = fileInput.files[0];
              if (!f) throw new Error('Select a file or paste Base64 audio');
              audioB64 = await toBase64(f);
            }
            const res = await fetch('/detect', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ audio_base64: audioB64, language: lang })
            });
            const txt = await res.text();
            document.getElementById('result').textContent = txt;
          } catch (err) {
            document.getElementById('result').textContent = JSON.stringify({ error: String(err) }, null, 2);
          } finally {
            btn.disabled = false;
          }
        }
        document.getElementById('send').addEventListener('click', detect);
        document.getElementById('copy').addEventListener('click', async () => {
          const txt = document.getElementById('result').textContent;
          try { await navigator.clipboard.writeText(txt); } catch(e) {}
        });
        function renderSummary(obj) {
          const badge = document.getElementById('status-badge');
          const summary = document.getElementById('summary-text');
          badge.className = 'badge';
          if (obj && typeof obj === 'object' && obj.classification) {
            const cls = String(obj.classification);
            const conf = typeof obj.confidence_score === 'number' ? Math.round(obj.confidence_score * 100) / 100 : obj.confidence_score;
            summary.textContent = `${cls} • ${conf}`;
            badge.textContent = cls;
            if (/AI-Generated/i.test(cls)) badge.className = 'badge badge--ai';
            if (/Human/i.test(cls)) badge.className = 'badge badge--human';
          } else {
            summary.textContent = 'Response';
            badge.textContent = 'JSON';
          }
        }
      </script>
    </body>
    </html>
    """

@app.get("/manifest.json", response_class=JSONResponse)
def manifest():
    return {
        "name": "AI Voice Detection",
        "short_name": "VoiceDetect",
        "start_url": "/app",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#2563eb",
        "icons": []
    }

@app.get("/sw.js", response_class=PlainTextResponse)
def service_worker():
    return """
    self.addEventListener('install', (event) => {
      self.skipWaiting();
    });
    self.addEventListener('activate', (event) => {
      event.waitUntil(clients.claim());
    });
    self.addEventListener('fetch', (event) => {
      event.respondWith(fetch(event.request));
    });
    """
@app.post("/detect", response_model=AudioResponse)
async def detect_voice(request: AudioRequest):
    """
    Analyzes the uploaded audio and returns whether it is AI-generated or Human.
    """
    # Validate language
    supported_languages = ["tamil", "english", "hindi", "malayalam", "telugu", "kannada"]
    if request.language.lower() not in supported_languages:
        # We can just warn or proceed, but let's strictly validate for now or just allow it.
        # The prompt says "Voice samples will be provided in five languages", implying these are the expected ones.
        pass 

    try:
        # 1. Decode Audio
        y, sr = decode_audio(request.audio_base64)
        
        # 2. Extract Features
        features = extract_features(y, sr)
        
        # 3. Predict
        result = classifier.predict(features)
        
        # 4. Construct Response
        return AudioResponse(
            classification=result["classification"],
            confidence_score=result["confidence_score"],
            explanation=result["explanation"],
            metadata={
                "duration_seconds": features["duration"],
                "detected_language": request.language,
                "features_summary": {k: v for k, v in features.items() if k != "duration" and k != "mfcc_mean"}
            }
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Internal Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error processing audio")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
