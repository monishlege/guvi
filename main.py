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
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 2rem; }
        .card { max-width: 800px; margin: 0 auto; border: 1px solid #eee; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        h1 { font-size: 1.6rem; margin-bottom: 1rem; }
        label { display: block; margin: .5rem 0 .25rem; font-weight: 600; }
        input[type=file], select, textarea { width: 100%; padding: .5rem; border: 1px solid #ccc; border-radius: 6px; }
        button { margin-top: 1rem; padding: .6rem 1rem; border: none; border-radius: 6px; background: #2563eb; color: #fff; cursor: pointer; }
        button:disabled { background: #9aa7c7; cursor: not-allowed; }
        pre { background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow: auto; }
        .row { display: grid; grid-template-columns: 1fr; gap: 1rem; }
        @media (min-width: 640px) { .row { grid-template-columns: 1fr 1fr; } }
        .small { font-size: .85rem; color: #555; }
        .links a { margin-right: .75rem; }
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
          <pre id="result">{ "status": "waiting for input" }</pre>
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
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 2rem; }
        .card { max-width: 720px; margin: 0 auto; border: 1px solid #eee; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        h1 { font-size: 1.4rem; margin-bottom: 1rem; }
        label { display: block; margin: .5rem 0 .25rem; font-weight: 600; }
        input[type=file], select, textarea { width: 100%; padding: .5rem; border: 1px solid #ccc; border-radius: 6px; }
        button { margin-top: 1rem; padding: .6rem 1rem; border: none; border-radius: 6px; background: #2563eb; color: #fff; cursor: pointer; }
        button:disabled { background: #9aa7c7; cursor: not-allowed; }
        pre { background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow: auto; }
        .row { display: grid; grid-template-columns: 1fr; gap: 1rem; }
        @media (min-width: 640px) { .row { grid-template-columns: 1fr 1fr; } }
        .small { font-size: .85rem; color: #555; }
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
          <pre id="result">{ "status": "waiting for input" }</pre>
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
