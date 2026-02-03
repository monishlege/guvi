import requests
import base64
import numpy as np
import soundfile as sf
import io
import json

API_BASE = "http://localhost:8001"

def create_dummy_audio_base64(seconds: float = 1.0, sr: int = 22050, freq: float = 440.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def demo_health():
    url = f"{API_BASE}/"
    r = requests.get(url, timeout=10)
    print("GET / ->", r.status_code, r.text)

def demo_detect(language: str, freq: float):
    url = f"{API_BASE}/detect"
    payload = {
        "audio_base64": create_dummy_audio_base64(seconds=1.0, sr=22050, freq=freq),
        "language": language
    }
    r = requests.post(url, json=payload, timeout=20)
    print(f"POST /detect ({language}) -> {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)

def main():
    demo_health()
    langs_freqs = [
        ("English", 440.0),
        ("Tamil", 523.25),
        ("Hindi", 392.0),
        ("Malayalam", 659.25),
        ("Telugu", 349.23),
        ("Kannada", 300.0),
    ]
    for lang, f in langs_freqs:
        demo_detect(lang, f)

if __name__ == "__main__":
    main()
