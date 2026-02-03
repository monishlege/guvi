from fastapi import FastAPI, HTTPException, Body
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

@app.get("/")
def health_check():
    return {"status": "active", "message": "AI Voice Detection System is running"}

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
