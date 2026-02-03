# AI-Generated Voice Detection System

This project is an API-based system designed to detect whether a voice sample is AI-generated or Human-generated. It supports multiple languages (Tamil, English, Hindi, Malayalam, Telugu, Kannada) and accepts Base64-encoded MP3 audio inputs.

## Features

- **API Endpoint**: RESTful API built with FastAPI.
- **Input Format**: JSON payload with Base64-encoded audio and language tag.
- **Output Format**: Structured JSON with classification, confidence score, and explanation.
- **Extensible Architecture**: Modular design allowing easy integration of trained ML models (PyTorch, TensorFlow, etc.).

## Project Structure

- `main.py`: The entry point for the FastAPI application.
- `model.py`: Contains the `VoiceClassifier` class. Currently implements a simulation logic. **This is where you should load your trained model.**
- `preprocessing.py`: Handles audio decoding and feature extraction using `librosa`.
- `requirements.txt`: List of dependencies.
- `test_api.py`: A script to test the API with dummy audio.

## Setup and Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**:
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`.

3. **Test the API**:
   You can use the provided test script:
   ```bash
   python test_api.py
   ```
   Or use `curl` / Postman:
   ```bash
   curl -X POST "http://localhost:8000/detect" \
        -H "Content-Type: application/json" \
        -d '{"audio_base64": "<BASE64_STRING>", "language": "English"}'
   ```

## Model Integration

To use a real AI detection model:
1. Train or download a model (e.g., ASVspoof baseline).
2. Place the model weights file in the project directory.
3. Update `model.py`:
   - Uncomment the model loading logic in `__init__`.
   - Implement the actual inference in `predict`.

## API Specification

### POST `/detect`

**Request Body**:
```json
{
  "audio_base64": "SUQzBAAAAA...",
  "language": "English"
}
```

**Response**:
```json
{
  "classification": "AI-Generated",
  "confidence_score": 0.95,
  "explanation": "High confidence based on spectral artifacts...",
  "metadata": { ... }
}
```
