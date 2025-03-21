# File: mitbih_ecg_api.py
import pandas as pd
import numpy as np
from scipy import signal
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import requests
import os
import io
import joblib
from scipy.signal import find_peaks
import json
import asyncio

app = FastAPI(title="MIT-BIH ECG Classification API")

app.mount("/static", StaticFiles(directory=".", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

categories = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 'F (Fusion)', 'Q (Unknown)']

def preprocess_signal(data, fs=360, window_size=256):
    try:
        segments = []
        for channel in range(data.shape[1]):
            b, a = signal.butter(3, [0.5, 40], btype='bandpass', fs=fs)
            filtered = signal.filtfilt(b, a, data[:, channel])
            filtered = (filtered - np.mean(filtered)) / np.std(filtered)
            channel_segments = []
            for i in range(0, len(filtered) - window_size + 1, window_size // 2):
                segment = filtered[i:i + window_size]
                if len(segment) == window_size:
                    channel_segments.append(segment)
            segments.append(channel_segments)
        if not segments:
            raise ValueError("No segments extracted from signal data")
        return np.stack(segments, axis=2)
    except Exception as e:
        raise ValueError(f"Error in preprocess_signal: {str(e)}")

def extract_features(segments):
    features = []
    for segment in segments:
        lead_ii = segment[:, 0]
        chest_lead = segment[:, 1]
        feat = [
            np.mean(lead_ii), np.std(lead_ii), np.max(lead_ii), np.min(lead_ii), len(signal.find_peaks(lead_ii)[0]),
            np.mean(chest_lead), np.std(chest_lead), np.max(chest_lead), np.min(chest_lead), len(signal.find_peaks(chest_lead)[0])
        ]
        features.append(feat)
    return np.array(features)

def calculate_heart_rate(signal, fs=360):
    lead_ii = signal[:, 0]
    # Lower height threshold and adjust distance
    peaks, _ = find_peaks(lead_ii, height=np.max(lead_ii) * 0.2, distance=fs * 0.3)  # 0.3s = ~200 BPM max
    print(f"Debug: Found {len(peaks)} peaks in signal of length {len(lead_ii)} samples")
    if len(peaks) < 2:
        return None
    rr_intervals = np.diff(peaks) / fs
    print(f"Debug: RR intervals (seconds): {rr_intervals[:5]}")
    heart_rate = 60 / np.median(rr_intervals)
    return round(heart_rate, 1)

model = None

@app.on_event("startup")
async def startup_event():
    global model
    model_path = "mitbih_rf_model.pkl"
    model_url = "https://www.dropbox.com/scl/fi/k9t70dx1jdvykfjy7in2o/mitbih_rf_model.pkl?rlkey=8bzg06ca63vx8roavlu08o42j&st=pd9xsc77&dl=1"  # Replace with your link
    if not os.path.exists(model_path):
        print("Downloading model...")
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    try:
        model = joblib.load(model_path)
        print("Trained Random Forest model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

@app.get("/health")
async def health_check():
    status = "healthy" if model is not None else "unhealthy: model not loaded"
    return {
        "status": status,
        "model_loaded": model is not None,
        "api_version": "1.0.0",
        "timestamp": pd.Timestamp.now().isoformat()
    }

async def classify_with_progress(signal_data):
    segments = preprocess_signal(signal_data, fs=360)
    total_segments = len(segments)
    features = []
    predictions = []
    batch_size = 20
    
    async def stream_generator():
        for i in range(0, total_segments, batch_size):
            batch_end = min(i + batch_size, total_segments)
            batch_segments = segments[i:batch_end]
            batch_features = extract_features(batch_segments)
            batch_predictions = model.predict(batch_features)
            
            features.extend(batch_features)
            predictions.extend(batch_predictions)
            
            progress = (batch_end / total_segments) * 100
            yield json.dumps({"progress": progress}) + "\n"
            await asyncio.sleep(0.01)
        
        features_array = np.array(features)
        heart_rate = calculate_heart_rate(signal_data)
        if predictions:
            final_class = np.bincount(predictions).argmax()
            confidence = np.mean(model.predict_proba(features_array)[:, final_class])
            segment_counts = {categories[i]: int(np.sum(np.array(predictions) == i)) for i in range(len(categories))}
            result = {
                "prediction": categories[final_class],
                "confidence": float(confidence),
                "segment_summary": segment_counts,
                "num_segments": total_segments,
                "heart_rate_bpm": heart_rate if heart_rate is not None else "Not enough peaks detected"
            }
        else:
            result = {
                "prediction": "Unknown",
                "confidence": 0.0,
                "segment_summary": {cat: 0 for cat in categories},
                "num_segments": 0,
                "heart_rate_bpm": "Not enough peaks detected"
            }
        yield json.dumps({"result": result}) + "\n"
    
    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.post("/classify-ecg/")
async def classify_ecg(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.xlsx'):
            raise HTTPException(status_code=400, detail="Only .xlsx files are supported")
        
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        if 'Lead_II' not in df.columns or 'Chest_Lead' not in df.columns:
            raise HTTPException(status_code=400, detail="Excel file must contain 'Lead_II' and 'Chest_Lead' columns")
        
        signal_data = df[['Lead_II', 'Chest_Lead']].values
        return await classify_with_progress(signal_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the MIT-BIH ECG Classification API. Upload an .xlsx file to /classify-ecg/"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
