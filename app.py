"""FastAPI service for the FMA genre classifier.
Run   : uvicorn app:app --reload
Needs : pip install fastapi uvicorn python-multipart joblib librosa scikit-learn numpy
Files : genre_model_fma.joblib (from 05_export_model.py), genre_dsp.py in the same folder."""
import os, tempfile
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import genre_dsp as g

app = FastAPI(title="Genre Classifier API")

# Enable CORS so your JavaScript frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's specific URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model once at startup
bundle = joblib.load("genre_model_fma.joblib")
model, cfg, classes = bundle["model"], bundle["cfg"], bundle["classes"]


@app.get("/health")
def health():
    return {"status": "ok", "front_end": cfg, "n_classes": len(classes)}


@app.get("/classes")
def get_classes():
    return {"classes": classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an audio file (mp3/wav). Returns the predicted genre."""
    suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        x = g.load_audio(path)                              # mono, 22,050 Hz, first 30 s, padded
        feats = g.clip_features(x, **g.core(cfg))           # 40-d MFCC mean/std vector
        genre = str(model.predict(feats.reshape(1, -1))[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio: {type(e).__name__}")
    finally:
        os.remove(path)
    return {"prediction": genre, "filename": file.filename}