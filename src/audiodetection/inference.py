import argparse
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "MelodyMachine/Deepfake-audio-detection-V2"
SAMPLE_RATE = 16000
MAX_SAMPLES = SAMPLE_RATE * 10  # 10 second limit for stability

def load_resources(device_name=None):
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Initializing {MODEL_ID} on {device}")
    
    # Feature extractor is CRITICAL for Wav2Vec2 normalization
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()
    
    return model, extractor, device

def process_audio(path):
    waveform, sr = torchaudio.load(path)
    
    # Force Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz (Wav2Vec2 requirement)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    
    # Slice to MAX_SAMPLES
    if waveform.shape[1] > MAX_SAMPLES:
        waveform = waveform[:, :MAX_SAMPLES]
        
    return waveform.squeeze().numpy()

def get_prediction(model, extractor, device, audio_path):
    audio = process_audio(audio_path)
    
    # Extractor handles the vital zero-mean/unit-variance scaling
    inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        # Convert logits to probabilities 0.0 - 1.0
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Map index to Label (usually 0=REAL, 1=FAKE for this model)
    id2label = model.config.id2label
    results = {id2label[i].upper(): float(probs[i]) for i in range(len(probs))}
    
    # Determine winner
    top_label = max(results, key=results.get)
    
    return {
        "label": top_label,
        "fake_conf": results.get("FAKE", results.get("LABEL_1", 0.0)),
        "real_conf": results.get("REAL", results.get("LABEL_0", 0.0))
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to wav file")
    args = parser.parse_args()

    model, extractor, device = load_resources()
    pred = get_prediction(model, extractor, device, args.file)

    print(f"\n--- INFERENCE REPORT ---")
    print(f"File       : {args.file}")
    print(f"Verdict    : {pred['label']}")
    print(f"Confidence :")
    print(f"  > FAKE: {pred['fake_conf']:.6f}")
    print(f"  > REAL: {pred['real_conf']:.6f}")
    print(f"------------------------\n")
