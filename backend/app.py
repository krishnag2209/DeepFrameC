"""
DeepFake Guardian — FastAPI backend
Video detection powered by prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-base)
No local checkpoint required — model is downloaded from HuggingFace on first run.
"""

import os
import sys
import uuid
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
import uvicorn
import torch

# Add videodetection to path
sys.path.insert(0, str(Path("src/videodetection").resolve()))

from bot import start_bot, stop_bot

app = FastAPI(title="DeepFake Guardian")

# ── Ensure temp dir exists ──────────────────────────────────────────────────
Path("temp").mkdir(exist_ok=True)

# ── Mount /static (plain HTML/CSS/JS frontend) ──────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Mount /assets (React dist) only if the build exists ────────────────────
_react_assets = Path("frontend/dist/assets")
if _react_assets.exists():
    app.mount("/assets", StaticFiles(directory=str(_react_assets)), name="assets")
    print("[startup] React assets mounted.")

# ── Jinja2 templates ────────────────────────────────────────────────────────
templates = Jinja2Templates(directory="templates")

# ── Model globals ────────────────────────────────────────────────────────────
PROCESSOR = None
MODEL = None
DEVICE = None


@app.on_event("startup")
async def startup_event():
    global PROCESSOR, MODEL, DEVICE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] Using device: {DEVICE}")

    try:
        # Import the ViT-based inference from DeepFrameC videodetection
        from inference import load_model as vit_load_model
        print("[startup] Loading ViT deepfake model from HuggingFace...")
        PROCESSOR, MODEL = vit_load_model(DEVICE)
        print("[startup] ViT model loaded successfully.")
    except Exception as e:
        print(f"[startup] Failed to load model: {e}")
        PROCESSOR = None
        MODEL = None

    # Start Telegram bot as background task (skips gracefully if no token)
    asyncio.create_task(start_bot())


@app.on_event("shutdown")
async def shutdown_event():
    await stop_bot()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Prefer React dist if built, fall back to plain HTML template
    react_index = Path("frontend/dist/index.html")
    if react_index.exists():
        return HTMLResponse(react_index.read_text(encoding="utf-8"))
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if MODEL is None or PROCESSOR is None:
        return JSONResponse(
            {"error": "Model not loaded. Check server logs for details."},
            status_code=503,
        )

    temp_path = Path("temp") / f"{uuid.uuid4()}_{file.filename}"

    try:
        temp_path.write_bytes(await file.read())

        result = await run_in_threadpool(_run_video_inference, str(temp_path))
        return result

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _run_video_inference(video_path: str) -> dict:
    """Run ViT-based video inference — called in thread pool."""
    import time
    from inference import extract_frames, run_inference, temporal_smooth, aggregate

    t0 = time.time()
    frames = extract_frames(video_path, n_frames=40)
    # Changed margin to 0.0 because custom model may not handle boundary crops cleanly
    per_frame = run_inference(frames, MODEL, DEVICE, use_tta=False, margin=0.0)

    if not per_frame:
        return {
            "verdict": "UNCERTAIN",
            "fake_prob": 0.5,
            "real_prob": 0.5,
            "confidence": 0.0,
            "elapsed": time.time() - t0,
            "warning": "No valid face detections found. Ensure the video contains clearly visible frontal faces.",
        }

    per_frame = temporal_smooth(per_frame, window=5)
    agg = aggregate(per_frame, threshold=0.50)

    return {
        "verdict": agg["verdict"].upper(),
        "fake_prob": round(agg["mean_deepfake_prob"] / 100, 4),
        "real_prob": round(agg["mean_real_prob"] / 100, 4),
        "confidence": round(agg["confidence"], 2),
        "deepfake_frames": agg["deepfake_frames"],
        "total_frames": agg["total_frames"],
        "deepfake_frame_pct": round(agg["deepfake_frame_pct"], 1),
        "uncertain": agg["uncertain"],
        "elapsed": round(time.time() - t0, 2),
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "not initialized",
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
