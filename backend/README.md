# DeepFake Guardian — Merged Project

A full-stack deepfake video detection web app deployable on Railway.

## What changed vs the original deploy

| | Before (DeepFakeGuardian_Deploy) | After (this merge) |
|---|---|---|
| Video model | Custom `DeepFakeDetector` — required a `checkpoints/best.pth` file that was never present → always returned "Model not loaded" | **ViT-base** (`prithivMLmods/Deep-Fake-Detector-v2-Model`) — downloads automatically from HuggingFace on first boot, no checkpoint file needed |
| Face detection | None (whole-frame) | MediaPipe (primary) + Haar cascade fallback |
| Confidence fix | Hardcoded `[0.5, 0.5, 0.5]` normalisation → caused 50/50 outputs | Reads actual mean/std from ViTImageProcessor |
| Result data | `verdict`, `fake_prob`, `real_prob`, `elapsed` | + `confidence`, `deepfake_frames`, `total_frames`, `deepfake_frame_pct`, `uncertain` |
| Frontend | Shows Fake/Real % and time | Also shows confidence %, frame breakdown, uncertainty warning |

## Project structure

```
.
├── app.py                    # FastAPI backend (entry point)
├── requirements.txt          # Python deps (includes transformers)
├── nixpacks.toml             # Railway build: installs Python + Node, builds React
├── railway.json
├── Procfile
├── runtime.txt
├── bot.py                    # Telegram bot (optional)
├── src/
│   └── videodetection/
│       └── inference.py      # ViT-based video inference (from DeepFrameC)
├── frontend/                 # React + Vite frontend (source)
│   ├── src/App.jsx           # Updated to show richer results
│   ├── package.json
│   └── vite.config.js
├── static/                   # Fallback HTML/CSS/JS frontend
├── templates/                # Jinja2 fallback template
└── temp/                     # Temp dir for uploaded videos (auto-created)
```

## Local development

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Build the React frontend
cd frontend && npm install && npm run build && cd ..

# 3. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The ViT model (~330 MB) is downloaded from HuggingFace on **first startup** and cached at `~/.cache/huggingface/`. On Railway this happens during the first request — subsequent requests are fast.

## Railway deployment

1. Push this folder to a GitHub repo
2. Connect the repo to Railway
3. Railway auto-detects `nixpacks.toml` and:
   - Installs Python deps
   - Runs `npm ci && npm run build` for the React frontend
   - Starts the FastAPI server
4. Set the `PORT` env variable (Railway does this automatically)

### Optional: Telegram bot
Set `TELEGRAM_TOKEN` in Railway environment variables to enable the bot.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `PORT` | Yes (auto-set by Railway) | HTTP port |
| `TELEGRAM_TOKEN` | No | Telegram bot token |
| `HF_HOME` | No | Override HuggingFace cache dir (e.g. `/data/hf_cache` for a persistent volume) |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/predict` | Upload a video file, returns JSON verdict |
| `GET` | `/health` | Health check — includes `model_loaded` status |

### `/predict` response

```json
{
  "verdict": "DEEPFAKE",
  "fake_prob": 0.873,
  "real_prob": 0.127,
  "confidence": 87.3,
  "deepfake_frames": 28,
  "total_frames": 34,
  "deepfake_frame_pct": 82.4,
  "uncertain": false,
  "elapsed": 12.45
}
```
