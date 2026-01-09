# Web UI (FastAPI)

This folder contains a lightweight web interface for running DeepFake detection on uploaded videos.

## Run locally

1) Install dependencies:

- `pip install -r web/requirements.txt`

2) Start the server:

- `uvicorn web.server:app --reload --port 8000`

3) Open the UI:

- http://127.0.0.1:8000

## Notes

- The server calls `predict_deepfake` from `deep_fake_detect_app.py`.
- Uploaded files are stored in `uploads/` (ignored by git).
- Supported methods currently include: `plain_frames`, `MRI`, `fusion`, `temporal`.
