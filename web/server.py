import os
import uuid
import asyncio
from contextlib import contextmanager
import sys
from typing import Optional, Dict
import time
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from starlette.requests import Request


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DF_ROOT = PROJECT_ROOT  # Structure is now flattened
UPLOAD_DIR = os.path.join(DF_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Progress tracking
job_progress: Dict[str, Dict] = {}


@contextmanager
def chdir(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


app = FastAPI(title="DeepFake Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), 'static')
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
app.mount('/static', StaticFiles(directory=static_dir), name='static')
templates = Jinja2Templates(directory=templates_dir)


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


def _load_predict_function():
    """Load predict_deepfake from module or directly from file as a fallback."""
    if DF_ROOT not in sys.path:
        sys.path.insert(0, DF_ROOT)
    try:
        from deep_fake_detect_app import predict_deepfake  # type: ignore
        return predict_deepfake
    except ModuleNotFoundError:
        import importlib.util
        module_path = os.path.join(DF_ROOT, 'deep_fake_detect_app.py')
        if not os.path.isfile(module_path):
            raise
        spec = importlib.util.spec_from_file_location('deep_fake_detect_app_fallback', module_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return getattr(mod, 'predict_deepfake')


async def run_detection(video_path: str, method: str, temperature: float = 1.0, 
                       temporal_window: int = 5, job_id: Optional[str] = None):
    """Run deepfake detection with enhanced error handling and progress tracking"""
    try:
        # Update progress
        if job_id:
            job_progress[job_id] = {
                'status': 'processing',
                'stage': 'initializing',
                'progress': 0,
                'message': 'Starting detection...'
            }
        
        # Import inside to avoid heavy import at startup
        with chdir(DF_ROOT):
            predict_deepfake = _load_predict_function()
            
            # Validate method
            valid_methods = ['plain_frames', 'MRI', 'fusion', 'temporal']
            if method not in valid_methods:
                raise HTTPException(status_code=422, detail=f"Invalid method. Must be one of: {valid_methods}")
            
            if job_id:
                job_progress[job_id].update({
                    'stage': 'extracting_faces',
                    'progress': 20,
                    'message': 'Extracting faces from video...'
                })
            
            # Run detection
            fake_prob, real_prob, pred = predict_deepfake(
                input_videofile=video_path,
                df_method=method,
                debug=False,
                verbose=False,
                temperature=temperature,
                overwrite_faces=True,
                overwrite_mri=False,
                json_out=False,
                temporal_window=temporal_window
            )
            
            if pred is None:
                if job_id:
                    job_progress[job_id].update({
                        'status': 'error',
                        'message': 'No faces detected in video'
                    })
                raise HTTPException(status_code=400, detail="No faces/frames detected in the video. Please ensure the video contains visible faces.")
            
            label = 'REAL' if pred == 0 else 'DEEP-FAKE'
            prob = real_prob if pred == 0 else fake_prob
            
            result = {
                'label': label,
                'probability_percent': round(float(prob) * 100.0, 2),
                'method': method,
                'fake_probability': round(float(fake_prob) * 100.0, 2),
                'real_probability': round(float(real_prob) * 100.0, 2)
            }
            
            if job_id:
                job_progress[job_id].update({
                    'status': 'completed',
                    'stage': 'done',
                    'progress': 100,
                    'message': 'Detection completed',
                    'result': result
                })
            
            return result
    
    except HTTPException:
        if job_id:
            job_progress[job_id].update({
                'status': 'error',
                'message': 'Detection failed'
            })
        raise
    except FileNotFoundError as e:
        if job_id:
            job_progress[job_id].update({
                'status': 'error',
                'message': f'File not found: {str(e)}'
            })
        raise HTTPException(status_code=404, detail=f"Required file not found: {str(e)}")
    except Exception as e:
        if job_id:
            job_progress[job_id].update({
                'status': 'error',
                'message': f'Error: {str(e)}'
            })
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post('/api/detect')
async def api_detect(file: UploadFile = File(...), method: str = Form('plain_frames'), 
                    temperature: float = Form(1.0), temporal_window: int = Form(5),
                    background_tasks: BackgroundTasks = None):
    """API endpoint for deepfake detection with enhanced validation"""
    # Validate method
    valid_methods = ['plain_frames', 'MRI', 'fusion', 'temporal']
    if method not in valid_methods:
        raise HTTPException(status_code=422, detail=f"Invalid method. Must be one of: {valid_methods}")
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        raise HTTPException(status_code=422, detail="Unsupported file type. Supported: .mp4, .mov, .avi, .mkv, .webm")
    
    # Validate parameters
    if temperature <= 0:
        raise HTTPException(status_code=422, detail="Temperature must be > 0")
    if temporal_window < 1:
        raise HTTPException(status_code=422, detail="Temporal window must be >= 1")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save upload
    suffix = os.path.splitext(file.filename)[1]
    dest = os.path.join(UPLOAD_DIR, f"{job_id}{suffix}")
    
    try:
        content = await file.read()
        # Check file size (limit to 500MB)
        if len(content) > 500 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size: 500MB")
        
        with open(dest, 'wb') as f:
            f.write(content)
        
        # Initialize progress
        job_progress[job_id] = {
            'status': 'uploaded',
            'stage': 'uploading',
            'progress': 10,
            'message': 'File uploaded successfully'
        }
        
        # Run detection
        result = await run_detection(dest, method, temperature, temporal_window, job_id)
        
        return JSONResponse({
            'status': 'ok',
            'job_id': job_id,
            'result': result
        })
    
    except HTTPException:
        raise
    except Exception as e:
        if job_id in job_progress:
            job_progress[job_id]['status'] = 'error'
            job_progress[job_id]['message'] = str(e)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        # Clean up uploaded file after processing (optional)
        try:
            if os.path.exists(dest):
                # Keep for debugging, or uncomment to delete:
                # os.remove(dest)
                pass
        except Exception:
            pass


@app.get('/api/progress/{job_id}')
async def get_progress(job_id: str):
    """Get progress for a detection job"""
    if job_id not in job_progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(job_progress[job_id])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)


