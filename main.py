import os
import io
import torch
import librosa
import requests
import tempfile
import numpy as np
import soundfile as sf
import time
import base64
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Form, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# Add these new imports for environment variables and HF authentication
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RunPod specific logging
logger.info("🚀 Starting IndicF5 TTS API server for RunPod deployment...")

# RunPod port handling - check for RunPod's environment variables
RUNPOD_PORT = os.getenv("RUNPOD_TCP_PORT_8000")
DEFAULT_PORT = os.getenv("PORT", "8000")
SERVER_PORT = int(RUNPOD_PORT or DEFAULT_PORT)

logger.info(f"🌐 Server will start on port {SERVER_PORT}")

# Initialize FastAPI app
app = FastAPI(
    title="IndicF5 TTS Production API - KC Voice Only",
    description="🎯 Simple Malayalam Text-to-Speech API using KC Voice - No Reference Audio Required",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (optional)
security = HTTPBearer(auto_error=False)

# Global variables
model = None
device = None

# Simplified Pydantic models - NO reference audio required
class TTSRequest(BaseModel):
    text: str
    language: str = "malayalam"

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_base64: Optional[str] = None
    sample_rate: int = 24000
    inference_time: float = 0.0
    language: str = ""
    voice: str = "KC Voice"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: Optional[str] = None
    gpu_available: bool
    gpu_info: Optional[Dict[str, Any]] = None
    supported_languages: List[str]
    voice_loaded: bool
    runpod_info: Optional[Dict[str, Any]] = None

# Configuration
SUPPORTED_LANGUAGES = [
    "assamese", "bengali", "gujarati", "hindi", "kannada",
    "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu"
]

# KC Voice configuration (only voice used) - FIXED REFERENCE
KC_VOICE = {
    "name": "KC Voice",
    "url": "https://raw.githubusercontent.com/Aparna0112/voicerecording-_TTS/main/KC%20Voice.wav",
    "ref_text": "ഹലോ ഇത് അപരനെ അല്ലേ ഞാൻ ജഗദീപ് ആണ് വിളിക്കുന്നത് ഇപ്പോൾ ഫ്രീയാണോ സംസാരിക്കാമോ",
    "sample_rate": None,
    "audio_data": None,
    "temp_file_path": None  # Store the temp file path for reuse
}

def get_runpod_info():
    """Get RunPod specific environment information"""
    return {
        "pod_id": os.getenv("RUNPOD_POD_ID"),
        "public_ip": os.getenv("RUNPOD_PUBLIC_IP"),
        "tcp_port": os.getenv("RUNPOD_TCP_PORT_8000"),
        "gpu_count": os.getenv("RUNPOD_GPU_COUNT"),
        "cpu_count": os.getenv("RUNPOD_CPU_COUNT"),
        "memory_gb": os.getenv("RUNPOD_MEM_GB"),
        "container_id": os.getenv("RUNPOD_CONTAINER_ID"),
    }

def authenticate_huggingface():
    """Authenticate with Hugging Face using token from .env file"""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            logger.info("✅ Authenticated with Hugging Face using token from environment")
            return True
        except Exception as e:
            logger.error(f"❌ HF Authentication failed: {str(e)}")
            return False
    else:
        logger.error("❌ No HF_TOKEN found in environment variables")
        return False

def get_device():
    """Get the best available device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🎮 Using GPU: {gpu_name}")
        except:
            logger.info("🎮 Using GPU: CUDA device available")
        return device
    else:
        device = torch.device("cpu")
        logger.info("🖥️ Using CPU")
        return device

def load_audio_from_url(url: str, timeout: int = 30):
    """Load audio from URL with better error handling"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()
        
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        return sample_rate, audio_data
    except Exception as e:
        logger.error(f"Error loading audio from URL {url}: {str(e)}")
        return None, None

def normalize_audio(audio):
    """Normalize audio to prevent clipping"""
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio)) * 0.95
    return audio

def preload_kc_voice():
    """Preload KC voice audio at startup and create permanent temp file"""
    logger.info("📂 Preloading KC voice audio...")
    try:
        sample_rate, audio_data = load_audio_from_url(KC_VOICE["url"])
        if sample_rate is not None and audio_data is not None:
            # Normalize and prepare audio
            audio_data = normalize_audio(audio_data)
            duration = len(audio_data) / sample_rate
            KC_VOICE["sample_rate"] = sample_rate
            KC_VOICE["audio_data"] = audio_data
            
            # Create permanent temp file for the reference audio
            temp_dir = os.getenv("TEMP_DIR", "/app/temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, "kc_voice_reference.wav")
            sf.write(temp_file_path, audio_data, samplerate=sample_rate, format='WAV')
            KC_VOICE["temp_file_path"] = temp_file_path
            
            logger.info(f"✅ Loaded {KC_VOICE['name']} (SR: {sample_rate}, Length: {duration:.2f}s)")
            logger.info(f"📁 KC Voice reference saved to: {temp_file_path}")
            return True
        else:
            logger.warning(f"❌ Failed to load {KC_VOICE['name']}")
            return False
    except Exception as e:
        logger.error(f"Error preloading {KC_VOICE['name']}: {str(e)}")
        return False

def ensure_directories():
    """Ensure necessary directories exist"""
    dirs = [
        os.getenv("MODEL_CACHE_DIR", "/app/model_cache"),
        os.getenv("TEMP_DIR", "/app/temp"),
        "/app/logs"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"📁 Directory ensured: {dir_path}")

# API Authentication (optional)
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for production use"""
    if credentials is None:
        return None
    
    api_key = os.getenv("API_KEY")
    if api_key and credentials.credentials != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.on_event("startup")
async def startup_event():
    """Initialize the model and resources on startup"""
    global model, device
    
    try:
        logger.info("🔧 Starting initialization process...")
        
        # Ensure directories exist
        ensure_directories()
        
        # Log RunPod environment info
        runpod_info = get_runpod_info()
        logger.info(f"🏃 RunPod Environment: {runpod_info}")
        
        # Authenticate with Hugging Face first
        auth_success = authenticate_huggingface()
        if not auth_success:
            logger.error("⚠️ HF authentication failed, model loading will likely fail")
            # Don't return here, continue trying to load model
        
        # Get device
        device = get_device()
        
        # Check if IndicF5 is properly installed
        logger.info("🚀 Loading IndicF5 model...")
        
        try:
            from transformers import AutoModel
            
            repo_id = "ai4bharat/IndicF5"
            cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")
            
            # Load model with proper configuration
            model = AutoModel.from_pretrained(
                repo_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir
            )
            
            if not torch.cuda.is_available():
                model = model.to(device)
            
            model.eval()
            logger.info("✅ IndicF5 model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.info("💡 Trying alternative loading method...")
            
            try:
                from transformers import AutoModel
                
                model = AutoModel.from_pretrained(
                    "ai4bharat/IndicF5",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    cache_dir=cache_dir
                )
                
                model = model.to(device)
                model.eval()
                logger.info("✅ IndicF5 model loaded with alternative method!")
                
            except Exception as e2:
                logger.error(f"Alternative method also failed: {str(e2)}")
                logger.error("❌ Model loading failed completely")
                model = None
        
        # Preload KC voice - CRITICAL for no-upload functionality
        voice_loaded = preload_kc_voice()
        
        if model is not None and voice_loaded:
            logger.info("🎉 API is ready for production with KC Voice - NO REFERENCE AUDIO UPLOAD NEEDED!")
        else:
            logger.error("⚠️ API started but model or voice is not loaded")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        model = None

@app.get("/")
async def root():
    """API information and health check"""
    runpod_info = get_runpod_info()
    return {
        "message": "IndicF5 TTS Production API - KC Voice Only (No Reference Audio Upload Required)",
        "version": "2.2.0",
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "voice": "KC Voice (Pre-loaded)",
        "gpu_available": torch.cuda.is_available(),
        "supported_languages": SUPPORTED_LANGUAGES,
        "runpod_info": runpod_info,
        "usage": "Just send text - no audio upload needed!",
        "endpoints": {
            "POST /generate": "Generate speech using KC voice (text only)",
            "POST /generate_audio": "Generate speech and return as audio file",
            "GET /voice_info": "Get KC voice information",
            "GET /health": "Detailed health check"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                "gpu_memory_reserved": torch.cuda.memory_reserved(0),
            }
        except:
            gpu_info = {"gpu_available": True, "details": "GPU info unavailable"}
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else None,
        gpu_available=torch.cuda.is_available(),
        gpu_info=gpu_info,
        supported_languages=SUPPORTED_LANGUAGES,
        voice_loaded=KC_VOICE["audio_data"] is not None,
        runpod_info=get_runpod_info()
    )

@app.get("/voice_info")
async def get_voice_info():
    """Get KC voice information"""
    return {
        "voice": {
            "name": KC_VOICE["name"],
            "ref_text": KC_VOICE["ref_text"],
            "available": KC_VOICE["audio_data"] is not None,
            "sample_rate": KC_VOICE.get("sample_rate"),
            "duration": len(KC_VOICE["audio_data"]) / KC_VOICE["sample_rate"] if KC_VOICE["audio_data"] is not None and KC_VOICE["sample_rate"] else None,
            "temp_file_ready": KC_VOICE["temp_file_path"] is not None
        }
    }

# MAIN ENDPOINT - NO REFERENCE AUDIO UPLOAD REQUIRED
@app.post("/generate", response_model=TTSResponse)
async def generate_speech(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form(default="malayalam", description="Target language"),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Generate speech using KC voice - NO REFERENCE AUDIO UPLOAD REQUIRED"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    if KC_VOICE["audio_data"] is None or KC_VOICE["temp_file_path"] is None:
        raise HTTPException(
            status_code=503,
            detail="KC voice not loaded properly. Please restart the server."
        )
    
    if language.lower() not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Language {language} not supported. Supported: {SUPPORTED_LANGUAGES}"
        )
    
    try:
        start_time = time.time()
        
        # Use the pre-loaded KC voice reference file
        temp_audio_path = KC_VOICE["temp_file_path"]
        ref_text = KC_VOICE["ref_text"]
        
        # Generate speech
        logger.info(f"🎯 Generating with KC Voice: {text[:50]}...")
        inference_start = time.time()
        
        with torch.no_grad():
            generated_audio = model(
                text,
                ref_audio_path=temp_audio_path,
                ref_text=ref_text
            )
        
        inference_time = time.time() - inference_start
        
        # Process audio
        if isinstance(generated_audio, torch.Tensor):
            generated_audio = generated_audio.cpu().numpy()
            
        if generated_audio.dtype == np.int16:
            generated_audio = generated_audio.astype(np.float32) / 32768.0
        elif generated_audio.dtype == np.int32:
            generated_audio = generated_audio.astype(np.float32) / 2147483648.0
        
        generated_audio = normalize_audio(generated_audio)
        
        # Convert to base64
        buffer = io.BytesIO()
        sf.write(buffer, generated_audio, 24000, format='WAV')
        audio_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return TTSResponse(
            success=True,
            message=f"✅ Speech generated using KC Voice for {language}!",
            audio_base64=audio_base64,
            sample_rate=24000,
            inference_time=inference_time,
            language=language,
            voice="KC Voice"
        )
        
    except Exception as e:
        logger.error(f"❌ Error with KC Voice synthesis: {str(e)}")
        return TTSResponse(
            success=False,
            message=f"❌ Error: {str(e)}",
            sample_rate=24000,
            language=language,
            voice="KC Voice"
        )

# AUDIO FILE ENDPOINT - NO REFERENCE AUDIO UPLOAD REQUIRED
@app.post("/generate_audio")
async def generate_speech_audio(
    text: str = Form(..., description="Text to synthesize"),
    language: str = Form(default="malayalam", description="Target language"),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Generate speech using KC voice and return as audio file - NO REFERENCE AUDIO UPLOAD REQUIRED"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    if KC_VOICE["audio_data"] is None or KC_VOICE["temp_file_path"] is None:
        raise HTTPException(
            status_code=503,
            detail="KC voice not loaded properly. Please restart the server."
        )
    
    try:
        # Use the pre-loaded KC voice reference file
        temp_audio_path = KC_VOICE["temp_file_path"]
        ref_text = KC_VOICE["ref_text"]
        
        # Generate speech
        logger.info(f"🎯 Generating audio file with KC Voice: {text[:50]}...")
        
        with torch.no_grad():
            generated_audio = model(
                text,
                ref_audio_path=temp_audio_path,
                ref_text=ref_text
            )
        
        # Process audio
        if isinstance(generated_audio, torch.Tensor):
            generated_audio = generated_audio.cpu().numpy()
            
        if generated_audio.dtype == np.int16:
            generated_audio = generated_audio.astype(np.float32) / 32768.0
        elif generated_audio.dtype == np.int32:
            generated_audio = generated_audio.astype(np.float32) / 2147483648.0
        
        generated_audio = normalize_audio(generated_audio)
        
        # Return as audio file
        buffer = io.BytesIO()
        sf.write(buffer, generated_audio, 24000, format='WAV')
        buffer.seek(0)
        
        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=kc_voice_{language}_{int(time.time())}.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import socket
    
    def find_free_port():
        """Find a free port to use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    # Use the SERVER_PORT we determined earlier
    port = SERVER_PORT or find_free_port()
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    logger.info(f"🔍 Health Check: http://{host}:{port}/health")
    logger.info(f"🎤 KC Voice TTS: http://{host}:{port}/generate")
    
    # Production server configuration optimized for RunPod
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=1,
        loop="asyncio",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
        reload=False
    )
