#!/usr/bin/env python3
"""
Arabic Sentiment Classification API
واجهة برمجة التطبيقات لتصنيف المشاعر العربية

This API provides HTTP endpoints for sentiment classification of Arabic text.

Usage:
    python src/api/api.py --model outputs/arabic_sentiment_model/best_model --port 8001
    
    Or with config:
    python src/api/api.py --config config/sentiment_config.yaml --port 8001

API Endpoints:
    POST /predict - Single text prediction
    POST /predict/batch - Multiple texts prediction
    GET /health - Health check
    GET /model/info - Model information
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import uvicorn
import argparse
import logging
from pathlib import Path
import sys
import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
model_path = project_root / "src" / "model"
for _p in {project_root, model_path, Path(__file__).resolve().parent}:
    p_str = str(_p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from src.model.predictor import ArabicSentimentPredictor
from src.model.config import get_config, SentimentLabels
from src.model.validator import ArabicValidator

# Setup logging
logging.basicConfig(  # still configure (harmless) but will be disabled
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from transformers import logging as hf_logging  # type: ignore
    hf_logging.set_verbosity_error()
except Exception:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Helper discovery functions
# -------------------------------------------------
def discover_config(explicit: Optional[str]) -> Optional[Path]:
    """Return Path to config file if exists. Priority: explicit -> default under project root."""
    if explicit:
        cfg = Path(explicit)
        if not cfg.is_absolute():
            cfg = (Path.cwd() / cfg).resolve()
        return cfg if cfg.exists() else None
    default_cfg = project_root / "config" / "sentiment_config.yaml"
    return default_cfg if default_cfg.exists() else None

def candidate_model_paths(from_config: Optional[Path]) -> List[Path]:
    """Yield ordered candidate model directories (absolute)."""
    candidates: List[Path] = []
    # New preferred structure
    candidates.append(project_root / "outputs" / "arabic_sentiment_model" / "best_model")
    candidates.append(project_root / "outputs" / "arabic_sentiment_model")
    # Config-derived
    if from_config is not None:
        try:
            cfg = get_config(str(from_config))
            out_dir = cfg.get("output_dir", "outputs/arabic_sentiment_model")
            out_dir_path = (project_root / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir)
            candidates.append(out_dir_path / "best_model")
            candidates.append(out_dir_path)
        except Exception:
            pass
    # Legacy fallbacks (relative to project root)
    candidates.append(project_root / "outputs" / "arabic_sentiment_model" / "best_model")
    candidates.append(project_root / "outputs" / "arabic_sentiment_model")
    candidates.append(project_root / "outputs" / "best_model")
    candidates.append(project_root / "best_model")
    # Deduplicate preserving order
    seen = set()
    ordered: List[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered

def resolve_model_path(explicit: Optional[str], cfg_path: Optional[Path]) -> Path:
    if explicit:
        mp = Path(explicit)
        if not mp.is_absolute():
            mp = (Path.cwd() / mp).resolve()
        if not mp.exists():
            logger.error(f"Specified --model not found: {mp}")
            sys.exit(1)
        return mp
    # Auto detection
    for cand in candidate_model_paths(cfg_path):
        if cand.exists():
            return cand
    # Not found
    logger.error("❌ No model found via auto-detection.")
    logger.info("Tried paths (ordered):")
    for c in candidate_model_paths(cfg_path):
        logger.info(f"  - {c}")
    sys.exit(1)


# Thread pool for CPU-intensive predictions
executor = ThreadPoolExecutor(max_workers=4)

# -------------------------------
# Pydantic Models for Request/Response
# -------------------------------

class TextRequest(BaseModel):
    """Single text prediction request"""
    text: str = Field(..., description="Arabic text to analyze", min_length=1, max_length=5000)
    return_probabilities: bool = Field(default=True, description="Return probability distribution")
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('النص لا يمكن أن يكون فارغاً')
        return v

class BatchTextRequest(BaseModel):
    """Batch text prediction request"""
    texts: List[str] = Field(..., description="List of Arabic texts", min_items=1, max_items=100)
    return_probabilities: bool = Field(default=False, description="Return probability distribution")
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError('قائمة النصوص فارغة')
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'النص رقم {i+1} فارغ')
        return v

class PredictionResponse(BaseModel):
    """Single prediction response"""
    success: bool
    text: str
    cleaned_text: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    method: Optional[str] = None
    emojis_found: Optional[List[str]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    success: bool
    predictions: List[PredictionResponse]
    total_texts: int
    successful_predictions: int
    failed_predictions: int
    processing_time_ms: float

class ModelInfo(BaseModel):
    """Model information response"""
    model_path: str
    labels: List[str]
    label_mapping: Dict[str, int]
    device: str
    model_type: Optional[str] = None
    training_date: Optional[str] = None
    best_metric: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    uptime_seconds: float
    total_predictions: int
    version: str = "1.0.0"

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: str

# -------------------------------
# FastAPI Application
# -------------------------------

class ArabicSentimentAPI:
    """Arabic Sentiment Classification API"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize API with model
        
        Args:
            model_path: Path to trained model directory
            config_path: Path to config file (optional)
        """
        self.app = FastAPI(
            title="Arabic Sentiment Classification API",
            description="واجهة برمجة التطبيقات لتصنيف المشاعر في النصوص العربية",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.model_path = model_path
        self.config_path = config_path
        self.predictor = None
        self.validator = ArabicValidator()
        self.start_time = time.time()
        self.total_predictions = 0
        
        # Setup CORS
        self.setup_cors()
        
        # Setup routes
        self.setup_routes()
        
        # Load model
        if model_path:
            self.load_model(model_path)
    
    def setup_cors(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify allowed origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def load_model(self, model_path: str):
        """Load the sentiment model"""
        try:
            logger.info(f"Loading model from: {model_path}")
            self.predictor = ArabicSentimentPredictor(model_path)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_routes(self):
        """Setup API routes"""
        
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "message": "Arabic Sentiment Classification API",
                "description": "واجهة برمجة التطبيقات لتصنيف المشاعر العربية",
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy" if self.predictor else "model_not_loaded",
                model_loaded=self.predictor is not None,
                timestamp=datetime.now().isoformat(),
                uptime_seconds=time.time() - self.start_time,
                total_predictions=self.total_predictions
            )
        
        @self.app.get("/model/info", response_model=ModelInfo)
        async def model_info():
            """Get model information"""
            if not self.predictor:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )
            
            info = self.predictor.get_model_info()
            return ModelInfo(
                model_path=info.get("model_path", ""),
                labels=info.get("labels", SentimentLabels.LABELS),
                label_mapping=info.get("label_mapping", SentimentLabels.LABEL_TO_ID),
                device=info.get("device", "cpu"),
                model_type=info.get("model", {}).get("pretrained_name"),
                training_date=info.get("training_completed"),
                best_metric=info.get("best_metric")
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_single(request: TextRequest):
            """
            Predict sentiment for a single text
            
            Example:
                curl -X POST "http://localhost:8001/predict" \
                     -H "Content-Type: application/json" \
                     -d '{"text": "هذا المنتج ممتاز جداً"}'
            """
            if not self.predictor:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )
            
            start_time = time.time()
            
            try:
                # Run prediction in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor,
                    self.predictor.predict_single,
                    request.text,
                    request.return_probabilities
                )
                
                self.total_predictions += 1
                processing_time = (time.time() - start_time) * 1000
                
                return PredictionResponse(
                    success=result.get("success", False),
                    text=result.get("text", request.text),
                    cleaned_text=result.get("cleaned_text"),
                    prediction=result.get("prediction"),
                    confidence=result.get("confidence"),
                    probabilities=result.get("probabilities"),
                    method=result.get("method"),
                    emojis_found=result.get("emojis_found"),
                    error=result.get("error"),
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed: {str(e)}"
                )
        
        @self.app.post("/predict/batch", response_model=BatchPredictionResponse)
        async def predict_batch(request: BatchTextRequest):
            """
            Predict sentiment for multiple texts
            
            Example:
                curl -X POST "http://localhost:8001/predict/batch" \
                     -H "Content-Type: application/json" \
                     -d '{"texts": ["نص أول", "نص ثاني", "نص ثالث"]}'
            """
            if not self.predictor:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )
            
            start_time = time.time()
            
            try:
                # Process batch in thread pool
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    executor,
                    self.predictor.predict_batch,
                    request.texts,
                    request.return_probabilities
                )
                
                self.total_predictions += len(request.texts)
                processing_time = (time.time() - start_time) * 1000
                
                # Convert results to response format
                predictions = []
                successful = 0
                failed = 0
                
                for i, result in enumerate(results):
                    pred_response = PredictionResponse(
                        success=result.get("success", False),
                        text=result.get("text", request.texts[i]),
                        cleaned_text=result.get("cleaned_text"),
                        prediction=result.get("prediction"),
                        confidence=result.get("confidence"),
                        probabilities=result.get("probabilities"),
                        method=result.get("method"),
                        emojis_found=result.get("emojis_found"),
                        error=result.get("error")
                    )
                    predictions.append(pred_response)
                    
                    if result.get("success"):
                        successful += 1
                    else:
                        failed += 1
                
                return BatchPredictionResponse(
                    success=True,
                    predictions=predictions,
                    total_texts=len(request.texts),
                    successful_predictions=successful,
                    failed_predictions=failed,
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Batch prediction failed: {str(e)}"
                )
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions"""
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error=exc.detail,
                    timestamp=datetime.now().isoformat()
                ).dict()
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions"""
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Internal server error",
                    detail=str(exc),
                    timestamp=datetime.now().isoformat()
                ).dict()
            )

# -------------------------------
# CLI and Main Entry Point
# -------------------------------

def main():
    """Main entry point for the API"""
    parser = argparse.ArgumentParser(
        description="Arabic Sentiment Classification API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with trained model
    python api.py --model outputs/arabic_sentiment_model/best_model --port 8001
    
    # Start with config file
    python api.py --config config/sentiment_config.yaml --port 8001
    
    # Start with custom host
    python api.py --model outputs/arabic_sentiment_model/best_model --host 0.0.0.0 --port 8080
    
API Usage:
    # Single prediction
    curl -X POST "http://localhost:8001/predict" \\
         -H "Content-Type: application/json" \\
         -d '{"text": "هذا المنتج رائع"}'
    
    # Batch prediction
    curl -X POST "http://localhost:8001/predict/batch" \\
         -H "Content-Type: application/json" \\
         -d '{"texts": ["نص أول", "نص ثاني"]}'
    
    # Health check
    curl "http://localhost:8001/health"
    
    # Model info
    curl "http://localhost:8001/model/info"
    
Notes:
    If --config is omitted the server will attempt to auto-detect 'config/sentiment_config.yaml'.
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the server (default: 8001)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)"
    )
    
    args = parser.parse_args()

    cfg_path = discover_config(args.config)
    model_path = resolve_model_path(args.model, cfg_path)
    
    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"❌ Model directory not found: {model_path}")
        logger.info("\n📋 To fix this issue:\n")
        logger.info("1. Check if you have trained a model:")
        logger.info("   python main.py status\n")
        logger.info("2. If no model exists, train one:")
        logger.info("   python main.py train --config config/sentiment_config.yaml\n")
        logger.info("3. If model exists elsewhere, specify its path:")
        logger.info("   python api.py --model /correct/path/to/model\n")
        logger.info("4. Common model locations:")
        logger.info("   - outputs/arabic_sentiment_model/best_model")
        logger.info("   - outputs/best_model")
        logger.info("   - checkpoints/best_model")
        sys.exit(1)
    
    # Create API instance
    api = ArabicSentimentAPI(model_path=str(model_path), config_path=str(cfg_path) if cfg_path else None)
    
    # Start server
    # Intentionally not logging startup details (logging suppressed)
    
    try:
        # Force quiet uvicorn (ignore provided log-level to meet silent requirement)
        uvicorn.run(
            api.app,
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level="critical",  # silence access & error logs
            access_log=False
        )
    except KeyboardInterrupt:
        logger.info("⏹️ Server stopped by user")
    except Exception as e:
        logger.error(f"💥 Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Expose a module-level FastAPI instance for `uvicorn api.api:app`
try:
    _cfg = discover_config(None)
    _model = resolve_model_path(None, _cfg)
    app = ArabicSentimentAPI(model_path=str(_model), config_path=str(_cfg) if _cfg else None).app
except SystemExit:
    # Model not found; provide a minimal placeholder app with error endpoint
    placeholder = FastAPI(title="Arabic Sentiment API (Uninitialized)")
    @placeholder.get("/error")
    async def _error():  # type: ignore
        return {"error": "Model not found during module import. Run with training or specify --model."}
    app = placeholder