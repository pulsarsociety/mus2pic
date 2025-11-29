"""
FastAPI web application for mus2pic - Lightweight and fast!
"""
import os
# Set ALL Hugging Face cache environment variables before loading any other modules
# This ensures downloads go to a single location
base_dir = "/data/.hf_home"
hub_cache_dir = os.path.join(base_dir, "hub")
os.makedirs(base_dir, exist_ok=True)
os.makedirs(hub_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = base_dir
os.environ["HF_HUB_CACHE"] = hub_cache_dir  # Explicitly set hub cache location
os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache_dir  # Compatibility for older versions

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import time
import asyncio
import random
import hashlib
from contextlib import redirect_stdout
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import sys
from audio_processor import download_audio, analyze_audio, audio_to_prompt_v3, audio_to_prompt_v4, get_genre_from_metadata, get_genre_from_spotify, infer_genre_from_features, normalize_genre
from model_options import generate_with_lcm, generate_with_sd14, generate_with_sd21, generate_with_turbo, generate_with_lcm_sd21, generate_with_sdxl_lightning, MODEL_REGISTRY

app = FastAPI(title="Mus2Pic", description="Transform music into visual art")

# Global pipeline cache - pre-load default model at startup
# This ensures the model is ready immediately and stays in VRAM
_global_pipelines = {}

@app.on_event("startup")
async def startup_event():
    """Pre-load Turbo model at startup for faster first request"""
    # Lazy import to avoid import-time errors
    try:
        from model_options import get_turbo_pipeline
        import torch
        
        device, dtype = ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)
        
        print("\n" + "="*60)
        print("🚀 STARTUP: Pre-loading Turbo model (SDXL Turbo)")
        print("="*60)
        print(f"Device: {device}, Dtype: {dtype}")
        
        # Pre-load the Turbo model (fast + high quality)
        # This loads it into VRAM immediately, so first request is instant
        pipeline = get_turbo_pipeline()
        _global_pipelines["turbo"] = pipeline
        print("✓ Turbo model (SDXL Turbo) loaded and ready in VRAM")
        print("="*60 + "\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not pre-load Turbo model: {e}")
        print("   Model will be loaded on first request instead")
        print("="*60 + "\n")

# Configuration
UPLOAD_FOLDER = 'static/generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Thread pool for parallel image generation
executor = ThreadPoolExecutor(max_workers=4)

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except:
            pass
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request models
class PromptRequest(BaseModel):
    url: str
    duration: Optional[int] = None  # Duration in seconds. None or 0 means full length
    prompt_version: str = "v3"  # "v1" or "v3" - which prompt generation function to use
    band_name: Optional[str] = None  # Optional: override extracted band name
    song_title: Optional[str] = None  # Optional: override extracted song title
    features: Optional[dict] = None  # Optional: pre-extracted audio features (to avoid re-processing)

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_steps: int = 8
    num_variations: int = 1
    model: str = "lcm"  # Model selection: lcm, sd14, sd21, turbo

class LLMPromptRequest(BaseModel):
    """Request model for LLM-based prompt generation (no audio analysis)"""
    song_title: str
    artist: str
    api_key: str  # OpenAI API key provided by user
    genre: Optional[str] = None
    additional_context: Optional[str] = None  # e.g., "acoustic version", "live"
    temperature: float = 0.8  # Creativity level (0.0-1.0)

class ValidateAPIKeyRequest(BaseModel):
    """Request model for API key validation"""
    api_key: str

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page"""
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/direct", response_class=HTMLResponse)
async def direct():
    """Serve the direct prompt page"""
    with open("templates/direct.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    favicon_path = os.path.join("static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type='image/x-icon')
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/api/models")
async def get_models():
    """Get available models with their metadata"""
    return {
        "success": True,
        "models": MODEL_REGISTRY
    }

@app.post("/api/validate-api-key")
async def validate_api_key(request: ValidateAPIKeyRequest):
    """Validate an OpenAI API key provided by the user"""
    try:
        from llm_prompt_generator import validate_api_key
        result = validate_api_key(request.api_key)
        return {
            "success": True,
            **result
        }
    except ImportError as e:
        return {
            "success": False,
            "valid": False,
            "message": f"LLM module not available: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "valid": False,
            "message": str(e)
        }

@app.post("/api/generate-prompt-llm")
async def generate_prompt_llm(request: LLMPromptRequest):
    """
    Generate SDXL Turbo optimized prompt using GPT-4o-mini.
    
    This is an alternative to audio-based prompt generation.
    Uses LLM reasoning to create visually evocative prompts from song metadata.
    
    No audio analysis needed - just provide song title, artist, API key, and optionally genre.
    """
    try:
        song_title = request.song_title.strip()
        artist = request.artist.strip()
        api_key = request.api_key.strip()
        genre = request.genre.strip() if request.genre else None
        additional_context = request.additional_context.strip() if request.additional_context else None
        temperature = min(max(request.temperature, 0.0), 1.0)  # Clamp to valid range
        
        if not song_title:
            raise HTTPException(status_code=400, detail="Song title is required")
        if not artist:
            raise HTTPException(status_code=400, detail="Artist name is required")
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required")
        
        # Import and use LLM generator
        from llm_prompt_generator import generate_prompt_with_llm
        
        prompt, generation_time, usage = generate_prompt_with_llm(
            song_title=song_title,
            artist=artist,
            api_key=api_key,
            genre=genre,
            additional_context=additional_context,
            temperature=temperature
        )
        
        return {
            "success": True,
            "prompt": prompt,
            "negative_prompt": "",  # SDXL Turbo doesn't use negative prompts
            "song": song_title,
            "band": artist,
            "genre_hint": genre,
            "generation_method": "llm",
            "llm_stats": {
                "model": usage["model"],
                "generation_time": round(generation_time, 3),
                "tokens_used": usage["total_tokens"],
                "estimated_cost_usd": usage["estimated_cost_usd"]
            }
        }
        
    except ValueError as e:
        # API key invalid
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

@app.post("/api/extract-metadata")
async def extract_metadata(request: PromptRequest):
    """Extract metadata (band/song) and audio features from YouTube URL without generating prompt"""
    try:
        youtube_url = request.url.strip()
        
        if not youtube_url:
            raise HTTPException(status_code=400, detail="YouTube URL is required")
        
        # Parse duration - None or 0 means full length
        duration = request.duration if request.duration and request.duration > 0 else None
        
        # Check if band/song is in cache (skip verification if cached)
        from audio_processor import load_cache, get_cached_entry
        cache = load_cache()
        
        # Suppress console output during processing
        f = StringIO()
        with redirect_stdout(f):
            # Download audio and get metadata
            audio_path, band_name, song_title = download_audio(youtube_url)
            
            # Check if this band/song combination is in cache
            cached_entry = get_cached_entry(cache, band_name, song_title)
            is_cached = cached_entry is not None
            
            # Also extract audio features (we'll need them anyway)
            features = analyze_audio(audio_path, duration=duration)
        
        # Clean up audio file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        
        # Convert numpy types to native Python types for JSON serialization
        # Return full features dict so we can reuse it
        import numpy as np
        
        # Helper to safely convert numpy types to native Python types
        def to_python(value, key_name="unknown"):
            """Convert numpy types to native Python types for JSON serialization."""
            try:
                if value is None:
                    return None
                # Handle numpy arrays (including 0-d arrays)
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        # 0-d array (scalar array) - convert to Python scalar
                        result = value.item()
                        print(f"   DEBUG [to_python/{key_name}]: np.ndarray (0-d) -> {type(result).__name__}", file=sys.stderr)
                        return result
                    else:
                        # Multi-dimensional array
                        result = value.tolist()
                        print(f"   DEBUG [to_python/{key_name}]: np.ndarray ({value.ndim}-d) -> list", file=sys.stderr)
                        return result
                # Handle numpy scalars
                elif isinstance(value, (np.integer, np.floating)):
                    result = float(value) if isinstance(value, np.floating) else int(value)
                    print(f"   DEBUG [to_python/{key_name}]: {type(value).__name__} -> {type(result).__name__}", file=sys.stderr)
                    return result
                # Handle lists/tuples recursively
                elif isinstance(value, (list, tuple)):
                    return [to_python(item, f"{key_name}[{i}]") for i, item in enumerate(value)]
                # Already a native Python type
                return value
            except Exception as e:
                print(f"   ERROR [to_python/{key_name}]: Failed to convert {type(value).__name__}: {e}", file=sys.stderr)
                # Return a safe default
                if isinstance(value, (int, float)):
                    return float(value) if isinstance(value, float) else int(value)
                return None
        
        # Helper to safely convert scalar values
        def safe_float(key, default=0.0):
            """Safely convert a feature value to float, handling numpy types."""
            val = features.get(key, default)
            converted = to_python(val)
            try:
                return float(converted)
            except (ValueError, TypeError):
                return default
        
        # Convert all features - handle all possible keys
        print(f"🔍 DEBUG [app.py]: Starting feature serialization, {len(features)} keys", file=sys.stderr)
        serialized_features = {}
        for key in features.keys():
            try:
                # Safe type checking - don't try to format numpy arrays
                val = features[key]
                val_type = type(val).__name__
                if isinstance(val, np.ndarray):
                    val_str = f"np.ndarray(shape={val.shape})"
                elif isinstance(val, (np.integer, np.floating)):
                    val_str = f"{val_type}({float(val)})"
                else:
                    val_str = str(val)[:50]  # Limit length
                print(f"   Processing key: {key}, type: {val_type}, value: {val_str}", file=sys.stderr)
                serialized_features[key] = to_python(features[key], key)
                print(f"   ✓ Converted {key}: {type(serialized_features[key]).__name__}", file=sys.stderr)
            except Exception as e:
                print(f"   ✗ ERROR converting {key}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                # Use a safe default
                serialized_features[key] = 0.0 if key not in ['mfcc_mean', 'mfcc_std'] else []
        
        # Ensure we have all expected keys with defaults (including new scientific features)
        expected_keys = {
            # Basic features
            'tempo': 0, 'brightness': 0, 'spectral_centroid': 0, 'energy': 0,
            'spectral_rolloff': 0, 'spectral_bandwidth': 0, 'spectral_contrast': 0,
            'zcr': 0, 'chroma_std': 0, 'rhythm_stability': 0, 'harmonicity': 0,
            'mfcc_mean': [], 'mfcc_std': [], 'raw_energy': 0,
            # NEW scientific features
            'is_major_key': True, 'mode_confidence': 0.5,
            'spectral_flux': 0.5, 'onset_density': 0.5, 'onset_density_raw': 0,
            'hp_ratio': 0.5, 'dynamic_range': 0.5, 'dynamic_range_db': 0,
            'pitch_register': 0.5, 'attack_sharpness': 0.5,
            'valence_scientific': 0.5, 'arousal_scientific': 0.5
        }
        for key, default in expected_keys.items():
            if key not in serialized_features:
                serialized_features[key] = default
        
        # Try to get genre from Spotify for LLM mode prefill
        genre_hint = None
        try:
            raw_genres = get_genre_from_spotify(band_name, song_title, skip_swap_detection=False)
            if raw_genres:
                genre_hint = normalize_genre(raw_genres)
        except Exception:
            pass  # Genre is optional, don't fail if we can't get it
        
        return {
            'success': True,
            'band': band_name,
            'song': song_title,
            'genre': genre_hint,  # For LLM mode prefill
            'features': serialized_features,
            'is_cached': is_cached  # Flag to indicate if band/song is in cache
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-prompt")
async def generate_prompt(request: PromptRequest):
    """Generate prompt from YouTube URL"""
    try:
        youtube_url = request.url.strip()
        
        if not youtube_url:
            raise HTTPException(status_code=400, detail="YouTube URL is required")
        
        # Parse duration - None or 0 means full length
        duration = request.duration if request.duration and request.duration > 0 else None
        
        # Suppress console output during processing
        f = StringIO()
        with redirect_stdout(f):
            # Step 1: Use provided features if available (from metadata extraction), otherwise extract
            audio_path = None
            if request.features:
                # Use pre-extracted features (from metadata extraction step)
                # Normalize features to ensure all values are native Python types
                import numpy as np
                def normalize_features(feat_dict):
                    """Convert all numpy types in features dict to native Python types."""
                    normalized = {}
                    for key, value in feat_dict.items():
                        if isinstance(value, np.ndarray):
                            normalized[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            normalized[key] = float(value) if isinstance(value, np.floating) else int(value)
                        elif isinstance(value, (list, tuple)):
                            normalized[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
                        elif isinstance(value, (int, float, str, bool, type(None))):
                            normalized[key] = value
                        else:
                            # Fallback: try to convert to float
                            try:
                                normalized[key] = float(value)
                            except (ValueError, TypeError):
                                normalized[key] = value
                    return normalized
                features = normalize_features(request.features)
                # No need to download audio again - we already have features
                if request.band_name and request.song_title:
                    band_name = request.band_name.strip()
                    song_title = request.song_title.strip()
                else:
                    # Still need metadata if not provided, but we can skip audio download
                    # Just get metadata from a minimal download (or skip if we have band/song)
                    pass  # We already have band/song from metadata extraction
                    band_name = None
                    song_title = None
            else:
                # Extract metadata and features normally
                if request.band_name and request.song_title:
                    # Use provided band/song, still need to download for analysis
                    audio_path, _, _ = download_audio(youtube_url)
                    band_name = request.band_name.strip()
                    song_title = request.song_title.strip()
                else:
                    # Extract metadata normally
                    audio_path, band_name, song_title = download_audio(youtube_url)
                
                # Step 2: Analyze audio with specified duration
                features = analyze_audio(audio_path, duration=duration)
            
            # Step 3: Generate prompt (v1, v3, or v4)
            genre_hint = None
            genre_source = None
            
            if request.prompt_version in ["v3", "v4"]:
                # Try to get genre from multiple sources (in order of preference)
                raw_genres = None
                genre_source = None
                
                # 1. Try metadata first (fastest, if available)
                # Only try metadata if we have audio_path (not when using pre-extracted features)
                metadata_genre = None
                if audio_path and os.path.exists(audio_path):
                    metadata_genre = get_genre_from_metadata(audio_path)
                if metadata_genre:
                    # Convert string to list for normalize_genre
                    raw_genres = [metadata_genre]
                    genre_source = "metadata"
                else:
                    # 2. Try Spotify API (most accurate, with caching)
                    # get_genre_from_spotify returns raw genres list (or None)
                    # Skip swap detection if user provided corrected values
                    skip_swap = request.band_name is not None and request.song_title is not None
                    raw_genres = get_genre_from_spotify(band_name, song_title, skip_swap_detection=skip_swap)
                    if raw_genres:
                        genre_source = "spotify"
                    else:
                        # 3. Fallback: infer from audio features
                        inferred_genre = infer_genre_from_features(features)
                        if inferred_genre:
                            raw_genres = [inferred_genre]
                            genre_source = "inferred"
                        else:
                            # 4. Final fallback: use default
                            genre_source = "default"
                            raw_genres = None  # Will normalize to "abstract" in prompt generator
                
                # Choose prompt generator based on version
                if request.prompt_version == "v4":
                    # V4: Soul-capturing prompts optimized for Turbo
                    prompt, negative_prompt = audio_to_prompt_v4(
                        features, 
                        band_name=band_name, 
                        song_title=song_title,
                        raw_genres=raw_genres
                    )
                else:
                    # V3: Genre-specific visual vocabularies
                    prompt, negative_prompt = audio_to_prompt_v3(
                        features, 
                        band_name=band_name, 
                        song_title=song_title,
                        raw_genres=raw_genres
                    )
                # Get normalized genre for response
                refined_genre = normalize_genre(raw_genres) if raw_genres else "abstract"
            else:
                # Use v1 (default) - no genre needed
                genre_source = "none"
                refined_genre = None
                prompt, negative_prompt = audio_to_prompt_v3(features, band_name=band_name, song_title=song_title, raw_genres=raw_genres)
        
        # Clean up audio file (only if we downloaded it)
        if audio_path:
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
        
        # Convert numpy types to native Python types for JSON serialization
        import numpy as np
        
        def safe_float(val, default=0.0, decimals=4):
            """Safely convert value to float."""
            try:
                if isinstance(val, np.ndarray):
                    val = val.item() if val.ndim == 0 else val.flat[0]
                return round(float(val), decimals)
            except:
                return default
        
        # Return ALL features including new scientific ones
        return {
            'success': True,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'band': band_name,
            'song': song_title,
            'from_cache': False,  # Prompt caching removed
            'genre_hint': refined_genre if request.prompt_version in ["v3", "v4"] else None,
            'genre_source': genre_source,
            'features': {
                # Basic features
                'tempo': safe_float(features.get('tempo', 0), decimals=0),
                'brightness': safe_float(features.get('brightness', 0), decimals=2),
                'energy': safe_float(features.get('energy', 0)),
                'raw_energy': safe_float(features.get('raw_energy', 0)),
                'harmonicity': safe_float(features.get('harmonicity', 0)),
                'rhythm_stability': safe_float(features.get('rhythm_stability', 0)),
                # NEW scientific features
                'is_major_key': bool(features.get('is_major_key', True)),
                'mode_confidence': safe_float(features.get('mode_confidence', 0.5)),
                'spectral_flux': safe_float(features.get('spectral_flux', 0.5)),
                'onset_density': safe_float(features.get('onset_density', 0.5)),
                'onset_density_raw': safe_float(features.get('onset_density_raw', 0)),
                'hp_ratio': safe_float(features.get('hp_ratio', 0.5)),
                'dynamic_range': safe_float(features.get('dynamic_range', 0.5)),
                'dynamic_range_db': safe_float(features.get('dynamic_range_db', 0)),
                'pitch_register': safe_float(features.get('pitch_register', 0.5)),
                'attack_sharpness': safe_float(features.get('attack_sharpness', 0.5)),
                'valence_scientific': safe_float(features.get('valence_scientific', 0.5)),
                'arousal_scientific': safe_float(features.get('arousal_scientific', 0.5)),
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_single_image(prompt, negative_prompt, output_path, num_steps, variation_index, total_variations, progress_queue=None, width=512, height=512, seed=None, model="lcm"):
    """Generate a single image (runs in thread pool)"""
    try:
        # Send initial progress update
        if progress_queue:
            try:
                progress_queue.put_nowait({
                    'type': 'progress',
                    'variation': variation_index + 1,
                    'total_variations': total_variations,
                    'step': 0,
                    'total_steps': num_steps,
                    'progress': (variation_index / total_variations) * 100
                })
            except:
                pass
        
        # Create progress callback wrapper
        def callback(step, total_steps):
            if progress_queue:
                # Ensure step is an integer (handle tensor conversion)
                if hasattr(step, 'item'):
                    step_int = int(step.item())
                elif isinstance(step, (int, float)):
                    step_int = int(step)
                else:
                    step_int = int(step)
                
                # Calculate overall progress across all variations
                base_progress = (variation_index / total_variations) * 100
                variation_progress = (step_int / total_steps) * (100 / total_variations)
                overall_progress = base_progress + variation_progress
                try:
                    progress_queue.put_nowait({
                        'type': 'progress',
                        'variation': variation_index + 1,
                        'total_variations': total_variations,
                        'step': step_int,  # Use integer step
                        'total_steps': total_steps,
                        'progress': min(overall_progress, 99)
                    })
                except Exception as e:
                    # Queue full, skip update (non-fatal)
                    pass
        
        # Select model function
        model_functions = {
            "lcm": generate_with_lcm,
            "sd14": generate_with_sd14,
            "sd21": generate_with_sd21,
            "turbo": generate_with_turbo,
            "lcm_sd21": generate_with_lcm_sd21,
            "sdxl_lightning": generate_with_sdxl_lightning
        }
        
        if model not in model_functions:
            model = "lcm"  # Default to LCM if invalid model
        
        generate_func = model_functions[model]
        
        # Models that support progress callbacks and return (path, seed, time)
        models_with_callbacks = ["lcm", "lcm_sd21", "sdxl_lightning"]
        
        # Call the selected model function
        if model in models_with_callbacks:
            output_path, seed_used, gen_time = generate_func(
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=output_path,
                num_inference_steps=num_steps,
                progress_callback=callback if progress_queue else None,
                width=width,
                height=height,
                seed=seed
            )
        else:
            # For other models, we need to add timing manually
            start_time = time.time()
            result = generate_func(
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_path=output_path,
                num_inference_steps=num_steps
            )
            gen_time = time.time() - start_time
            
            # Other models return different formats, normalize them to (path, seed, time)
            if isinstance(result, tuple):
                if len(result) >= 2:
                    output_path, seed_used = result[0], result[1]
                else:
                    output_path, seed_used = result[0], None
            else:
                output_path = result
                seed_used = None
            
            # Generate a seed for consistency if not provided
            if seed_used is None:
                seed_used = random.randint(0, 2**32 - 1) if seed is None else seed
        
        # Send generation time update
        if progress_queue:
            try:
                progress_queue.put_nowait({
                    'type': 'generation_time',
                    'variation': variation_index + 1,
                    'time': gen_time
                })
            except:
                pass
        
        # Send completion progress for this variation
        if progress_queue:
            try:
                progress_queue.put_nowait({
                    'type': 'progress',
                    'variation': variation_index + 1,
                    'total_variations': total_variations,
                    'step': num_steps,
                    'total_steps': num_steps,
                    'progress': min(((variation_index + 1) / total_variations) * 100, 99)
                })
            except:
                pass
        
        return output_path, seed_used, gen_time
    except Exception as e:
        if progress_queue:
            try:
                progress_queue.put_nowait({
                    'type': 'error',
                    'variation': variation_index + 1,
                    'error': str(e)
                })
            except:
                pass
        raise


@app.post("/api/generate-image")
async def generate_image(request: ImageRequest):
    """Generate image(s) from prompt - supports multiple variations (parallel)"""
    try:
        prompt = request.prompt.strip()
        negative_prompt = request.negative_prompt.strip()
        num_steps = request.num_steps
        num_variations = min(max(request.num_variations, 1), 4)  # Limit to 1-4 variations
        model = request.model if request.model in MODEL_REGISTRY else "lcm"
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Use default steps for model if not specified
        if num_steps == 8 and model in MODEL_REGISTRY:
            num_steps = MODEL_REGISTRY[model].get("default_steps", 8)
        
        # Generate at full size (512x512)
        width = 512
        height = 512
        
        timestamp = int(time.time())
        
        # Suppress console output during image generation
        f = StringIO()
        with redirect_stdout(f):
            # Generate multiple variations in parallel
            loop = asyncio.get_event_loop()
            tasks = []
            
            for i in range(num_variations):
                filename = f'poster_{timestamp}_{i}.png'
                output_path = os.path.join(UPLOAD_FOLDER, filename)
                
                # Submit to thread pool for parallel execution
                task = loop.run_in_executor(
                    executor,
                    generate_single_image,
                    prompt,
                    negative_prompt,
                    output_path,
                    num_steps,
                    i,
                    num_variations,
                    None,  # No progress callback for HTTP endpoint
                    width,
                    height,
                    None,  # No seed - generate unique variations
                    model  # Model selection
                )
                tasks.append((task, filename))
            
            # Wait for all images to complete
            image_urls = []
            generation_times = []
            for task, filename in tasks:
                try:
                    output_path, seed_used, gen_time = await task
                    image_urls.append({
                        'url': f'/static/generated/{filename}',
                        'filename': filename
                    })
                    generation_times.append(gen_time)
                except Exception as e:
                    print(f"Error generating variation: {e}")
                    generation_times.append(None)
        
        return {
            'success': True,
            'images': image_urls,
            'count': len(image_urls),
            'generation_times': generation_times,
            'total_time': sum(t for t in generation_times if t is not None)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Wait for client to send generation request
            data = await websocket.receive_json()
            
            if data.get('type') == 'generate':
                prompt = data.get('prompt', '').strip()
                negative_prompt = data.get('negative_prompt', '').strip()
                num_steps = data.get('num_steps', 8)
                num_variations = min(max(data.get('num_variations', 1), 1), 4)
                model = data.get('model', 'lcm')
                if model not in MODEL_REGISTRY:
                    model = 'lcm'
                
                if not prompt:
                    await manager.send_personal_message({
                        'type': 'error',
                        'message': 'Prompt is required'
                    }, websocket)
                    continue
                
                # Use default steps for model if not specified
                if num_steps == 8 and model in MODEL_REGISTRY:
                    num_steps = MODEL_REGISTRY[model].get("default_steps", 8)
                
                # Generate at full size (512x512)
                width = 512
                height = 512
                
                timestamp = int(time.time())
                image_urls = []
                generation_times = []
                
                # Create progress queue for communication between threads and async
                progress_queue = Queue()
                monitor_running = True
                
                # Task to monitor progress queue and send updates via WebSocket
                async def progress_monitor():
                    while monitor_running:
                        try:
                            # Check for progress updates (non-blocking)
                            try:
                                progress_data = progress_queue.get_nowait()
                                print(f"Progress update: {progress_data}")  # Debug
                                await manager.send_personal_message(progress_data, websocket)
                                
                                # If error, break
                                if progress_data.get('type') == 'error':
                                    break
                            except:
                                # Queue empty, wait a bit
                                await asyncio.sleep(0.1)
                        except Exception as e:
                            print(f"Progress monitor error: {e}")  # Debug
                            break
                
                # Generate images in parallel
                loop = asyncio.get_event_loop()
                tasks = []
                
                for i in range(num_variations):
                    filename = f'poster_{timestamp}_{i}.png'
                    output_path = os.path.join(UPLOAD_FOLDER, filename)
                    
                    # Submit to thread pool
                    task = loop.run_in_executor(
                        executor,
                        generate_single_image,
                        prompt,
                        negative_prompt,
                        output_path,
                        num_steps,
                        i,
                        num_variations,
                        progress_queue,
                        width,
                        height,
                        None,  # No seed - generate unique variations
                        model  # Model selection
                    )
                    tasks.append((task, filename))
                
                # Start progress monitor
                monitor_task = asyncio.create_task(progress_monitor())
                
                # Also start a fallback progress simulator in case callbacks don't work
                async def fallback_progress():
                    """Fallback progress if callbacks don't work"""
                    await asyncio.sleep(1)  # Wait a bit for real callbacks
                    step = 0
                    while monitor_running and step < num_steps:
                        await asyncio.sleep(0.5)  # Update every 0.5 seconds
                        if not monitor_running:
                            break
                        step += 1
                        # Only send if no real progress has been received recently
                        try:
                            progress_queue.put_nowait({
                                'type': 'progress',
                                'variation': 1,
                                'total_variations': num_variations,
                                'step': min(step, num_steps),
                                'total_steps': num_steps,
                                'progress': min((step / num_steps) * 90, 90)  # Cap at 90% for fallback
                            })
                        except:
                            pass  # Queue might be full, that's okay
                
                fallback_task = asyncio.create_task(fallback_progress())
                
                # Wait for all images
                for task, filename in tasks:
                    try:
                        output_path, seed_used, gen_time = await task
                        image_urls.append({
                            'url': f'/static/generated/{filename}',
                            'filename': filename
                        })
                        generation_times.append(gen_time)
                    except Exception as e:
                        await manager.send_personal_message({
                            'type': 'error',
                            'message': f'Error generating image: {str(e)}'
                        }, websocket)
                        generation_times.append(None)
                
                # Cancel fallback
                fallback_task.cancel()
                try:
                    await fallback_task
                except asyncio.CancelledError:
                    pass
                
                # Stop monitor and send completion
                monitor_running = False
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
                
                await manager.send_personal_message({
                    'type': 'complete',
                    'progress': 100,
                    'images': image_urls,
                    'count': len(image_urls),
                    'generation_times': generation_times,
                    'total_time': sum(t for t in generation_times if t is not None)
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_personal_message({
            'type': 'error',
            'message': str(e)
        }, websocket)
        manager.disconnect(websocket)

@app.get("/static/generated/{filename}")
async def serve_image(filename: str):
    """Serve generated images"""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type='image/png')
    raise HTTPException(status_code=404, detail="Image not found")

if __name__ == '__main__':
    import uvicorn
    import sys
    
    # Support reload parameter from command line
    reload = '--reload' in sys.argv or 'reload' in sys.argv
    port = 9090
    host = "0.0.0.0"
    
    # Parse port from command line if provided
    if '--port' in sys.argv:
        try:
            port_idx = sys.argv.index('--port')
            port = int(sys.argv[port_idx + 1])
        except (IndexError, ValueError):
            pass
    
    # Parse host from command line if provided
    if '--host' in sys.argv:
        try:
            host_idx = sys.argv.index('--host')
            host = sys.argv[host_idx + 1]
        except IndexError:
            pass
    
    # When running with reload, we need to use string format for uvicorn to work properly
    # However, when running python app.py directly, the module name is __main__, not app
    # So we need to check if we're being run directly or imported
    if reload:
        # For reload to work, use string format
        # This assumes you're running: uvicorn app:app --reload
        # If running python app.py --reload, it won't work - use uvicorn command instead
        import importlib.util
        spec = importlib.util.find_spec("app")
        if spec is not None and spec.origin and "app.py" in spec.origin:
            # Module can be imported as "app"
            uvicorn.run(
                "app:app",  # Module:attribute format
                host=host,
                port=port,
                reload=reload,
                reload_dirs=[os.getcwd()] if reload else None
            )
        else:
            # Can't import as "app", use app object directly (reload won't work)
            print("⚠️  Warning: Reload mode requires running with: uvicorn app:app --reload")
            print("   Running without reload...")
            uvicorn.run(
                app,
                host=host,
                port=port,
                reload=False
            )
    else:
        # Without reload, we can pass the app object directly
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False
        )
