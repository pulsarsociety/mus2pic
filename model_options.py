"""
Alternative image generation models optimized for M1 MacBook Air
Choose based on your priority: speed, quality, or memory usage
"""

import torch
from diffusers import DiffusionPipeline, LCMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from PIL import Image
import threading
import os
import shutil
from pathlib import Path
try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Global model cache with thread safety (separate caches for each model)
_pipe_cache = {}  # Dictionary to store different model pipelines
_cache_lock = threading.Lock()
_cache_load_count = {}  # Track how many times each model was loaded
_cache_access_order = []  # Track access order for LRU eviction

# Configuration
MAX_CACHED_MODELS = 4  # Maximum number of models to keep in memory (M1 VRAM is tight)
ENABLE_CPU_OFFLOAD = False  # Set to True for SDXL on 8GB models (adds latency on MPS)
ENABLE_ATTENTION_SLICING = True  # Reduces memory but adds small slowdown (~10-20%)


def get_device_dtype():
    """
    Get device and dtype for M-series Mac optimization.
    Returns (device, dtype) tuple.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16  # float16 is much faster on MPS (2-3x speedup)
    return "cpu", torch.float32


def _enforce_cache_limit(max_models=MAX_CACHED_MODELS):
    """
    Enforce cache limit by evicting oldest model (LRU).
    This prevents memory issues on M1 Macs with limited VRAM.
    """
    global _pipe_cache, _cache_access_order
    
    if len(_pipe_cache) > max_models:
        # Remove oldest accessed model
        while len(_pipe_cache) > max_models and _cache_access_order:
            oldest_key = _cache_access_order.pop(0)
            if oldest_key in _pipe_cache:
                print(f"🧹 Evicting model from cache: {oldest_key} (freeing VRAM)")
                del _pipe_cache[oldest_key]
                # Clear MPS cache to free VRAM
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                break


def _update_cache_access(cache_key):
    """Update access order for LRU eviction."""
    global _cache_access_order
    if cache_key in _cache_access_order:
        _cache_access_order.remove(cache_key)
    _cache_access_order.append(cache_key)

# Model registry with metadata
MODEL_REGISTRY = {
    "lcm": {
        "name": "LCM (Latent Consistency Model)",
        "description": "Fastest option - generates in 4-8 steps. Best balance of speed and quality.",
        "expected_time": "6-12 seconds",
        "memory": "4-5GB",
        "quality": "Good",
        "speed": "Very Fast",
        "default_steps": 8,
        "function": "generate_with_lcm"
    },
    "sd14": {
        "name": "Stable Diffusion 1.4",
        "description": "Smallest model - best for limited RAM (8GB). Good quality, slightly older.",
        "expected_time": "15-25 seconds",
        "memory": "3.5-4GB",
        "quality": "Good",
        "speed": "Medium",
        "default_steps": 20,
        "function": "generate_with_sd14"
    },
    "sd21": {
        "name": "Stable Diffusion 2.1",
        "description": "Better quality than 1.5 - improved prompt understanding. Requires more RAM.",
        "expected_time": "20-30 seconds",
        "memory": "4-5GB",
        "quality": "Very Good",
        "speed": "Medium",
        "default_steps": 20,
        "function": "generate_with_sd21"
    },
    "turbo": {
        "name": "SDXL Turbo",
        "description": "Very fast with high quality - can generate in 1-4 steps. Requires 8GB+ RAM.",
        "expected_time": "8-15 seconds",
        "memory": "6-8GB",
        "quality": "Excellent",
        "speed": "Fast",
        "default_steps": 1,
        "function": "generate_with_turbo"
    },
    "lcm_sd21": {
        "name": "LCM SD2.1",
        "description": "LCM based on SD2.1 - fast generation (4-8 steps) with better prompt understanding than SD1.5 LCM.",
        "expected_time": "8-15 seconds",
        "memory": "5-6GB",
        "quality": "Very Good",
        "speed": "Very Fast",
        "default_steps": 8,
        "function": "generate_with_lcm_sd21"
    },
    "sdxl_lightning": {
        "name": "SDXL Lightning",
        "description": "Ultra-fast SDXL model - generates in 1-4 steps with excellent quality. Requires 8GB+ RAM.",
        "expected_time": "5-10 seconds",
        "memory": "7-9GB",
        "quality": "Excellent",
        "speed": "Ultra Fast",
        "default_steps": 4,
        "function": "generate_with_sdxl_lightning"
    }
}

def get_lcm_pipeline(model_name="lcm", model_id="SimianLuo/LCM_Dreamshaper_v7"):
    """
    Get or create cached LCM pipeline (thread-safe).
    Unified function for both LCM and LCM SD2.1.
    Model is loaded once and reused for all requests.
    
    Args:
        model_name: Cache key name ("lcm" or "lcm_sd21")
        model_id: Hugging Face model ID
    """
    global _pipe_cache, _cache_load_count
    import time as time_module
    
    cache_key = model_name
    
    # Fast path: check cache first (no lock needed for read)
    if cache_key in _pipe_cache and _pipe_cache[cache_key] is not None:
        _update_cache_access(cache_key)
        return _pipe_cache[cache_key]
    
    # Slow path: need to load model (acquire lock)
    with _cache_lock:
        # Double-check after acquiring lock (another thread might have loaded it)
        if cache_key in _pipe_cache and _pipe_cache[cache_key] is not None:
            _update_cache_access(cache_key)
            return _pipe_cache[cache_key]
        
        # Clear MPS cache before loading new model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Enforce cache limit before loading
        _enforce_cache_limit()
        
        load_start = time_module.time()
        print("\n" + "="*60)
        print(f"LOADING {model_name.upper()} MODEL (First time - will be cached)")
        print("="*60)
        
        device, dtype = get_device_dtype()
        
        print(f"Loading {model_name} model on {device}...")
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(device)
            
            # Enable attention slicing for memory efficiency (optional, adds small slowdown)
            # Note: Disable for LCM if you get black images - it can cause issues on MPS
            if ENABLE_ATTENTION_SLICING and model_name != "lcm":
                pipe.enable_attention_slicing()
            
            # Optional CPU offload (adds latency on MPS)
            if ENABLE_CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
            
            # Use LCM scheduler for fast generation
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            
            # MPS workaround: Ensure VAE is properly configured for MPS
            if device == "mps" and hasattr(pipe, 'vae'):
                # Force VAE to use float16 on MPS to avoid black images
                if hasattr(pipe.vae, 'to'):
                    pipe.vae = pipe.vae.to(dtype=torch.float16)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and model_name == "lcm_sd21":
                print("⚠️ Low VRAM – falling back to LCM 1.5")
                # Try to free memory
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                # Fallback to regular LCM
                return get_lcm_pipeline("lcm", "SimianLuo/LCM_Dreamshaper_v7")
            raise
        
        load_time = time_module.time() - load_start
        _pipe_cache[cache_key] = pipe
        _cache_load_count[cache_key] = _cache_load_count.get(cache_key, 0) + 1
        _update_cache_access(cache_key)
        print(f"✓ Model loaded and cached on {device} (took {load_time:.2f}s)")
        print(f"  Cache load count: {_cache_load_count[cache_key]} (should be 1 if caching works!)")
        print("="*60 + "\n")
        return _pipe_cache[cache_key]


def get_lcm_sd21_pipeline():
    """
    Get or create cached LCM SD2.1 pipeline (thread-safe).
    Uses unified get_lcm_pipeline() function.
    """
    try:
        return get_lcm_pipeline("lcm_sd21", "latent-consistency/lcm-sd21")
    except Exception:
        # Fallback: Use SD2.1 base with LCM scheduler
        print("  Note: Using SD2.1 base with LCM scheduler (LCM-specific model not found)")
        return get_lcm_pipeline("lcm_sd21", "stabilityai/stable-diffusion-2-1")


def get_sdxl_lightning_pipeline(num_inference_steps=4):
    """
    Get or create cached SDXL Lightning pipeline (thread-safe).
    Model is loaded once and reused for all requests.
    
    Args:
        num_inference_steps: Number of inference steps (2, 4, or 8) - determines which checkpoint to load
    """
    global _pipe_cache, _cache_load_count
    import time as time_module
    
    # Determine checkpoint based on steps (use closest available: 2, 4, or 8)
    if num_inference_steps <= 2:
        step_key = 2
        checkpoint_name = "sdxl_lightning_2step_unet.safetensors"
    elif num_inference_steps <= 4:
        step_key = 4
        checkpoint_name = "sdxl_lightning_4step_unet.safetensors"
    else:
        step_key = 8
        checkpoint_name = "sdxl_lightning_8step_unet.safetensors"
    
    # Cache key includes step count to cache different checkpoints
    cache_key = f"sdxl_lightning_{step_key}step"
    
    # Fast path: check cache first
    if cache_key in _pipe_cache and _pipe_cache[cache_key] is not None:
        _update_cache_access(cache_key)
        return _pipe_cache[cache_key]
    
    # Slow path: need to load model
    with _cache_lock:
        # Double-check after acquiring lock
        if cache_key in _pipe_cache and _pipe_cache[cache_key] is not None:
            _update_cache_access(cache_key)
            return _pipe_cache[cache_key]
        
        # Clear MPS cache before loading new model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Enforce cache limit before loading
        _enforce_cache_limit()
        
        load_start = time_module.time()
        print("\n" + "="*60)
        print(f"LOADING SDXL LIGHTNING MODEL ({step_key}-step checkpoint)")
        print("="*60)
        print("⚠️  Requires ~8GB RAM - may not work on 8GB M1 Air")
        
        device, dtype = get_device_dtype()
        
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors and huggingface_hub are required for SDXL Lightning. Install with: pip install safetensors huggingface_hub")
        
        # SDXL Lightning requires loading base SDXL + Lightning UNet checkpoint
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        lightning_repo = "ByteDance/SDXL-Lightning"
        
        print(f"Loading SDXL Lightning model on {device}...")
        print(f"  Base model: {base_model}")
        print(f"  Lightning checkpoint: {checkpoint_name} (for {step_key} steps)")
        
        # Load base SDXL model
        print("  Loading base SDXL model...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None
        )
        
        # Load Lightning UNet checkpoint
        print(f"  Loading Lightning UNet checkpoint: {checkpoint_name}")
        unet = UNet2DConditionModel.from_config(base_model, subfolder="unet")
        checkpoint_path = hf_hub_download(lightning_repo, checkpoint_name)
        unet.load_state_dict(load_file(checkpoint_path, device=device))
        unet = unet.to(device).to(dtype)
        
        # Replace UNet in pipeline
        pipe.unet = unet
        
        # Set scheduler to use "trailing" timesteps (required for Lightning)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing"
        )
        
        pipe = pipe.to(device)
        
        # Enable attention slicing for memory efficiency
        pipe.enable_attention_slicing()
        
        # Optional CPU offload (adds latency on MPS)
        if ENABLE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        
        load_time = time_module.time() - load_start
        _pipe_cache[cache_key] = pipe
        _cache_load_count[cache_key] = _cache_load_count.get(cache_key, 0) + 1
        _update_cache_access(cache_key)
        print(f"✓ Model loaded and cached on {device} (took {load_time:.2f}s)")
        print(f"  Cache load count: {_cache_load_count[cache_key]} (should be 1 if caching works!)")
        print("="*60 + "\n")
        return _pipe_cache[cache_key]


def generate_with_lcm(prompt, negative_prompt, output_path="generated_poster_lcm.png", num_inference_steps=8, progress_callback=None, width=512, height=512, seed=None):
    """
    LCM (Latent Consistency Model) - FASTEST option for M1
    - Generates images in 4-8 steps (vs 20-30 for regular SD)
    - ~1 second per image on M1
    - Similar quality to SD 1.5
    - Memory: ~4-5GB
    - Uses cached model for speed
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Negative prompt
        output_path: Path to save generated image
        num_inference_steps: Number of inference steps (4-8 recommended)
        progress_callback: Optional callback function(step, total_steps) for progress updates
        width: Image width (default 512, smaller for faster preview)
        height: Image height (default 512, smaller for faster preview)
        seed: Random seed for reproducibility (None for random)
    
    Returns:
        tuple: (output_path, seed_used) - Path to saved image and the seed used
    
    Install: pip install diffusers transformers accelerate
    """
    try:
        import time as time_module
        start_time = time_module.time()
        
        # Get cached pipeline (loads on first call)
        pipe = get_lcm_pipeline()
        
        model_load_time = time_module.time() - start_time
        cache_key = "lcm"
        load_count = _cache_load_count.get(cache_key, 0)
        if model_load_time > 1.0:  # If it took more than 1 second, model was loaded
            if load_count == 1:
                print(f"✓ Model loaded (first time): {model_load_time:.2f}s - will be cached for future requests")
            else:
                print(f"⚠️  Model loading took {model_load_time:.2f}s (load count: {load_count})")
                print(f"⚠️  WARNING: Cache may not be working! Model loaded {load_count} times.")
        else:
            print(f"✓ Using cached model (cache access: {model_load_time*1000:.1f}ms, load count: {load_count})")
        
        # LCM is optimized for 4-8 steps. Using more steps doesn't improve quality much
        if num_inference_steps > 8:
            print(f"⚠️  Note: LCM works best with 4-8 steps. Using {num_inference_steps} steps may be slower.")
        
        # Create diffusers-compatible callback wrapper
        # Use callback_on_step_end for newer diffusers API
        callback_to_use = None
        if progress_callback:
            def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                """Callback for newer diffusers API (callback_on_step_end)"""
                try:
                    # step_index should be an integer (0, 1, 2, 3...)
                    # Convert to int if it's a tensor or other type
                    if hasattr(step_index, 'item'):
                        step = int(step_index.item())
                    elif isinstance(step_index, (int, float)):
                        step = int(step_index)
                    else:
                        step = int(step_index)
                    
                    # step_index is 0-based, convert to 1-based for display
                    progress_callback(step + 1, num_inference_steps)
                except Exception as e:
                    # Log but don't break generation
                    print(f"Callback error (non-fatal): {e}")
                
                # Return callback_kwargs (required by diffusers)
                return callback_kwargs
            
            callback_to_use = callback_on_step_end
        
        # Generate seed if not provided
        import random
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Set up generator with seed for reproducibility (if provided)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipe.device)
            generator.manual_seed(seed)
        
        generation_start = time_module.time()
        print(f"Generating with {num_inference_steps} steps (using cached model, seed: {seed})...")
        
        # LCM guidance scale: use 1.0 (lower guidance can cause black images on MPS)
        guidance_scale = 1.0
        
        # Use no_grad() for MPS to avoid black images (inference_mode can cause issues on MPS)
        # For CUDA/CPU, use inference_mode for better performance
        if pipe.device.type == "mps":
            context_manager = torch.no_grad()
        else:
            context_manager = torch.inference_mode()
        
        with context_manager:
            # Use callback_on_step_end for newer diffusers API (avoids deprecation warnings)
            call_kwargs = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'width': width,
                'height': height,
            }
            
            # Add generator if seed provided
            if generator is not None:
                call_kwargs['generator'] = generator
            
            # Add callback if provided
            if callback_to_use:
                call_kwargs['callback_on_step_end'] = callback_to_use
            
            image = pipe(**call_kwargs).images[0]
        
        generation_time = time_module.time() - generation_start
        total_time = time_module.time() - start_time
        
        # Save full-size image
        image.save(output_path)
        print(f"✓ Saved to: {output_path} (seed: {seed})")
        print(f"  Generation time: {generation_time:.2f}s, Total time: {total_time:.2f}s\n")
        
        return output_path, seed, generation_time
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def generate_with_sd14(prompt, negative_prompt, output_path="generated_poster_sd14.png", num_inference_steps=20):
    """
    Stable Diffusion 1.4 - SMALLEST model
    - Slightly older but smaller
    - Memory: ~3.5-4GB
    - Good quality, slightly less than SD 1.5
    - Best for: Limited RAM (8GB M1 Air)
    """
    print("\n" + "="*60)
    print("USING STABLE DIFFUSION 1.4 - SMALLEST")
    print("="*60)
    
    # Clear MPS cache before loading
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    device, dtype = get_device_dtype()
    
    try:
        model_id = "CompVis/stable-diffusion-v1-4"
        
        print(f"Loading SD 1.4 on {device}...")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        
        # Enable attention slicing for memory efficiency
        pipe.enable_attention_slicing()
        
        print(f"Generating with {num_inference_steps} steps...")
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                width=512,
                height=512,
            ).images[0]
        
        image.save(output_path)
        print(f"✓ Saved to: {output_path}\n")
        return output_path
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def generate_with_sd21(prompt, negative_prompt, output_path="generated_poster_sd21.png", num_inference_steps=20):
    """
    Stable Diffusion 2.1 - Better quality than 1.5
    - Improved prompt understanding
    - Memory: ~4-5GB
    - Slightly slower than 1.5
    - Best for: Better quality when you have 16GB RAM
    """
    print("\n" + "="*60)
    print("USING STABLE DIFFUSION 2.1 - BETTER QUALITY")
    print("="*60)
    
    # Clear MPS cache before loading
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    device, dtype = get_device_dtype()
    
    try:
        model_id = "stabilityai/stable-diffusion-2-1"
        
        print(f"Loading SD 2.1 on {device}...")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        
        # Enable attention slicing for memory efficiency (optional)
        if ENABLE_ATTENTION_SLICING:
            pipe.enable_attention_slicing()
        
        # Optional CPU offload for SD2.1 on 8GB models
        if ENABLE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        
        print(f"Generating with {num_inference_steps} steps...")
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                width=512,
                height=512,
            ).images[0]
        
        image.save(output_path)
        print(f"✓ Saved to: {output_path}\n")
        return output_path
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def generate_with_turbo(prompt, negative_prompt, output_path="generated_poster_turbo.png", num_inference_steps=1):
    """
    SDXL Turbo - Very fast, good quality
    - Can generate in 1-4 steps
    - Memory: ~6-8GB (heavier)
    - Best quality but needs more RAM
    - Best for: 16GB+ M1 Air, when you want best quality + speed
    """
    print("\n" + "="*60)
    print("USING SDXL TURBO - FAST + HIGH QUALITY")
    print("="*60)
    print("⚠️  Requires ~8GB RAM - may not work on 8GB M1 Air")
    
    # Clear MPS cache before loading
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    device, dtype = get_device_dtype()
    
    try:
        model_id = "stabilityai/sdxl-turbo"
        
        print(f"Loading SDXL Turbo on {device}...")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        
        # Enable attention slicing for memory efficiency
        pipe.enable_attention_slicing()
        
        # Optional CPU offload for SDXL on 8GB models
        if ENABLE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        
        print(f"Generating with {num_inference_steps} step(s)...")
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,  # Turbo uses 0.0 guidance
                width=512,
                height=512,
            ).images[0]
        
        image.save(output_path)
        print(f"✓ Saved to: {output_path}\n")
        return output_path
        
    except Exception as e:
        print(f"Error: {e}")
        print("If you got OOM (Out of Memory), try SD 1.4 or LCM instead")
        raise


def generate_with_lcm_sd21(prompt, negative_prompt, output_path="generated_poster_lcm_sd21.png", num_inference_steps=8, progress_callback=None, width=512, height=512, seed=None):
    """
    LCM SD2.1 - Fast generation with better prompt understanding
    - Generates images in 4-8 steps (vs 20-30 for regular SD2.1)
    - Better prompt understanding than SD1.5-based LCM
    - Memory: ~5-6GB
    - Uses cached model for speed
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Negative prompt
        output_path: Path to save generated image
        num_inference_steps: Number of inference steps (4-8 recommended)
        progress_callback: Optional callback function(step, total_steps) for progress updates
        width: Image width (default 512)
        height: Image height (default 512)
        seed: Random seed for reproducibility (None for random)
    
    Returns:
        tuple: (output_path, seed_used, generation_time)
    """
    try:
        import time as time_module
        start_time = time_module.time()
        
        # Get cached pipeline (loads on first call)
        pipe = get_lcm_sd21_pipeline()
        
        model_load_time = time_module.time() - start_time
        cache_key = "lcm_sd21"
        load_count = _cache_load_count.get(cache_key, 0)
        if model_load_time > 1.0:
            if load_count == 1:
                print(f"✓ Model loaded (first time): {model_load_time:.2f}s - will be cached for future requests")
            else:
                print(f"⚠️  Model loading took {model_load_time:.2f}s (load count: {load_count})")
        else:
            print(f"✓ Using cached model (cache access: {model_load_time*1000:.1f}ms, load count: {load_count})")
        
        # LCM is optimized for 4-8 steps
        if num_inference_steps > 8:
            print(f"⚠️  Note: LCM works best with 4-8 steps. Using {num_inference_steps} steps may be slower.")
        
        # Create callback wrapper
        callback_to_use = None
        if progress_callback:
            def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                try:
                    if hasattr(step_index, 'item'):
                        step = int(step_index.item())
                    elif isinstance(step_index, (int, float)):
                        step = int(step_index)
                    else:
                        step = int(step_index)
                    progress_callback(step + 1, num_inference_steps)
                except Exception as e:
                    print(f"Callback error (non-fatal): {e}")
                return callback_kwargs
            callback_to_use = callback_on_step_end
        
        # Generate seed if not provided
        import random
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Set up generator with seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipe.device)
            generator.manual_seed(seed)
        
        generation_start = time_module.time()
        print(f"Generating with {num_inference_steps} steps (using cached model, seed: {seed})...")
        
        # LCM guidance scale: use 1.0 (lower guidance can cause black images on MPS)
        guidance_scale = 1.0
        
        with torch.inference_mode():
            call_kwargs = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'width': width,
                'height': height,
            }
            
            if generator is not None:
                call_kwargs['generator'] = generator
            
            if callback_to_use:
                call_kwargs['callback_on_step_end'] = callback_to_use
            
            image = pipe(**call_kwargs).images[0]
        
        generation_time = time_module.time() - generation_start
        total_time = time_module.time() - start_time
        
        image.save(output_path)
        print(f"✓ Saved to: {output_path} (seed: {seed})")
        print(f"  Generation time: {generation_time:.2f}s, Total time: {total_time:.2f}s\n")
        
        return output_path, seed, generation_time
        
    except Exception as e:
        print(f"Error: {e}")
        raise


def generate_with_sdxl_lightning(prompt, negative_prompt, output_path="generated_poster_sdxl_lightning.png", num_inference_steps=4, progress_callback=None, width=512, height=512, seed=None):
    """
    SDXL Lightning - Ultra-fast SDXL generation
    - Generates images in 1-4 steps (ultra-fast)
    - Excellent quality with SDXL base
    - Memory: ~7-9GB (heavier)
    - Uses cached model for speed
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Negative prompt
        output_path: Path to save generated image
        num_inference_steps: Number of inference steps (1-4 recommended, model is 4-step)
        progress_callback: Optional callback function(step, total_steps) for progress updates
        width: Image width (default 512, SDXL supports up to 1024)
        height: Image height (default 512, SDXL supports up to 1024)
        seed: Random seed for reproducibility (None for random)
    
    Returns:
        tuple: (output_path, seed_used, generation_time)
    """
    try:
        import time as time_module
        start_time = time_module.time()
        
        # Get cached pipeline (loads on first call) - pass steps to load correct checkpoint
        pipe = get_sdxl_lightning_pipeline(num_inference_steps)
        
        model_load_time = time_module.time() - start_time
        
        # Determine which checkpoint was loaded
        if num_inference_steps <= 2:
            step_key = 2
        elif num_inference_steps <= 4:
            step_key = 4
        else:
            step_key = 8
        cache_key = f"sdxl_lightning_{step_key}step"
        
        load_count = _cache_load_count.get(cache_key, 0)
        if model_load_time > 1.0:
            if load_count == 1:
                print(f"✓ Model loaded (first time): {model_load_time:.2f}s - will be cached for future requests")
            else:
                print(f"⚠️  Model loading took {model_load_time:.2f}s (load count: {load_count})")
        else:
            print(f"✓ Using cached model (cache access: {model_load_time*1000:.1f}ms, load count: {load_count})")
        
        # SDXL Lightning supports 2, 4, or 8 steps
        if num_inference_steps not in [2, 4, 8]:
            print(f"⚠️  Note: SDXL Lightning works best with 2, 4, or 8 steps. Using {num_inference_steps} steps (loaded {step_key}-step checkpoint).")
        
        # Create callback wrapper
        callback_to_use = None
        if progress_callback:
            def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                try:
                    if hasattr(step_index, 'item'):
                        step = int(step_index.item())
                    elif isinstance(step_index, (int, float)):
                        step = int(step_index)
                    else:
                        step = int(step_index)
                    progress_callback(step + 1, num_inference_steps)
                except Exception as e:
                    print(f"Callback error (non-fatal): {e}")
                return callback_kwargs
            callback_to_use = callback_on_step_end
        
        # Generate seed if not provided
        import random
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Set up generator with seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=pipe.device)
            generator.manual_seed(seed)
        
        generation_start = time_module.time()
        print(f"Generating with {num_inference_steps} steps (using cached model, seed: {seed})...")
        
        # SDXL Lightning requires guidance_scale=0
        with torch.inference_mode():
            call_kwargs = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': 0.0,  # SDXL Lightning requires guidance_scale=0
                'width': width,
                'height': height,
            }
            
            if generator is not None:
                call_kwargs['generator'] = generator
            
            if callback_to_use:
                call_kwargs['callback_on_step_end'] = callback_to_use
            
            image = pipe(**call_kwargs).images[0]
        
        generation_time = time_module.time() - generation_start
        total_time = time_module.time() - start_time
        
        image.save(output_path)
        print(f"✓ Saved to: {output_path} (seed: {seed})")
        print(f"  Generation time: {generation_time:.2f}s, Total time: {total_time:.2f}s\n")
        
        return output_path, seed, generation_time
        
    except Exception as e:
        print(f"Error: {e}")
        print("If you got OOM (Out of Memory), try LCM or SD 1.4 instead")
        raise


# Model comparison guide
MODEL_COMPARISON = {
    "lcm": {
        "function": generate_with_lcm,
        "speed": "⭐⭐⭐⭐⭐ (Fastest - 4-8 steps)",
        "quality": "⭐⭐⭐⭐ (Good)",
        "memory": "⭐⭐⭐⭐ (4-5GB)",
        "best_for": "Speed priority",
        "steps": 4
    },
    "sd14": {
        "function": generate_with_sd14,
        "speed": "⭐⭐⭐ (Standard - 20 steps)",
        "quality": "⭐⭐⭐⭐ (Good)",
        "memory": "⭐⭐⭐⭐⭐ (3.5-4GB - Smallest)",
        "best_for": "8GB RAM M1 Air",
        "steps": 20
    },
    "sd15": {
        "function": None,  # Your current implementation
        "speed": "⭐⭐⭐ (Standard - 20-30 steps)",
        "quality": "⭐⭐⭐⭐ (Very Good)",
        "memory": "⭐⭐⭐⭐ (4-5GB)",
        "best_for": "Balance of speed/quality",
        "steps": 20
    },
    "sd21": {
        "function": generate_with_sd21,
        "speed": "⭐⭐⭐ (Standard - 20 steps)",
        "quality": "⭐⭐⭐⭐⭐ (Excellent)",
        "memory": "⭐⭐⭐⭐ (4-5GB)",
        "best_for": "Quality priority",
        "steps": 20
    },
    "turbo": {
        "function": generate_with_turbo,
        "speed": "⭐⭐⭐⭐⭐ (Very Fast - 1-4 steps)",
        "quality": "⭐⭐⭐⭐⭐ (Excellent)",
        "memory": "⭐⭐ (6-8GB - Heavy)",
        "best_for": "16GB+ RAM, best quality",
        "steps": 1
    },
    "lcm_sd21": {
        "function": generate_with_lcm_sd21,
        "speed": "⭐⭐⭐⭐⭐ (Very Fast - 4-8 steps)",
        "quality": "⭐⭐⭐⭐ (Very Good)",
        "memory": "⭐⭐⭐ (5-6GB)",
        "best_for": "Better prompt understanding + speed",
        "steps": 8
    },
    "sdxl_lightning": {
        "function": generate_with_sdxl_lightning,
        "speed": "⭐⭐⭐⭐⭐ (Ultra Fast - 1-4 steps)",
        "quality": "⭐⭐⭐⭐⭐ (Excellent)",
        "memory": "⭐⭐ (7-9GB - Very Heavy)",
        "best_for": "16GB+ RAM, ultra-fast + excellent quality",
        "steps": 4
    }
}


def print_model_comparison():
    """Print a comparison table of all models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON FOR M1 MACBOOK AIR")
    print("="*80)
    print(f"{'Model':<10} {'Speed':<30} {'Quality':<15} {'Memory':<20} {'Best For':<20}")
    print("-"*80)
    for name, info in MODEL_COMPARISON.items():
        print(f"{name.upper():<10} {info['speed']:<30} {info['quality']:<15} {info['memory']:<20} {info['best_for']:<20}")
    print("="*80)
    print("\n💡 RECOMMENDATIONS:")
    print("  • 8GB RAM M1 Air: Use 'sd14' or 'lcm'")
    print("  • 16GB RAM M1 Air: Use 'lcm' (fastest) or 'sd21' (best quality)")
    print("  • Want speed: Use 'lcm' (4 steps)")
    print("  • Want quality: Use 'sd21' or 'turbo' (if you have 16GB+)")
    print("="*80 + "\n")


def get_huggingface_cache_dir():
    """Get the Hugging Face cache directory"""
    cache_dir = os.getenv("HF_HOME")
    if cache_dir:
        return os.path.join(cache_dir, "hub")
    
    # Default location
    home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "huggingface", "hub")


def remove_model_cache(model_key):
    """
    Remove cached model files from disk to free up space.
    
    Args:
        model_key: One of "turbo", "sd14", "sd21", "lcm", "lcm_sd21", "sdxl_lightning"
    
    Returns:
        tuple: (success: bool, message: str, freed_space_mb: float)
    """
    # Map model keys to their Hugging Face model IDs
    model_id_map = {
        "turbo": "stabilityai/sdxl-turbo",
        "sd14": "CompVis/stable-diffusion-v1-4",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "lcm": "SimianLuo/LCM_Dreamshaper_v7",
        "lcm_sd21": "latent-consistency/lcm-sd21",  # Primary, fallback is sd21
        "sdxl_lightning": "ByteDance/SDXL-Lightning-4step"
    }
    
    if model_key not in model_id_map:
        return False, f"Unknown model key: {model_key}. Available: {', '.join(model_id_map.keys())}", 0.0
    
    model_id = model_id_map[model_key]
    cache_dir = get_huggingface_cache_dir()
    
    # Convert model ID to cache path format (models--org--name)
    cache_path = model_id.replace("/", "--")
    full_path = os.path.join(cache_dir, f"models--{cache_path}")
    
    if not os.path.exists(full_path):
        # Also check for alternative paths (some models might be stored differently)
        # Try searching for the model name in the cache
        if not os.path.exists(cache_dir):
            return False, f"Hugging Face cache directory not found: {cache_dir}", 0.0
        
        # Search for any directory containing the model name
        found = False
        for item in os.listdir(cache_dir):
            if cache_path.lower() in item.lower() or model_id.split("/")[-1].lower() in item.lower():
                full_path = os.path.join(cache_dir, item)
                found = True
                break
        
        if not found:
            return False, f"Model cache not found: {model_id} (searched in {cache_dir})", 0.0
    
    # Calculate size before deletion
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(full_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except Exception as e:
        return False, f"Error calculating size: {str(e)}", 0.0
    
    size_mb = total_size / (1024 * 1024)
    
    # Remove the cache directory
    try:
        shutil.rmtree(full_path)
        freed_space_mb = size_mb
        
        # Also clear in-memory cache if model is loaded
        global _pipe_cache
        if model_key in _pipe_cache:
            _pipe_cache[model_key] = None
            del _pipe_cache[model_key]
        
        return True, f"Successfully removed {model_id} cache ({size_mb:.2f} MB freed)", freed_space_mb
    except Exception as e:
        return False, f"Error removing cache: {str(e)}", 0.0


def list_model_caches():
    """
    List all model caches and their sizes.
    
    Returns:
        dict: Dictionary mapping model keys to (exists: bool, size_mb: float, path: str)
    """
    cache_dir = get_huggingface_cache_dir()
    model_id_map = {
        "turbo": "stabilityai/sdxl-turbo",
        "sd14": "CompVis/stable-diffusion-v1-4",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "lcm": "SimianLuo/LCM_Dreamshaper_v7",
        "lcm_sd21": "latent-consistency/lcm-sd21",
        "sdxl_lightning": "ByteDance/SDXL-Lightning-4step"
    }
    
    results = {}
    
    if not os.path.exists(cache_dir):
        for key in model_id_map.keys():
            results[key] = (False, 0.0, None)
        return results
    
    for key, model_id in model_id_map.items():
        cache_path = model_id.replace("/", "--")
        full_path = os.path.join(cache_dir, f"models--{cache_path}")
        
        if not os.path.exists(full_path):
            # Try to find alternative path
            found_path = None
            for item in os.listdir(cache_dir):
                if cache_path.lower() in item.lower() or model_id.split("/")[-1].lower() in item.lower():
                    found_path = os.path.join(cache_dir, item)
                    break
            
            if found_path and os.path.exists(found_path):
                full_path = found_path
            else:
                results[key] = (False, 0.0, None)
                continue
        
        # Calculate size
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(full_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
        except Exception:
            total_size = 0
        
        size_mb = total_size / (1024 * 1024)
        results[key] = (True, size_mb, full_path)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        # Cleanup mode
        if len(sys.argv) > 2:
            model_key = sys.argv[2]
            success, message, freed_mb = remove_model_cache(model_key)
            print(message)
            if success:
                print(f"✓ Freed {freed_mb:.2f} MB of disk space")
            sys.exit(0 if success else 1)
        else:
            print("Usage: python model_options.py clean <model_key>")
            print("Available models: turbo, sd14, sd21, lcm, lcm_sd21, sdxl_lightning")
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "list":
        # List caches
        print("\n" + "="*80)
        print("MODEL CACHE STATUS")
        print("="*80)
        caches = list_model_caches()
        total_size = 0.0
        for key, (exists, size_mb, path) in caches.items():
            if exists:
                print(f"{key:15} | {size_mb:8.2f} MB | {path}")
                total_size += size_mb
            else:
                print(f"{key:15} | Not cached")
        print("="*80)
        print(f"Total cached: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
        print("="*80 + "\n")
    else:
        print_model_comparison()

