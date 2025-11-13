import librosa
import numpy as np
from diffusers import StableDiffusionPipeline
import yt_dlp
import torch
from PIL import Image, ImageDraw, ImageFont
import os

# Check device for M1 Mac
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# 1. Extract audio and metadata from YouTube
def download_audio_and_metadata(youtube_url):
    import glob
    
    # Try with ffmpeg first, fallback to direct download
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        },
    }
    
    # Check if ffmpeg is available
    import shutil
    has_ffmpeg = shutil.which('ffmpeg') is not None
    
    if has_ffmpeg:
        # Use ffmpeg to convert to wav
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
        expected_ext = 'wav'
    else:
        # Download in original format (librosa can handle most formats)
        expected_ext = None
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        title = info.get('title', 'Unknown')
        artist = info.get('uploader', 'Unknown Artist')
        duration = info.get('duration', 0)
    
    # Find the downloaded file
    if expected_ext:
        audio_path = f'temp_audio.{expected_ext}'
    else:
        # Find any temp_audio file
        temp_files = glob.glob('temp_audio.*')
        if temp_files:
            audio_path = temp_files[0]
        else:
            raise FileNotFoundError("Could not find downloaded audio file")
    
    return audio_path, title, artist, duration

# 2. Analyze audio with more features
def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, duration=60)  # First 60 sec for better analysis
    
    # Tempo and rhythm
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Energy and dynamics
    energy = np.mean(librosa.feature.rms(y=y))
    energy_std = np.std(librosa.feature.rms(y=y))
    
    # Harmonic and percussive
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-10)
    
    # Chroma (key/mode)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    dominant_key = np.argmax(chroma_mean)
    
    return {
        'tempo': float(tempo),
        'brightness': float(spectral_centroid),
        'energy': float(energy),
        'energy_variation': float(energy_std),
        'harmonic_ratio': float(harmonic_ratio),
        'zero_crossing_rate': float(zero_crossing_rate),
        'spectral_rolloff': float(spectral_rolloff),
        'dominant_key': int(dominant_key)
    }

# 3. Enhanced prompt generation based on audio features
def audio_to_prompt(features, title="", artist=""):
    tempo = features['tempo']
    energy = features['energy']
    brightness = features['brightness']
    harmonic_ratio = features['harmonic_ratio']
    energy_variation = features['energy_variation']
    
    # Genre/style inference
    style_parts = []
    
    if tempo > 140:
        style_parts.append("electronic")
        composition = "chaotic, dynamic, asymmetric, futuristic"
    elif tempo > 100:
        style_parts.append("energetic")
        composition = "energetic, balanced, rhythmic"
    else:
        style_parts.append("ambient")
        composition = "minimal, calm, centered, spacious"
    
    # Color palette based on energy and brightness
    if energy > 0.15:
        if brightness > 3000:
            colors = "vibrant neon colors, electric blues and purples, high contrast, glowing"
        else:
            colors = "deep rich colors, dark reds and oranges, dramatic lighting"
    elif energy > 0.08:
        colors = "balanced color palette, warm tones, moderate contrast"
    else:
        if harmonic_ratio > 0.5:
            colors = "muted pastels, soft blues and greens, gentle gradients"
        else:
            colors = "monochromatic, subtle grays, minimal color"
    
    # Mood based on harmonic content
    if harmonic_ratio > 0.6:
        mood = "melodic, harmonious, smooth"
    else:
        mood = "rhythmic, percussive, textured"
    
    # Energy variation adds dynamism
    if energy_variation > 0.05:
        dynamism = "dynamic movement, flowing shapes, organic forms"
    else:
        dynamism = "static composition, geometric patterns, structured"
    
    # Build comprehensive prompt
    prompt = (
        f"music poster design, {composition} composition, "
        f"{colors}, {mood} atmosphere, {dynamism}, "
        f"modern minimalist design, abstract art, high quality, professional, "
        f"album cover style, typography space, poster art"
    )
    
    return prompt

# 4. Generate image using Stable Diffusion (optimized for M1)
def generate_image(prompt, device="cpu", num_inference_steps=30):
    print(f"Loading Stable Diffusion model on {device}...")
    
    # Use a lighter, faster model for M1 Mac
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    if device == "mps":
        pipe = pipe.to("mps")
    elif device == "cuda":
        pipe = pipe.to("cuda")
    
    print(f"Generating poster image... (this may take 1-2 minutes on M1)")
    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            width=512,
            height=768  # Portrait orientation for poster
        ).images[0]
    
    return image

# 5. Add text overlay to create final poster
def add_text_to_poster(image, title, artist, output_path):
    # Resize to poster dimensions (can be adjusted)
    poster_width = 1080
    poster_height = 1350  # 4:5 aspect ratio, good for posters
    image = image.resize((poster_width, poster_height), Image.Resampling.LANCZOS)
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Try to load a nice font, fallback to default
    try:
        # Try system fonts on Mac
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 72)
        artist_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except:
        try:
            title_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 72)
            artist_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 48)
        except:
            title_font = ImageFont.load_default()
            artist_font = ImageFont.load_default()
    
    # Calculate text positions (centered, near bottom)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    artist_bbox = draw.textbbox((0, 0), artist, font=artist_font)
    
    title_width = title_bbox[2] - title_bbox[0]
    artist_width = artist_bbox[2] - artist_bbox[0]
    
    title_x = (poster_width - title_width) // 2
    title_y = poster_height - 200
    
    artist_x = (poster_width - artist_width) // 2
    artist_y = poster_height - 120
    
    # Draw text with outline for readability
    # Draw outline (black)
    for adj in range(-2, 3):
        for adj2 in range(-2, 3):
            if adj != 0 or adj2 != 0:
                draw.text((title_x + adj, title_y + adj2), title, font=title_font, fill=(0, 0, 0, 255))
                draw.text((artist_x + adj, artist_y + adj2), artist, font=artist_font, fill=(0, 0, 0, 255))
    
    # Draw main text (white)
    draw.text((title_x, title_y), title, font=title_font, fill=(255, 255, 255, 255))
    draw.text((artist_x, artist_y), artist, font=artist_font, fill=(255, 255, 255, 255))
    
    # Save the final poster
    image.save(output_path, quality=95)
    print(f"✅ Poster saved: {output_path}")

# Main function
def create_music_poster(youtube_url, output_path=None):
    """
    Main function to create a music poster from a YouTube URL.
    
    Args:
        youtube_url: YouTube URL of the music video
        output_path: Output file path (default: poster_[title].png)
    """
    temp_audio = None
    try:
        print("=" * 60)
        print("🎵 Music Poster Generator")
        print("=" * 60)
        
        # Step 1: Download audio and get metadata
        print("\n📥 Downloading audio from YouTube...")
        audio_path, title, artist, duration = download_audio_and_metadata(youtube_url)
        temp_audio = audio_path
        print(f"✅ Downloaded: {title} by {artist}")
        
        # Step 2: Analyze audio
        print("\n🎼 Analyzing audio features...")
        features = analyze_audio(audio_path)
        print(f"   Tempo: {features['tempo']:.1f} BPM")
        print(f"   Energy: {features['energy']:.3f}")
        print(f"   Brightness: {features['brightness']:.1f}")
        
        # Step 3: Generate prompt
        print("\n✨ Generating creative prompt...")
        prompt = audio_to_prompt(features, title, artist)
        print(f"   Prompt: {prompt[:100]}...")
        
        # Step 4: Generate image
        device = get_device()
        print(f"\n🎨 Generating poster image (using {device.upper()})...")
        image = generate_image(prompt, device=device)
        
        # Step 5: Add text and save
        if output_path is None:
            # Clean title for filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')[:50]
            output_path = f"poster_{safe_title}.png"
        
        print("\n📝 Adding text overlay...")
        add_text_to_poster(image, title, artist, output_path)
        
        print("\n" + "=" * 60)
        print(f"🎉 Success! Poster created: {output_path}")
        print("=" * 60)
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
            print(f"\n🧹 Cleaned up temporary files")

# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mus2pic.py <youtube_url> [output_path]")
        print("\nExample:")
        print("  python mus2pic.py 'https://www.youtube.com/watch?v=...'")
        print("  python mus2pic.py 'https://www.youtube.com/watch?v=...' my_poster.png")
        sys.exit(1)
    
    youtube_url = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    create_music_poster(youtube_url, output_path)