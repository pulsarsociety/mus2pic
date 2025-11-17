import librosa
import numpy as np
import yt_dlp
import re
import os
import time
import json
import random
import hashlib
import sys
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from model_options import generate_with_lcm

# Try to import mutagen for metadata extraction
try:
    from mutagen.easyid3 import EasyID3
    from mutagen import File
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    print("Warning: mutagen not available. Genre extraction from metadata will be disabled.")

# Try to import spotipy for Spotify API genre extraction
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
    print("Warning: spotipy not available. Spotify genre extraction will be disabled.")
    print("Install with: pip install spotipy")

# Cache file path
CACHE_FILE = "band_song_cache.json"
CACHE_DIR = "cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Helper function to extract video ID from URL
def extract_video_id(url):
    """Extract video ID from YouTube URL, removing playlist parameters."""
    # Extract video ID using regex
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if match:
        video_id = match.group(1)
        # Return clean URL with just the video ID
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

# Cache management functions
def load_cache():
    """Load the band-song cache from JSON file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load cache file: {e}")
            return {}
    return {}

def save_cache(cache):
    """Save the band-song cache to JSON file."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save cache file: {e}")

def get_cached_entry(cache, band, song):
    """Get cached entry for a band-song combination."""
    key = f"{band}|||{song}"  # Use ||| as separator to avoid conflicts
    return cache.get(key, None)

def save_cached_entry(cache, band, song, prompt, negative_prompt, seed, image_path, features=None):
    """Save an entry to the cache."""
    key = f"{band}|||{song}"
    cache[key] = {
        "band": band,
        "song": song,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "image_path": image_path
    }
    # Store features if provided
    if features:
        cache[key]["tempo"] = float(features.get('tempo', 0))
        cache[key]["brightness"] = float(features.get('brightness', 0))
        cache[key]["energy"] = float(features.get('energy', 0))
    save_cache(cache)

# Genre extraction from metadata
def get_genre_from_metadata(file_path):
    """
    Extract genre from audio file metadata.
    Returns genre string or None if not available.
    """
    if not MUTAGEN_AVAILABLE:
        return None
    
    try:
        # Try EasyID3 first (for MP3 files)
        try:
            audio = EasyID3(file_path)
            genre = audio.get('genre', [None])[0]
            if genre:
                return genre.lower()
        except:
            pass
        
        # Fallback: Try generic File for other formats (M4A, etc.)
        try:
            audio_file = File(file_path)
            if audio_file is not None:
                # Try common tag formats
                for tag_key in ['TIT1', 'TCON', 'genre', 'GENRE']:
                    if tag_key in audio_file:
                        genre = str(audio_file[tag_key][0]).lower()
                        if genre:
                            return genre
            return None
        except:
            return None
    except Exception as e:
        print(f"Warning: Could not extract genre from metadata: {e}")
        return None

def normalize_genre(raw_genres):
    """
    Normalize Spotify genres into a consistent visual category.
    Chooses the most specific or visually descriptive one.
    """
    GENRE_PRIORITY = {
        # --- Metal hierarchy ---
        "symphonic metal": 10,
        "progressive metal": 9,
        "gothic metal": 9,
        "power metal": 8,
        "folk metal": 8,
        "black metal": 7,
        "death metal": 7,
        "doom metal": 7,
        "metalcore": 7,
        "metal": 5,
        # --- Rock hierarchy ---
        "psychedelic rock": 9,
        "progressive rock": 9,
        "alternative rock": 8,
        "classic rock": 8,
        "hard rock": 8,
        "indie rock": 7,
        "grunge": 7,
        "punk rock": 7,
        "rock": 5,
        # --- Electronic / pop ---
        "synthpop": 9,
        "electropop": 8,
        "dance pop": 7,
        "pop": 6,
        "electronic": 6,
        "techno": 6,
        "edm": 6,
        # --- Folk / world ---
        "folk rock": 8,
        "folk": 7,
        "world": 6,
        # --- Abstract / experimental ---
        "post-rock": 8,
        "ambient": 7,
        "avant-garde": 7,
        "drone": 6,
        "slowcore": 6,
        "southern gothic": 6,
        "abstract": 5,
    }
    
    if not raw_genres:
        return "abstract"
    
    # Handle both list and string inputs
    if isinstance(raw_genres, str):
        raw_genres = [raw_genres]
    
    raw_genres = [g.lower() for g in raw_genres]
    
    # --- Score each genre ---
    scores = {}
    for g in raw_genres:
        for key, weight in GENRE_PRIORITY.items():
            if key in g:  # fuzzy match
                scores[key] = max(scores.get(key, 0), weight)
    
    if not scores:
        return "abstract"
    
    # --- Pick the highest-weighted genre ---
    best_genre = max(scores, key=scores.get)
    return best_genre

def get_genre_from_spotify(band_name, song_title=None, cache_file="genre_cache.json"):
    """
    Attempts to get genre from Spotify API using only band/artist name.
    Returns raw genres list for refinement.
    Falls back to None if not found.
    Caches locally to save API calls.
    """
    if not SPOTIPY_AVAILABLE:
        return None
    
    # --- Load cache if exists ---
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except:
            cache = {}
    else:
        cache = {}
    
    # Cache key is only band name (not song title)
    key = band_name.lower().strip()
    if key in cache:
        # Return cached raw genres (could be list or None)
        cached = cache[key]
        if cached != "abstract":
            # Print cached genres to stderr (not suppressed by redirect_stdout)
            if isinstance(cached, list) and cached:
                print(f"🎵 Spotify raw genres (cached) for '{band_name}': {cached}", file=sys.stderr)
            return cached
        return None
    
    # --- Spotify credentials (env vars with fallback to your credentials) ---
    # Get credentials from: https://developer.spotify.com/dashboard
    # Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file (optional)
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "5637c929e8794d9ea917d12963507696")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "a162aa98a7f4481c83e57c835e2057fa")
    
    if not client_id or not client_secret:
        cache[key] = "abstract"
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        return None
    
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret))
        # Search for artist directly (not track)
        results = sp.search(q=f"artist:{band_name}", type='artist', limit=1)
        if not results['artists']['items']:
            # Fallback: try track search and get artist from first result
            results = sp.search(q=band_name, type='track', limit=1)
            if not results['tracks']['items']:
                cache[key] = "abstract"
                with open(cache_file, "w") as f:
                    json.dump(cache, f)
                print(f"⚠️  No artist found on Spotify for '{band_name}'", file=sys.stderr)
                return None
            artist_id = results['tracks']['items'][0]['artists'][0]['id']
        else:
            artist_id = results['artists']['items'][0]['id']
        
        artist = sp.artist(artist_id)
        raw_genres = artist.get('genres', [])
        
        # Print raw genres from Spotify to stderr (not suppressed by redirect_stdout)
        if raw_genres:
            print(f"🎵 Spotify raw genres for '{band_name}': {raw_genres}", file=sys.stderr)
        else:
            print(f"⚠️  No genres found on Spotify for '{band_name}'", file=sys.stderr)
        
        # Cache raw genres (list)
        if raw_genres:
            cache[key] = raw_genres
        else:
            cache[key] = "abstract"
        
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        
        return raw_genres if raw_genres else None
    except Exception as e:
        print(f"Warning: Could not get genre from Spotify: {e}", file=sys.stderr)
        cache[key] = "abstract"
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        return None

def infer_genre_from_features(features):
    """
    Infer genre from audio features as fallback when metadata is not available.
    Returns genre string or None.
    """
    tempo = features.get('tempo', 120)
    energy = features.get('energy', 0.5)
    spectral_centroid = features.get('spectral_centroid', 3000)
    
    # Simple genre inference based on features
    if tempo > 140 and energy > 0.7:
        return "electronic"
    elif tempo > 120 and energy > 0.6:
        return "rock"
    elif tempo > 100 and energy > 0.5:
        return "metal"
    elif tempo < 90 and energy < 0.4:
        return "ambient"
    else:
        return None  # Return None to use default genre in v3

# 1. Extract audio from YouTube and get metadata
def download_audio(youtube_url):
    """
    Download audio from YouTube and extract metadata.
    Returns: (audio_path, band_name, song_title)
    """
    # Extract just the video ID to avoid playlist issues
    clean_url = extract_video_id(youtube_url)
    print(f"Downloading from: {clean_url}")
    
    # First try: Download as audio file that librosa can read directly (mp3, m4a)
    # This avoids FFmpeg conversion issues
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'noplaylist': True,  # Only download the video, not the playlist
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        },
        'quiet': False,
        'no_warnings': False,
    }
    
    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=True)
            # Get the actual filename that was downloaded
            filename = ydl.prepare_filename(info)
            # If it has an extension, use it; otherwise try common extensions
            if os.path.exists(filename):
                audio_path = filename
            else:
                # Try common extensions
                audio_path = None
                for ext in ['m4a', 'mp3', 'webm', 'opus', 'wav']:
                    test_path = f'temp_audio.{ext}'
                    if os.path.exists(test_path):
                        audio_path = test_path
                        break
                if audio_path is None:
                    audio_path = 'temp_audio.m4a'  # Default
            
            # Extract metadata
            title = info.get('title', 'Unknown')
            uploader = info.get('uploader', 'Unknown Artist')
            
            # First, try to get artist and track directly from metadata (most reliable)
            # yt-dlp often provides these fields directly
            metadata_artist = info.get('artist') or info.get('creator')
            metadata_track = info.get('track')
            
            # Try to parse band and song from title (don't rely on uploader/creator)
            # Common patterns: "Band - Song", "Song - Band", "Band: Song", etc.
            band = None
            song = None
            
            # Priority 1: Use metadata fields if available
            if metadata_artist and metadata_track:
                band = metadata_artist
                song = metadata_track
                print(f"Using metadata fields: Band='{band}', Song='{song}'", file=sys.stderr)
            elif metadata_artist:
                # Have artist but no track, try to get song from title
                band = metadata_artist
                song = title
                print(f"Using metadata artist with title: Band='{band}', Song='{song}'", file=sys.stderr)
            
            # Priority 2: Parse from title if metadata not available
            if band and song:
                pass  # Already have band and song from metadata
            # Pattern 1: "Band - Song" (most common)
            elif ' - ' in title:
                parts = title.split(' - ', 1)
                potential_band = parts[0].strip()
                potential_song = parts[1].strip()
                
                # Heuristics to determine which is band vs song:
                # - Band names are usually shorter and more consistent
                # - Song titles can be longer and more varied
                # - Check if first part contains common band indicators
                if len(potential_band) <= 50 and len(potential_song) <= 100:
                    # Both reasonable lengths, check for indicators
                    # If second part has "Official", "Lyrics", "Video", etc., first is likely band
                    if any(word in potential_song.lower() for word in ['official', 'lyrics', 'video', 'hq', 'hd', '4k']):
                        band = potential_band
                        song = potential_song
                    # If first part is very short (< 15 chars), likely band
                    elif len(potential_band) < 15:
                        band = potential_band
                        song = potential_song
                    # Default: first is band, second is song (most common pattern)
                    else:
                        band = potential_band
                        song = potential_song
                else:
                    # Default to first is band
                    band = potential_band
                    song = potential_song
            
            # Pattern 2: "Song by Band"
            elif not band and ' by ' in title.lower():
                parts = title.lower().split(' by ', 1)
                song = parts[0].strip()
                band = parts[1].strip()
            
            # Pattern 3: "Song (Band)" or "Song [Band]" - but skip if it's just quality indicators
            elif not band and ' (' in title and title.endswith(')'):
                # Check if the content in parentheses looks like a quality indicator, not a band name
                idx = title.rfind(' (')
                paren_content = title[idx+2:-1].strip().lower()
                # Skip if it's a quality indicator (HD, 4K, live, official, etc.)
                quality_indicators = ['hd', '4k', '8k', 'live', 'official', 'lyrics', 'video', 'audio', 'hq', 'remastered', 'remaster']
                if paren_content not in quality_indicators and len(paren_content) > 2:
                    song = title[:idx].strip()
                    band = title[idx+2:-1].strip()
            elif not band and ' [' in title and title.endswith(']'):
                idx = title.rfind(' [')
                song = title[:idx].strip()
                band = title[idx+2:-1].strip()
            
            # Pattern 4: "Band: Song"
            elif not band and ':' in title and title.count(':') == 1:
                parts = title.split(':', 1)
                band = parts[0].strip()
                song = parts[1].strip()
            
            # Pattern 5: "Band | Song"
            elif not band and '|' in title:
                parts = title.split('|', 1)
                band = parts[0].strip()
                song = parts[1].strip()
            
            # Pattern 6: Check description for "Band - Song" or "Song · Artist" pattern
            # Do this BEFORE fallback to ensure we catch description patterns
            if not band or not song:
                description = info.get('description', '')
                # Look for pattern like "Nightwish - High Hopes" (most common in description)
                desc_match = re.search(r'^([^-·\n]+)\s*[-–—]\s*([^\n(]+)', description, re.MULTILINE)
                if desc_match:
                    band = desc_match.group(1).strip()
                    song = desc_match.group(2).strip()
                    # Remove common suffixes from song
                    song = re.sub(r'\s*\(.*?\)$', '', song)  # Remove (live), (HD), etc.
                    song = song.strip()
                    print(f"Found 'Band - Song' pattern in description: Band='{band}', Song='{song}'", file=sys.stderr)
                else:
                    # Look for pattern like "High Hopes · Nightwish"
                    desc_match = re.search(r'^([^·\n]+)\s*[·•]\s*([^\n]+)', description, re.MULTILINE)
                    if desc_match:
                        song = desc_match.group(1).strip()
                        band = desc_match.group(2).strip()
                        print(f"Found 'Song · Artist' pattern in description: Band='{band}', Song='{song}'", file=sys.stderr)
            
            # Pattern 7: Handle Japanese brackets 「Band」Song or 「Band」Song lyrics
            if not band and '「' in title and '」' in title:
                # Extract text between Japanese brackets as band
                jp_match = re.search(r'「([^」]+)」', title)
                if jp_match:
                    band = jp_match.group(1).strip()
                    # Remove the bracket part and clean up
                    song = re.sub(r'「[^」]+」', '', title).strip()
                    # Remove common suffixes
                    song = re.sub(r'\s*\(.*?\)$', '', song)  # Remove (HD), (live), etc.
                    song = re.sub(r'\s*lyrics.*$', '', song, flags=re.IGNORECASE)
                    song = song.strip()
                    print(f"Found Japanese bracket pattern: Band='{band}', Song='{song}'", file=sys.stderr)
            
            # Fallback: if no pattern matched, use title as song and try to extract from uploader
            if not band or not song:
                song = title
                # Only use uploader if it doesn't look like a generic channel name
                uploader_lower = uploader.lower()
                if not any(suffix in uploader_lower for suffix in ['topic', 'vevo', 'official', 'channel', 'music', 'records', 'label']):
                    band = uploader
                else:
                    # Try to extract from title - use first word or first few words
                    words = title.split()
                    if len(words) > 1:
                        # Use first 1-3 words as potential band name
                        band = ' '.join(words[:min(3, len(words))])
                    else:
                        band = 'Unknown Artist'
            
            # Clean up band name - remove common suffixes and channel indicators
            band = re.sub(r'\s*-\s*Topic$', '', band, flags=re.IGNORECASE)
            band = re.sub(r'\s*VEVO$', '', band, flags=re.IGNORECASE)
            band = re.sub(r'\s*Official.*$', '', band, flags=re.IGNORECASE)
            band = re.sub(r'\s*\[.*?\]$', '', band)  # Remove [Official Video] etc
            band = re.sub(r'\s*\(.*?\)$', '', band)  # Remove (Official Audio) etc
            band = re.sub(r'^[「『]|[''」』]$', '', band)  # Remove Japanese brackets if still present
            band = band.strip()
            
            # Clean up song name - remove common video indicators
            song = re.sub(r'\s*\[.*?\]$', '', song)  # Remove [Official Video] etc
            song = re.sub(r'\s*\(.*?\)$', '', song)  # Remove (HD), (live), (Official Audio) etc
            song = re.sub(r'\s*-\s*Official.*$', '', song, flags=re.IGNORECASE)
            song = re.sub(r'\s*lyrics.*$', '', song, flags=re.IGNORECASE)  # Remove "lyrics", "lyrics HD", etc.
            song = re.sub(r'\s*\(HD\)$', '', song, flags=re.IGNORECASE)  # Remove (HD) suffix
            song = re.sub(r'^[「『]|[''」』]$', '', song)  # Remove Japanese brackets if still present
            song = song.strip()
            
            return audio_path, band, song
    except Exception as e:
        print(f"First attempt failed: {e}")
        print("Trying alternative download method...")
        
        # Second try: Without post-processing (no FFmpeg needed)
        ydl_opts_alt = {
            'format': 'worstaudio[ext=m4a]/worstaudio[ext=mp3]/worstaudio/worst',
            'outtmpl': 'temp_audio.%(ext)s',
            'noplaylist': True,
            'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts_alt) as ydl:
                info = ydl.extract_info(clean_url, download=True)
                filename = ydl.prepare_filename(info)
                if os.path.exists(filename):
                    audio_path = filename
                else:
                    audio_path = None
                    for ext in ['m4a', 'mp3', 'webm', 'opus']:
                        test_path = f'temp_audio.{ext}'
                        if os.path.exists(test_path):
                            audio_path = test_path
                            break
                    if audio_path is None:
                        audio_path = 'temp_audio.m4a'
                
                # Extract metadata (same logic as first attempt)
                title = info.get('title', 'Unknown')
                uploader = info.get('uploader', 'Unknown Artist')
                
                # First, try metadata fields
                metadata_artist = info.get('artist') or info.get('creator')
                metadata_track = info.get('track')
                
                band = None
                song = None
                
                if metadata_artist and metadata_track:
                    band = metadata_artist
                    song = metadata_track
                elif metadata_artist:
                    band = metadata_artist
                    song = title
                elif ' - ' in title:
                    parts = title.split(' - ', 1)
                    potential_band = parts[0].strip()
                    potential_song = parts[1].strip()
                    
                    if len(potential_band) <= 50 and len(potential_song) <= 100:
                        if any(word in potential_song.lower() for word in ['official', 'lyrics', 'video', 'hq', 'hd', '4k']):
                            band = potential_band
                            song = potential_song
                        elif len(potential_band) < 15:
                            band = potential_band
                            song = potential_song
                        else:
                            band = potential_band
                            song = potential_song
                    else:
                        band = potential_band
                        song = potential_song
                elif ' by ' in title.lower():
                    parts = title.lower().split(' by ', 1)
                    song = parts[0].strip()
                    band = parts[1].strip()
                elif ' (' in title and title.endswith(')'):
                    idx = title.rfind(' (')
                    song = title[:idx].strip()
                    band = title[idx+2:-1].strip()
                elif ' [' in title and title.endswith(']'):
                    idx = title.rfind(' [')
                    song = title[:idx].strip()
                    band = title[idx+2:-1].strip()
                elif ':' in title and title.count(':') == 1:
                    parts = title.split(':', 1)
                    band = parts[0].strip()
                    song = parts[1].strip()
                elif '|' in title:
                    parts = title.split('|', 1)
                    band = parts[0].strip()
                    song = parts[1].strip()
                
                # Fallback
                if not band or not song:
                    song = title
                    uploader_lower = uploader.lower()
                    if not any(suffix in uploader_lower for suffix in ['topic', 'vevo', 'official', 'channel', 'music', 'records', 'label']):
                        band = uploader
                    else:
                        words = title.split()
                        if len(words) > 1:
                            band = ' '.join(words[:min(3, len(words))])
                        else:
                            band = 'Unknown Artist'
                
                # Clean up band name
                band = re.sub(r'\s*-\s*Topic$', '', band, flags=re.IGNORECASE)
                band = re.sub(r'\s*VEVO$', '', band, flags=re.IGNORECASE)
                band = re.sub(r'\s*Official.*$', '', band, flags=re.IGNORECASE)
                band = re.sub(r'\s*\[.*?\]$', '', band)
                band = re.sub(r'\s*\(.*?\)$', '', band)
                band = band.strip()
                
                # Clean up song name
                song = re.sub(r'\s*\[.*?\]$', '', song)
                song = re.sub(r'\s*\(.*?\)$', '', song)
                song = re.sub(r'\s*-\s*Official.*$', '', song, flags=re.IGNORECASE)
                song = song.strip()
                
                return audio_path, band, song
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            raise

# 2. Analyze audio
def analyze_audio(audio_path, duration=None):
    """
    Enhanced audio analysis extracting multiple features for better prompt generation.

    Args:
        audio_path: Path to the audio file
        duration: Duration in seconds to analyze. None means full length.

    Returns:
        Dictionary with comprehensive audio features
    """
    # Load audio - if duration is None, load full length
    if duration is None:
        y, sr = librosa.load(audio_path)  # Full length
    else:
        y, sr = librosa.load(audio_path, duration=duration)  # Specific duration

    # Basic features
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    energy = np.mean(librosa.feature.rms(y=y))

    # Spectral rolloff - indicates brightness/darkness (frequency above which 85% of energy is contained)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))

    # Zero crossing rate - indicates percussiveness/noise content
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Chroma - harmonic content (captures tonal complexity)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_std = np.std(chroma)  # Harmonic complexity/variation

    # MFCC - timbre characteristics (texture of sound)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # Rhythm stability (beat consistency)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    rhythm_stability = 1.0 - np.clip(np.std(onset_env) / (np.mean(onset_env) + 1e-6), 0, 1)

    # Harmonic vs percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonicity = np.mean(librosa.feature.rms(y=y_harmonic)) / (energy + 1e-6)

    # Spectral bandwidth - indicates "purity" vs "noisiness"
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Spectral contrast - indicates dynamic range and texture
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

    return {
        'tempo': float(tempo),
        'brightness': float(spectral_centroid),
        'spectral_centroid': float(spectral_centroid),
        'energy': float(energy),
        'spectral_rolloff': float(spectral_rolloff),
        'spectral_bandwidth': float(spectral_bandwidth),
        'spectral_contrast': float(spectral_contrast),
        'zcr': float(zcr),
        'chroma_std': float(chroma_std),
        'rhythm_stability': float(np.clip(rhythm_stability, 0, 1)),
        'harmonicity': float(np.clip(harmonicity, 0, 1)),
        'mfcc_mean': mfcc_mean.tolist(),
        'mfcc_std': mfcc_std.tolist()
    }

def audio_to_prompt(features, band_name=None, song_title=None, enable_variation=False):
    tempo = features.get('tempo', 120)
    energy = features.get('energy', 0.5)
    spectral_centroid = features.get('spectral_centroid', 3000)
    mode = features.get('mode', 'major')
    valence = features.get('valence', 0.5)
    danceability = features.get('danceability', 0.5)
    instrumentalness = features.get('instrumentalness', 0.5)
    rhythm_stability = features.get('rhythm_stability', 0.5)

    # --- Composition ---
    if tempo > 140:
        composition = "chaotic energetic asymmetry"
    elif tempo > 110:
        composition = "dynamic rhythmic motion"
    elif tempo > 90:
        composition = "flowing balanced symmetry"
    else:
        composition = "minimal serene centered"

    # --- Colors ---
    if energy > 0.7:
        colors = "vivid neon high contrast"
    elif energy > 0.4:
        colors = "rich warm tones"
    else:
        colors = "pastel muted soft"

    # --- Mood ---
    if valence > 0.7:
        mood = "euphoric uplifting"
    elif valence > 0.4:
        mood = "nostalgic introspective"
    else:
        mood = "dark mysterious"

    # --- Texture ---
    if spectral_centroid > 5000:
        texture = "sharp crystalline detail"
    elif spectral_centroid > 2500:
        texture = "smooth modern surface"
    else:
        texture = "soft atmospheric haze"

    # --- Style ---
    style = "geometric" if mode == "major" else "organic"

    # --- Rhythm ---
    rhythm = "structured repetition" if rhythm_stability > 0.6 else "fluid motion"

    # --- Lighting ---
    if energy > 0.7 and valence < 0.4:
        lighting = "harsh contrast red-blue glow"
    elif valence > 0.7:
        lighting = "bright golden light"
    else:
        lighting = "dim ambient shadows"

    # --- Theme (genre-like mood fusion) ---
    if instrumentalness > 0.7 and energy < 0.5:
        theme = "ambient cinematic depth"
    elif energy > 0.8 and valence < 0.4:
        theme = "chaotic aggression and tension"
    elif valence > 0.7 and danceability > 0.6:
        theme = "playful rhythmic geometry"
    else:
        theme = "atmospheric emotional balance"

    # --- Band style memory (simple dictionary hook) ---
    band_styles = {
        "Brutus": "stormy lighting, shattered textures, crimson-blue tones",
        "Her Last Sight": "futuristic decay, neon shards, metallic reflections"
    }
    style_hint = band_styles.get(band_name, "")

    # --- Optional variation ---
    if enable_variation:
        color_mods = ["vivid", "intense", "dynamic", "saturated"]
        comp_mods = ["asymmetry", "rhythmic design", "visual pulse"]
        colors = f"{random.choice(color_mods)} colors"
        composition = f"{random.choice(comp_mods)} composition"

    # --- Context tags ---
    band_tag = f"inspired by the music of {band_name}" if band_name else ""
    song_tag = f"titled '{song_title}'" if song_title else ""

    # --- Final prompt (weighted hierarchy) ---
    prompt = (
        f"music-inspired abstract art poster, {theme}, {composition}, "
        f"{colors}, {mood}, {texture}, {style} shapes, {rhythm}, "
        f"{lighting}, {style_hint}, {band_tag}, {song_tag}, "
        f"modern design, cinematic lighting, detailed, high quality"
    )

    # --- Negative prompt ---
    negative_prompt = (
        "text, logo, watermark, words, lowres, jpeg artifacts, cropped, "
        "collage, grain, border, frame, blurry, distorted face, "
        "unfocused, overexposed, underexposed, duplicate, noisy, deformed"
    )

    return prompt.strip(), negative_prompt.strip()


def audio_to_prompt_v3(features, band_name=None, song_title=None, raw_genres=None):
    """
    HEAVILY IMPROVED: Feature-driven prompt generation that captures musical soul.
    Uses ACTUAL audio analysis instead of defaults or random selection.
    Every decision is now tied to measurable audio characteristics.
    """

    # -------------------------------
    # 1. Normalize genre
    # -------------------------------
    genre_hint = normalize_genre(raw_genres) if raw_genres else "abstract"
    genre_key = genre_hint.lower()

    # -------------------------------
    # 2. Extract REAL features (no more defaults)
    # -------------------------------
    tempo = features.get("tempo", 120)
    energy = features.get("energy", 0.5)
    brightness = features.get("spectral_centroid", 3000)
    rolloff = features.get("spectral_rolloff", 5000)
    bandwidth = features.get("spectral_bandwidth", 2000)
    contrast = features.get("spectral_contrast", 20)
    zcr = features.get("zcr", 0.1)
    chroma_std = features.get("chroma_std", 0.3)
    rhythm_stability = features.get("rhythm_stability", 0.5)
    harmonicity = features.get("harmonicity", 0.5)

    # Print extracted features for debugging
    print(f"\n🎵 Audio Features Extracted:", file=sys.stderr)
    print(f"   Tempo: {tempo:.1f} BPM", file=sys.stderr)
    print(f"   Energy: {energy:.3f}", file=sys.stderr)
    print(f"   Brightness: {brightness:.1f} Hz", file=sys.stderr)
    print(f"   Rolloff: {rolloff:.1f} Hz", file=sys.stderr)
    print(f"   Harmonicity: {harmonicity:.3f}", file=sys.stderr)
    print(f"   Chroma Std: {chroma_std:.3f}", file=sys.stderr)
    print(f"   Rhythm Stability: {rhythm_stability:.3f}", file=sys.stderr)
    print(f"   ZCR: {zcr:.4f}", file=sys.stderr)

    # -------------------------------
    # 3. MOOD - Driven by harmonic complexity + energy + harmonicity
    # -------------------------------
    if chroma_std < 0.2 and energy > 0.6:
        mood = "aggressive focused intensity"  # Simple harmonics + high energy = raw aggression
    elif chroma_std > 0.4 and harmonicity > 0.6:
        mood = "complex emotional depth"  # Rich harmonics = layered emotion
    elif harmonicity < 0.3 and zcr > 0.15:
        mood = "chaotic raw energy"  # Percussive/noisy = chaos
    elif chroma_std > 0.5 and energy < 0.4:
        mood = "melancholic complexity"  # Complex but quiet = sad sophistication
    elif energy < 0.25:
        mood = "meditative stillness"  # Very low energy = ambient/calm
    elif harmonicity > 0.7 and energy > 0.5:
        mood = "euphoric uplifting power"  # Harmonic + energetic = uplifting
    else:
        mood = "balanced introspective atmosphere"

    # -------------------------------
    # 4. LIGHTING/TONE - Driven by brightness + rolloff + contrast
    # -------------------------------
    if rolloff > 7000 and brightness > 4000:
        tone = "piercing bright highlights, sharp high-contrast lighting"  # Very bright = sharp
    elif rolloff < 3000 and brightness < 2000:
        tone = "deep shadows, heavy darkness, muted tones"  # Dark = shadowy
    elif contrast > 25:
        tone = "dramatic chiaroscuro, bold light-dark interplay"  # High contrast = dramatic
    elif brightness > 3500:
        tone = "luminous ethereal glow, soft radiance"  # Bright but not harsh
    else:
        tone = "balanced cinematic lighting, atmospheric depth"

    # -------------------------------
    # 5. MOTION - Driven by tempo + rhythm stability
    # -------------------------------
    if tempo > 160 and rhythm_stability < 0.5:
        motion = "violent chaotic explosions, erratic movement"  # Fast + unstable = chaos
    elif tempo > 160 and rhythm_stability > 0.7:
        motion = "relentless machine-like precision, driving force"  # Fast + stable = mechanical
    elif tempo > 130 and rhythm_stability > 0.6:
        motion = "dynamic rhythmic pulse, energetic flow"  # Fast + somewhat stable = energetic
    elif tempo < 80:
        motion = "glacial weight, slow deliberate movement"  # Very slow = heavy
    elif tempo < 100:
        motion = "contemplative drift, gentle sway"  # Slow = gentle
    elif rhythm_stability < 0.4:
        motion = "unpredictable organic shifts, flowing chaos"  # Unstable = organic
    else:
        motion = "steady measured progression, balanced rhythm"

    # -------------------------------
    # 6. TEXTURE - Driven by harmonicity + bandwidth
    # -------------------------------
    if harmonicity > 0.7 and bandwidth < 1500:
        texture = "pristine crystalline clarity, smooth polished surfaces"  # Pure + narrow = clean
    elif harmonicity > 0.6:
        texture = "layered ethereal atmosphere, soft translucent veils"  # Harmonic = smooth
    elif harmonicity < 0.3 and zcr > 0.15:
        texture = "jagged fractured surfaces, raw abrasive grain"  # Percussive = rough
    elif bandwidth > 3000:
        texture = "dense noisy complexity, rich gritty detail"  # Wide bandwidth = dense
    else:
        texture = "dynamic balanced depth, varied tactile quality"

    # -------------------------------
    # 7. COMPOSITION - Driven by energy + rhythm stability
    # -------------------------------
    if energy < 0.25:
        composition = "minimal sparse arrangement, vast negative space, centered focus"
    elif energy < 0.5 and rhythm_stability > 0.6:
        composition = "symmetrical layered depth, architectural balance, geometric order"
    elif energy > 0.7 and rhythm_stability < 0.5:
        composition = "explosive asymmetric chaos, fractured angles, dynamic instability"
    elif energy > 0.6:
        composition = "bold dramatic composition, aggressive positioning, powerful presence"
    else:
        composition = "organic flowing arrangement, natural balance, gentle curves"

    # -------------------------------
    # 8. GENRE-SPECIFIC FLAVOR - FEATURE-DRIVEN selection
    # -------------------------------
    # Each genre gets 4 flavors: high_energy, low_energy, complex, simple
    genre_flavor_map = {
        "symphonic metal": {
            "high_energy": "apocalyptic orchestral storm, divine fury",
            "low_energy": "gothic cathedral grandeur, sacred solemnity",
            "complex": "symphonic architecture, layered epic construction",
            "simple": "ethereal celestial power, pure transcendence"
        },
        "progressive metal": {
            "high_energy": "fractal chaos, mathematical violence",
            "low_energy": "cosmic contemplation, dimensional drift",
            "complex": "geometric impossibility, nested structures",
            "simple": "astral minimalism, spatial clarity"
        },
        "gothic metal": {
            "high_energy": "romantic devastation, beautiful violence",
            "low_energy": "velvet decay, elegant sorrow",
            "complex": "baroque darkness, ornate shadows",
            "simple": "somber ritual, pure melancholy"
        },
        "folk metal": {
            "high_energy": "tribal warfare, primal force",
            "low_energy": "ancient forest whispers, mystic earth",
            "complex": "mythological tapestry, layered legend",
            "simple": "runic simplicity, elemental power"
        },
        "power metal": {
            "high_energy": "heroic blaze, triumphant ascension",
            "low_energy": "mythic horizon, distant glory",
            "complex": "epic narrative, legendary saga",
            "simple": "pure valor, radiant courage"
        },
        "metalcore": {
            "high_energy": "industrial collapse, urban warfare",
            "low_energy": "post-apocalyptic silence, ruins",
            "complex": "existential fragmentation, layered despair",
            "simple": "brutal honesty, raw confrontation"
        },
        "death metal": {
            "high_energy": "visceral brutality, savage chaos",
            "low_energy": "doomed inevitability, crushing weight",
            "complex": "technical precision, surgical violence",
            "simple": "primal aggression, pure brutality"
        },
        "black metal": {
            "high_energy": "frozen fury, arctic violence",
            "low_energy": "desolate wasteland, empty void",
            "complex": "atmospheric misanthropy, layered hatred",
            "simple": "raw primordial darkness, pure evil"
        },
        "doom metal": {
            "high_energy": "earth-crushing weight, tectonic power",
            "low_energy": "funeral procession, eternal mourning",
            "complex": "psychedelic despair, hallucinogenic sorrow",
            "simple": "monolithic doom, pure heaviness"
        },
        "post-rock": {
            "high_energy": "cathartic crescendo, explosive release",
            "low_energy": "infinite horizon, peaceful dissolution",
            "complex": "narrative journey, evolving landscape",
            "simple": "minimal beauty, sparse elegance"
        },
        "ambient": {
            "high_energy": "cosmic event, nebula birth",
            "low_energy": "absolute stillness, void meditation",
            "complex": "layered dimensions, sonic architecture",
            "simple": "pure tone, essential frequency"
        },
        "psychedelic rock": {
            "high_energy": "kaleidoscopic explosion, fractal madness",
            "low_energy": "dreamy dissolution, gentle hallucination",
            "complex": "recursive patterns, infinite depth",
            "simple": "color-washed simplicity, pure distortion"
        },
        "progressive rock": {
            "high_energy": "complex virtuosity, technical mastery",
            "low_energy": "conceptual meditation, thoughtful progression",
            "complex": "narrative architecture, structured epic",
            "simple": "retro-futurist clarity, clean vision"
        },
        "alternative rock": {
            "high_energy": "urban intensity, gritty rebellion",
            "low_energy": "suburban melancholy, quiet desperation",
            "complex": "layered angst, textured emotion",
            "simple": "raw honesty, stripped-down truth"
        },
        "classic rock": {
            "high_energy": "stadium power, timeless energy",
            "low_energy": "vintage warmth, nostalgic glow",
            "complex": "blues-rock tapestry, soulful layers",
            "simple": "pure rock essence, analog clarity"
        },
        "grunge": {
            "high_energy": "explosive frustration, distorted rage",
            "low_energy": "apathetic haze, numb detachment",
            "complex": "layered alienation, textured despair",
            "simple": "raw unpolished truth, stripped authenticity"
        },
        "indie rock": {
            "high_energy": "joyful chaos, playful rebellion",
            "low_energy": "bedroom intimacy, quiet reflection",
            "complex": "clever arrangement, artistic layering",
            "simple": "DIY purity, honest simplicity"
        },
        "punk rock": {
            "high_energy": "anarchic fury, three-chord rage",
            "low_energy": "punk meditation, quiet resistance",
            "complex": "political tapestry, message layers",
            "simple": "raw defiance, pure rebellion"
        },
        "hard rock": {
            "high_energy": "amplified aggression, driving power",
            "low_energy": "heavy contemplation, weighted thought",
            "complex": "layered riffs, textured heaviness",
            "simple": "pure rock force, essential drive"
        },
        "pop": {
            "high_energy": "neon euphoria, vibrant celebration",
            "low_energy": "soft intimacy, gentle sweetness",
            "complex": "polished production, crafted perfection",
            "simple": "pure catchiness, minimal hooks"
        },
        "synthpop": {
            "high_energy": "digital ecstasy, electric joy",
            "low_energy": "retro nostalgia, warm synth glow",
            "complex": "layered synthesis, electronic depth",
            "simple": "clean digital lines, pure pop"
        },
        "electronic": {
            "high_energy": "cybernetic pulse, digital assault",
            "low_energy": "ambient circuitry, soft electrons",
            "complex": "algorithmic complexity, coded layers",
            "simple": "pure waveform, essential signal"
        },
        "edm": {
            "high_energy": "festival explosion, peak euphoria",
            "low_energy": "comedown drift, gentle pulse",
            "complex": "drop architecture, build-up tension",
            "simple": "four-on-floor purity, simple kick"
        },
        "techno": {
            "high_energy": "industrial drive, relentless machine",
            "low_energy": "minimal groove, hypnotic loop",
            "complex": "modular maze, technical precision",
            "simple": "raw kick, pure rhythm"
        },
        "folk": {
            "high_energy": "dancing celebration, joyful tradition",
            "low_energy": "fireside storytelling, quiet wisdom",
            "complex": "cultural tapestry, heritage layers",
            "simple": "acoustic purity, honest song"
        },
        "folk rock": {
            "high_energy": "electrified tradition, amplified roots",
            "low_energy": "contemplative countryside, pastoral calm",
            "complex": "narrative folk, story weaving",
            "simple": "simple truth, folk honesty"
        },
        "abstract": {
            "high_energy": "formless energy, undefined power",
            "low_energy": "shapeless calm, ambient void",
            "complex": "conceptual depth, abstract layers",
            "simple": "pure abstraction, essential form"
        },
    }

    # Get flavor dictionary for this genre
    flavor_dict = genre_flavor_map.get(genre_key, genre_flavor_map["abstract"])

    # SMART SELECTION based on actual audio features
    if energy > 0.65:
        flavor = flavor_dict["high_energy"]
    elif chroma_std > 0.45 or harmonicity > 0.65:
        flavor = flavor_dict["complex"]
    elif energy < 0.35:
        flavor = flavor_dict["low_energy"]
    else:
        flavor = flavor_dict["simple"]

    print(f"   Selected flavor: '{flavor}' (genre: {genre_key})", file=sys.stderr)

    # -------------------------------
    # 9. COLOR PALETTE - Driven by brightness + energy + genre
    # -------------------------------
    if energy > 0.7 and brightness > 4000:
        colors = "saturated neon brilliance, electric intensity"
    elif energy > 0.6 and rolloff > 6000:
        colors = "vivid high-energy chromas, bold hues"
    elif brightness < 2000 and energy < 0.4:
        colors = "desaturated darkness, muted shadows, deep blacks"
    elif harmonicity > 0.65:
        colors = "harmonious balanced palette, rich warm tones"
    elif zcr > 0.15:
        colors = "harsh contrasting colors, jarring combinations"
    else:
        colors = "atmospheric gradient, nuanced transitions"

    # -------------------------------
    # 10. FINAL SDXL-TURBO OPTIMIZED PROMPT
    # -------------------------------
    # Structure: Identity → Genre → Mood → Visual Elements → Technical Quality
    prompt = (
        f"music poster art representing {band_name or 'the artist'}, "
        f"track: {song_title or 'this song'}. "
        f"{genre_hint} aesthetic. "
        f"{flavor}. "
        f"{mood}. "
        f"{tone}. "
        f"{motion}. "
        f"{texture}. "
        f"{colors}. "
        f"{composition}. "
        f"abstract symbolic forms, no text, no faces, no letters, "
        f"professional poster design, volumetric atmosphere, dramatic depth, "
        f"high detail, cinematic quality, artistic coherence"
    )

    # -------------------------------
    # 11. NEGATIVE PROMPT (Turbo-optimized)
    # -------------------------------
    negative_prompt = (
        "text, letters, words, watermark, logo, signature, typography, "
        "human faces, people, portraits, photorealism, photography, "
        "messy collage, random clutter, low quality, blurry, artifacts, "
        "oversharpened, jpeg compression, noisy, deformed, distorted"
    )

    print(f"\n✨ Prompt generated with feature-driven logic\n", file=sys.stderr)

    return prompt.strip(), negative_prompt.strip()


# Hardcoded YouTube URL for testing
YOUTUBE_URL = 'https://www.youtube.com/watch?v=Sk-geMg8Tyc&list=RDSk-geMg8Tyc'

def format_time(seconds):
    """Format seconds into a readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"

def main(regenerate=False, fixed_seed=None, duration=None):
    """
    Main script to extract audio, analyze it, and generate a prompt for image generation.
    
    Args:
        regenerate: If True, bypass cache and generate new image even if cached entry exists.
        fixed_seed: Seed to use for generation. Options:
            - None: Use cached seed if available, otherwise generate random
            - int: Use this specific seed (for fixed-seed testing)
            - "cached": Always use cached seed if available (for prompt comparison testing)
        duration: Duration in seconds to analyze. None means full length.
    """
    start_time = time.time()
    print("Starting audio extraction and analysis...")
    print(f"Using YouTube URL: {YOUTUBE_URL}\n")
    
    # Load cache
    cache = load_cache()
    
    step_times = {}
    band = None
    song = None
    prompt = None
    negative_prompt = None
    seed_used = None
    image_path = None
    
    try:
        # Step 1: Download audio from YouTube and get metadata
        step_start = time.time()
        print("Step 1: Downloading audio from YouTube...")
        audio_path, band, song = download_audio(YOUTUBE_URL)
        step_time = time.time() - step_start
        step_times['Download'] = step_time
        print(f"Audio downloaded to: {audio_path}")
        print(f"Band: {band}, Song: {song}")
        print(f"⏱️  Step 1 time: {format_time(step_time)}\n")
        
        # Get cached entry (always check, even if regenerating, to get seed)
        cached_entry = get_cached_entry(cache, band, song)
        
        # Determine seed to use (for stable baseline comparison)
        if isinstance(fixed_seed, int):
            # Use explicitly provided seed
            seed_used = fixed_seed
            print(f"🔒 Using fixed seed: {seed_used}\n")
        elif fixed_seed == "cached" and cached_entry:
            # Force use cached seed for prompt comparison testing
            seed_used = cached_entry['seed']
            print(f"🔒 Using cached seed for fixed-seed testing: {seed_used}\n")
        elif cached_entry:
            # Default: Use cached seed if available (enables prompt comparison testing)
            seed_used = cached_entry['seed']
            if regenerate:
                print(f"🔒 Using cached seed {seed_used} for stable baseline (compare prompt changes)\n")
        # Otherwise seed_used stays None and will be generated randomly
        
        # Check cache before generating (unless regenerating)
        if not regenerate and cached_entry:
            print(f"\n{'='*60}")
            print("✓ CACHED ENTRY FOUND - Reusing previous generation")
            print(f"{'='*60}")
            print(f"Band: {cached_entry['band']}")
            print(f"Song: {cached_entry['song']}")
            print(f"Seed: {cached_entry['seed']}")
            print(f"Image: {cached_entry['image_path']}")
            print(f"{'='*60}\n")
            
            prompt = cached_entry['prompt']
            negative_prompt = cached_entry['negative_prompt']
            seed_used = cached_entry['seed']
            image_path = cached_entry['image_path']
            
            # Verify image file exists
            if os.path.exists(image_path):
                print(f"✓ Using cached image: {image_path}")
                print(f"💡 Tip: Use --regenerate to generate a new variation")
                print(f"💡 Tip: Use --seed <number> or --seed cached for fixed-seed testing\n")
                return prompt, negative_prompt, image_path
            else:
                print(f"⚠️  Cached image not found at {image_path}, generating new one...\n")
                cached_entry = None  # Will generate new one
        
        # Step 2: Analyze audio features (only if not using cache)
        if cached_entry is None:
            step_start = time.time()
            duration_text = f"{duration}s" if duration else "full length"
            print(f"Step 2: Analyzing audio features ({duration_text})...")
            features = analyze_audio(audio_path, duration=duration)
            step_time = time.time() - step_start
            step_times['Analysis'] = step_time
            print(f"Audio features extracted:")
            print(f"  - Tempo: {features['tempo']:.2f} BPM")
            print(f"  - Brightness: {features['brightness']:.2f}")
            print(f"  - Energy: {features['energy']:.4f}")
            print(f"⏱️  Step 2 time: {format_time(step_time)}\n")
            
            # Step 3: Generate prompt for image generation
            step_start = time.time()
            print("Step 3: Generating prompt for image generation...")
            prompt, negative_prompt = audio_to_prompt(features, band_name=band, song_title=song)
            step_time = time.time() - step_start
            step_times['Prompt Generation'] = step_time
            print(f"\n{'='*60}")
            print("GENERATED PROMPT FOR IMAGE GENERATION:")
            print(f"{'='*60}")
            print(prompt)
            print(f"\nNEGATIVE PROMPT:")
            print(negative_prompt)
            print(f"{'='*60}")
            print(f"⏱️  Step 3 time: {format_time(step_time)}\n")
            
            # Step 4: Generate image
            step_start = time.time()
            print("Step 4: Generating image from prompt...")
            # Generate seed if not already set (for reproducibility)
            if seed_used is None:
                seed_used = random.randint(0, 2**32 - 1)
                print(f"🎲 Generated new random seed: {seed_used}")
            else:
                print(f"🔒 Using fixed seed: {seed_used} (for stable baseline comparison)")
            
            # Create cache directory path for image
            safe_band = "".join(c for c in band if c.isalnum() or c in (' ', '-', '_')).strip()[:30]
            safe_song = "".join(c for c in song if c.isalnum() or c in (' ', '-', '_')).strip()[:30]
            safe_band = safe_band.replace(' ', '_')
            safe_song = safe_song.replace(' ', '_')
            image_filename = f"{safe_band}-{safe_song}.png"
            image_path = os.path.join(CACHE_DIR, image_filename)
            
            # LCM works best with 4-8 steps (much faster than 30!)
            image_path, seed_used, gen_time = generate_with_lcm(
                prompt, 
                negative_prompt, 
                output_path=image_path,
                num_inference_steps=4,
                seed=seed_used
            )
            step_time = time.time() - step_start
            step_times['Image Generation'] = step_time
            print(f"⏱️  Step 4 time: {format_time(step_time)}\n")
            
            # Save to cache (always save seed for future fixed-seed testing)
            save_cached_entry(cache, band, song, prompt, negative_prompt, seed_used, image_path)
            print(f"✓ Saved to cache: {band} - {song} (seed: {seed_used})\n")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print timing summary
        if step_times:
            print(f"\n{'='*60}")
            print("⏱️  TIMING SUMMARY")
            print(f"{'='*60}")
            for step_name, step_time in step_times.items():
                percentage = (step_time / total_time) * 100
                print(f"  {step_name:<20}: {format_time(step_time):<12} ({percentage:5.1f}%)")
            print(f"{'='*60}")
            print(f"  {'TOTAL TIME':<20}: {format_time(total_time):<12} (100.0%)")
            print(f"{'='*60}\n")
        
        print(f"Generated image saved to: {image_path}\n")
        
        return prompt, negative_prompt, image_path
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n✗ Error occurred after {format_time(total_time)}: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    
    # Parse command-line arguments
    regenerate = "--regenerate" in sys.argv or "-r" in sys.argv
    fixed_seed = None
    
    # Check for --seed argument
    if "--seed" in sys.argv:
        seed_idx = sys.argv.index("--seed")
        if seed_idx + 1 < len(sys.argv):
            seed_arg = sys.argv[seed_idx + 1]
            if seed_arg.lower() == "cached":
                fixed_seed = "cached"
            else:
                try:
                    fixed_seed = int(seed_arg)
                except ValueError:
                    print(f"⚠️  Invalid seed value: {seed_arg}. Using default behavior.")
                    fixed_seed = None
    
    # Print mode information
    if regenerate:
        print("🔄 Regenerate mode: Bypassing cache to create new variation")
        if fixed_seed == "cached":
            print("🔒 Fixed-seed mode: Using cached seed for prompt comparison testing")
        elif isinstance(fixed_seed, int):
            print(f"🔒 Fixed-seed mode: Using seed {fixed_seed} for stable baseline")
        print()
    
    prompt, negative_prompt, image_path = main(regenerate=regenerate, fixed_seed=fixed_seed)