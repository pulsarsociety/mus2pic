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
        "thrash metal": 7,
        "nu metal": 6,
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
        "post-rock": 8,
        "rock": 5,
        
        # --- Jazz hierarchy ---
        "free jazz": 9,
        "fusion": 9,
        "bebop": 8,
        "cool jazz": 8,
        "smooth jazz": 7,
        "jazz funk": 7,
        "contemporary jazz": 7,
        "jazz": 6,
        
        # --- Blues hierarchy ---
        "delta blues": 8,
        "electric blues": 7,
        "blues rock": 7,
        "blues": 6,
        
        # --- Soul / R&B / Funk ---
        "neo soul": 8,
        "funk": 7,
        "soul": 7,
        "r&b": 6,
        "rhythm and blues": 6,
        "motown": 7,
        "disco": 6,
        
        # --- Hip-Hop / Rap ---
        "conscious hip hop": 8,
        "trap": 7,
        "boom bap": 7,
        "gangsta rap": 7,
        "hip hop": 6,
        "rap": 6,
        
        # --- Electronic / EDM ---
        "synthpop": 9,
        "electropop": 8,
        "techno": 7,
        "house": 7,
        "trance": 7,
        "drum and bass": 7,
        "dubstep": 7,
        "edm": 6,
        "electronic": 6,
        
        # --- Pop ---
        "dance pop": 7,
        "indie pop": 7,
        "pop": 6,
        
        # --- Country / Folk ---
        "folk rock": 8,
        "americana": 7,
        "bluegrass": 7,
        "country": 6,
        "folk": 7,
        
        # --- World / Regional ---
        "afrobeat": 8,
        "reggae": 7,
        "latin": 6,
        "world": 6,
        "bossa nova": 7,
        "flamenco": 7,
        
        # --- Classical / Orchestral ---
        "contemporary classical": 8,
        "orchestral": 7,
        "classical": 6,
        "opera": 7,
        "baroque": 7,
        
        # --- Experimental / Ambient ---
        "avant-garde": 9,
        "experimental": 8,
        "ambient": 7,
        "drone": 6,
        "noise": 6,
        "industrial": 7,
        
        # --- Other ---
        "slowcore": 6,
        "shoegaze": 7,
        "emo": 6,
        "ska": 6,
        "gospel": 6,
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

def get_genre_from_spotify(band_name, song_title=None, cache_file="genre_cache.json", skip_swap_detection=False):
    """
    ROBUST multi-fallback Spotify genre lookup.
    Tries multiple search strategies to find the artist.
    Returns raw genres list for refinement.
    Caches locally to save API calls.
    
    Args:
        band_name: Artist/band name to search for
        song_title: Optional song title
        cache_file: Cache file path
        skip_swap_detection: If True, skip trying song as artist (use when values are user-verified)
    
    Fallback strategies:
    1. Direct artist search: "artist:{band_name}"
    2. Track search with both: "track:{song_title} artist:{band_name}"
    3. Reverse search (in case swap): "track:{band_name} artist:{song_title}" (skipped if skip_swap_detection=True)
    4. Generic track search: "{band_name}"
    5. Combined search: "{band_name} {song_title}"
    6. Cleaned band name (remove common suffixes)
    """
    if not SPOTIPY_AVAILABLE:
        return None
    
    # --- Load cache ---
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except:
            cache = {}
    else:
        cache = {}
    
    # Cache key combines both band and song for better accuracy
    cache_key = f"{band_name.lower().strip()}|||{song_title.lower().strip() if song_title else ''}"
    
    if cache_key in cache:
        cached = cache[cache_key]
        if cached and cached != "abstract":
            if isinstance(cached, list) and cached:
                print(f"🎵 Spotify genres (cached): {cached}", file=sys.stderr)
            return cached if cached != "abstract" else None
        return None
    
    # --- Spotify credentials ---
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "5637c929e8794d9ea917d12963507696")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "a162aa98a7f4481c83e57c835e2057fa")
    
    if not client_id or not client_secret:
        cache[cache_key] = "abstract"
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        return None
    
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret))
        
        # Clean band name - remove common junk
        def clean_artist_name(name):
            """Remove common YouTube channel suffixes and junk."""
            if not name:
                return name
            cleaned = name
            # Remove common suffixes
            suffixes = [
                ' - Topic', ' Topic', 'VEVO', 'Official', 
                'Channel', 'Music', 'Records', 'Label',
                'HD', 'HQ', '4K', 'Lyrics'
            ]
            for suffix in suffixes:
                cleaned = re.sub(rf'\s*-?\s*{suffix}\s*$', '', cleaned, flags=re.IGNORECASE)
            # Remove brackets/parens at end
            cleaned = re.sub(r'\s*[\[\(].*?[\]\)]\s*$', '', cleaned)
            return cleaned.strip()
        
        cleaned_band = clean_artist_name(band_name)
        cleaned_song = clean_artist_name(song_title) if song_title else None
        
        artist_id = None
        found_artist_name = None
        search_method = None
        
        # Helper function to search and score artist matches
        def search_artist_with_scoring(search_term):
            """Search for artist and return best match with score."""
            try:
                results = sp.search(q=f"artist:{search_term}", type='artist', limit=10)
                if not results['artists']['items']:
                    return None, 0, None
                
                best_match = None
                best_score = 0
                search_lower = search_term.lower().strip()
                
                for artist in results['artists']['items']:
                    artist_name = artist['name']
                    artist_lower = artist_name.lower().strip()
                    
                    # Calculate match score
                    score = 0
                    
                    # Exact match (case-insensitive) = highest score
                    if artist_lower == search_lower:
                        score = 1000
                    # Single word exact match
                    elif len(search_lower.split()) == 1 and len(artist_lower.split()) == 1:
                        if search_lower == artist_lower:
                            score = 500
                        else:
                            score = 0
                    # Search term is single word, check if it matches artist's first word (preferred)
                    elif len(search_lower.split()) == 1:
                        artist_first = artist_lower.split()[0]
                        artist_last = artist_lower.split()[-1]
                        if search_lower == artist_first:
                            score = 400
                        elif search_lower == artist_last:
                            score = 20  # Last word match is suspicious
                        else:
                            score = 0
                    # All words match
                    elif set(search_lower.split()) == set(artist_lower.split()):
                        score = 800
                    # Search term is contained in artist name
                    elif search_lower in artist_lower:
                        if artist_lower.startswith(search_lower):
                            score = 200
                        else:
                            score = 50
                    # Artist name is contained in search term
                    elif artist_lower in search_lower:
                        score = 300
                    # Word overlap
                    else:
                        search_words = set(search_lower.split())
                        artist_words = set(artist_lower.split())
                        overlap = len(search_words & artist_words)
                        if overlap > 0:
                            score = overlap * 10
                    
                    # Penalize if artist name is much longer
                    if len(artist_lower) > len(search_lower) * 2:
                        score = max(0, score - 50)
                    
                    if score > best_score:
                        best_score = score
                        best_match = artist
                
                # Minimum score threshold
                min_score = 100 if len(search_lower.split()) == 1 else 50
                
                if best_match and best_score >= min_score:
                    return best_match, best_score, best_match['name']
                else:
                    return None, best_score, None
            except Exception as e:
                print(f"  Search failed for '{search_term}': {e}", file=sys.stderr)
                return None, 0, None
        
        # === STRATEGY 1: Try extracted "band" as artist ===
        print(f"🔍 Strategy 1: Trying '{cleaned_band}' as artist", file=sys.stderr)
        match1, score1, name1 = search_artist_with_scoring(cleaned_band)
        if match1:
            artist_id = match1['id']
            found_artist_name = name1
            search_method = f"band as artist (score: {score1})"
            print(f"✓ Found: {found_artist_name} (score: {score1})", file=sys.stderr)
        
        # === STRATEGY 2: Try extracted "song" as artist (in case of swap) ===
        # Skip this if user has already verified the values (skip_swap_detection=True)
        if cleaned_song and not skip_swap_detection:
            print(f"🔍 Strategy 2: Trying '{cleaned_song}' as artist (possible swap)", file=sys.stderr)
            match2, score2, name2 = search_artist_with_scoring(cleaned_song)
            if match2:
                # If song as artist has better score, use it (swap detected)
                # When scores are equal, prefer band match (more reliable)
                if not artist_id or (score2 > score1):
                    artist_id = match2['id']
                    found_artist_name = name2
                    search_method = f"song as artist - SWAP DETECTED (score: {score2})"
                    print(f"✓ SWAP DETECTED! Found: {found_artist_name} (score: {score2})", file=sys.stderr)
                else:
                    print(f"  Found '{name2}' but band match was better (score: {score2} vs {score1})", file=sys.stderr)
        elif skip_swap_detection:
            print(f"⏭️  Skipping swap detection (user-verified values)", file=sys.stderr)
        
        # === STRATEGY 3: Track + Artist search ===
        if not artist_id and cleaned_song:
            print(f"🔍 Strategy 3: Searching track '{cleaned_song}' by '{cleaned_band}'", file=sys.stderr)
            try:
                results = sp.search(q=f"track:{cleaned_song} artist:{cleaned_band}", type='track', limit=5)
                if results['tracks']['items']:
                    # Pick the result with most matching artist name
                    best_match = None
                    best_score = 0
                    for track in results['tracks']['items']:
                        for artist in track['artists']:
                            # Simple fuzzy match
                            artist_lower = artist['name'].lower()
                            band_lower = cleaned_band.lower()
                            if band_lower in artist_lower or artist_lower in band_lower:
                                score = len(set(band_lower.split()) & set(artist_lower.split()))
                                if score > best_score:
                                    best_score = score
                                    best_match = artist
                    
                    if best_match:
                        artist_id = best_match['id']
                        found_artist_name = best_match['name']
                        search_method = "track+artist search"
                        print(f"✓ Found: {found_artist_name}", file=sys.stderr)
                    else:
                        # Just take first result
                        artist_id = results['tracks']['items'][0]['artists'][0]['id']
                        found_artist_name = results['tracks']['items'][0]['artists'][0]['name']
                        search_method = "track search (first result)"
                        print(f"⚠️  Using first result: {found_artist_name}", file=sys.stderr)
            except Exception as e:
                print(f"  Strategy 3 failed: {e}", file=sys.stderr)
        
        # === STRATEGY 4: Generic track search ===
        if not artist_id:
            print(f"🔍 Strategy 4: Generic track search '{cleaned_band}'", file=sys.stderr)
            try:
                results = sp.search(q=cleaned_band, type='track', limit=1)
                if results['tracks']['items']:
                    artist_id = results['tracks']['items'][0]['artists'][0]['id']
                    found_artist_name = results['tracks']['items'][0]['artists'][0]['name']
                    search_method = "generic track search"
                    print(f"✓ Found: {found_artist_name}", file=sys.stderr)
            except Exception as e:
                print(f"  Strategy 4 failed: {e}", file=sys.stderr)
        
        # === STRATEGY 5: Combined search (both terms together) ===
        if not artist_id and cleaned_song:
            combined = f"{cleaned_band} {cleaned_song}"
            print(f"🔍 Strategy 5: Combined search '{combined}'", file=sys.stderr)
            try:
                results = sp.search(q=combined, type='track', limit=1)
                if results['tracks']['items']:
                    artist_id = results['tracks']['items'][0]['artists'][0]['id']
                    found_artist_name = results['tracks']['items'][0]['artists'][0]['name']
                    search_method = "combined search"
                    print(f"✓ Found: {found_artist_name}", file=sys.stderr)
            except Exception as e:
                print(f"  Strategy 5 failed: {e}", file=sys.stderr)
        
        # === STRATEGY 6: Try original uncleaned names ===
        if not artist_id and cleaned_band != band_name:
            print(f"🔍 Strategy 6: Trying original name '{band_name}'", file=sys.stderr)
            try:
                results = sp.search(q=f"artist:{band_name}", type='artist', limit=1)
                if results['artists']['items']:
                    artist_id = results['artists']['items'][0]['id']
                    found_artist_name = results['artists']['items'][0]['name']
                    search_method = "original name (uncleaned)"
                    print(f"✓ Found: {found_artist_name}", file=sys.stderr)
            except Exception as e:
                print(f"  Strategy 6 failed: {e}", file=sys.stderr)
        
        # === GIVE UP ===
        if not artist_id:
            cache[cache_key] = "abstract"
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            print(f"❌ No artist found after all strategies for '{band_name}'", file=sys.stderr)
            return None
        
        # === GET GENRES ===
        artist = sp.artist(artist_id)
        raw_genres = artist.get('genres', [])
        
        if raw_genres:
            print(f"🎵 Spotify genres for '{found_artist_name}': {raw_genres} (via {search_method})", file=sys.stderr)
            cache[cache_key] = raw_genres
        else:
            print(f"⚠️  No genres for '{found_artist_name}' (via {search_method})", file=sys.stderr)
            cache[cache_key] = "abstract"
        
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        
        return raw_genres if raw_genres else None
        
    except Exception as e:
        print(f"❌ Spotify API error: {e}", file=sys.stderr)
        cache[cache_key] = "abstract"
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
            
            # Pattern 3: "Song (Band)" or "Song [Band]" - but skip if it's just quality indicators or venues
            elif not band and ' (' in title and title.endswith(')'):
                # Check if the content in parentheses looks like a quality indicator, venue, or band name
                idx = title.rfind(' (')
                paren_content = title[idx+2:-1].strip().lower()
                # Skip if it's a quality indicator (HD, 4K, live, official, etc.)
                quality_indicators = ['hd', '4k', '8k', 'live', 'official', 'lyrics', 'video', 'audio', 'hq', 'remastered', 'remaster']
                # Skip if it's a venue/location indicator
                venue_indicators = ['live at', 'at ', 'venue', 'concert', 'festival', 'session', 'studio', 'recording', 'from']
                is_venue = any(venue in paren_content for venue in venue_indicators)
                if paren_content not in quality_indicators and not is_venue and len(paren_content) > 2:
                    song = title[:idx].strip()
                    band = title[idx+2:-1].strip()
            elif not band and ' [' in title and title.endswith(']'):
                idx = title.rfind(' [')
                bracket_content = title[idx+2:-1].strip().lower()
                # Check if bracket content is a venue/location
                venue_indicators = ['live at', 'at ', 'venue', 'concert', 'festival', 'session', 'studio', 'recording', 'from']
                is_venue = any(venue in bracket_content for venue in venue_indicators)
                if not is_venue:
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
            # Remove parentheses content if it looks like a venue/location
            venue_pattern = r'\s*\([^)]*(?:live\s+at|at\s+|venue|concert|festival|session|studio|recording|from)[^)]*\)'
            band = re.sub(venue_pattern, '', band, flags=re.IGNORECASE)
            band = re.sub(r'\s*\(.*?\)$', '', band)  # Remove remaining (Official Audio) etc
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
            
            # Aggressive cleanup - final pass to remove all remaining junk
            def aggressive_cleanup(text):
                """Final cleanup pass to remove all remaining junk."""
                if not text:
                    return text
                
                # Remove everything after common indicators
                text = re.split(r'\s*[\|\–\—]\s*Official', text, flags=re.IGNORECASE)[0]
                text = re.split(r'\s*[\|\–\—]\s*Lyrics', text, flags=re.IGNORECASE)[0]
                text = re.split(r'\s*[\|\–\—]\s*Audio', text, flags=re.IGNORECASE)[0]
                
                # Remove all parentheses and brackets with content
                text = re.sub(r'\s*[\[\(].*?[\]\)]', '', text)
                
                # Remove trailing quality indicators
                text = re.sub(r'\s*(HD|HQ|4K|8K|Official|Lyric|Video|Audio).*$', '', text, flags=re.IGNORECASE)
                
                return text.strip()
            
            # Apply aggressive cleanup to both band and song AFTER extraction
            band = aggressive_cleanup(band)
            song = aggressive_cleanup(song)
            
            # Detect potential issues with extracted metadata
            # Check if band name looks like a song title or venue name
            venue_indicators = ['live at', 'at ', 'venue', 'concert', 'festival', 'session', 'studio', 'recording']
            if any(indicator in band.lower() for indicator in venue_indicators):
                print(f"⚠️  Warning: Band name '{band}' looks like a venue/location. May need manual correction.", file=sys.stderr)
            
            # Check if band name looks like a song title (long, descriptive phrases)
            if len(band.split()) > 4 and len(band) > 30:
                print(f"⚠️  Warning: Band name '{band}' is unusually long. Possible band/song swap?", file=sys.stderr)
            
            # Check if song name is very short (could be band name)
            if len(song.split()) <= 1 and len(song) < 10:
                print(f"⚠️  Warning: Song name '{song}' is very short. Possible band/song swap?", file=sys.stderr)
            
            print(f"📋 Extracted metadata: Band='{band}', Song='{song}'", file=sys.stderr)
            
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
                # Remove parentheses content if it looks like a venue/location
                venue_pattern = r'\s*\([^)]*(?:live\s+at|at\s+|venue|concert|festival|session|studio|recording|from)[^)]*\)'
                band = re.sub(venue_pattern, '', band, flags=re.IGNORECASE)
                band = re.sub(r'\s*\(.*?\)$', '', band)
                band = band.strip()
                
                # Clean up song name
                song = re.sub(r'\s*\[.*?\]$', '', song)
                song = re.sub(r'\s*\(.*?\)$', '', song)
                song = re.sub(r'\s*-\s*Official.*$', '', song, flags=re.IGNORECASE)
                song = song.strip()
                
                # Aggressive cleanup - final pass to remove all remaining junk (same as first attempt)
                def aggressive_cleanup(text):
                    """Final cleanup pass to remove all remaining junk."""
                    if not text:
                        return text
                    
                    # Remove everything after common indicators
                    text = re.split(r'\s*[\|\–\—]\s*Official', text, flags=re.IGNORECASE)[0]
                    text = re.split(r'\s*[\|\–\—]\s*Lyrics', text, flags=re.IGNORECASE)[0]
                    text = re.split(r'\s*[\|\–\—]\s*Audio', text, flags=re.IGNORECASE)[0]
                    
                    # Remove all parentheses and brackets with content
                    text = re.sub(r'\s*[\[\(].*?[\]\)]', '', text)
                    
                    # Remove trailing quality indicators
                    text = re.sub(r'\s*(HD|HQ|4K|8K|Official|Lyric|Video|Audio).*$', '', text, flags=re.IGNORECASE)
                    
                    return text.strip()
                
                # Apply aggressive cleanup to both band and song AFTER extraction
                band = aggressive_cleanup(band)
                song = aggressive_cleanup(song)
                
                # Detect potential issues with extracted metadata (same as first attempt)
                venue_indicators = ['live at', 'at ', 'venue', 'concert', 'festival', 'session', 'studio', 'recording']
                if any(indicator in band.lower() for indicator in venue_indicators):
                    print(f"⚠️  Warning: Band name '{band}' looks like a venue/location. May need manual correction.", file=sys.stderr)
                
                if len(band.split()) > 4 and len(band) > 30:
                    print(f"⚠️  Warning: Band name '{band}' is unusually long. Possible band/song swap?", file=sys.stderr)
                
                if len(song.split()) <= 1 and len(song) < 10:
                    print(f"⚠️  Warning: Song name '{song}' is very short. Possible band/song swap?", file=sys.stderr)
                
                print(f"📋 Extracted metadata: Band='{band}', Song='{song}'", file=sys.stderr)
                
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
    SDXL-TURBO OPTIMIZED: Concise, high-impact prompts under 77 tokens.
    Handles missing genre gracefully.
    """
    
    genre_hint = normalize_genre(raw_genres) if raw_genres else "abstract"
    genre_key = genre_hint.lower()
    has_genre = (genre_hint != "abstract")  # Flag to check if we have real genre
    
    # Extract features
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
    
    print(f"\n🎵 Audio Features Extracted:", file=sys.stderr)
    print(f"   Tempo: {tempo:.1f} BPM", file=sys.stderr)
    print(f"   Energy: {energy:.3f}", file=sys.stderr)
    print(f"   Brightness: {brightness:.1f} Hz", file=sys.stderr)
    print(f"   Harmonicity: {harmonicity:.3f}", file=sys.stderr)
    print(f"   Rhythm Stability: {rhythm_stability:.3f}", file=sys.stderr)
    
    # MOOD (single powerful phrase)
    if chroma_std < 0.2 and energy > 0.6:
        mood = "aggressive intensity"
    elif chroma_std > 0.4 and harmonicity > 0.6:
        mood = "emotional depth"
    elif harmonicity < 0.3 and zcr > 0.15:
        mood = "chaotic energy"
    elif chroma_std > 0.5 and energy < 0.4:
        mood = "melancholic"
    elif energy < 0.25:
        mood = "meditative calm"
    elif harmonicity > 0.7 and energy > 0.5:
        mood = "euphoric"
    else:
        mood = "introspective"
    
    # LIGHTING (concise)
    if rolloff > 7000 and brightness > 4000:
        lighting = "sharp bright contrast"
    elif rolloff < 3000 and brightness < 2000:
        lighting = "deep shadows"
    elif contrast > 25:
        lighting = "dramatic chiaroscuro"
    else:
        lighting = "cinematic glow"
    
    # MOTION (short)
    if tempo > 160 and rhythm_stability < 0.5:
        motion = "explosive chaos"
    elif tempo > 160:
        motion = "relentless drive"
    elif tempo < 80:
        motion = "glacial weight"
    elif rhythm_stability < 0.4:
        motion = "organic flow"
    else:
        motion = "rhythmic pulse"
    
    # COMPOSITION (minimal)
    if energy < 0.25:
        composition = "minimal centered"
    elif energy > 0.7:
        composition = "explosive asymmetric"
    else:
        composition = "balanced depth"
    
    # COLORS (short)
    if energy > 0.7 and brightness > 4000:
        colors = "neon electric"
    elif brightness < 2000 and energy < 0.4:
        colors = "dark muted"
    elif harmonicity > 0.65:
        colors = "warm harmonious"
    else:
        colors = "atmospheric gradient"
    
    # GENRE FLAVOR - MUCH shorter variants
    genre_flavors = {
        # Metal
        "symphonic metal": ["orchestral storm", "cathedral grandeur", "epic layers", "celestial power"],
        "progressive metal": ["fractal geometry", "cosmic drift", "technical precision", "astral minimal"],
        "gothic metal": ["romantic darkness", "velvet decay", "baroque shadows", "somber ritual"],
        "folk metal": ["tribal force", "mystic forest", "mythic tapestry", "runic power"],
        "power metal": ["heroic blaze", "mythic glory", "epic saga", "pure valor"],
        "metalcore": ["industrial collapse", "apocalyptic ruins", "fragmented despair", "brutal honesty"],
        "death metal": ["visceral brutality", "crushing doom", "technical violence", "primal fury"],
        "black metal": ["frozen fury", "desolate void", "atmospheric hate", "raw darkness"],
        "doom metal": ["earth-crushing", "funeral procession", "psychedelic despair", "monolithic"],
        "thrash metal": ["relentless aggression", "speed violence", "razor precision", "raw power"],
        "nu metal": ["urban chaos", "digital rage", "industrial angst", "modern brutality"],
        "metal": ["heavy force", "dark power", "aggressive weight", "metallic drive"],
        
        # Rock
        "post-rock": ["cathartic crescendo", "infinite horizon", "narrative journey", "minimal beauty"],
        "psychedelic rock": ["kaleidoscopic", "dreamy dissolution", "recursive patterns", "color distortion"],
        "progressive rock": ["technical mastery", "conceptual meditation", "structured epic", "retro-futurist"],
        "alternative rock": ["urban grit", "suburban melancholy", "layered angst", "raw honesty"],
        "classic rock": ["stadium power", "vintage warmth", "blues tapestry", "analog essence"],
        "grunge": ["distorted rage", "apathetic haze", "textured despair", "stripped truth"],
        "indie rock": ["playful chaos", "bedroom intimacy", "clever layers", "DIY purity"],
        "punk rock": ["anarchic fury", "quiet resistance", "political message", "raw defiance"],
        "hard rock": ["amplified aggression", "heavy contemplation", "layered riffs", "pure force"],
        "rock": ["electric energy", "driving rhythm", "guitar power", "raw sound"],
        
        # Jazz
        "free jazz": ["chaotic improvisation", "spontaneous freedom", "abstract expression", "wild energy"],
        "fusion": ["hybrid complexity", "electric sophistication", "groove synthesis", "technical flow"],
        "bebop": ["rapid complexity", "harmonic adventure", "virtuosic speed", "jazz essence"],
        "cool jazz": ["sophisticated calm", "elegant restraint", "smooth intellectualism", "refined atmosphere"],
        "smooth jazz": ["polished grooves", "radio elegance", "commercial sophistication", "easy listening"],
        "jazz funk": ["syncopated groove", "electric soul", "rhythmic fusion", "funky sophistication"],
        "contemporary jazz": ["modern elegance", "evolved tradition", "fresh sophistication", "current expression"],
        "jazz": ["swing sophistication", "improvisational flow", "blue note mood", "syncopated elegance"],
        
        # Blues
        "delta blues": ["raw authenticity", "rural soul", "acoustic pain", "roots essence"],
        "electric blues": ["amplified emotion", "urban grit", "powerful bends", "electric soul"],
        "blues rock": ["guitar fury", "rock intensity", "blues power", "electric passion"],
        "blues": ["deep feeling", "soulful pain", "twelve-bar truth", "emotional depth"],
        
        # Soul / R&B / Funk
        "neo soul": ["modern warmth", "contemporary soul", "smooth evolution", "refined emotion"],
        "funk": ["syncopated groove", "tight pocket", "rhythmic punch", "percussive soul"],
        "soul": ["emotional depth", "vocal power", "heartfelt expression", "pure feeling"],
        "r&b": ["smooth grooves", "vocal intimacy", "rhythmic soul", "contemporary feeling"],
        "rhythm and blues": ["soulful rhythm", "emotional groove", "classic feel", "heartfelt beats"],
        "motown": ["polished soul", "pop elegance", "Detroit sound", "classic groove"],
        "disco": ["mirror ball energy", "dance euphoria", "funky glamour", "groove celebration"],
        
        # Hip-Hop / Rap
        "conscious hip hop": ["lyrical depth", "social awareness", "intellectual flow", "message power"],
        "trap": ["heavy 808s", "hi-hat rolls", "dark atmosphere", "southern sound"],
        "boom bap": ["classic drums", "sample soul", "90s essence", "golden era"],
        "gangsta rap": ["street reality", "raw narrative", "west coast sound", "hard truth"],
        "hip hop": ["rhythmic flow", "urban poetry", "breakbeat soul", "street culture"],
        "rap": ["verbal dexterity", "rhythmic speech", "lyrical prowess", "flow mastery"],
        
        # Electronic / EDM
        "synthpop": ["digital ecstasy", "retro glow", "layered synthesis", "clean digital"],
        "electropop": ["neon pop", "electronic hooks", "synthetic charm", "digital pop"],
        "techno": ["relentless machine", "hypnotic loop", "modular precision", "raw rhythm"],
        "house": ["four-on-floor", "disco evolution", "club pulse", "dance spirit"],
        "trance": ["euphoric build", "epic progression", "emotional journey", "hypnotic ascent"],
        "drum and bass": ["breakbeat fury", "sub-bass power", "rapid energy", "jungle evolution"],
        "dubstep": ["wobble bass", "half-time power", "sub frequency", "bass aggression"],
        "edm": ["festival explosion", "gentle pulse", "drop architecture", "pure kick"],
        "electronic": ["cybernetic pulse", "ambient circuitry", "algorithmic depth", "pure waveform"],
        
        # Pop
        "dance pop": ["infectious energy", "club euphoria", "pop hooks", "dance floor"],
        "indie pop": ["quirky charm", "DIY sophistication", "bedroom production", "alternative hooks"],
        "pop": ["euphoric celebration", "soft intimacy", "polished perfection", "pure hooks"],
        
        # Country / Folk
        "folk rock": ["electrified roots", "pastoral calm", "narrative weaving", "folk honesty"],
        "americana": ["rootsy authenticity", "heartland soul", "storytelling tradition", "rural poetry"],
        "bluegrass": ["acoustic virtuosity", "mountain harmony", "rapid picking", "traditional roots"],
        "country": ["heartland stories", "twang emotion", "rural authenticity", "Nashville sound"],
        "folk": ["dancing tradition", "fireside storytelling", "cultural tapestry", "acoustic purity"],
        
        # World / Regional
        "afrobeat": ["polyrhythmic power", "funk fusion", "political groove", "African pulse"],
        "reggae": ["island rhythm", "offbeat groove", "roots vibration", "Caribbean soul"],
        "latin": ["rhythmic passion", "tropical heat", "dance culture", "Latin fire"],
        "world": ["cultural fusion", "global rhythm", "ethnic texture", "world sound"],
        "bossa nova": ["cool sophistication", "Brazilian elegance", "jazz samba", "tropical smooth"],
        "flamenco": ["passionate drama", "Spanish fire", "guitar intensity", "emotional dance"],
        
        # Classical / Orchestral
        "contemporary classical": ["modern composition", "evolved tradition", "experimental orchestration", "current classical"],
        "orchestral": ["symphonic grandeur", "instrumental power", "ensemble majesty", "orchestral sweep"],
        "classical": ["timeless elegance", "formal beauty", "compositional mastery", "refined tradition"],
        "opera": ["dramatic vocals", "theatrical grandeur", "emotional intensity", "vocal drama"],
        "baroque": ["ornate complexity", "period elegance", "contrapuntal beauty", "historical richness"],
        
        # Experimental / Ambient
        "avant-garde": ["boundary breaking", "experimental edge", "unconventional form", "artistic risk"],
        "experimental": ["sonic exploration", "unconventional structure", "pushing boundaries", "artistic innovation"],
        "ambient": ["cosmic stillness", "void meditation", "layered dimensions", "pure frequency"],
        "drone": ["sustained tones", "minimal evolution", "meditative hum", "sonic persistence"],
        "noise": ["textural chaos", "anti-musical", "sonic assault", "pure sound"],
        "industrial": ["mechanical aggression", "factory rhythm", "harsh electronics", "dystopian sound"],
        
        # Other
        "slowcore": ["glacial patience", "minimal emotion", "sparse beauty", "slow burn"],
        "shoegaze": ["guitar wash", "dreamy noise", "reverb depth", "wall of sound"],
        "emo": ["emotional intensity", "confessional lyrics", "melodic angst", "heartfelt pain"],
        "ska": ["upbeat offbeat", "horn section", "bouncing rhythm", "Caribbean punk"],
        "gospel": ["spiritual power", "choir majesty", "religious fervor", "vocal testimony"],
        
        "abstract": ["flowing forms", "ethereal shapes", "pure expression", "sonic essence"],
    }
    
    flavors = genre_flavors.get(genre_key, genre_flavors["abstract"])
    
    if energy > 0.65:
        flavor = flavors[0]
    elif chroma_std > 0.45 or harmonicity > 0.65:
        flavor = flavors[2]
    elif energy < 0.35:
        flavor = flavors[1]
    else:
        flavor = flavors[3]
    
    if has_genre:
        print(f"   Genre: {genre_hint}, Flavor: {flavor}", file=sys.stderr)
    else:
        print(f"   No genre found - using feature-driven description: {flavor}", file=sys.stderr)
    
    # ==========================================
    # PROMPT STRUCTURE: Different for abstract vs genre-known
    # ==========================================
    
    if has_genre:
        # Has genre: use genre-based structure
        prompt = (
            f"abstract {genre_hint} music poster, {flavor}, {mood} mood, "
            f"{lighting}, {motion}, {colors} colors, {composition}, "
            f"geometric symbolic forms, volumetric depth, no text, no faces"
        )
    else:
        # No genre: use purely descriptive structure based on audio features
        # Replace "abstract abstract" with more descriptive terms
        prompt = (
            f"music poster art, {flavor}, {mood} mood, "
            f"{lighting}, {motion}, {colors} colors, {composition}, "
            f"geometric symbolic forms, volumetric depth, no text, no faces"
        )
    
    # Negative prompt (barely used in SDXL Turbo, but doesn't hurt)
    negative_prompt = (
        "text, letters, words, watermark, faces, people, portraits, "
        "photo, realistic, messy, blurry, low quality"
    )
    
    # PROPER token counting - rough estimate by word count
    word_count = len(prompt.split())
    est_tokens = int(word_count * 1.4)  # Conservative estimate
    
    print(f"   Words: {word_count}, Est. tokens: ~{est_tokens}/77", file=sys.stderr)
    
    if est_tokens > 75:
        print(f"   ⚠️  WARNING: Prompt may be too long!", file=sys.stderr)
    
    print(f"✨ Concise prompt generated\n", file=sys.stderr)
    
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