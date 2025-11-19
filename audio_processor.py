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

# Try to import requests for YouTube page scraping
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. YouTube page metadata extraction will be disabled.", file=sys.stderr)

# Cache file path
CACHE_FILE = "band_song_cache.json"
YOUTUBE_METADATA_CACHE_FILE = "youtube_metadata_cache.json"
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

def get_video_id_from_url(url):
    """Extract just the video ID (11 characters) from YouTube URL."""
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if match:
        return match.group(1)
    return None

# YouTube metadata cache management functions
def load_youtube_metadata_cache():
    """Load the YouTube video ID -> band/song cache from JSON file."""
    if os.path.exists(YOUTUBE_METADATA_CACHE_FILE):
        try:
            with open(YOUTUBE_METADATA_CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load YouTube metadata cache file: {e}", file=sys.stderr)
            return {}
    return {}

def save_youtube_metadata_cache(cache):
    """Save the YouTube video ID -> band/song cache to JSON file."""
    try:
        with open(YOUTUBE_METADATA_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save YouTube metadata cache file: {e}", file=sys.stderr)

def get_cached_youtube_metadata(video_id):
    """Get cached band/song for a YouTube video ID."""
    cache = load_youtube_metadata_cache()
    entry = cache.get(video_id)
    if entry:
        return entry.get('band'), entry.get('song')
    return None, None

def save_youtube_metadata(video_id, band, song):
    """Save band/song metadata for a YouTube video ID."""
    cache = load_youtube_metadata_cache()
    cache[video_id] = {
        'band': band,
        'song': song
    }
    save_youtube_metadata_cache(cache)

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
    
    # Cache key is just the band name (genres are artist-level, not song-level)
    cache_key = band_name.lower().strip()
    
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
        # Cache key is just band name
        cache_key = band_name.lower().strip()
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
        
        # === STRATEGY 1: Try extracted "band" as artist (ONLY STRATEGY) ===
        print(f"🔍 Strategy 1: Trying '{cleaned_band}' as artist", file=sys.stderr)
        match1, score1, name1 = search_artist_with_scoring(cleaned_band)
        if match1:
            artist_id = match1['id']
            found_artist_name = name1
            search_method = f"band as artist (score: {score1})"
            print(f"✓ Found: {found_artist_name} (score: {score1})", file=sys.stderr)
        
        # === GIVE UP ON SPOTIFY IF STRATEGY 1 FAILS ===
        if not artist_id:
            print(f"❌ No artist found in Spotify (Strategy 1 failed) for '{band_name}'", file=sys.stderr)
            # Try MusicBrainz as fallback
            print(f"🔄 Falling back to MusicBrainz API...", file=sys.stderr)
            musicbrainz_genres = get_genre_from_musicbrainz(band_name, cache_file=cache_file)
            if musicbrainz_genres:
                return musicbrainz_genres
            # Cache as abstract if both fail
            cache[cache_key] = "abstract"
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            return None
        
        # === GET GENRES ===
        artist = sp.artist(artist_id)
        raw_genres = artist.get('genres', [])
        
        if raw_genres:
            print(f"🎵 Spotify genres for '{found_artist_name}': {raw_genres} (via {search_method})", file=sys.stderr)
            cache[cache_key] = raw_genres
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            return raw_genres
        else:
            print(f"⚠️  No genres for '{found_artist_name}' in Spotify (via {search_method})", file=sys.stderr)
            # Try MusicBrainz as fallback when Spotify has no genres
            print(f"🔄 Falling back to MusicBrainz API...", file=sys.stderr)
            musicbrainz_genres = get_genre_from_musicbrainz(band_name, cache_file=cache_file)
            if musicbrainz_genres:
                return musicbrainz_genres
            # Cache as abstract if both fail
            cache[cache_key] = "abstract"
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            return None
        
    except Exception as e:
        print(f"❌ Spotify API error: {e}", file=sys.stderr)
        # Try MusicBrainz as fallback
        print(f"🔄 Falling back to MusicBrainz API...", file=sys.stderr)
        musicbrainz_genres = get_genre_from_musicbrainz(band_name, cache_file=cache_file)
        if musicbrainz_genres:
            return musicbrainz_genres
        # Cache key is just band name
        cache_key = band_name.lower().strip()
        cache[cache_key] = "abstract"
        with open(cache_file, "w") as f:
            json.dump(cache, f)
        return None

def get_genre_from_musicbrainz(band_name, cache_file="genre_cache.json"):
    """
    Get genres from MusicBrainz API as fallback when Spotify fails.
    Uses the same cache format as Spotify genres.
    
    Args:
        band_name: Artist/band name to search for
        cache_file: Cache file path (same as Spotify cache)
    
    Returns:
        List of genres or None if not found
    """
    if not REQUESTS_AVAILABLE:
        return None
    
    # --- Load cache (same file as Spotify) ---
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except:
            cache = {}
    else:
        cache = {}
    
    # Cache key is just the band name (same as Spotify)
    cache_key = band_name.lower().strip()
    
    # Check cache first - if we already have genres (from Spotify or MusicBrainz), return them
    if cache_key in cache:
        cached = cache[cache_key]
        if cached and cached != "abstract":
            if isinstance(cached, list) and cached:
                print(f"🎵 MusicBrainz genres (cached): {cached}", file=sys.stderr)
            return cached if cached != "abstract" else None
        # If cached as "abstract", don't try again
        return None
    
    try:
        # MusicBrainz API endpoint
        base_url = "https://musicbrainz.org/ws/2"
        headers = {
            'User-Agent': 'mus2pic/1.0 (https://github.com/yourusername/mus2pic)',
            'Accept': 'application/json'
        }
        
        # Rate limiting: max 1 request per second
        time.sleep(1.1)  # Be safe, wait 1.1 seconds
        
        # Search for artist
        search_url = f"{base_url}/artist"
        params = {
            'query': f'artist:"{band_name}"',
            'limit': 5,
            'fmt': 'json'
        }
        
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        artists = data.get('artists', [])
        
        if not artists:
            print(f"⚠️  No artist found in MusicBrainz for '{band_name}'", file=sys.stderr)
            cache[cache_key] = "abstract"
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            return None
        
        # Find best matching artist (exact match preferred)
        best_artist = None
        search_lower = band_name.lower().strip()
        
        for artist in artists:
            artist_name = artist.get('name', '').lower().strip()
            if artist_name == search_lower:
                best_artist = artist
                break
        
        # If no exact match, use first result
        if not best_artist:
            best_artist = artists[0]
        
        artist_mbid = best_artist.get('id')
        found_artist_name = best_artist.get('name', band_name)
        
        if not artist_mbid:
            print(f"⚠️  No MBID found for '{found_artist_name}'", file=sys.stderr)
            cache[cache_key] = "abstract"
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            return None
        
        # Rate limiting: wait before lookup
        time.sleep(1.1)
        
        # Lookup artist details with tags (tags often contain genres)
        lookup_url = f"{base_url}/artist/{artist_mbid}"
        params = {
            'inc': 'tags',
            'fmt': 'json'
        }
        
        response = requests.get(lookup_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        artist_data = response.json()
        
        # Extract tags (MusicBrainz uses tags for genres)
        tags = artist_data.get('tags', [])
        genres = []
        
        # Get tags with count > 0 (more reliable)
        for tag in tags:
            tag_name = tag.get('name', '').lower().strip()
            count = tag.get('count', 0)
            if tag_name and count > 0:
                genres.append(tag_name)
        
        # If no tags, try to get from disambiguation or aliases
        if not genres:
            # Check for genre-like information in other fields
            disambiguation = artist_data.get('disambiguation', '')
            if disambiguation:
                # Sometimes genre info is in disambiguation
                pass
        
        if genres:
            print(f"🎵 MusicBrainz genres for '{found_artist_name}': {genres}", file=sys.stderr)
            cache[cache_key] = genres
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            return genres
        else:
            print(f"⚠️  No genres/tags found for '{found_artist_name}' in MusicBrainz", file=sys.stderr)
            cache[cache_key] = "abstract"
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ MusicBrainz API request error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ MusicBrainz API error: {e}", file=sys.stderr)
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

# Primary method: Extract videoAttributeViewModel from YouTube page JSON
def extract_video_attribute_from_page(video_id):
    """
    Extract videoAttributeViewModel (title/subtitle) from YouTube page JSON.
    This contains structured music metadata that YouTube displays.
    
    Returns: (artist, song) or (None, None) if not found
    """
    if not REQUESTS_AVAILABLE:
        return None, None
    
    try:
        # Fetch the YouTube page
        url = f'https://www.youtube.com/watch?v={video_id}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text
        
        # Look for ytInitialData in the HTML
        # It's usually in a script tag: var ytInitialData = {...};
        patterns = [
            r'var\s+ytInitialData\s*=\s*({.+?});',
            r'window\["ytInitialData"\]\s*=\s*({.+?});',
            r'ytInitialData\s*=\s*({.+?});',
        ]
        
        json_data = None
        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL)
            if match:
                try:
                    json_data = json.loads(match.group(1))
                    break
                except json.JSONDecodeError:
                    continue
        
        if not json_data:
            return None, None
        
        # Navigate through the JSON structure to find videoAttributeViewModel
        # Structure: contents -> twoColumnWatchNextResults -> results -> results -> contents
        # Or: horizontalCardListRenderer -> cards -> videoAttributeViewModel
        
        def find_video_attribute(obj, path=""):
            """Recursively search for videoAttributeViewModel"""
            if isinstance(obj, dict):
                if 'videoAttributeViewModel' in obj:
                    vm = obj['videoAttributeViewModel']
                    title = vm.get('title', '').strip()
                    subtitle = vm.get('subtitle', '').strip()
                    if title and subtitle:
                        return subtitle, title  # subtitle is artist, title is song
                
                # Also check for horizontalCardListRenderer
                if 'horizontalCardListRenderer' in obj:
                    cards = obj['horizontalCardListRenderer'].get('cards', [])
                    for card in cards:
                        if 'videoAttributeViewModel' in card:
                            vm = card['videoAttributeViewModel']
                            title = vm.get('title', '').strip()
                            subtitle = vm.get('subtitle', '').strip()
                            if title and subtitle:
                                return subtitle, title
                
                # Recursively search nested structures
                for key, value in obj.items():
                    result = find_video_attribute(value, f"{path}.{key}")
                    if result[0] and result[1]:
                        return result
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    result = find_video_attribute(item, f"{path}[{i}]")
                    if result[0] and result[1]:
                        return result
            
            return None, None
        
        return find_video_attribute(json_data)
        
    except Exception as e:
        # Silently fail - this is a fallback method
        return None, None

# Helper function to extract band and song from YouTube metadata
def extract_band_song_from_metadata(info):
    """
    Extract band name and song title from YouTube metadata info dict.
    Returns: (band_name, song_title)
    """
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
    
    return band, song

# 1. Extract audio from YouTube and get metadata
def download_audio(youtube_url):
    """
    Download audio from YouTube and extract metadata.
    Returns: (audio_path, band_name, song_title)
    """
    # Extract just the video ID to avoid playlist issues
    clean_url = extract_video_id(youtube_url)
    video_id = get_video_id_from_url(youtube_url)
    
    # Check cache first
    if video_id:
        cached_band, cached_song = get_cached_youtube_metadata(video_id)
        if cached_band and cached_song:
            print(f"✅ Using cached metadata for video {video_id}: Band='{cached_band}', Song='{cached_song}'", file=sys.stderr)
            # We still need to download the audio, but we'll use the cached metadata
            use_cached_metadata = True
        else:
            use_cached_metadata = False
    else:
        use_cached_metadata = False
    
    print(f"Downloading from: {clean_url}")
    
    # PRIMARY METHOD: Try to extract from YouTube page JSON first (most reliable)
    # Skip if we have cached metadata
    if not use_cached_metadata:
        if video_id:
            page_artist, page_song = extract_video_attribute_from_page(video_id)
        else:
            page_artist, page_song = None, None
        if page_artist and page_song:
            print(f"✅ Extracted from YouTube page: Band='{page_artist}', Song='{page_song}'", file=sys.stderr)
            # We still need to download the audio, but we'll use the extracted metadata
            use_page_metadata = True
        else:
            use_page_metadata = False
    else:
        use_page_metadata = False
    
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
            
            # Use cached metadata if available, otherwise use page metadata, otherwise fall back to yt-dlp metadata extraction
            if use_cached_metadata:
                band, song = cached_band, cached_song
            elif use_page_metadata:
                band, song = page_artist, page_song
            else:
                # Extract metadata using shared function (fallback method)
                band, song = extract_band_song_from_metadata(info)
            
            # Save to cache if we have a video ID and successfully extracted metadata
            if video_id and band and song and not use_cached_metadata:
                save_youtube_metadata(video_id, band, song)
                print(f"💾 Saved metadata to cache for video {video_id}", file=sys.stderr)
            
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
                
                # Use cached metadata if available, otherwise use page metadata, otherwise fall back to yt-dlp metadata extraction
                if use_cached_metadata:
                    band, song = cached_band, cached_song
                elif use_page_metadata:
                    band, song = page_artist, page_song
                else:
                    # Extract metadata using shared function (fallback method)
                    band, song = extract_band_song_from_metadata(info)
                
                # Save to cache if we have a video ID and successfully extracted metadata
                if video_id and band and song and not use_cached_metadata:
                    save_youtube_metadata(video_id, band, song)
                    print(f"💾 Saved metadata to cache for video {video_id}", file=sys.stderr)
                
                return audio_path, band, song
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            raise

# 2. Analyze audio
def analyze_audio(audio_path, duration=None, offset=5.0):
    """
    Enhanced audio analysis with proper scaling and beat detection.
    """
    # Load audio with offset
    if duration is None:
        y, sr = librosa.load(audio_path, offset=offset)
    else:
        y, sr = librosa.load(audio_path, offset=offset, duration=duration)
    
    # === TEMPO - Try to detect correct BPM (avoid half-time detection) ===
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Check for half-time detection - if tempo is very low, try doubling
    if tempo < 100:
        # Also try with tightness parameter for better detection
        tempo_alt, _ = librosa.beat.beat_track(y=y, sr=sr, tightness=100)
        # If alternative is significantly higher and makes sense, use it
        if tempo_alt > tempo * 1.5 and tempo_alt < 200:
            tempo_float = float(tempo)
            tempo_alt_float = float(tempo_alt)
            print(f"   ⚠️  Tempo correction: {tempo_float:.1f} → {tempo_alt_float:.1f} BPM", file=sys.stderr)
            tempo = tempo_alt
        else:
            # Just double it if it seems like half-time
            tempo_doubled = tempo * 2
            tempo_float = float(tempo)
            tempo_doubled_float = float(tempo_doubled)
            print(f"   ⚠️  Half-time detected, doubling: {tempo_float:.1f} → {tempo_doubled_float:.1f} BPM", file=sys.stderr)
            tempo = tempo_doubled
    
    # === ENERGY - Normalize RMS to 0-1 scale ===
    rms = librosa.feature.rms(y=y)[0]
    raw_energy = np.mean(rms)
    
    # RMS typically ranges 0.01 (quiet) to 0.35 (loud/compressed)
    # Normalize to 0-1 scale for easier thresholding
    energy_normalized = np.clip((raw_energy - 0.01) / 0.29, 0, 1)
    
    # Convert to native Python types before formatting
    raw_energy_float = float(raw_energy) if isinstance(raw_energy, (np.integer, np.floating, np.ndarray)) else float(raw_energy)
    energy_norm_float = float(energy_normalized) if isinstance(energy_normalized, (np.integer, np.floating, np.ndarray)) else float(energy_normalized)
    print(f"   DEBUG - Raw RMS: {raw_energy_float:.4f} → Normalized: {energy_norm_float:.3f}", file=sys.stderr)
    
    # === REST OF FEATURES ===
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_std = np.std(chroma)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    rhythm_stability = 1.0 - np.clip(np.std(onset_env) / (np.mean(onset_env) + 1e-6), 0, 1)
    
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonicity = np.mean(librosa.feature.rms(y=y_harmonic)) / (raw_energy + 1e-6)
    
    # Helper to safely convert numpy values to Python native types
    def to_native_float(value, name="unknown"):
        """Convert numpy scalar/array to native Python float."""
        try:
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    result = float(value.item())
                    print(f"   DEBUG [{name}]: np.ndarray (0-d) -> {type(result).__name__} = {result}", file=sys.stderr)
                    return result
                else:
                    # Multi-dimensional array - shouldn't happen for scalars
                    print(f"   DEBUG [{name}]: np.ndarray ({value.ndim}-d) -> converting first element", file=sys.stderr)
                    return float(value.flat[0])
            elif isinstance(value, (np.integer, np.floating)):
                result = float(value)
                print(f"   DEBUG [{name}]: {type(value).__name__} -> {type(result).__name__} = {result}", file=sys.stderr)
                return result
            else:
                result = float(value)
                print(f"   DEBUG [{name}]: {type(value).__name__} -> {type(result).__name__} = {result}", file=sys.stderr)
                return result
        except Exception as e:
            print(f"   ERROR [{name}]: Failed to convert {type(value).__name__} = {value}: {e}", file=sys.stderr)
            return 0.0
    
    # Safe helper to convert values to string for debugging
    def safe_str(value):
        """Safely convert value to string for debugging."""
        try:
            if isinstance(value, np.ndarray):
                return f"np.ndarray(shape={value.shape}, dtype={value.dtype})"
            elif isinstance(value, (np.integer, np.floating)):
                return f"{type(value).__name__}({float(value)})"
            else:
                return str(value)
        except:
            return f"<unprintable {type(value).__name__}>"
    
    print(f"\n🔍 DEBUG: Converting features to native Python types...", file=sys.stderr)
    print(f"   tempo type: {type(tempo).__name__}, value: {safe_str(tempo)}", file=sys.stderr)
    print(f"   spectral_centroid type: {type(spectral_centroid).__name__}, value: {safe_str(spectral_centroid)}", file=sys.stderr)
    print(f"   energy_normalized type: {type(energy_normalized).__name__}, value: {safe_str(energy_normalized)}", file=sys.stderr)
    print(f"   raw_energy type: {type(raw_energy).__name__}, value: {safe_str(raw_energy)}", file=sys.stderr)
    print(f"   rhythm_stability type: {type(rhythm_stability).__name__}, value: {safe_str(rhythm_stability)}", file=sys.stderr)
    print(f"   harmonicity type: {type(harmonicity).__name__}, value: {safe_str(harmonicity)}", file=sys.stderr)
    print(f"   mfcc_mean type: {type(mfcc_mean).__name__}, shape: {mfcc_mean.shape if isinstance(mfcc_mean, np.ndarray) else 'N/A'}", file=sys.stderr)
    print(f"   mfcc_std type: {type(mfcc_std).__name__}, shape: {mfcc_std.shape if isinstance(mfcc_std, np.ndarray) else 'N/A'}", file=sys.stderr)
    
    result = {
        'tempo': to_native_float(tempo, 'tempo'),
        'brightness': to_native_float(spectral_centroid, 'brightness'),
        'spectral_centroid': to_native_float(spectral_centroid, 'spectral_centroid'),
        'energy': to_native_float(energy_normalized, 'energy'),  # Use normalized energy
        'raw_energy': to_native_float(raw_energy, 'raw_energy'),     # Keep raw for reference
        'spectral_rolloff': to_native_float(spectral_rolloff, 'spectral_rolloff'),
        'spectral_bandwidth': to_native_float(spectral_bandwidth, 'spectral_bandwidth'),
        'spectral_contrast': to_native_float(spectral_contrast, 'spectral_contrast'),
        'zcr': to_native_float(zcr, 'zcr'),
        'chroma_std': to_native_float(chroma_std, 'chroma_std'),
        'rhythm_stability': to_native_float(np.clip(rhythm_stability, 0, 1), 'rhythm_stability'),
        'harmonicity': to_native_float(np.clip(harmonicity, 0, 1), 'harmonicity'),
        'mfcc_mean': mfcc_mean.tolist() if isinstance(mfcc_mean, np.ndarray) else list(mfcc_mean),
        'mfcc_std': mfcc_std.tolist() if isinstance(mfcc_std, np.ndarray) else list(mfcc_std)
    }
    
    print(f"✅ DEBUG: All features converted. Checking result types...", file=sys.stderr)
    for key, value in result.items():
        if isinstance(value, (list, tuple)):
            print(f"   {key}: {type(value).__name__} (length {len(value)})", file=sys.stderr)
        else:
            print(f"   {key}: {type(value).__name__} = {value}", file=sys.stderr)
    
    return result
def audio_to_prompt_v3(features, band_name=None, song_title=None, raw_genres=None):
    """
    ENHANCED: Genre-specific visual languages with token budget optimization.
    """
    
    genre_hint = normalize_genre(raw_genres) if raw_genres else "abstract"
    genre_key = genre_hint.lower()
    has_genre = (genre_hint != "abstract")
    
    # Extract features (keep as is)
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
    
    # Genre categories for visual treatment
    is_metal = "metal" in genre_key
    is_aggressive = any(x in genre_key for x in ["metal", "punk", "hardcore", "grunge"])
    is_electronic = any(x in genre_key for x in ["electronic", "techno", "edm", "house", "synthpop"])
    is_jazz = "jazz" in genre_key or "blues" in genre_key
    is_ambient = any(x in genre_key for x in ["ambient", "drone", "post-rock"])
    is_classical = any(x in genre_key for x in ["classical", "orchestral", "opera"])
    
    # === MOOD (keep existing logic) ===
    if chroma_std < 0.2 and energy > 0.55:
        mood = "aggressive intensity"
    elif energy > 0.55 and tempo > 140 and is_aggressive:
        mood = "explosive power"
    elif chroma_std > 0.4 and harmonicity > 0.6:
        mood = "complex emotional depth"
    elif harmonicity < 0.3 and zcr > 0.15:
        mood = "chaotic energy"
    elif chroma_std > 0.5 and energy < 0.4:
        mood = "melancholic"
    elif energy < 0.25:
        mood = "meditative calm"
    elif harmonicity > 0.65 and energy > 0.45:
        mood = "euphoric power"
    else:
        mood = "introspective"
    
    # === LIGHTING (keep existing) ===
    if rolloff > 7000 and brightness > 4000:
        lighting = "sharp bright contrast"
    elif rolloff < 3000 and brightness < 2000:
        lighting = "deep shadows"
    elif contrast > 25:
        lighting = "dramatic chiaroscuro"
    elif energy > 0.6:
        lighting = "bold dramatic lighting"
    else:
        lighting = "cinematic glow"
    
    # === MOTION (improved) ===
    if tempo > 150 and rhythm_stability < 0.5:
        motion = "explosive chaos"
    elif tempo > 150 and rhythm_stability > 0.6:
        motion = "relentless drive"
    elif tempo > 120 and energy > 0.7:
        motion = "aggressive rhythmic force"
    elif tempo > 130 and energy > 0.5:
        motion = "dynamic rhythmic drive"
    elif tempo < 80:
        motion = "glacial weight"
    elif rhythm_stability < 0.4:
        motion = "organic flow"
    else:
        motion = "rhythmic pulse"
    
    # === COLORS (keep existing) ===
    if energy > 0.65 and brightness > 4000:
        colors = "neon electric"
    elif energy > 0.55 and is_metal:
        colors = "bold saturated"
    elif brightness < 2000 and energy < 0.4:
        colors = "dark muted"
    elif harmonicity > 0.65:
        colors = "rich harmonious"
    else:
        colors = "atmospheric gradient"
    
    # === COMPOSITION (keep existing) ===
    if energy < 0.25:
        composition = "minimal centered"
    elif energy > 0.55 and tempo > 140:
        composition = "explosive asymmetric"
    elif energy > 0.5:
        composition = "dynamic bold arrangement"
    else:
        composition = "balanced depth"
    
    # === GENRE FLAVORS (your existing dict - keep it) ===
    genre_flavors = {
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
        "free jazz": ["chaotic improvisation", "spontaneous freedom", "abstract expression", "wild energy"],
        "fusion": ["hybrid complexity", "electric sophistication", "groove synthesis", "technical flow"],
        "bebop": ["rapid complexity", "harmonic adventure", "virtuosic speed", "jazz essence"],
        "cool jazz": ["sophisticated calm", "elegant restraint", "smooth intellectualism", "refined atmosphere"],
        "smooth jazz": ["polished grooves", "radio elegance", "commercial sophistication", "easy listening"],
        "jazz funk": ["syncopated groove", "electric soul", "rhythmic fusion", "funky sophistication"],
        "contemporary jazz": ["modern elegance", "evolved tradition", "fresh sophistication", "current expression"],
        "jazz": ["swing sophistication", "improvisational flow", "blue note mood", "syncopated elegance"],
        "delta blues": ["raw authenticity", "rural soul", "acoustic pain", "roots essence"],
        "electric blues": ["amplified emotion", "urban grit", "powerful bends", "electric soul"],
        "blues rock": ["guitar fury", "rock intensity", "blues power", "electric passion"],
        "blues": ["deep feeling", "soulful pain", "twelve-bar truth", "emotional depth"],
        "neo soul": ["modern warmth", "contemporary soul", "smooth evolution", "refined emotion"],
        "funk": ["syncopated groove", "tight pocket", "rhythmic punch", "percussive soul"],
        "soul": ["emotional depth", "vocal power", "heartfelt expression", "pure feeling"],
        "r&b": ["smooth grooves", "vocal intimacy", "rhythmic soul", "contemporary feeling"],
        "rhythm and blues": ["soulful rhythm", "emotional groove", "classic feel", "heartfelt beats"],
        "motown": ["polished soul", "pop elegance", "Detroit sound", "classic groove"],
        "disco": ["mirror ball energy", "dance euphoria", "funky glamour", "groove celebration"],
        "conscious hip hop": ["lyrical depth", "social awareness", "intellectual flow", "message power"],
        "trap": ["heavy 808s", "hi-hat rolls", "dark atmosphere", "southern sound"],
        "boom bap": ["classic drums", "sample soul", "90s essence", "golden era"],
        "gangsta rap": ["street reality", "raw narrative", "west coast sound", "hard truth"],
        "hip hop": ["rhythmic flow", "urban poetry", "breakbeat soul", "street culture"],
        "rap": ["verbal dexterity", "rhythmic speech", "lyrical prowess", "flow mastery"],
        "synthpop": ["digital ecstasy", "retro glow", "layered synthesis", "clean digital"],
        "electropop": ["neon pop", "electronic hooks", "synthetic charm", "digital pop"],
        "techno": ["relentless machine", "hypnotic loop", "modular precision", "raw rhythm"],
        "house": ["four-on-floor", "disco evolution", "club pulse", "dance spirit"],
        "trance": ["euphoric build", "epic progression", "emotional journey", "hypnotic ascent"],
        "drum and bass": ["breakbeat fury", "sub-bass power", "rapid energy", "jungle evolution"],
        "dubstep": ["wobble bass", "half-time power", "sub frequency", "bass aggression"],
        "edm": ["festival explosion", "gentle pulse", "drop architecture", "pure kick"],
        "electronic": ["cybernetic pulse", "ambient circuitry", "algorithmic depth", "pure waveform"],
        "dance pop": ["infectious energy", "club euphoria", "pop hooks", "dance floor"],
        "indie pop": ["quirky charm", "DIY sophistication", "bedroom production", "alternative hooks"],
        "pop": ["euphoric celebration", "soft intimacy", "polished perfection", "pure hooks"],
        "folk rock": ["electrified roots", "pastoral calm", "narrative weaving", "folk honesty"],
        "americana": ["rootsy authenticity", "heartland soul", "storytelling tradition", "rural poetry"],
        "bluegrass": ["acoustic virtuosity", "mountain harmony", "rapid picking", "traditional roots"],
        "country": ["heartland stories", "twang emotion", "rural authenticity", "Nashville sound"],
        "folk": ["dancing tradition", "fireside storytelling", "cultural tapestry", "acoustic purity"],
        "afrobeat": ["polyrhythmic power", "funk fusion", "political groove", "African pulse"],
        "reggae": ["island rhythm", "offbeat groove", "roots vibration", "Caribbean soul"],
        "latin": ["rhythmic passion", "tropical heat", "dance culture", "Latin fire"],
        "world": ["cultural fusion", "global rhythm", "ethnic texture", "world sound"],
        "bossa nova": ["cool sophistication", "Brazilian elegance", "jazz samba", "tropical smooth"],
        "flamenco": ["passionate drama", "Spanish fire", "guitar intensity", "emotional dance"],
        "contemporary classical": ["modern composition", "evolved tradition", "experimental orchestration", "current classical"],
        "orchestral": ["symphonic grandeur", "instrumental power", "ensemble majesty", "orchestral sweep"],
        "classical": ["timeless elegance", "formal beauty", "compositional mastery", "refined tradition"],
        "opera": ["dramatic vocals", "theatrical grandeur", "emotional intensity", "vocal drama"],
        "baroque": ["ornate complexity", "period elegance", "contrapuntal beauty", "historical richness"],
        "avant-garde": ["boundary breaking", "experimental edge", "unconventional form", "artistic risk"],
        "experimental": ["sonic exploration", "unconventional structure", "pushing boundaries", "artistic innovation"],
        "ambient": ["cosmic stillness", "void meditation", "layered dimensions", "pure frequency"],
        "drone": ["sustained tones", "minimal evolution", "meditative hum", "sonic persistence"],
        "noise": ["textural chaos", "anti-musical", "sonic assault", "pure sound"],
        "industrial": ["mechanical aggression", "factory rhythm", "harsh electronics", "dystopian sound"],
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
    
    # ==========================================
    # NEW: GENRE-SPECIFIC VISUAL VOCABULARIES
    # ==========================================
    
    # Define visual elements by genre category
    if is_metal:
        visual_elements = [
            "fractured shards",
            "angular brutality",
            "jagged spikes",
            "shattered crystalline forms",
            "apocalyptic architecture"
        ]
        texture_style = [
            "corroded metal texture",
            "industrial decay",
            "rough brutal surfaces",
            "distressed materials"
        ]
        atmosphere = [
            "smoke and ash",
            "forge fire glow",
            "volcanic atmosphere",
            "storm clouds"
        ]
    
    elif is_electronic:
        visual_elements = [
            "geometric circuitry",
            "digital grid patterns",
            "wireframe structures",
            "holographic forms",
            "cybernetic shapes"
        ]
        texture_style = [
            "neon glow edges",
            "pixelated gradients",
            "chrome reflections",
            "LED light trails"
        ]
        atmosphere = [
            "digital fog",
            "laser beams",
            "data streams",
            "electric haze"
        ]
    
    elif is_jazz:
        visual_elements = [
            "fluid organic curves",
            "improvisational shapes",
            "syncopated patterns",
            "abstract instrumental forms",
            "flowing silhouettes"
        ]
        texture_style = [
            "smoky gradients",
            "brushstroke texture",
            "vinyl grain",
            "liquid ink flow"
        ]
        atmosphere = [
            "dimmed stage lighting",
            "spotlight beams",
            "cigarette smoke haze",
            "intimate club ambiance"
        ]
    
    elif is_ambient:
        visual_elements = [
            "floating particles",
            "ethereal wisps",
            "nebula clouds",
            "infinite space",
            "weightless forms"
        ]
        texture_style = [
            "soft diffusion",
            "translucent layers",
            "gaussian blur",
            "gossamer fabric"
        ]
        atmosphere = [
            "cosmic void",
            "celestial mist",
            "aurora shimmer",
            "endless depth"
        ]
    
    elif is_classical:
        visual_elements = [
            "ornate scrollwork",
            "architectural columns",
            "baroque flourishes",
            "symmetrical mandala",
            "elegant filigree"
        ]
        texture_style = [
            "gold leaf accents",
            "marble surfaces",
            "velvet richness",
            "aged parchment"
        ]
        atmosphere = [
            "concert hall lighting",
            "chandelier glow",
            "theatrical curtains",
            "cathedral rays"
        ]
    
    else:  # Default/Rock/Pop
        visual_elements = [
            "abstract geometric forms",
            "dynamic shapes",
            "flowing patterns",
            "symbolic imagery",
            "expressive marks"
        ]
        texture_style = [
            "screen-print texture",
            "poster grain",
            "spray paint effects",
            "analog warmth"
        ]
        atmosphere = [
            "volumetric depth",
            "atmospheric haze",
            "stage lighting",
            "cinematic mood"
        ]
    
    # Select specific elements based on features
    if energy > 0.7:
        visual_elem = visual_elements[0] if len(visual_elements) > 0 else "dynamic forms"
        texture = texture_style[0] if len(texture_style) > 0 else "textured surface"
    elif energy < 0.3:
        visual_elem = visual_elements[1] if len(visual_elements) > 1 else "minimal shapes"
        texture = texture_style[-1] if len(texture_style) > 0 else "soft texture"
    else:
        visual_elem = visual_elements[2] if len(visual_elements) > 2 else "balanced forms"
        texture = texture_style[1] if len(texture_style) > 1 else "layered texture"
    
    atmo = atmosphere[0] if harmonicity > 0.6 else atmosphere[-1] if len(atmosphere) > 1 else atmosphere[0]
    
    if has_genre:
        print(f"   Genre: {genre_hint}, Flavor: {flavor}", file=sys.stderr)
        print(f"   Visual: {visual_elem}, {texture}, {atmo}", file=sys.stderr)
    else:
        print(f"   No genre - Feature-driven: {flavor}", file=sys.stderr)
    
    # ==========================================
    # FINAL PROMPT CONSTRUCTION
    # ==========================================
    
    if has_genre:
        prompt = (
            f"abstract {genre_hint} poster, {flavor}, {mood}, "
            f"{visual_elem}, {texture}, {atmo}, "
            f"{lighting}, {motion}, {colors} colors, {composition}, "
            f"professional design, no text, no faces"
        )
    else:
        prompt = (
            f"music poster art, {flavor}, {mood}, "
            f"{visual_elem}, {texture}, {atmo}, "
            f"{lighting}, {motion}, {colors} colors, {composition}, "
            f"professional design, no text, no faces"
        )
    
    negative_prompt = (
        "text, letters, words, watermark, faces, people, portraits, "
        "photo, realistic, messy, blurry, low quality"
    )
    
    word_count = len(prompt.split())
    est_tokens = int(word_count * 1.4)
    
    print(f"   Words: {word_count}, Est. tokens: ~{est_tokens}/77", file=sys.stderr)
    
    if est_tokens > 75:
        print(f"   ⚠️  WARNING: Prompt may be too long!", file=sys.stderr)
    
    print(f"✨ Genre-specific visual prompt generated\n", file=sys.stderr)
    
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