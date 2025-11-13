#!/usr/bin/env python3
"""
Test script to extract and display YouTube metadata.
Tests different methods to extract band name and song title.
"""

import yt_dlp
import json
import re

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:v\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url

def test_youtube_metadata(url):
    """Extract and display all available metadata from YouTube URL"""
    
    # Extract video ID
    video_id = extract_video_id(url)
    clean_url = f"https://www.youtube.com/watch?v={video_id}"
    print("=" * 80)
    print(f"Testing URL: {url}")
    print(f"Clean URL: {clean_url}")
    print("=" * 80)
    
    # Configure yt-dlp to extract info without downloading
    ydl_opts = {
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,  # Get full metadata
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
            
            print("\n" + "=" * 80)
            print("ALL AVAILABLE METADATA FIELDS:")
            print("=" * 80)
            
            # Print all available fields
            for key, value in sorted(info.items()):
                if key not in ['formats', 'thumbnails', 'automatic_captions', 'subtitles', 'requested_formats']:
                    # Truncate long values
                    if isinstance(value, (list, dict)):
                        print(f"{key}: {type(value).__name__} with {len(value)} items")
                    elif isinstance(value, str) and len(value) > 100:
                        print(f"{key}: {value[:100]}...")
                    else:
                        print(f"{key}: {value}")
            
            print("\n" + "=" * 80)
            print("KEY METADATA FIELDS:")
            print("=" * 80)
            
            # Key fields we're interested in
            key_fields = [
                'title',
                'uploader',
                'uploader_id',
                'channel',
                'channel_id',
                'artist',
                'creator',
                'album',
                'track',
                'description',
                'tags',
            ]
            
            for field in key_fields:
                value = info.get(field)
                if value:
                    print(f"\n{field.upper()}:")
                    if isinstance(value, list):
                        for item in value:
                            print(f"  - {item}")
                    else:
                        print(f"  {value}")
            
            print("\n" + "=" * 80)
            print("ATTEMPTING TO EXTRACT BAND NAME AND SONG TITLE:")
            print("=" * 80)
            
            title = info.get('title', 'Unknown')
            uploader = info.get('uploader', 'Unknown Artist')
            artist = info.get('artist')
            creator = info.get('creator')
            channel = info.get('channel')
            description = info.get('description', '')
            
            print(f"\nTitle: {title}")
            print(f"Uploader: {uploader}")
            print(f"Artist: {artist}")
            print(f"Creator: {creator}")
            print(f"Channel: {channel}")
            
            # Method 1: Check description first (most reliable for fan uploads)
            print("\n--- Method 1: Check Description First ---")
            band = None
            song = None
            
            description = info.get('description', '')
            if description:
                # Look for "Band - Song" pattern
                desc_match = re.search(r'^([^-·\n]+)\s*[-–—]\s*([^\n(]+)', description, re.MULTILINE)
                if desc_match:
                    band = desc_match.group(1).strip()
                    song = desc_match.group(2).strip()
                    song = re.sub(r'\s*\(.*?\)$', '', song)  # Remove (live), (HD), etc.
                    song = song.strip()
                    print(f"  Found 'Band - Song' in description: Band='{band}', Song='{song}'")
            
            # Method 2: Parse from title if description didn't work
            if not band or not song:
                print("\n--- Method 2: Parse from Title ---")
                if ' - ' in title:
                    parts = title.split(' - ', 1)
                    potential_band = parts[0].strip()
                    potential_song = parts[1].strip()
                    
                    print(f"  Found ' - ' separator")
                    print(f"  Part 1: '{potential_band}'")
                    print(f"  Part 2: '{potential_song}'")
                    
                    if len(potential_band) <= 50 and len(potential_song) <= 100:
                        if any(word in potential_song.lower() for word in ['official', 'lyrics', 'video', 'hq', 'hd', '4k']):
                            band = potential_band
                            song = potential_song
                            print(f"  → Detected indicators in part 2, using: Band='{band}', Song='{song}'")
                        elif len(potential_band) < 15:
                            band = potential_band
                            song = potential_song
                            print(f"  → Part 1 is short, using: Band='{band}', Song='{song}'")
                        else:
                            band = potential_band
                            song = potential_song
                            print(f"  → Default: Band='{band}', Song='{song}'")
                    else:
                        band = potential_band
                        song = potential_song
                        print(f"  → Using: Band='{band}', Song='{song}'")
                
                elif ' by ' in title.lower():
                    parts = title.lower().split(' by ', 1)
                    song = parts[0].strip()
                    band = parts[1].strip()
                    print(f"  Found ' by ' pattern: Band='{band}', Song='{song}'")
                
                elif ' (' in title and title.endswith(')'):
                    idx = title.rfind(' (')
                    paren_content = title[idx+2:-1].strip().lower()
                    quality_indicators = ['hd', '4k', '8k', 'live', 'official', 'lyrics', 'video', 'audio', 'hq', 'remastered', 'remaster']
                    if paren_content not in quality_indicators and len(paren_content) > 2:
                        song = title[:idx].strip()
                        band = title[idx+2:-1].strip()
                        print(f"  Found ' (Band)' pattern: Band='{band}', Song='{song}'")
                    else:
                        print(f"  Skipped ' ({title[idx+2:-1]})' - looks like quality indicator")
                
                # Handle Japanese brackets
                elif '「' in title and '」' in title:
                    jp_match = re.search(r'「([^」]+)」', title)
                    if jp_match:
                        band = jp_match.group(1).strip()
                        song = re.sub(r'「[^」]+」', '', title).strip()
                        song = re.sub(r'\s*\(.*?\)$', '', song)
                        song = re.sub(r'\s*lyrics.*$', '', song, flags=re.IGNORECASE)
                        song = song.strip()
                        print(f"  Found Japanese bracket pattern: Band='{band}', Song='{song}'")
            
            elif ':' in title and title.count(':') == 1:
                parts = title.split(':', 1)
                band = parts[0].strip()
                song = parts[1].strip()
                print(f"  Found ':' separator: Band='{band}', Song='{song}'")
            
            else:
                print(f"  No pattern matched in title")
                song = title
                uploader_lower = uploader.lower()
                if not any(suffix in uploader_lower for suffix in ['topic', 'vevo', 'official', 'channel', 'music', 'records', 'label']):
                    band = uploader
                    print(f"  → Using uploader as band: Band='{band}', Song='{song}'")
                else:
                    words = title.split()
                    if len(words) > 1:
                        band = ' '.join(words[:min(3, len(words))])
                        print(f"  → Extracted from title: Band='{band}', Song='{song}'")
                    else:
                        band = 'Unknown Artist'
                        print(f"  → Fallback: Band='{band}', Song='{song}'")
            
            # Clean up
            band_original = band
            song_original = song
            
            band = re.sub(r'\s*-\s*Topic$', '', band, flags=re.IGNORECASE)
            band = re.sub(r'\s*VEVO$', '', band, flags=re.IGNORECASE)
            band = re.sub(r'\s*Official.*$', '', band, flags=re.IGNORECASE)
            band = re.sub(r'\s*\[.*?\]$', '', band)
            band = re.sub(r'\s*\(.*?\)$', '', band)
            band = band.strip()
            
            song = re.sub(r'\s*\[.*?\]$', '', song)
            song = re.sub(r'\s*\(.*?\)$', '', song)
            song = re.sub(r'\s*-\s*Official.*$', '', song, flags=re.IGNORECASE)
            song = song.strip()
            
            print(f"\n  Final Result:")
            print(f"    Band: '{band}' (was: '{band_original}')")
            print(f"    Song: '{song}' (was: '{song_original}')")
            
            # Method 2: Try to extract from description
            print("\n--- Method 2: Extract from Description ---")
            if description:
                # Look for common patterns in description
                desc_lower = description.lower()
                print(f"  Description length: {len(description)} chars")
                print(f"  First 200 chars: {description[:200]}...")
                
                # Check if description mentions artist/song
                artist_patterns = [
                    r'artist[:\s]+([^\n]+)',
                    r'performed by[:\s]+([^\n]+)',
                    r'by[:\s]+([^\n]+)',
                ]
                for pattern in artist_patterns:
                    match = re.search(pattern, description, re.IGNORECASE)
                    if match:
                        print(f"  Found artist pattern '{pattern}': {match.group(1).strip()}")
            
            # Method 3: Check tags
            print("\n--- Method 3: Check Tags ---")
            tags = info.get('tags', [])
            if tags:
                print(f"  Tags: {tags[:10]}")  # First 10 tags
                # Look for artist name in tags
                if band:
                    band_words = band.lower().split()
                    for tag in tags:
                        tag_lower = tag.lower()
                        if any(word in tag_lower for word in band_words if len(word) > 3):
                            print(f"  → Tag '{tag}' might relate to band '{band}'")
            
            print("\n" + "=" * 80)
            print("SUMMARY:")
            print("=" * 80)
            print(f"Extracted Band: '{band}'")
            print(f"Extracted Song: '{song}'")
            print(f"Title: '{title}'")
            print(f"Uploader: '{uploader}'")
            if artist:
                print(f"Artist (metadata): '{artist}'")
            if creator:
                print(f"Creator (metadata): '{creator}'")
            if channel:
                print(f"Channel (metadata): '{channel}'")
            
            return {
                'band': band,
                'song': song,
                'title': title,
                'uploader': uploader,
                'artist': artist,
                'creator': creator,
                'channel': channel,
                'all_metadata': info
            }
            
    except Exception as e:
        print(f"\n❌ Error extracting metadata: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_url = "https://youtu.be/NI80xDyuVJ4?si=knKmZPzkbG49p8BN"
    result = test_youtube_metadata(test_url)
    
    if result:
        print("\n" + "=" * 80)
        print("✅ Test completed successfully!")
        print("=" * 80)

