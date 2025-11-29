"""
LLM-based prompt generator for SDXL Turbo
Uses GPT-4o-mini to generate optimized image prompts from song metadata

No audio analysis needed - just song title, artist, and genre
"""

from typing import Optional
import time

# Get OpenAI client with provided API key
def get_openai_client(api_key: str):
    """Get OpenAI client with the provided API key"""
    if not api_key or not api_key.strip():
        raise ValueError("OpenAI API key is required. Please enter your API key.")
    
    api_key = api_key.strip()
    
    # Basic validation
    if not api_key.startswith("sk-"):
        raise ValueError("Invalid API key format. OpenAI keys start with 'sk-'")
    
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError(
            "OpenAI package not installed. Install with: pip install openai"
        )


# System prompt optimized for SDXL Turbo
# Based on: https://huggingface.co/stabilityai/sdxl-turbo
SDXL_TURBO_SYSTEM_PROMPT = """You create poster art that captures what songs MEAN to fans - their identity, their emotions, their memories.

## CORE PRINCIPLE: FANS USE MUSIC TO EXPRESS WHO THEY ARE

When someone loves a song, they're not thinking about the literal lyrics. They're thinking:
- "This is MY anthem"
- "This song gets me"
- "This is how I feel"
- "This is my vibe/energy"

Your job: Capture that FEELING as visual art they'd proudly hang on their wall.


## CRITICAL - NEVER BE LITERAL:
- "Flowers" = empowerment energy, NOT actual flowers
- "Firework" = inner power exploding, NOT literal fireworks
- "Thunder" = powerful presence, NOT weather
- "Rain" = emotional atmosphere, NOT precipitation

## TECHNICAL (SDXL Turbo):
- Maximum 50 words
- NO text/words
- NO detailed faces (silhouettes work great)
- Strong colors, dramatic lighting

## OUTPUT:
Return ONLY the prompt. No explanations.

masterpiece, best quality"""


def generate_prompt_with_llm(
    song_title: str,
    artist: str,
    api_key: str,
    genre: Optional[str] = None,
    additional_context: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.8,
    max_retries: int = 2
) -> tuple[str, float, dict]:
    """
    Generate SDXL Turbo optimized prompt using GPT-4o-mini.
    
    Args:
        song_title: Name of the song
        artist: Artist/band name
        api_key: OpenAI API key (provided by user)
        genre: Optional genre hint (if known)
        additional_context: Optional extra context (e.g., "acoustic version", "live recording")
        model: OpenAI model to use (default: gpt-4o-mini)
        temperature: Creativity level (0.0-1.0, higher = more creative)
        max_retries: Number of retries on failure
    
    Returns:
        tuple: (prompt, generation_time_seconds, usage_dict)
    
    Raises:
        ValueError: If API key invalid
        Exception: On API errors after retries
    """
    client = get_openai_client(api_key)
    
    # Build user message
    user_message = f'Song: "{song_title}"\nArtist: {artist}'
    if genre:
        user_message += f"\nGenre: {genre}"
    if additional_context:
        user_message += f"\nContext: {additional_context}"
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SDXL_TURBO_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=100,  # Shorter for SDXL Turbo (40-50 words optimal)
                temperature=temperature
            )
            
            generation_time = time.time() - start_time
            
            prompt = response.choices[0].message.content.strip()
            
            # Clean up prompt (remove quotes if LLM added them)
            if prompt.startswith('"') and prompt.endswith('"'):
                prompt = prompt[1:-1]
            if prompt.startswith("'") and prompt.endswith("'"):
                prompt = prompt[1:-1]
            
            # Remove any "Prompt:" prefix the LLM might add
            if prompt.lower().startswith("prompt:"):
                prompt = prompt[7:].strip()
            
            # Ensure prompt ends with quality keywords (SDXL Turbo optimized)
            quality_keywords = ["masterpiece", "best quality", "high quality"]
            has_quality = any(kw in prompt.lower() for kw in quality_keywords)
            if not has_quality:
                prompt = prompt.rstrip(".,") + ", masterpiece, best quality"
            
            # Get usage stats
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "model": model,
                "estimated_cost_usd": _estimate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    model
                )
            }
            
            return prompt, generation_time, usage
            
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1)  # Brief delay before retry
                continue
            raise last_error


def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate API cost in USD"""
    # Pricing as of 2025 (per 1M tokens)
    pricing = {
        # GPT-4o series
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        # GPT-4.1 series (newer, recommended)
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},  # Cheapest option!
        # GPT-5 series
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},  # Ultra cheap!
    }
    
    if model not in pricing:
        model = "gpt-4o-mini"  # Default pricing
    
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]
    
    return round(input_cost + output_cost, 6)


# Available models for prompt generation (cheapest to most expensive)
AVAILABLE_MODELS = {
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "description": "Best balance of cost and quality",
        "cost_per_prompt": "~$0.0003",
        "recommended": True
    },
    "gpt-4.1-nano": {
        "name": "GPT-4.1 Nano", 
        "description": "Cheaper, good for high volume",
        "cost_per_prompt": "~$0.0001",
        "recommended": False
    },
    "gpt-4.1-mini": {
        "name": "GPT-4.1 Mini",
        "description": "Newer model, slightly better quality",
        "cost_per_prompt": "~$0.0005",
        "recommended": False
    },
}


def validate_api_key(api_key: str) -> dict:
    """Validate an OpenAI API key provided by the user"""
    if not api_key or not api_key.strip():
        return {
            "valid": False,
            "message": "Please enter your OpenAI API key"
        }
    
    api_key = api_key.strip()
    
    # Basic format check
    if not api_key.startswith("sk-"):
        return {
            "valid": False,
            "message": "Invalid format. OpenAI API keys start with 'sk-'"
        }
    
    # Mask the key for display
    masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 12 else "sk-***"
    
    # Try a minimal API call to verify the key works
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Use a minimal call to check key validity
        response = client.models.list()
        
        return {
            "valid": True,
            "masked_key": masked_key,
            "message": "API key is valid!"
        }
    except ImportError:
        return {
            "valid": False,
            "message": "OpenAI package not installed. Run: pip install openai"
        }
    except Exception as e:
        error_msg = str(e)
        if "invalid_api_key" in error_msg.lower() or "incorrect api key" in error_msg.lower():
            return {
                "valid": False,
                "masked_key": masked_key,
                "message": "Invalid API key. Check your key at platform.openai.com"
            }
        elif "rate_limit" in error_msg.lower():
            # Rate limited means key is valid
            return {
                "valid": True,
                "masked_key": masked_key,
                "message": "API key is valid! (rate limited, try again in a moment)"
            }
        else:
            return {
                "valid": False,
                "masked_key": masked_key,
                "message": f"Error: {error_msg[:80]}"
            }


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("LLM Prompt Generator - Test Mode")
    print("="*60)
    
    # Get API key from command line or prompt
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = input("\nEnter your OpenAI API key (sk-...): ").strip()
    
    if not api_key:
        print("\n❌ API key is required")
        print("\nUsage: python llm_prompt_generator.py <api_key>")
        print("\nTo get an API key:")
        print("1. Go to https://platform.openai.com/api-keys")
        print("2. Create a new API key")
        sys.exit(1)
    
    # Validate the key
    print("\nValidating API key...")
    status = validate_api_key(api_key)
    
    if not status['valid']:
        print(f"❌ {status['message']}")
        sys.exit(1)
    
    print(f"✅ {status['message']}")
    
    # Test with some songs
    test_songs = [
        ("Blinding Lights", "The Weeknd", "Synthwave/Pop"),
        ("Bohemian Rhapsody", "Queen", "Progressive Rock"),
        ("Lose Yourself", "Eminem", "Hip Hop"),
        ("Take Five", "Dave Brubeck", "Jazz"),
    ]
    
    print("\n" + "-"*60)
    print("Generating prompts for test songs...")
    print("-"*60)
    
    for title, artist, genre in test_songs:
        try:
            prompt, gen_time, usage = generate_prompt_with_llm(title, artist, api_key, genre)
            print(f"\n🎵 {title} - {artist} ({genre})")
            print(f"📝 Prompt: {prompt}")
            print(f"⏱️  Time: {gen_time:.2f}s | Tokens: {usage['total_tokens']} | Cost: ${usage['estimated_cost_usd']:.6f}")
        except Exception as e:
            print(f"\n❌ Error for {title}: {e}")
    
    print("\n" + "="*60)

