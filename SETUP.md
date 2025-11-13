# Setup Guide

## Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mus2pic.git
   cd mus2pic
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (optional but recommended):**
   ```bash
   cp .env.example .env
   # Edit .env and add your Spotify API credentials
   ```

## Spotify API Setup (Optional)

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Copy your Client ID and Client Secret
4. Add them to your `.env` file:
   ```env
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   ```

**Note:** Without Spotify credentials, the app will use demo credentials (rate-limited) or fall back to metadata/inference-based genre detection.

## Running the Application

### Development Mode (with auto-reload):
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 9090
```

### Production Mode:
```bash
python app.py
```

Then open: **http://localhost:9090**

## First Run

On first run, models will be downloaded automatically from Hugging Face:
- **LCM**: ~10GB (first model, recommended)
- **SDXL Lightning**: ~14GB (base SDXL + checkpoint)
- **Other models**: Varies by model

Models are cached in `~/.cache/huggingface/hub/` and won't be re-downloaded.

## Troubleshooting

### Import Errors
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

### Out of Memory
- Start with LCM model (smallest)
- Reduce `MAX_CACHED_MODELS` in `model_options.py`
- Close other applications

### Model Download Issues
- Check internet connection
- Models download on first use
- Can take 10-30 minutes depending on connection

### Port Already in Use
- Change port: `uvicorn app:app --port 8000`
- Or kill process using port 9090

## Next Steps

- Read [README.md](README.md) for usage instructions
- Check [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
- Explore the codebase and customize for your needs!

