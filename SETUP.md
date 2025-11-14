# Setup Guide

## Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mus2pic.git
   cd mus2pic
   ```

2. **Create and activate virtual environment:**
   
   **Option A: Using `uv` (recommended - 10-100x faster):**
   ```bash
   # Install uv if you don't have it: curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
   
   **Option B: Using standard venv:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   
   **Using `uv` (faster):**
   ```bash
   uv pip install -r requirements.txt
   ```
   
   **Or using pip:**
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

**"Could not import module 'AutoImageProcessor'" or diffusers import errors:**
- This means `transformers` is missing or incompatible
- **Quick fix script:**
  ```bash
  ./fix_dependencies.sh
  ```
- **Manual fix:**
  ```bash
  uv pip install --upgrade --force-reinstall "torchvision>=0.15.0" "transformers>=4.30.0,<5.0.0" "diffusers>=0.21.0"
  # Or with pip:
  pip install --upgrade --force-reinstall "torchvision>=0.15.0" "transformers>=4.30.0,<5.0.0" "diffusers>=0.21.0"
  ```
  
  **Note:** `torchvision` is required for `transformers` to work properly (needed for `AutoImageProcessor`).

**General import errors:**
- Make sure virtual environment is activated
- Run: `uv pip install -r requirements.txt` (or `pip install -r requirements.txt`)
- If issues persist, try: `uv pip install --upgrade -r requirements.txt`

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

