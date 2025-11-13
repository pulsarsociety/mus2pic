# 🎵 Mus2Pic - Transform Music into Visual Art

A web application that generates AI-powered poster art from YouTube music videos. Analyze audio features, extract genres, and create stunning visual representations of your favorite songs.

![Mus2Pic](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎵 **YouTube Audio Analysis** - Extract audio features (tempo, energy, brightness, valence) from any YouTube video
- 🎨 **AI-Powered Prompt Generation** - Two versions (V1 & V2) that create prompts based on musical characteristics
- 🎭 **Genre Detection** - Automatic genre detection from:
  - Audio file metadata
  - Spotify API (with caching)
  - Audio feature inference
- 🖼️ **Multiple AI Models** - Support for 6 different image generation models:
  - **LCM** (Latent Consistency Model) - Fastest, 4-8 steps
  - **LCM SD2.1** - Fast with better prompt understanding
  - **SDXL Lightning** - Ultra-fast SDXL, 1-4 steps
  - **SDXL Turbo** - Fast SDXL generation
  - **Stable Diffusion 1.4** - Smallest model
  - **Stable Diffusion 2.1** - Best quality
- ⚡ **Optimized for M1/M2 Macs** - Memory-efficient with automatic cache management
- 🌐 **Modern Web Interface** - Real-time progress updates via WebSocket
- 📊 **Audio Features Display** - Visualize tempo, energy, and brightness metrics
- 🎯 **Configurable Analysis** - Choose audio duration (30s, 60s, or full length)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- macOS (M1/M2 recommended) or Linux/Windows
- 8GB+ RAM (16GB recommended for SDXL models)
- ~10GB free disk space for model downloads

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/mus2pic.git
cd mus2pic
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables (optional):**
```bash
cp .env.example .env
# Edit .env and add your Spotify API credentials (optional)
```

### Running the Application

**Option 1: Using uvicorn (recommended for development):**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 9090
```

**Option 2: Using Python directly:**
```bash
python app.py
```

Then open your browser to: **http://localhost:9090**

## 📖 Usage

1. **Enter YouTube URL** - Paste any YouTube video URL
2. **Configure Options** (optional):
   - Audio Duration: 30s, 60s, or Full Length
   - Prompt Version: V1 (classic) or V2 (genre-aware)
   - Model: Choose from 6 available models
   - Inference Steps: Adjust based on model (4-8 for LCM, 20 for SD)
3. **Generate Prompt** - Click "Generate Prompt from Music"
4. **Edit Prompt** (optional) - Customize the generated prompt
5. **Generate Image** - Click "Generate Image" and watch real-time progress
6. **View Result** - Your generated poster appears below

## 🎛️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Spotify API (optional - for genre detection)
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret

# Hugging Face Cache (optional)
HF_HOME=/path/to/cache
```

### Model Configuration

Edit `model_options.py` to adjust:

- `MAX_CACHED_MODELS` - Maximum models in memory (default: 4)
- `ENABLE_CPU_OFFLOAD` - Enable CPU offloading for low VRAM (default: False)
- `ENABLE_ATTENTION_SLICING` - Reduce memory usage (default: True)

## 🧪 Testing

Test YouTube metadata extraction:
```bash
python test_youtube_metadata.py
```

Test model cache management:
```bash
python model_options.py list  # List cached models
python model_options.py clean <model_key>  # Remove model cache
```

## 📁 Project Structure

```
mus2pic/
├── app.py                 # FastAPI web application
├── test.py                # Core audio analysis & prompt generation
├── model_options.py       # Image generation models
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── generated/        # Generated images (gitignored)
│   └── favicon.*         # Website icons
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── README.md            # This file
```

## 🎨 Models Comparison

| Model | Speed | Quality | Memory | Best For |
|-------|-------|---------|--------|----------|
| LCM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 4-5GB | Speed priority |
| LCM SD2.1 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 5-6GB | Better prompts + speed |
| SDXL Lightning | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 7-9GB | Ultra-fast + quality |
| SDXL Turbo | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 6-8GB | Fast + quality |
| SD 1.4 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 3.5-4GB | Low RAM (8GB) |
| SD 2.1 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4-5GB | Quality priority |

## 🔧 Troubleshooting

### Black Images
- Ensure guidance scale is set correctly (1.0 for LCM)
- Try disabling attention slicing: `ENABLE_ATTENTION_SLICING = False`
- Use `torch.no_grad()` instead of `torch.inference_mode()` on MPS

### Out of Memory (OOM)
- Reduce `MAX_CACHED_MODELS` to 2
- Enable CPU offload: `ENABLE_CPU_OFFLOAD = True`
- Use smaller models (SD 1.4 or LCM)
- Close other applications

### Slow Generation
- Use float16 (already enabled for MPS)
- Disable attention slicing for speed
- Use LCM model (fastest)
- Reduce inference steps (4-8 for LCM)

### Model Cache Management
```bash
# List all cached models and sizes
python model_options.py list

# Remove specific model to free space
python model_options.py clean turbo  # Removes SDXL Turbo cache
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - For amazing diffusion models
- [LCM](https://github.com/latent-consistency/lcm) - For fast generation
- [librosa](https://librosa.org/) - For audio analysis
- [FastAPI](https://fastapi.tiangolo.com/) - For the web framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

Made with ❤️ for music and art lovers

