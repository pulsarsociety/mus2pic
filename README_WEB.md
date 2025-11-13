# Mus2Pic Web Application

A beautiful, minimalistic web interface for generating music-inspired posters from YouTube videos.

## Features

- 🎵 **YouTube URL Input** - Paste any YouTube URL
- 🎨 **Auto-Generated Prompts** - AI analyzes the music and creates a prompt
- ✏️ **Editable Prompts** - Customize the prompt before generating
- 🖼️ **Fast Image Generation** - Uses LCM model for quick results (4-8 steps)
- 📊 **Audio Features Display** - See tempo, brightness, and energy metrics

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have all the required packages (torch, diffusers, librosa, etc.)

## Running the Application

1. Start the FastAPI server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. **Bonus**: Check out automatic API documentation at:
```
http://localhost:5000/docs
```

## Usage

1. **Enter YouTube URL**: Paste a YouTube video URL in the input field
2. **Generate Prompt**: Click "Generate Prompt from Music" - this will:
   - Download and analyze the audio
   - Extract musical features (tempo, energy, brightness)
   - Generate a prompt based on the music
3. **Edit Prompt** (optional): Modify the prompt or negative prompt to your liking
4. **Generate Image**: Click "Generate Image" to create your poster
5. **View Result**: Your generated poster will appear below

## Tips

- **Inference Steps**: 4-8 steps are recommended for LCM (faster, good quality)
- **Prompt Editing**: Feel free to modify the generated prompt to add your own creative touches
- **Different Music**: Try different genres to see how the prompts change!

## Technical Details

- **Backend**: FastAPI (Python) - Fast, lightweight, async-capable
- **Frontend**: Vanilla HTML/CSS/JavaScript (no frameworks)
- **Model**: LCM (Latent Consistency Model) for fast generation
- **Audio Analysis**: librosa for feature extraction
- **API Docs**: Automatic OpenAPI docs available at `/docs`

## Troubleshooting

- If image generation fails, check that you have enough RAM (4-5GB free)
- Make sure all dependencies are installed correctly
- Check the browser console for any JavaScript errors

Enjoy creating music-inspired art! 🎨🎵

