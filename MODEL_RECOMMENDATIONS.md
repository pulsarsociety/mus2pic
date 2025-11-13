# Image Generation Models for M1 MacBook Air

## Quick Comparison

| Model | Speed | Quality | Memory | Best For |
|-------|-------|---------|--------|----------|
| **LCM** | ⭐⭐⭐⭐⭐ (4-8 steps) | ⭐⭐⭐⭐ | 4-5GB | **Speed priority** ⚡ |
| **SD 1.4** | ⭐⭐⭐ (20 steps) | ⭐⭐⭐⭐ | 3.5-4GB | **8GB RAM** 💾 |
| **SD 1.5** (current) | ⭐⭐⭐ (20-30 steps) | ⭐⭐⭐⭐ | 4-5GB | Balance ⚖️ |
| **SD 2.1** | ⭐⭐⭐ (20 steps) | ⭐⭐⭐⭐⭐ | 4-5GB | **Quality priority** 🎨 |
| **SDXL Turbo** | ⭐⭐⭐⭐⭐ (1-4 steps) | ⭐⭐⭐⭐⭐ | 6-8GB | 16GB+ RAM 🚀 |

## Recommendations by Use Case

### 🚀 **Want Speed? → Use LCM**
- **4-8 steps** vs 20-30 for regular SD
- ~1 second per image on M1
- Similar quality to SD 1.5
- **Install**: Already included in `model_options.py`

**Usage:**
```python
from model_options import generate_with_lcm

image_path = generate_with_lcm(prompt, negative_prompt, num_inference_steps=4)
```

### 💾 **Have 8GB RAM? → Use SD 1.4**
- Smallest model (~3.5GB)
- Good quality
- Standard speed

**Usage:**
```python
from model_options import generate_with_sd14

image_path = generate_with_sd14(prompt, negative_prompt)
```

### 🎨 **Want Best Quality? → Use SD 2.1**
- Better prompt understanding
- Excellent quality
- Works well on 16GB M1 Air

**Usage:**
```python
from model_options import generate_with_sd21

image_path = generate_with_sd21(prompt, negative_prompt)
```

### ⚡ **Have 16GB+ RAM? → Use SDXL Turbo**
- Best quality + speed combo
- Can generate in 1-4 steps
- Needs 6-8GB RAM

**Usage:**
```python
from model_options import generate_with_turbo

image_path = generate_with_turbo(prompt, negative_prompt, num_inference_steps=1)
```

## How to Switch Models in test.py

### Option 1: Import and use directly
```python
# At the top of test.py
from model_options import generate_with_lcm

# In main() function, replace:
# image_path = generate_image(prompt, negative_prompt)
# With:
image_path = generate_with_lcm(prompt, negative_prompt, num_inference_steps=4)
```

### Option 2: Modify generate_image function
Edit the `generate_image` function in `test.py` to support model switching, or just replace the function call.

## Installation Requirements

All models use the same dependencies:
```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install Pillow
```

## Performance Benchmarks (M1 Air)

- **LCM**: ~1-2 seconds per image (4 steps)
- **SD 1.4/1.5**: ~30-60 seconds per image (20 steps)
- **SD 2.1**: ~40-70 seconds per image (20 steps)
- **SDXL Turbo**: ~5-10 seconds per image (1 step) - if you have enough RAM

## Memory Usage Tips

1. **Close other apps** when generating images
2. **Use float16** (already enabled for MPS)
3. **Reduce image size** if you get OOM errors (512x512 is good)
4. **Lower inference steps** for faster generation
5. **Use SD 1.4 or LCM** if you have 8GB RAM

## My Recommendation for You

Based on M1 MacBook Air:
- **If you have 8GB RAM**: Use **LCM** (fastest) or **SD 1.4** (smallest)
- **If you have 16GB RAM**: Use **LCM** for speed or **SD 2.1** for quality
- **Current setup (SD 1.5)**: Good balance, but LCM would be 5-10x faster!

Try LCM first - it's the best upgrade for speed without sacrificing quality!

