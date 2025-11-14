#!/bin/bash
# Script to fix diffusers/transformers import issues

echo "🔧 Fixing diffusers and transformers dependencies..."
echo ""

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "✓ Using uv (faster)"
    INSTALLER="uv pip"
else
    echo "⚠️  Using pip (consider installing uv for faster installs)"
    INSTALLER="pip"
fi

echo ""
echo "Step 1: Uninstalling old versions..."
$INSTALLER uninstall -y transformers diffusers accelerate 2>/dev/null || true

echo ""
echo "Step 2: Installing compatible versions..."
$INSTALLER install --upgrade --force-reinstall \
    "torchvision>=0.15.0" \
    "transformers>=4.30.0,<5.0.0" \
    "diffusers>=0.21.0" \
    "accelerate>=0.20.0"

echo ""
echo "Step 3: Verifying installation..."
python3 -c "
try:
    from transformers import AutoImageProcessor
    print('✓ transformers imported successfully')
except ImportError as e:
    print(f'✗ transformers import failed: {e}')
    exit(1)

try:
    from diffusers import DiffusionPipeline
    print('✓ diffusers imported successfully')
except ImportError as e:
    print(f'✗ diffusers import failed: {e}')
    exit(1)

print('')
print('✅ All dependencies installed correctly!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dependencies fixed! You can now run the application."
else
    echo ""
    echo "❌ Installation failed. Try running manually:"
    echo "   $INSTALLER install --upgrade --force-reinstall \"transformers>=4.30.0,<5.0.0\" \"diffusers>=0.21.0\""
fi

