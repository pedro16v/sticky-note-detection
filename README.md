# Sticky Note Detector

A Python script that automatically detects and optionally transcribes sticky notes from images using computer vision and OCR or LLM-based transcription.

## Features

- Detects sticky notes in multiple colors:
  - Standard: yellow, pink, orange, blue, green, purple, red, cyan
  - Light variants: light yellow, light pink, light blue, light green
- Multiple transcription options:
  - Local OCR using Tesseract
  - LLM-based transcription using Claude or ChatGPT (better for handwriting)
- Visual overlay showing detected notes
- JSON output for programmatic access
- Debug mode for troubleshooting detection issues
- Intelligent size filtering to remove false positives

## Installation

1. Install required packages:
```bash
pip install opencv-python numpy pytesseract pillow requests
```

2. Install Tesseract OCR (only needed if using local OCR):
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt-get install tesseract-ocr`
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

3. For LLM transcription, get an API key:
   - Claude: Get an API key from [Anthropic](https://www.anthropic.com/)
   - ChatGPT: Get an API key from [OpenAI](https://platform.openai.com/)

## Usage

### Basic Detection (No Transcription)
```bash
python sticky_note_detector.py /path/to/image.jpg
```

### Detection with Local OCR Transcription
```bash
python sticky_note_detector.py /path/to/image.jpg --transcribe
```

### Detection with Claude Transcription
```bash
# Using environment variable
export ANTHROPIC_API_KEY="your-api-key"
python sticky_note_detector.py /path/to/image.jpg --llm claude

# Or directly with API key
python sticky_note_detector.py /path/to/image.jpg --llm claude --api-key "your-api-key"
```

### Detection with ChatGPT Transcription
```bash
# Using environment variable
export OPENAI_API_KEY="your-api-key"
python sticky_note_detector.py /path/to/image.jpg --llm chatgpt

# Or directly with API key
python sticky_note_detector.py /path/to/image.jpg --llm chatgpt --api-key "your-api-key"
```

### Debug Mode
```bash
python sticky_note_detector.py /path/to/image.jpg --debug
```

### All Options
```bash
python sticky_note_detector.py /path/to/image.jpg \
    --transcribe \
    --llm claude \
    --api-key "your-key" \
    --debug \
    --tesseract-path /custom/path/to/tesseract
```

## Output

The script creates a timestamped output directory containing:

- `overlay_result.jpg` - Original image with detected sticky notes highlighted
- `detection_results.json` - Machine-readable detection results
- `summary.txt` - Human-readable summary of detections
- `cropped_note_X.jpg` - Individual sticky note crops (only with transcription)
- `cropped_note_X_preprocessed.jpg` - Preprocessed images for OCR (only with local OCR)

## Transcription Methods

### Local OCR (Tesseract)
- Free and runs locally
- Good for printed text
- May struggle with handwriting
- Requires Tesseract installation

### LLM Transcription (Claude/ChatGPT)
- Excellent for handwritten text
- Better context understanding
- Requires API key and internet connection
- Costs per API call

## Detection Parameters

The detector uses:
- Color-based detection in HSV space
- Shape validation (area, aspect ratio, extent, vertices)
- Confidence scoring based on multiple validation criteria
- Non-aggressive clustering to prevent over-merging of nearby notes
- Relative size filtering to remove unreasonably small false positives

### Size Filtering
The detector automatically filters out detections that are too small compared to other detected notes:
- Removes detections smaller than 10% of the median size
- Removes detections smaller than 70% of the 30th percentile size
- Ensures all detections meet an absolute minimum size threshold

This prevents false positives from tiny colored regions while preserving legitimate sticky notes of varying sizes.

## Troubleshooting

If not detecting enough sticky notes:
1. Use `--debug` to see what's being filtered out
2. Check lighting conditions in the image
3. Ensure sticky notes have good contrast with background
4. Consider adjusting the validation parameters in the code

If getting too many false positives:
1. The size filter should remove most tiny false detections automatically
2. Check if non-sticky note objects match the color ranges
3. Consider adjusting color ranges for your specific sticky notes

If LLM transcription fails:
1. Check your API key is valid
2. Ensure you have internet connectivity
3. Check API rate limits or quotas
4. Try the other LLM provider

## Example

```bash
# Quick detection without transcription
python sticky_note_detector.py photo.jpg

# Detection with local OCR
python sticky_note_detector.py photo.jpg --transcribe

# Detection with Claude (best for handwriting)
export ANTHROPIC_API_KEY="your-key"
python sticky_note_detector.py photo.jpg --llm claude

# Detection with ChatGPT
export OPENAI_API_KEY="your-key"
python sticky_note_detector.py photo.jpg --llm chatgpt

# Debug mode to see filtering details
python sticky_note_detector.py photo.jpg --debug
``` 