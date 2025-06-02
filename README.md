# Sticky Note Detector

A Python script that automatically detects and optionally transcribes sticky notes from images using computer vision and OCR or LLM-based transcription.

## Features

- Detects sticky notes in multiple colors:
  - Standard: yellow, orange, blue, green, purple, red (includes pink), cyan (broad range)
  - Light variants: light yellow, light blue, light green, light red
- Multiple transcription options:
  - Local OCR using Tesseract
  - LLM-based transcription using Claude or ChatGPT (better for handwriting)
- Visual overlay showing detected notes
- JSON output for programmatic access
- Debug mode for troubleshooting detection issues
- Intelligent size filtering to remove false positives
- **Environment variable configuration** for easy customization

## Installation

1. Install required packages:◊
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR (only needed if using local OCR):
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt-get install tesseract-ocr`
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

3. For LLM transcription, get an API key:
   - Claude: Get an API key from [Anthropic](https://www.anthropic.com/)
   - ChatGPT: Get an API key from [OpenAI](https://platform.openai.com/)

## Configuration

### Environment Variables

Copy `config.env.example` to `.env` and customize the settings:

```bash
cp config.env.example .env
```

Key configuration options:

**API Keys:**
```bash
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Detection Parameters:**
```bash
DEBUG_MODE=false
MIN_CONFIDENCE=40
OVERLAP_THRESHOLD=0.3
SIZE_RATIO_THRESHOLD=0.1
```

**Color Ranges (HSV format):**
```bash
YELLOW_LOWER=22,100,120
YELLOW_UPPER=30,255,255
# ... (see config.env.example for all colors)
```

**Shape Validation:**
```bash
MIN_AREA_RATIO=0.00002
MAX_AREA_RATIO=0.25
MIN_ASPECT_RATIO=0.2
MAX_ASPECT_RATIO=5.0
```

### Command Line Override

Command line arguments take precedence over environment variables:

```bash
# Override debug mode
python sticky_note_detector.py image.jpg --debug

# Override API key
python sticky_note_detector.py image.jpg --llm claude --api-key "your-key"

# Override tesseract path
python sticky_note_detector.py image.jpg --tesseract-path /custom/path
```

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

### Detection with Ollama (local LLM - privacy-focused, cost-free)
```bash
# First, install and start Ollama
# Visit https://ollama.ai for installation instructions

# Pull a vision model (one-time setup)
ollama pull llava

# Run detection with local LLM
python sticky_note_detector.py /path/to/image.jpg --llm ollama

# Use a different model
python sticky_note_detector.py /path/to/image.jpg --llm ollama --ollama-model llava:13b

# Configure custom Ollama URL (if not using default localhost:11434)
export OLLAMA_URL=http://your-server:11434
python sticky_note_detector.py /path/to/image.jpg --llm ollama
```

### Debug Mode
```bash
python sticky_note_detector.py photo.jpg --debug

# Open the HTML report in your browser for easy review
open sticky_notes_output_*/detection_report.html
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
- `detection_report.html` - **Interactive HTML report with table view of all notes**
- `cropped_note_X.jpg` - Individual sticky note crops (only with transcription)
- `cropped_note_X_preprocessed.jpg` - Preprocessed images for OCR (only with local OCR)

**When using the HTML editor:**
- `edited_detection_results.json` - Filtered results with metadata about deletions and edits
- `sticky_notes_transcription.md` - Clean markdown list of transcribed text

### HTML Report Features

The HTML report provides an interactive table view with:
- **📸 Larger Images**: Enhanced 250x250px thumbnails for better readability
- **🔍 Foldable Technical Details**: Color, bounding box, area, and confidence data hidden by default
  - Click "📊 Technical Details" to expand/collapse technical information
  - Keeps the interface clean while preserving access to all data
- **✂️ Break Note Functionality**: Split detections that contain multiple notes
  - Click "✂️ Break Note" to add sub-rows for additional text entries
  - Each sub-note can be edited independently
  - Sub-notes are included in exports with proper numbering (e.g., "Note #5.sub1")
  - Dismiss unwanted sub-notes with "✖️ Dismiss" button
- **🖼️ Full Image Overlay**: View the complete detection results
  - Click "🖼️ View Full Image" to see the overlay_result.jpg in a modal
  - Shows all detected notes highlighted on the original image
  - Close with Escape key or click outside the image
- **✨ Enhanced Editing Capabilities**:
  - **Edit Mode**: Toggle between view and edit modes
  - **Text Editing**: Click to edit transcribed text directly in the browser
  - **Save Edits**: Download edited results as JSON file with sub-note support
  - **Export Markdown**: Download transcriptions as a numbered markdown list
  - **🗑️ Row Deletion**: Mark false positives for removal from final results
    - Click delete button to mark/unmark rows as false positives
    - Deleted rows are visually crossed out and excluded from exports
    - Summary automatically updates to show active vs deleted counts
    - Restore functionality to undo deletions
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🎨 Modern UI**: Clean styling with hover effects and visual feedback

Simply open `detection_report.html` in any web browser to review all detected notes in a user-friendly format.

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
- **✨ NEW: Built-in rate limiting and retry logic**
  - Automatic retry with exponential backoff on rate limit errors
  - Intelligent delay parsing from API error messages
  - 500ms delay between API calls to prevent rate limits
  - Up to 3 retry attempts for failed requests

### Local LLM Transcription (Ollama)
- **🔒 Privacy-focused**: All processing happens locally
- **💰 Cost-free**: No API charges after initial setup
- **🚀 Fast**: No network latency once model is loaded
- **📱 Offline**: Works without internet connection
- Excellent for handwritten text (with vision models like LLaVA)
- Requires Ollama installation and model download
- Uses more local compute resources
- Model quality depends on chosen model size

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

If Ollama transcription fails:
1. Ensure Ollama is installed and running: `ollama serve`
2. Check if the model is available: `ollama list`
3. Pull the model if missing: `ollama pull llava`
4. Verify Ollama is accessible: `curl http://localhost:11434/api/tags`
5. Check if you have sufficient RAM for the model
6. Try a smaller model if needed: `--ollama-model llava:7b`

If you encounter rate limit errors:
1. The script automatically retries with delays - just wait
2. For heavy usage, consider upgrading your API plan
3. Use local OCR for large batches to avoid API costs
4. Process images in smaller batches if needed
5. **Consider using Ollama for unlimited local processing**

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

# Detection with Ollama (local LLM - privacy-focused, cost-free)
ollama pull llava  # One-time setup
python sticky_note_detector.py photo.jpg --llm ollama

# Debug mode to see filtering details
python sticky_note_detector.py photo.jpg --debug
```