# Sticky Note Detector

A Python script that automatically detects and optionally transcribes sticky notes from images using computer vision and OCR.

## Features

- Detects sticky notes in multiple colors (yellow, pink, orange, blue, green, light variations)
- Optional OCR transcription using Tesseract
- Visual overlay showing detected notes
- JSON output for programmatic access
- Debug mode for troubleshooting detection issues

## Installation

1. Install required packages:
```bash
pip install opencv-python numpy pytesseract pillow
```

2. Install Tesseract OCR (only needed if using transcription):
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt-get install tesseract-ocr`
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### Basic Detection (No Transcription)
```bash
python sticky_note_detector.py /path/to/image.jpg
```

### Detection with OCR Transcription
```bash
python sticky_note_detector.py /path/to/image.jpg --transcribe
```

### Debug Mode
```bash
python sticky_note_detector.py /path/to/image.jpg --debug
```

### All Options
```bash
python sticky_note_detector.py /path/to/image.jpg --transcribe --debug --tesseract-path /custom/path/to/tesseract
```

## Output

The script creates a timestamped output directory containing:

- `overlay_result.jpg` - Original image with detected sticky notes highlighted
- `detection_results.json` - Machine-readable detection results
- `summary.txt` - Human-readable summary of detections
- `cropped_note_X.jpg` - Individual sticky note crops (only with --transcribe)
- `cropped_note_X_preprocessed.jpg` - Preprocessed images for OCR (only with --transcribe)

## Detection Parameters

The detector uses:
- Color-based detection in HSV space
- Shape validation (area, aspect ratio, extent, vertices)
- Confidence scoring based on multiple validation criteria
- Non-aggressive clustering to prevent over-merging of nearby notes

## Troubleshooting

If not detecting enough sticky notes:
1. Use `--debug` to see what's being filtered out
2. Check lighting conditions in the image
3. Ensure sticky notes have good contrast with background
4. Consider adjusting the validation parameters in the code

## Example

```bash
# Quick detection without OCR
python sticky_note_detector.py photo.jpg

# Full processing with text extraction
python sticky_note_detector.py photo.jpg --transcribe

# Debug mode to see filtering details
python sticky_note_detector.py photo.jpg --debug
``` 