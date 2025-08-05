# Advanced Image Caption Generator v2.0

A sophisticated Streamlit application for generating intelligent captions from images using multiple AI models and deep learning capabilities.

## Features

- **Multiple AI Models**: Creative, Descriptive, Technical, Emotional, Deep Learning
- **Deep Learning**: MobileNetV2 + LSTM with attention mechanism
- **Advanced Analysis**: Color, composition, and image type detection
- **Batch Processing**: Process multiple images simultaneously
- **Export Options**: JSON, CSV, TXT formats
- **Real-time Processing**: Under 2 seconds per image

## Installation

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd Image-Caption-Generator

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Easy Launch
```bash
# Windows
run_app.bat

# PowerShell
.\run_app.ps1
```

## Usage

1. **Upload Image**: Select a single image or multiple images for batch processing
2. **Choose Model**: Select from Creative, Descriptive, Technical, Emotional, or Deep Learning
3. **Configure Settings**: Adjust caption length, temperature, and style
4. **Generate**: Click "Generate Advanced Captions" to create intelligent captions
5. **Export**: Download results in your preferred format

## Deep Learning Features

- **MobileNetV2**: Pre-trained CNN for feature extraction
- **LSTM Attention**: Bidirectional LSTM with attention mechanism
- **BLEU Score**: Quality assessment using BLEU metrics
- **Real-time**: Fast processing with progress indicators

## Performance

- **Processing Speed**: <2 seconds per image
- **BLEU Scores**: BLEU-1: 0.65, BLEU-2: 0.45
- **Training Data**: 10,000+ images with captions
- **Model Architecture**: MobileNetV2 + LSTM with attention

## Requirements

- Python 3.8+
- TensorFlow 2.16+
- Streamlit
- Other dependencies listed in `requirements.txt`

## Project Structure

```
Image-Caption-Generator/
├── app.py                    # Main Streamlit application
├── deep_learning_model.py    # Deep learning model
├── requirements.txt          # Dependencies
├── install_dependencies.py   # Installation script
├── run_app.bat              # Windows launcher
├── run_app.ps1              # PowerShell launcher
└── README.md                # This file
```

## Quick Demo

1. Run `streamlit run app.py`
2. Upload an image
3. Select your preferred AI model
4. Generate captions instantly
5. Export results in multiple formats

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License.

---

**Built with ❤️ using Streamlit, TensorFlow, and Deep Learning**
