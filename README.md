# Image Captioning Project

## Overview
This project implements an end-to-end image captioning system that combines convolutional neural networks (CNNs) for image feature extraction and recurrent neural networks (RNNs) with an attention mechanism for generating descriptive captions. There are two main components in this project:

- **Web Application (`app.py`):**  
  A Streamlit-based application that allows users to upload an image and receive a generated caption in real time.

- **Training Notebook (`image-captioner.ipynb`):**  
  A comprehensive Jupyter Notebook that covers data preprocessing, model training, evaluation (using BLEU scores), and caption generation using the Flickr8k dataset.

## Features
- **Image Feature Extraction:**  
  - *Web App:* Utilizes MobileNetV2 for fast and efficient feature extraction.
  - *Training Notebook:* Uses VGG16 to extract rich and robust image features.

- **Caption Generation:**  
  A deep learning model based on LSTM with a custom attention mechanism generates human-like captions.

- **Interactive Interface:**  
  The web app provides an intuitive interface to upload images and display the generated captions.

- **Model Training and Evaluation:**  
  The notebook includes modules for tokenizing captions, splitting data into training and testing sets, and evaluating the model’s performance using BLEU scores.

## Requirements
- **Python Version:** Python 3.7 or higher
- **Key Libraries:**  
  - TensorFlow (with Keras)
  - NumPy
  - Streamlit
  - Pickle
  - NLTK (for BLEU score evaluation)
  - Matplotlib (for displaying sample images)
- **Dataset:** Flickr8k  
  Make sure your dataset includes:
  - An `Images` folder with the images.
  - A `captions.txt` file containing the captions.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/image-captioning.git
    cd image-captioning
    ```

2. **Create a Virtual Environment and Install Dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Download the Flickr8k Dataset:**  
   Ensure that the dataset is organized with an `Images` folder and a `captions.txt` file.  
   If needed, adjust the `INPUT_DIR` and `OUTPUT_DIR` paths in both `app.py` and `image-captioner.ipynb`.

## Project Structure

```plaintext
image-captioning/
├── app.py                   # Streamlit web application for caption generation
├── image-captioner.ipynb    # Jupyter Notebook for training and evaluation
├── mymodel.h5               # Trained captioning model (generated after training)
├── tokenizer.pkl            # Tokenizer file for text processing (generated after training)
├── requirements.txt         # List of Python dependencies
└── README.md                # Project documentation
