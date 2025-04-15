import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Custom LSTM class to ignore the "time_major" keyword argument
from tensorflow.keras.layers import LSTM as BaseLSTM
class CustomLSTM(BaseLSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super(CustomLSTM, self).__init__(*args, **kwargs)

# Use streamlit caching to avoid reloading every time
@st.cache(allow_output_mutation=True)
def load_feature_extractor():
    base_model = MobileNetV2(weights="imagenet")
    # Use the second last layer's output as features
    extractor = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return extractor

@st.cache(allow_output_mutation=True)
def load_caption_model():
    try:
        model = tf.keras.models.load_model('mymodel.h5', custom_objects={'LSTM': CustomLSTM})
        return model
    except Exception as e:
        st.error("Error loading caption model: " + str(e))
        return None

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except Exception as e:
        st.error("Error loading tokenizer: " + str(e))
        return None

def word_from_index(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

def generate_caption(model, image_features, tokenizer, max_length=34):
    caption = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        next_index = np.argmax(yhat)
        next_word = word_from_index(next_index, tokenizer)
        if next_word is None:
            break
        caption += " " + next_word
        if next_word == "endseq":
            break
    return caption.replace("startseq", "").replace("endseq", "").strip()

# Streamlit page settings
st.set_page_config(page_title="Smart Caption Generator", page_icon="📸")
st.title("Smart Image Captioning App")
st.markdown("Upload an image and let our model generate a caption!")

# Image uploader widget
uploaded_image = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Processing image..."):
        # Load and preprocess image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Load models/tokenizer
        feature_extractor = load_feature_extractor()
        caption_model = load_caption_model()
        tokenizer = load_tokenizer()

        if caption_model and tokenizer:
            features = feature_extractor.predict(image, verbose=0)
            caption_text = generate_caption(caption_model, features, tokenizer)
        else:
            caption_text = "Model or tokenizer unavailable."

    # Custom styled output
    st.markdown(
        f'<div style="border-left: 4px solid #007ACC; padding: 10px; margin-top: 20px;">'
        f'<h3>Generated Caption:</h3>'
        f'<p style="font-size: 18px; color: #333;">{caption_text}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
