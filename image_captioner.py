# %% [markdown]
# # Image Captioning with Attention Mechanism
# 
# This notebook demonstrates how to extract image features, train an attention-based captioning model, and evaluate caption generation on the Flickr8k dataset.

# %%
import os
import pickle
import numpy as np
from math import ceil
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, RepeatVector, Activation, Dot, Lambda, concatenate
from nltk.translate.bleu_score import corpus_bleu

# %% [markdown]
# ## 1. Image Feature Extraction Using VGG16
# 
# We load VGG16 and modify it to output features from its penultimate layer.

# %%
# Load pre-trained VGG16 and modify the network
base_model = VGG16(weights='imagenet')
feature_extractor = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
print("Feature extractor summary:")
print(feature_extractor.summary())

# %% [markdown]
# ## 2. Extract and Save Image Features
# 
# Extract features from the dataset images and save them for later use.
# 
# **Note:** Update `INPUT_DIR` and `OUTPUT_DIR` based on your environment.

# %%
INPUT_DIR = '/kaggle/input/flickr8k'
OUTPUT_DIR = '/kaggle/working'
img_folder = os.path.join(INPUT_DIR, 'Images')

image_features = {}
print("Extracting image features...")
for img_name in tqdm(os.listdir(img_folder)):
    img_path = os.path.join(img_folder, img_name)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    features = feature_extractor.predict(image, verbose=0)
    image_id = os.path.splitext(img_name)[0]
    image_features[image_id] = features

features_path = os.path.join(OUTPUT_DIR, 'vgg_img_features.pkl')
with open(features_path, 'wb') as f:
    pickle.dump(image_features, f)
print("Features saved at:", features_path)

# %% [markdown]
# ## 3. Load and Preprocess Caption Data
# 
# Create a mapping from image IDs to captions. Then clean the captions by lowercasing, removing non-alphabet characters, and adding start/end tokens.

# %%
captions_path = os.path.join(INPUT_DIR, 'captions.txt')
with open(captions_path, 'r') as f:
    next(f)  # Skip header
    captions_text = f.read()

captions_map = defaultdict(list)
for line in captions_text.split('\n'):
    tokens = line.split(',')
    if len(tokens) < 2:
        continue
    img_id = os.path.splitext(tokens[0])[0]
    cap = " ".join(tokens[1:]).strip()
    captions_map[img_id].append(cap)

print("Total captions before cleaning:", sum(len(caps) for caps in captions_map.values()))

# Function to clean captions
def clean_captions(mapping):
    for key, caps in mapping.items():
        for i in range(len(caps)):
            cap = caps[i].lower()
            cap = re.sub(r'[^a-z\s]', '', cap)
            cap = re.sub(r'\s+', ' ', cap).strip()
            caps[i] = 'startseq ' + cap + ' endseq'

# Show an example before cleaning
sample_id = list(captions_map.keys())[0]
print("Example before cleaning:", captions_map[sample_id])
clean_captions(captions_map)
print("Example after cleaning:", captions_map[sample_id])

# %% [markdown]
# ## 4. Tokenize Captions and Determine Max Length
# 
# We convert the cleaned captions to sequences, build a tokenizer, and compute the vocabulary size and maximum caption length.

# %%
all_captions = [cap for caps in captions_map.values() for cap in caps]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(tokenizer.texts_to_sequences([cap])[0]) for cap in all_captions)
print("Vocabulary Size:", vocab_size)
print("Maximum Caption Length:", max_len)

# Save tokenizer for later use
tokenizer_file = os.path.join(OUTPUT_DIR, 'tokenizer.pkl')
with open(tokenizer_file, 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved at:", tokenizer_file)

# %% [markdown]
# ## 5. Train-Test Split
# 
# Split the image IDs into training and testing sets.

# %%
image_ids = list(captions_map.keys())
split = int(len(image_ids) * 0.9)
train_ids = image_ids[:split]
test_ids = image_ids[split:]
print("Training images:", len(train_ids))
print("Testing images:", len(test_ids))

# %% [markdown]
# ## 6. Data Generator for Training
# 
# This generator function yields batches of image features and corresponding caption sequences.
 
# %%
def data_generator(img_ids, cap_map, features, tok, max_length, vocab_sz, batch_size):
    X1_batch, X2_batch, y_batch = [], [], []
    counter = 0
    while True:
        for img_id in img_ids:
            if img_id not in features:
                continue
            for caption in cap_map[img_id]:
                seq = tok.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_sz)[0]
                    X1_batch.append(features[img_id][0])
                    X2_batch.append(in_seq)
                    y_batch.append(out_seq)
                    counter += 1
                    if counter == batch_size:
                        yield ([np.array(X1_batch), np.array(X2_batch)], np.array(y_batch))
                        X1_batch, X2_batch, y_batch = [], [], []
                        counter = 0

# %% [markdown]
# ## 7. Building the Attention-Based Captioning Model
# 
# We build an encoder-decoder model with attention. The encoder processes image features while the decoder handles the caption sequence.

# %%
# Encoder branch for image features
img_input = Input(shape=(4096,))
img_dropout = Dropout(0.5)(img_input)
img_dense = Dense(256, activation='relu')(img_dropout)
img_vector = RepeatVector(max_len)(img_dense)
img_encoded = Bidirectional(LSTM(256, return_sequences=True))(img_vector)

# Decoder branch for text sequences
seq_input = Input(shape=(max_len,))
seq_embedding = Embedding(vocab_size, 256, mask_zero=True)(seq_input)
seq_dropout = Dropout(0.5)(seq_embedding)
seq_encoded = Bidirectional(LSTM(256, return_sequences=True))(seq_dropout)

# Attention mechanism via dot product
attn = Dot(axes=[2, 2])([img_encoded, seq_encoded])
attn = Activation('softmax')(attn)
context = Lambda(lambda x: tf.einsum('ijk,ijl->ikl', x[0], x[1]))([attn, seq_encoded])
context_vector = tf.reduce_sum(context, axis=1)

# Combine context with image dense features
decoder_input = concatenate([context_vector, img_dense])
decoder_dense = Dense(256, activation='relu')(decoder_input)
output = Dense(vocab_size, activation='softmax')(decoder_dense)

# Define and compile model
caption_model = Model(inputs=[img_input, seq_input], outputs=output)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')
caption_model.summary()

# Visualize the model architecture
plot_model(caption_model, show_shapes=True)

# %% [markdown]
# ## 8. Model Training
# 
# Train the model with the data generator.

# %%
epochs = 50
batch_size = 32
steps = ceil(len(train_ids) / batch_size)
val_steps = ceil(len(test_ids) / batch_size)

for ep in range(epochs):
    print(f"Epoch {ep+1}/{epochs}")
    train_gen = data_generator(train_ids, captions_map, image_features, tokenizer, max_len, vocab_size, batch_size)
    val_gen = data_generator(test_ids, captions_map, image_features, tokenizer, max_len, vocab_size, batch_size)
    caption_model.fit(train_gen, epochs=1, steps_per_epoch=steps,
                      validation_data=val_gen, validation_steps=val_steps, verbose=1)

# Save the trained model
model_path = os.path.join(OUTPUT_DIR, 'mymodel.h5')
caption_model.save(model_path)
print("Model saved at:", model_path)

# %% [markdown]
# ## 9. Caption Generation and Evaluation
# 
# Define helper functions for generating captions and evaluate the model using BLEU scores.

# %%
def token_to_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None

def generate_caption(model, image_feat, tok, max_length):
    caption = 'startseq'
    for _ in range(max_length):
        seq = tok.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([image_feat, seq], verbose=0)
        predicted_token = np.argmax(yhat)
        word = token_to_word(predicted_token, tok)
        if word is None:
            break
        caption += " " + word
        if word == 'endseq':
            break
    return caption

# Evaluate on the test set using BLEU scores
actual_caps = []
predicted_caps = []

for img_id in tqdm(test_ids):
    true_caps = captions_map[img_id]
    pred_caption = generate_caption(caption_model, image_features[img_id], tokenizer, max_len)
    actual_caps.append([cap.split() for cap in true_caps])
    predicted_caps.append(pred_caption.split())

print("BLEU-1: %f" % corpus_bleu(actual_caps, predicted_caps, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual_caps, predicted_caps, weights=(0.5, 0.5, 0, 0)))

# %% [markdown]
# ## 10. Generate Captions for Sample Images
# 
# Display the actual captions along with the generated caption for a few sample images.

# %%
from PIL import Image

def display_caption(image_file):
    img_id = os.path.splitext(image_file)[0]
    img_path = os.path.join(INPUT_DIR, "Images", image_file)
    image = Image.open(img_path)
    print("Actual Captions:")
    for cap in captions_map[img_id]:
        print(cap)
    print("\nGenerated Caption:")
    print(generate_caption(caption_model, image_features[img_id], tokenizer, max_len))
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Display captions for some sample images
display_caption("101669240_b2d3e7f17b.jpg")
display_caption("1077546505_a4f6c4daa9.jpg")
display_caption("1002674143_1b742ab4b8.jpg")
display_caption("1032460886_4a598ed535.jpg")
display_caption("1032122270_ea6f0beedb.jpg")
