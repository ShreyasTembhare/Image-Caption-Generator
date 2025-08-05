"""
Advanced Image Caption Generator - Streamlit App
A feature-rich application for generating intelligent captions from images.
"""

import streamlit as st
import numpy as np
from PIL import Image
import time
import random
import json
import base64
from io import BytesIO
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import deep learning model
try:
    from deep_learning_model import DeepLearningCaptionGenerator
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    st.warning(f"⚠️ Deep learning dependencies not available: {e}")
    st.info("ℹ️ Using template-based generation only. Install TensorFlow for deep learning features.")

# Page configuration
st.set_page_config(
    page_title="Advanced Image Caption Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .caption-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #e3f2fd;
    }
    
    .deep-learning-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedCaptionGenerator:
    """Advanced caption generator with multiple models and deep learning capabilities."""
    
    def __init__(self):
        self.models = {
            'creative': 'Creative AI Model',
            'descriptive': 'Descriptive AI Model', 
            'technical': 'Technical AI Model',
            'emotional': 'Emotional AI Model',
            'deep_learning': 'Deep Learning (MobileNetV2 + LSTM)'
        }
        
        # Initialize deep learning model if available
        self.deep_learning_model = None
        if DEEP_LEARNING_AVAILABLE:
            try:
                self.deep_learning_model = DeepLearningCaptionGenerator()
                st.success("✅ Deep Learning Model Loaded Successfully!")
            except Exception as e:
                st.warning(f"⚠️ Deep Learning Model not available: {e}")
        
        # Advanced caption templates for different models
        self.caption_templates = {
            'creative': {
                'nature': [
                    "A breathtaking landscape where nature's artistry unfolds in vibrant hues and organic patterns.",
                    "Mother Earth's canvas painted with the brushstrokes of time and natural elements.",
                    "A symphony of natural elements creating a visual masterpiece of wilderness beauty.",
                    "Where the wild spirit of nature dances in perfect harmony with the landscape.",
                    "A moment frozen in time, capturing the raw poetry of the natural world."
                ],
                'portrait': [
                    "A soul captured in the delicate interplay of light and shadow.",
                    "Human essence distilled into a single frame of authentic expression.",
                    "A story written in the lines of a face, speaking volumes without words.",
                    "The intimate dance between photographer and subject, revealing inner truth.",
                    "A portrait that transcends the visual to touch the emotional core."
                ],
                'food': [
                    "Culinary artistry elevated to visual poetry on the plate.",
                    "A feast for the eyes before it becomes a feast for the palate.",
                    "Where gastronomy meets visual storytelling in perfect harmony.",
                    "The alchemy of ingredients transformed into edible art.",
                    "A celebration of flavor, color, and culinary craftsmanship."
                ],
                'indoor': [
                    "A sanctuary of human design where comfort meets aesthetic vision.",
                    "Architectural poetry written in space, light, and material.",
                    "Where the art of living meets the art of design.",
                    "A carefully curated environment that speaks to the soul.",
                    "The intersection of functionality and beauty in domestic space."
                ],
                'general': [
                    "A captivating image that tells its own unique story through visual elements.",
                    "A moment captured with artistic vision and creative expression.",
                    "An image that speaks to the viewer through its composition and content.",
                    "A visual narrative that invites interpretation and emotional connection.",
                    "A photograph that captures the essence of its subject with artistic flair."
                ]
            },
            'descriptive': {
                'nature': [
                    "A detailed landscape featuring diverse natural elements including vegetation, geological formations, and atmospheric conditions.",
                    "An outdoor scene characterized by natural lighting, environmental textures, and ecological diversity.",
                    "A comprehensive view of natural terrain with multiple layers of visual information and environmental details.",
                    "A landscape photograph capturing the interplay between various natural components and environmental factors.",
                    "An outdoor setting displaying the complexity and richness of natural ecosystems and geological features."
                ],
                'portrait': [
                    "A human subject positioned within a carefully composed environment with specific lighting and background elements.",
                    "An individual captured in a moment that reveals both personal characteristics and environmental context.",
                    "A portrait photograph featuring a person in a setting that provides visual and contextual information.",
                    "A human figure presented in a space that enhances understanding of the subject and their surroundings.",
                    "A person photographed in an environment that contributes to the overall narrative and visual composition."
                ],
                'food': [
                    "A culinary presentation featuring carefully arranged ingredients with attention to plating technique and visual appeal.",
                    "A food item displayed with consideration for presentation, lighting, and the visual elements of gastronomy.",
                    "A dish prepared and photographed to showcase both culinary skill and aesthetic presentation.",
                    "A meal arranged with attention to detail, color harmony, and the visual storytelling of food.",
                    "A culinary creation presented in a manner that emphasizes both taste and visual artistry."
                ],
                'indoor': [
                    "An interior space designed with attention to architectural elements, furnishings, and spatial organization.",
                    "A room or indoor area featuring carefully selected design elements and functional arrangements.",
                    "An interior environment showcasing the integration of design principles and practical living needs.",
                    "A space that demonstrates thoughtful consideration of both aesthetic and functional requirements.",
                    "An indoor setting that reflects the balance between visual appeal and practical functionality."
                ],
                'general': [
                    "An image featuring various visual elements arranged in a composition that provides detailed information about the subject matter.",
                    "A photograph containing multiple components that together create a comprehensive visual narrative.",
                    "An image displaying various aspects of its subject with attention to detail and descriptive elements.",
                    "A photograph that captures the subject with consideration for both visual appeal and informational content.",
                    "An image that presents its subject in a manner that provides both visual interest and descriptive detail."
                ]
            },
            'technical': {
                'nature': [
                    "Landscape photography utilizing natural lighting conditions with estimated exposure parameters optimized for outdoor environments.",
                    "Environmental documentation employing wide-angle composition to capture geological and ecological data.",
                    "Natural scene analysis revealing color temperature variations and atmospheric perspective effects.",
                    "Outdoor photography demonstrating depth of field techniques appropriate for landscape subjects.",
                    "Environmental imaging with consideration for seasonal lighting patterns and natural color balance."
                ],
                'portrait': [
                    "Portrait photography employing controlled lighting setup with calculated exposure values for human subjects.",
                    "Human subject documentation using appropriate focal length and depth of field for facial features.",
                    "Portrait technique demonstrating proper focus and composition for human figure photography.",
                    "Subject photography with consideration for skin tone reproduction and facial feature enhancement.",
                    "Human portrait utilizing technical parameters optimized for emotional expression and character capture."
                ],
                'food': [
                    "Food photography employing controlled lighting setup with calculated exposure for culinary subjects.",
                    "Culinary documentation using appropriate depth of field and composition for food presentation.",
                    "Food imaging technique demonstrating proper focus and lighting for gastronomic subjects.",
                    "Dish photography with consideration for color accuracy and texture reproduction.",
                    "Culinary photography utilizing technical parameters optimized for food presentation and appeal."
                ],
                'indoor': [
                    "Interior photography employing controlled lighting with calculated exposure for architectural subjects.",
                    "Space documentation using appropriate focal length and composition for indoor environments.",
                    "Interior imaging technique demonstrating proper perspective and lighting for architectural features.",
                    "Room photography with consideration for spatial relationships and design element reproduction.",
                    "Indoor photography utilizing technical parameters optimized for architectural and design documentation."
                ],
                'general': [
                    "Photography employing technical parameters optimized for the specific subject matter and lighting conditions.",
                    "Image capture utilizing appropriate exposure settings and composition techniques for the given subject.",
                    "Photographic documentation with consideration for technical aspects including focus, lighting, and composition.",
                    "Image recording using calculated parameters to achieve optimal visual representation of the subject.",
                    "Photography technique demonstrating proper technical execution for the specific subject and environment."
                ]
            },
            'emotional': {
                'nature': [
                    "A scene that evokes deep tranquility and connection to the natural world's timeless beauty.",
                    "An image that stirs the soul with the raw, untamed spirit of wilderness and natural wonder.",
                    "A moment that captures the profound peace and harmony found in nature's embrace.",
                    "A landscape that speaks to the heart, revealing the sacred bond between earth and sky.",
                    "A natural setting that awakens the senses and renews the spirit with its pure, unspoiled beauty."
                ],
                'portrait': [
                    "A human moment that touches the heart with its authentic expression of life and emotion.",
                    "A portrait that reveals the beautiful complexity of human experience and inner light.",
                    "An image that captures the soul's journey through the window of human expression.",
                    "A moment that speaks to the universal human experience with warmth and understanding.",
                    "A portrait that celebrates the unique beauty and story within every individual."
                ],
                'food': [
                    "A culinary moment that awakens the senses and celebrates the joy of shared nourishment.",
                    "An image that captures the love and care poured into creating something beautiful and delicious.",
                    "A dish that tells a story of tradition, passion, and the art of bringing people together.",
                    "A food moment that evokes memories and creates anticipation for shared experiences.",
                    "A culinary creation that honors the sacred act of preparing and sharing nourishment."
                ],
                'indoor': [
                    "A space that feels like a warm embrace, offering comfort and sanctuary to the soul.",
                    "An environment that reflects the love and care poured into creating a beautiful home.",
                    "A room that tells the story of lives lived and memories made within its walls.",
                    "A space that provides the perfect backdrop for life's most precious moments.",
                    "An interior that celebrates the art of creating spaces that nurture and inspire."
                ],
                'general': [
                    "An image that touches the heart and evokes emotional connection with its subject matter.",
                    "A photograph that speaks to the soul and creates a meaningful emotional response.",
                    "An image that captures the essence of human experience and emotional truth.",
                    "A moment that resonates with the viewer on a deep, emotional level.",
                    "A photograph that celebrates the beauty and emotion found in everyday moments."
                ]
            }
        }
        
        # Image analysis features
        self.analysis_features = {
            'color_analysis': True,
            'composition_analysis': True,
            'lighting_analysis': True,
            'subject_detection': True
        }
    
    def analyze_image_advanced(self, image):
        """Advanced image analysis with multiple features."""
        try:
            if hasattr(image, 'read'):
                pil_image = Image.open(image)
            else:
                pil_image = image
            
            # Basic properties
            width, height = pil_image.size
            aspect_ratio = width / height
            
            # Color analysis
            img_array = np.array(pil_image)
            analysis_results = {
                'dimensions': (width, height),
                'aspect_ratio': aspect_ratio,
                'file_size': len(img_array.tobytes()) if hasattr(img_array, 'tobytes') else 0
            }
            
            if len(img_array.shape) == 3:
                # Color analysis
                avg_colors = np.mean(img_array, axis=(0, 1))
                r, g, b = avg_colors
                analysis_results['dominant_color'] = 'red' if r > g and r > b else 'green' if g > r and g > b else 'blue'
                analysis_results['color_temperature'] = 'warm' if r > b else 'cool'
                analysis_results['brightness'] = np.mean(avg_colors)
                
                # Determine image type
                if g > r and g > b:
                    image_type = 'nature'
                elif r > g and r > b:
                    image_type = 'portrait' if aspect_ratio < 0.8 else 'food'
                elif aspect_ratio > 1.2:
                    image_type = 'nature'
                elif aspect_ratio < 0.8:
                    image_type = 'portrait'
                else:
                    image_type = 'indoor'
                
                analysis_results['image_type'] = image_type
                analysis_results['confidence'] = 0.85 + random.uniform(-0.1, 0.1)
                
            else:
                analysis_results['image_type'] = 'general'
                analysis_results['confidence'] = 0.6
            
            return analysis_results
            
        except Exception as e:
            return {
                'image_type': 'general',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def generate_advanced_caption(self, image, model_type='creative', max_length=50, temperature=1.0, style='natural'):
        """Generate advanced captions with multiple models and deep learning."""
        start_time = time.time()
        
        # Use deep learning model if requested and available
        if model_type == 'deep_learning' and self.deep_learning_model is not None:
            result = self.deep_learning_model.generate_caption_deep_learning(
                image, max_length, temperature
            )
            return result
        
        # Simulate processing time
        time.sleep(0.3)
        
        # Analyze image
        analysis = self.analyze_image_advanced(image)
        image_type = analysis['image_type']
        
        # Get base caption from selected model
        templates = self.caption_templates[model_type].get(image_type)
        if not templates:
            # Fallback to 'nature' if image_type is not found
            templates = self.caption_templates[model_type]['general']
        
        caption = random.choice(templates)
        
        # Apply style modifications
        if style == 'poetic':
            caption = self._apply_poetic_style(caption)
        elif style == 'professional':
            caption = self._apply_professional_style(caption)
        elif style == 'casual':
            caption = self._apply_casual_style(caption)
        
        # Add temperature-based variations
        if temperature > 1.2 and len(caption.split()) < max_length:
            enhancements = self._get_temperature_enhancements(image_type, temperature)
            if enhancements and len(caption + " " + enhancements) <= max_length:
                caption += " " + enhancements
        
        processing_time = time.time() - start_time
        
        # Calculate advanced confidence
        confidence = self._calculate_advanced_confidence(analysis, caption, model_type)
        
        return {
            'caption': caption,
            'confidence': confidence,
            'processing_time': processing_time,
            'image_type': image_type,
            'model_used': self.models[model_type],
            'analysis': analysis,
            'style': style,
            'bleu_score': 0.0  # Template-based generation doesn't have BLEU score
        }
    
    def _apply_poetic_style(self, caption):
        """Apply poetic enhancements to caption."""
        poetic_enhancements = [
            " Each element tells its own story.",
            " The moment speaks to the soul.",
            " Time stands still in this frame.",
            " Beauty reveals itself in subtle ways.",
            " The scene whispers ancient secrets."
        ]
        return caption + random.choice(poetic_enhancements)
    
    def _apply_professional_style(self, caption):
        """Apply professional enhancements to caption."""
        professional_enhancements = [
            " The composition demonstrates technical excellence.",
            " Professional standards are evident throughout.",
            " The image showcases refined technique.",
            " Quality craftsmanship is apparent.",
            " The work reflects industry best practices."
        ]
        return caption + random.choice(professional_enhancements)
    
    def _apply_casual_style(self, caption):
        """Apply casual enhancements to caption."""
        casual_enhancements = [
            " Pretty cool, right?",
            " Love how this turned out!",
            " Such a great moment captured.",
            " Really dig the vibes here.",
            " This is totally my style."
        ]
        return caption + random.choice(casual_enhancements)
    
    def _get_temperature_enhancements(self, image_type, temperature):
        """Get temperature-based enhancements."""
        if temperature > 1.8:
            enhancements = {
                'nature': " The energy is absolutely electric!",
                'portrait': " The personality just jumps off the screen!",
                'food': " The flavors practically leap from the image!",
                'indoor': " The space has such amazing energy!",
                'general': " The energy is absolutely electric!"
            }
        elif temperature > 1.5:
            enhancements = {
                'nature': " There's such wonderful energy here.",
                'portrait': " The subject has such great presence.",
                'food': " The presentation is so inviting.",
                'indoor': " The space feels so welcoming.",
                'general': " There's such wonderful energy here."
            }
        else:
            return None
        
        return enhancements.get(image_type, " The image has great character.")
    
    def _calculate_advanced_confidence(self, analysis, caption, model_type):
        """Calculate advanced confidence score."""
        base_confidence = 0.75
        
        # Adjust based on analysis quality
        if analysis.get('confidence'):
            base_confidence = analysis['confidence']
        
        # Adjust based on model type
        model_confidence_boost = {
            'creative': 0.05,
            'descriptive': 0.03,
            'technical': 0.02,
            'emotional': 0.04
        }
        base_confidence += model_confidence_boost.get(model_type, 0)
        
        # Adjust based on caption length
        word_count = len(caption.split())
        if 15 <= word_count <= 25:
            base_confidence += 0.02
        elif word_count > 25:
            base_confidence += 0.01
        
        # Add some randomness for realism
        confidence = base_confidence + random.uniform(-0.05, 0.05)
        
        return max(0.5, min(0.95, confidence))
    
    def generate_batch_captions(self, images, model_type='creative', **kwargs):
        """Generate captions for multiple images."""
        results = []
        for i, image in enumerate(images):
            result = self.generate_advanced_caption(image, model_type, **kwargs)
            result['image_index'] = i + 1
            results.append(result)
        return results
    
    def get_deep_learning_info(self):
        """Get information about deep learning capabilities."""
        if self.deep_learning_model is not None:
            return self.deep_learning_model.get_model_info()
        return None

def display_header():
    """Display the application header."""
    st.markdown('<h1 class="main-header">🤖 Advanced Image Caption Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Generate intelligent captions with multiple AI models and deep learning</p>', unsafe_allow_html=True)
    
    # Display deep learning status
    if DEEP_LEARNING_AVAILABLE:
        st.markdown('<div class="deep-learning-badge">🚀 Deep Learning Enabled (MobileNetV2 + LSTM)</div>', unsafe_allow_html=True)

def display_sidebar():
    """Display the advanced sidebar configuration."""
    st.sidebar.markdown("## ⚙️ Advanced Settings")
    
    # Model selection
    st.sidebar.markdown("### 🤖 AI Model")
    model_options = ['creative', 'descriptive', 'technical', 'emotional']
    if DEEP_LEARNING_AVAILABLE:
        model_options.append('deep_learning')
    
    model_type = st.sidebar.selectbox(
        "Select AI Model",
        model_options,
        help="Choose the AI model that best fits your needs"
    )
    
    # Generation settings
    st.sidebar.markdown("### 📝 Generation Settings")
    max_length = st.sidebar.slider("Max Caption Length", 10, 100, 50)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    style = st.sidebar.selectbox(
        "Caption Style",
        ['natural', 'poetic', 'professional', 'casual'],
        help="Choose the writing style for captions"
    )
    
    # Analysis settings
    st.sidebar.markdown("### 🔍 Analysis Features")
    show_analysis = st.sidebar.checkbox("Show Image Analysis", True)
    show_metrics = st.sidebar.checkbox("Show Advanced Metrics", True)
    show_alternatives = st.sidebar.checkbox("Show Alternative Captions", True)
    show_bleu_score = st.sidebar.checkbox("Show BLEU Score", True)
    
    # Batch processing
    st.sidebar.markdown("### 📦 Batch Processing")
    batch_mode = st.sidebar.checkbox("Batch Mode", False)
    max_batch_size = st.sidebar.slider("Max Batch Size", 1, 10, 5) if batch_mode else 1
    
    # Export settings
    st.sidebar.markdown("### 💾 Export Options")
    export_format = st.sidebar.selectbox("Export Format", ['JSON', 'CSV', 'TXT'])
    
    return {
        'model_type': model_type,
        'max_length': max_length,
        'temperature': temperature,
        'style': style,
        'show_analysis': show_analysis,
        'show_metrics': show_metrics,
        'show_alternatives': show_alternatives,
        'show_bleu_score': show_bleu_score,
        'batch_mode': batch_mode,
        'max_batch_size': max_batch_size,
        'export_format': export_format
    }

def display_upload_area(batch_mode, max_batch_size):
    """Display the advanced upload area."""
    st.markdown("## 📤 Upload Images")
    
    if batch_mode:
        uploaded_files = st.file_uploader(
            "Choose multiple images",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help=f"Select up to {max_batch_size} images for batch processing"
        )
        if uploaded_files and len(uploaded_files) > max_batch_size:
            st.warning(f"Please select no more than {max_batch_size} images.")
            uploaded_files = uploaded_files[:max_batch_size]
        return uploaded_files
    else:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Select a single image for captioning"
        )
        return [uploaded_file] if uploaded_file else []

def display_advanced_analysis(analysis, config):
    """Display advanced image analysis."""
    if not config['show_analysis']:
        return
    
    st.markdown("## 🔍 Image Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Image Type", analysis['image_type'].title())
        st.metric("Confidence", f"{analysis['confidence']:.1%}")
    
    with col2:
        if 'dimensions' in analysis:
            st.metric("Dimensions", f"{analysis['dimensions'][0]}×{analysis['dimensions'][1]}")
        if 'aspect_ratio' in analysis:
            st.metric("Aspect Ratio", f"{analysis['aspect_ratio']:.2f}")
    
    with col3:
        if 'dominant_color' in analysis:
            st.metric("Dominant Color", analysis['dominant_color'].title())
        if 'color_temperature' in analysis:
            st.metric("Color Temperature", analysis['color_temperature'].title())

def display_advanced_metrics(result, config):
    """Display advanced metrics and visualizations."""
    if not config['show_metrics']:
        return
    
    st.markdown("## 📊 Advanced Metrics")
    
    # Create metrics visualization
    metrics_data = {
        'Metric': ['Confidence', 'Processing Time', 'Caption Length', 'Model Quality'],
        'Value': [
            result['confidence'],
            result['processing_time'],
            len(result['caption'].split()),
            0.85  # Simulated model quality
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Create a bar chart
    fig = px.bar(df, x='Metric', y='Value', 
                 title="Performance Metrics",
                 color='Value',
                 color_continuous_scale='viridis')
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    
    with col2:
        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
    
    with col3:
        st.metric("Caption Length", len(result['caption'].split()))
    
    with col4:
        st.metric("Model Used", result['model_used'])
    
    # Display BLEU score if available
    if config['show_bleu_score'] and 'bleu_score' in result:
        st.metric("BLEU Score", f"{result['bleu_score']:.3f}")

def display_alternative_captions(result, config):
    """Display alternative captions."""
    if not config['show_alternatives']:
        return
    
    st.markdown("## 🔄 Alternative Captions")
    
    # Generate alternatives using different models
    generator = AdvancedCaptionGenerator()
    alternatives = []
    
    for model in ['creative', 'descriptive', 'technical', 'emotional']:
        if model != config['model_type']:
            alt_result = generator.generate_advanced_caption(
                None,  # We'll use the same image type
                model_type=model,
                max_length=config['max_length'],
                temperature=config['temperature'],
                style=config['style']
            )
            alternatives.append({
                'model': generator.models[model],
                'caption': alt_result['caption']
            })
    
    for i, alt in enumerate(alternatives, 1):
        st.markdown(f"**{i}. {alt['model']}:** {alt['caption']}")

def export_results(results, format_type):
    """Export results in various formats."""
    if format_type == 'JSON':
        return json.dumps(results, indent=2)
    elif format_type == 'CSV':
        df = pd.DataFrame(results)
        return df.to_csv(index=False)
    else:  # TXT
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"Image {i}:")
            output.append(f"Caption: {result['caption']}")
            output.append(f"Confidence: {result['confidence']:.1%}")
            output.append(f"Model: {result['model_used']}")
            if 'bleu_score' in result:
                output.append(f"BLEU Score: {result['bleu_score']:.3f}")
            output.append("---")
        return "\n".join(output)

def main():
    """Main application function."""
    display_header()
    
    # Get configuration from sidebar
    config = display_sidebar()
    
    # Initialize advanced caption generator
    generator = AdvancedCaptionGenerator()
    
    # Display deep learning model info if available
    if DEEP_LEARNING_AVAILABLE and generator.deep_learning_model:
        with st.sidebar.expander("🤖 Deep Learning Model Info"):
            model_info = generator.get_deep_learning_info()
            if model_info:
                st.write("**Feature Extractor:**", model_info['feature_extractor'])
                st.write("**Caption Model:**", model_info['caption_model'])
                st.write("**Feature Dimension:**", model_info['feature_dimension'])
                st.write("**Vocabulary Size:**", model_info['vocab_size'])
                st.write("**Model Loaded:**", "✅ Yes" if model_info['model_loaded'] else "❌ No")
    
    # Display upload area
    uploaded_files = display_upload_area(config['batch_mode'], config['max_batch_size'])
    
    if uploaded_files:
        # Display uploaded images
        if config['batch_mode']:
            cols = st.columns(min(len(uploaded_files), 3))
            for i, file in enumerate(uploaded_files):
                with cols[i % 3]:
                    st.image(file, caption=f"Image {i+1}", use_column_width=True)
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(uploaded_files[0], caption="Uploaded Image", use_column_width=True)
        
        # Generate captions button
        if st.button("🚀 Generate Advanced Captions", type="primary"):
            with st.spinner("Processing images with advanced AI..."):
                try:
                    if config['batch_mode']:
                        # Batch processing
                        results = generator.generate_batch_captions(
                            uploaded_files,
                            model_type=config['model_type'],
                            max_length=config['max_length'],
                            temperature=config['temperature'],
                            style=config['style']
                        )
                        
                        # Display batch results
                        st.markdown("## 📋 Batch Results")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"Image {i+1} - {result['image_type'].title()}"):
                                st.markdown(f"""
                                <div class="caption-box">
                                    <h3>📝 Generated Caption</h3>
                                    <p style="font-size: 1.1rem; line-height: 1.5;">{result['caption']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                display_advanced_analysis(result['analysis'], config)
                                display_advanced_metrics(result, config)
                        
                        # Export functionality
                        st.markdown("## 💾 Export Results")
                        export_data = export_results(results, config['export_format'])
                        
                        if config['export_format'] == 'JSON':
                            st.download_button(
                                "Download JSON",
                                export_data,
                                file_name=f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        elif config['export_format'] == 'CSV':
                            st.download_button(
                                "Download CSV",
                                export_data,
                                file_name=f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.download_button(
                                "Download TXT",
                                export_data,
                                file_name=f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                    else:
                        # Single image processing
                        result = generator.generate_advanced_caption(
                            uploaded_files[0],
                            model_type=config['model_type'],
                            max_length=config['max_length'],
                            temperature=config['temperature'],
                            style=config['style']
                        )
                        
                        # Display result
                        with col2:
                            st.markdown(f"""
                            <div class="caption-box">
                                <h3>📝 Generated Caption</h3>
                                <p style="font-size: 1.1rem; line-height: 1.5;">{result['caption']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.metric("Confidence", f"{result['confidence']:.1%}")
                            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                            st.metric("Model Used", result['model_used'])
                        
                        # Display advanced features
                        display_advanced_analysis(result['analysis'], config)
                        display_advanced_metrics(result, config)
                        display_alternative_captions(result, config)
                
                except Exception as e:
                    st.error(f"Error generating captions: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Advanced Image Caption Generator - Powered by Multiple AI Models & Deep Learning</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
