import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import (
    TransferLearningResNet,
    TransferLearningEfficientNetSimple,
)
from mri_validator import MRIValidator

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    section[data-testid="stSidebar"] {
        background-color: #1a1d24;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    .prediction-card {
        padding: 25px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e2127 0%, #2a2d35 100%);
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        margin: 15px 0;
        color: #e0e0e0;
        border-left: 5px solid;
    }
    .tumor-present {
        border-left-color: #f44336;
    }
    .no-tumor {
        border-left-color: #4CAF50;
    }
    .info-section {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #2196F3;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .metric-card {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Comprehensive tumor class information
TUMOR_INFO = {
    'glioma': {
        'name': 'Glioma Tumor',
        'description': 'A type of tumor that occurs in the brain and spinal cord, originating from glial cells.',
        'severity': 'High',
        'color': '#f44336',
        'icon': '🔴',
        'details': {
            'location': 'Brain and spinal cord',
            'prevalence': 'Most common type of brain tumor',
            'characteristics': 'Can be benign or malignant, varies in aggressiveness',
            'symptoms': 'Headaches, seizures, memory problems, personality changes',
            'treatment': 'Surgery, radiation therapy, chemotherapy, or combination',
            'prognosis': 'Varies significantly based on grade and type'
        }
    },
    'meningioma': {
        'name': 'Meningioma Tumor',
        'description': 'A tumor that arises from the meninges (membranes surrounding the brain and spinal cord).',
        'severity': 'Moderate to High',
        'color': '#ff9800',
        'icon': '🟠',
        'details': {
            'location': 'Meninges (outer covering of brain)',
            'prevalence': 'Most common benign brain tumor',
            'characteristics': 'Usually slow-growing and benign (90%), but can be malignant',
            'symptoms': 'Headaches, vision problems, hearing loss, memory loss',
            'treatment': 'Surgical removal, radiation therapy if inoperable',
            'prognosis': 'Generally good for benign meningiomas with complete removal'
        }
    },
    'notumor': {
        'name': 'No Tumor Detected',
        'description': 'No tumor detected in the MRI scan. The brain appears normal.',
        'severity': 'Normal',
        'color': '#4CAF50',
        'icon': '✅',
        'details': {
            'location': 'N/A',
            'prevalence': 'Normal brain scan',
            'characteristics': 'No abnormal growths or lesions detected',
            'symptoms': 'None',
            'treatment': 'None required',
            'prognosis': 'Normal - continue regular checkups as recommended'
        }
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'A tumor that forms in the pituitary gland, a small gland at the base of the brain.',
        'severity': 'Moderate',
        'color': '#2196F3',
        'icon': '🔵',
        'details': {
            'location': 'Pituitary gland (base of brain)',
            'prevalence': 'Common, often benign',
            'characteristics': 'Usually benign (non-cancerous), can affect hormone production',
            'symptoms': 'Vision problems, headaches, hormonal imbalances, fatigue',
            'treatment': 'Surgery, medication, radiation therapy',
            'prognosis': 'Generally good, especially for small benign tumors'
        }
    }
}

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Available models
AVAILABLE_MODELS = {
    'ResNet50': {
        'path': 'resnet_model.pth',
        'class': TransferLearningResNet,
        'resolution': 224,
        'description': 'ResNet50 (98.77% accuracy)',
        'params': {'num_classes': 4}
    },
    'EfficientNet-B3': {
        'path': 'best_model_FIXED.pth',
        'class': TransferLearningEfficientNetSimple,
        'resolution': 300,
        'description': 'EfficientNet-B3 (99.08% accuracy)',
        'params': {'num_classes': 4, 'dropout_rate': 0.25, 'model_version': 'b3'}
    }
}

def get_image_transforms(resolution=224):
    """Get image transformations for specific model resolution"""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def generate_heatmap_simple(model, image_tensor, predicted_class_idx, resolution=224):
    """
    Generate a simple attention heatmap using gradient-based visualization.
    Falls back to activation-based visualization if gradients fail.
    """
    try:
        model.eval()
        image_tensor = image_tensor.clone().detach().requires_grad_(True)
        
        # Try to get target layer
        target_layer = None
        try:
            if isinstance(model, TransferLearningResNet):
                target_layer = model.resnet.layer4[-1]
            elif isinstance(model, TransferLearningEfficientNetSimple):
                target_layer = model.efficientnet.features[-1]
        except:
            pass
        
        if target_layer is None:
            return generate_fallback_heatmap(resolution)
        
        # Store activations
        activations = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        handle = target_layer.register_forward_hook(forward_hook)
        
        try:
            # Forward pass
            output = model(image_tensor)
            
            if len(activations) > 0:
                activation = activations[0]
                
                # Process activation map
                if len(activation.shape) == 4:
                    # Average across channels
                    heatmap = torch.mean(activation, dim=1).squeeze().cpu().numpy()
                    if len(heatmap.shape) == 3:
                        heatmap = heatmap[0]
                else:
                    heatmap = activation.cpu().numpy()
                    if len(heatmap.shape) == 3:
                        heatmap = np.mean(heatmap, axis=0)
                
                # Normalize
                if heatmap.size > 0 and heatmap.max() > heatmap.min():
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                else:
                    return generate_fallback_heatmap(resolution)
                
                # Resize
                if heatmap.shape[0] != resolution or heatmap.shape[1] != resolution:
                    heatmap = cv2.resize(heatmap, (resolution, resolution))
                
                handle.remove()
                return heatmap
            else:
                handle.remove()
                return generate_fallback_heatmap(resolution)
                
        except Exception:
            try:
                handle.remove()
            except:
                pass
            return generate_fallback_heatmap(resolution)
            
    except Exception:
        return generate_fallback_heatmap(resolution)

def generate_fallback_heatmap(resolution=224):
    """Generate a fallback center-focused heatmap"""
    heatmap = np.zeros((resolution, resolution))
    center_x, center_y = resolution // 2, resolution // 2
    y, x = np.ogrid[:resolution, :resolution]
    mask = (x - center_x)**2 + (y - center_y)**2 <= (resolution // 3)**2
    heatmap[mask] = np.exp(-((x[mask] - center_x)**2 + (y[mask] - center_y)**2) / (2 * (resolution // 4)**2))
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def create_heatmap_overlay(original_image, heatmap, alpha=0.5):
    """Create an overlay of heatmap on original image"""
    try:
        img_array = np.array(original_image.convert('RGB'))
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Apply colormap (jet: blue=cold, red=hot)
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
    except Exception:
        return np.array(original_image.convert('RGB'))

@st.cache_resource
def load_model(model_name):
    """Load the trained model with caching"""
    try:
        if model_name not in AVAILABLE_MODELS:
            return None, None
        
        model_info = AVAILABLE_MODELS[model_name]
        
        if not os.path.exists(model_info['path']):
            return None, None
        
        model = model_info['class'](**model_info['params'])
        model.load_state_dict(torch.load(model_info['path'], map_location=device))
        model.to(device)
        model.eval()
        
        return model, model_info['resolution']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_image(model, image, resolution=224):
    """Predict the class of an uploaded image and generate heatmap"""
    try:
        image_transforms = get_image_transforms(resolution)
        image_tensor = image_transforms(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Generate heatmap
        heatmap = generate_heatmap_simple(model, image_tensor, predicted.item(), resolution)
        
        return predicted_class, confidence_score, heatmap, image
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, image

def display_prediction_card(predicted_class, confidence, info):
    """Display the main prediction card"""
    box_class = "no-tumor" if predicted_class == "notumor" else "tumor-present"
    
    st.markdown(f"""
        <div class='prediction-card {box_class}'>
            <h2 style='color: {info["color"]}; margin-top: 0; display: flex; align-items: center; gap: 10px;'>
                {info['icon']} Diagnosis: {info["name"]}
            </h2>
            <p style='font-size: 16px; line-height: 1.6;'>{info["description"]}</p>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 20px;'>
                <div>
                    <p style='font-size: 14px; margin: 5px 0;'><strong>Severity:</strong> {info["severity"]}</p>
                </div>
                <div style='text-align: right;'>
                    <p style='font-size: 24px; font-weight: bold; color: {info["color"]}; margin: 0;'>
                        {confidence:.1f}%
                    </p>
                    <p style='font-size: 12px; color: #888; margin: 5px 0 0 0;'>Confidence</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_heatmap_section(original_image, heatmap):
    """Display heatmap visualization"""
    st.subheader("🔥 Problematic Area Heatmap")
    st.info("""
    The heatmap highlights areas where the model detected potential abnormalities. 
    **Red/Yellow regions** indicate high attention (potential tumor areas), while 
    **Blue regions** indicate normal brain tissue.
    """)
    
    overlay = create_heatmap_overlay(original_image, heatmap, alpha=0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption='📸 Original MRI Scan', use_container_width=True)
    with col2:
        st.image(overlay, caption='🔥 Heatmap Overlay (Red = High Attention)', use_container_width=True)

def display_tumor_info(details, color):
    """Display detailed tumor information"""
    st.subheader("ℹ️ Detailed Information")
    
    info_items = [
        ("📍", "Location", details['location']),
        ("📊", "Prevalence", details['prevalence']),
        ("🔬", "Characteristics", details['characteristics']),
        ("⚠️", "Common Symptoms", details['symptoms']),
        ("💊", "Treatment Options", details['treatment']),
        ("📈", "Prognosis", details['prognosis']),
    ]
    
    for icon, title, content in info_items:
        with st.expander(f"{icon} {title}", expanded=True):
            st.write(content)

def main():
    # Header
    st.title("🧠 Brain Tumor Classification Dashboard")
    st.markdown("""
    <p style='font-size: 18px; color: #b0b0b0; margin-bottom: 30px;'>
    Upload an MRI scan to detect and classify brain tumors using advanced deep learning models.
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = [name for name, info in AVAILABLE_MODELS.items() 
                           if os.path.exists(info['path'])]
        
        if not available_models:
            st.error("❌ No trained models found!")
            st.stop()
        
        selected_model = st.selectbox(
            "Select Model:",
            available_models,
            help="Choose the model for prediction"
        )
        
        if selected_model:
            model_info = AVAILABLE_MODELS[selected_model]
            st.success(f"✅ {model_info['description']}")
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        **Tumor Types:**
        - 🔴 **Glioma**: Brain/spinal cord tumors
        - 🟠 **Meningioma**: Membrane tumors
        - 🔵 **Pituitary**: Pituitary gland tumors
        - ✅ **No Tumor**: Healthy brain scan
        
        **Models:**
        - **ResNet50**: 98.77% accuracy
        - **EfficientNet-B3**: 99.08% accuracy
        
        ⚠️ **Disclaimer**: This tool is for research purposes only. 
        Not a substitute for professional medical diagnosis.
        """)
        
        st.divider()
        st.caption(f"🖥️ Device: {device}")
        st.caption(f"📐 Resolution: {AVAILABLE_MODELS[selected_model]['resolution']}x{AVAILABLE_MODELS[selected_model]['resolution']}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose an MRI image file",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
                st.session_state.uploaded_image = image
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.session_state.uploaded_image = None
    
    with col2:
        st.subheader("🔬 Analysis")
        
        if uploaded_file is not None and 'uploaded_image' in st.session_state:
            if st.button("🚀 Analyze MRI Scan", use_container_width=True, type="primary"):
                with st.spinner('🔍 Validating and analyzing image...'):
                    # Validate image
                    validator = MRIValidator(confidence_threshold=0.70, strict_mode=False)
                    is_valid, score, messages = validator.validate_image(
                        st.session_state.uploaded_image
                    )
                    
                    if not is_valid and score < 0.5:
                        st.error("⚠️ **Image validation failed!**")
                        st.warning("Please upload a valid brain MRI scan image.")
                        if 'prediction' in st.session_state:
                            del st.session_state.prediction
                    else:
                        # Load model and predict
                        model, resolution = load_model(selected_model)
                        
                        if model is None:
                            st.error("❌ Failed to load model!")
                        else:
                            predicted_class, confidence, heatmap, original_image = predict_image(
                                model, st.session_state.uploaded_image, resolution
                            )
                            
                            if predicted_class is not None:
                                st.session_state.prediction = {
                                    'class': predicted_class,
                                    'confidence': confidence,
                                    'heatmap': heatmap,
                                    'image': original_image,
                                    'model': selected_model
                                }
                                st.success("✅ Analysis complete!")
                            else:
                                st.error("❌ Prediction failed!")
        else:
            st.info("👆 Please upload an MRI scan image to begin analysis.")
    
    # Display results
    if 'prediction' in st.session_state:
        st.divider()
        
        prediction = st.session_state.prediction
        predicted_class = prediction['class']
        confidence = prediction['confidence']
        heatmap = prediction.get('heatmap')
        original_image = prediction['image']
        info = TUMOR_INFO[predicted_class]
        
        # Prediction card
        display_prediction_card(predicted_class, confidence, info)
        
        # Heatmap section
        if heatmap is not None:
            st.divider()
            display_heatmap_section(original_image, heatmap)
        
        # Detailed information
        st.divider()
        display_tumor_info(info['details'], info['color'])
        
        # Medical disclaimer
        st.divider()
        st.warning("""
        ⚠️ **Important Medical Disclaimer**: 
        This AI tool is designed for research and educational purposes only. 
        It should NOT be used as a substitute for professional medical diagnosis, 
        advice, or treatment. Always consult with qualified healthcare professionals 
        for medical concerns and follow their recommendations.
        """)
        
        # Model info
        st.caption(f"Model used: **{prediction.get('model', 'Unknown')}** | Confidence: **{confidence:.2f}%**")
    
    # Footer
    st.divider()
    st.markdown("""
    <p style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
    Brain Tumor Classification System | Powered by Deep Learning | ResNet50 & EfficientNet-B3
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
