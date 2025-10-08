import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models import TransferLearningResNet
import torch.nn.functional as F

# Set page config
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
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
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e2127;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        margin: 10px 0;
        color: #e0e0e0;
    }
    .tumor-present {
        border-left: 5px solid #f44336;
    }
    .no-tumor {
        border-left: 5px solid #4CAF50;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    p, .css-10trblm, .css-16idsys {
        color: #e0e0e0 !important;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names and their descriptions
class_info = {
    'glioma': {
        'name': 'Glioma Tumor',
        'description': 'A type of tumor that occurs in the brain and spinal cord.',
        'severity': 'High',
        'color': '#f44336'
    },
    'meningioma': {
        'name': 'Meningioma Tumor',
        'description': 'A tumor that arises from the meninges (membranes surrounding the brain and spinal cord).',
        'severity': 'Moderate to High',
        'color': '#ff9800'
    },
    'notumor': {
        'name': 'No Tumor Detected',
        'description': 'No tumor detected in the MRI scan.',
        'severity': 'Normal',
        'color': '#4CAF50'
    },
    'pituitary': {
        'name': 'Pituitary Tumor',
        'description': 'A tumor that forms in the pituitary gland near the brain.',
        'severity': 'Moderate',
        'color': '#2196F3'
    }
}

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        model = TransferLearningResNet(num_classes=len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image):
    """Predict the class of an uploaded image"""
    # Apply transformations
    image_tensor = image_transforms(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # Get all class probabilities
    all_probs = probabilities[0].cpu().numpy() * 100
    
    return predicted_class, confidence_score, all_probs

def display_prediction_results(predicted_class, confidence, all_probs):
    """Display prediction results in a formatted way"""
    info = class_info[predicted_class]
    
    # Main prediction card
    box_class = "no-tumor" if predicted_class == "notumor" else "tumor-present"
    
    st.markdown(f"""
        <div class='prediction-box {box_class}'>
            <h2 style='color: {info["color"]}; margin-top: 0;'>🔍 Diagnosis: {info["name"]}</h2>
            <p style='font-size: 16px;'>{info["description"]}</p>
            <p style='font-size: 14px;'><strong>Severity Level:</strong> {info["severity"]}</p>
            <p style='font-size: 20px; font-weight: bold; color: {info["color"]};'>
                Confidence: {confidence:.2f}%
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display probability distribution
    st.subheader("📊 Probability Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e2127')
    ax.set_facecolor('#1e2127')
    colors = [class_info[cls]['color'] for cls in class_names]
    bars = ax.barh([class_info[cls]['name'] for cls in class_names], all_probs, color=colors)
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, all_probs)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2f}%', 
                ha='left', va='center', fontweight='bold', fontsize=10, color='white')
    
    ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Class Probability Distribution', fontsize=14, fontweight='bold', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['top'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['right'].set_color('#444')
    ax.set_xlim(0, 105)
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close()

def main():
    # Header
    st.title("🧠 Brain Tumor Classification Dashboard")
    st.markdown("""
        <p style='font-size: 18px; color: #b0b0b0;'>
        Upload an MRI scan image to detect and classify brain tumors using deep learning.
        </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        ### About This Tool
        This dashboard uses a deep learning model (ResNet50) trained to classify brain tumors into four categories:
        
        - **Glioma**: Brain/spinal cord tumors
        - **Meningioma**: Membrane tumors
        - **Pituitary**: Pituitary gland tumors
        - **No Tumor**: Healthy brain scan
        
        ### How to Use
        1. Upload an MRI scan image (JPG, JPEG, or PNG)
        2. Click "Analyze MRI Scan"
        3. View the prediction results
        
        ### Note
        ⚠️ This is a research tool and should not be used as a substitute for professional medical diagnosis.
        """)
        
        st.divider()
        st.markdown(f"**Device:** {device}")
        st.markdown(f"**Model:** ResNet50 Transfer Learning")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose an MRI image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan image in JPG, JPEG, or PNG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
            
            # Store image in session state
            st.session_state.uploaded_image = image
    
    with col2:
        st.subheader("🔬 Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Analyze MRI Scan", use_container_width=True):
                with st.spinner('Analyzing MRI scan... Please wait.'):
                    # Load model
                    model = load_model("resnet_model.pth")
                    
                    if model is not None:
                        # Make prediction
                        predicted_class, confidence, all_probs = predict_image(
                            model, st.session_state.uploaded_image
                        )
                        
                        # Store results in session state
                        st.session_state.prediction = {
                            'class': predicted_class,
                            'confidence': confidence,
                            'probabilities': all_probs
                        }
                        
                        st.success("✅ Analysis complete!")
        else:
            st.info("👆 Please upload an MRI scan image to begin analysis.")
    
    # Display results if available
    if 'prediction' in st.session_state:
        st.divider()
        st.header("📋 Detailed Results")
        display_prediction_results(
            st.session_state.prediction['class'],
            st.session_state.prediction['confidence'],
            st.session_state.prediction['probabilities']
        )
        
        # Additional information
        with st.expander("ℹ️ Understanding Your Results"):
            st.markdown("""
            ### Interpretation Guide
            
            - **Confidence Score**: Indicates how certain the model is about its prediction. 
              Higher percentages indicate greater confidence.
            
            - **Probability Distribution**: Shows the likelihood of each tumor type. 
              The predicted class will have the highest probability.
            
            ### Next Steps
            If a tumor is detected, please:
            1. Consult with a qualified medical professional
            2. Share these results with your healthcare provider
            3. Follow up with additional diagnostic tests as recommended
            
            ### Disclaimer
            This AI model is designed for research and educational purposes. 
            It should not replace professional medical advice, diagnosis, or treatment.
            Always seek the advice of qualified health providers with questions regarding medical conditions.
            """)
    
    # Footer
    st.divider()
    st.markdown("""
        <p style='text-align: center; color: #888; font-size: 12px;'>
        Brain Tumor Classification System | Powered by Deep Learning | ResNet50 Architecture
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

