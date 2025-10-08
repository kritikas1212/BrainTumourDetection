# 🧠 Brain Tumor Classification Dashboard Guide

## Overview
This interactive web dashboard allows you to upload MRI scan images and get real-time predictions for brain tumor classification using a trained deep learning model.

## Features
- 📤 **Easy Upload**: Drag and drop or browse to upload MRI scans
- 🔬 **Real-time Analysis**: Instant predictions using ResNet50 model
- 📊 **Detailed Results**: Confidence scores and probability distributions
- 🎨 **User-friendly Interface**: Clean, medical-grade dashboard design
- 💡 **Educational Information**: Learn about different tumor types

## Quick Start

### 1. Installation

First, make sure you have Python 3.8+ installed. Then install the required packages:

**For Mac:**
```bash
pip install -r requirements-mac.txt
```

**For Windows/Linux:**
```bash
pip install -r requirements.txt
```

If you encounter issues, you can install Streamlit separately:
```bash
pip install streamlit
```

### 2. Verify Model File

Make sure the trained model file `resnet_model.pth` is present in the project directory. If not, you'll need to train the model first using:
```bash
python train.py
```

### 3. Launch the Dashboard

Run the following command in your terminal:
```bash
streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

### 4. Using the Dashboard

1. **Upload an MRI Scan**
   - Click on "Browse files" or drag and drop an image
   - Supported formats: JPG, JPEG, PNG
   - The uploaded image will appear in the left panel

2. **Analyze the Scan**
   - Click the "🚀 Analyze MRI Scan" button
   - Wait for the model to process (usually takes a few seconds)
   - Results will appear in the right panel

3. **Interpret Results**
   - **Primary Diagnosis**: Shows the detected tumor type or "No Tumor"
   - **Confidence Score**: Indicates model certainty (higher is more confident)
   - **Probability Distribution**: Shows likelihood of each class
   - **Additional Information**: Expand for detailed explanations

## Tumor Classes

The model can classify four categories:

1. **Glioma Tumor** 🔴
   - Type: Brain and spinal cord tumors
   - Severity: High
   - Requires immediate medical attention

2. **Meningioma Tumor** 🟠
   - Type: Membrane tumors around brain/spinal cord
   - Severity: Moderate to High
   - Often benign but requires monitoring

3. **Pituitary Tumor** 🔵
   - Type: Pituitary gland tumors
   - Severity: Moderate
   - Can affect hormone production

4. **No Tumor** 🟢
   - Type: Normal brain scan
   - Severity: Normal
   - No tumor detected

## Tips for Best Results

1. **Image Quality**: Use clear, high-resolution MRI scans
2. **File Format**: PNG files typically preserve quality better than JPG
3. **Single Scan**: Upload one MRI scan at a time
4. **Proper Orientation**: Ensure the MRI scan is properly oriented

## Troubleshooting

### Dashboard Won't Start
```bash
# Try upgrading streamlit
pip install --upgrade streamlit

# Or reinstall
pip uninstall streamlit
pip install streamlit
```

### Model Not Found Error
```bash
# Verify the model file exists
ls resnet_model.pth

# If missing, train the model
python train.py
```

### Memory Issues
If you encounter memory errors:
- Close other applications
- Use CPU mode (automatically detected if no GPU available)
- Try processing smaller images

### Port Already in Use
If port 8501 is busy, specify a different port:
```bash
streamlit run app.py --server.port 8502
```

## Advanced Configuration

You can customize the dashboard behavior by creating a `.streamlit/config.toml` file:

```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

## Deployment Options

### Local Network Access
To make the dashboard accessible on your local network:
```bash
streamlit run app.py --server.address 0.0.0.0
```

### Cloud Deployment
You can deploy this dashboard to:
- **Streamlit Cloud** (Free): https://streamlit.io/cloud
- **Heroku**: https://www.heroku.com
- **AWS/GCP/Azure**: Using containerization

## Security and Privacy

⚠️ **Important Notes:**
- This tool is for research and educational purposes only
- Do NOT use as a substitute for professional medical diagnosis
- Patient data should be anonymized before uploading
- Consider HIPAA compliance if using with real patient data
- For production use, implement proper authentication and encryption

## Technical Details

- **Model Architecture**: ResNet50 with transfer learning
- **Input Size**: 224x224 pixels (RGB)
- **Preprocessing**: ImageNet normalization
- **Framework**: PyTorch 2.4.1
- **Interface**: Streamlit 1.28.0

## Support and Documentation

For issues or questions:
1. Check the troubleshooting section above
2. Review the main README.md
3. Check the code comments in `app.py`

## License and Disclaimer

This is a research tool and should not be used for clinical diagnosis. Always consult with qualified medical professionals for health-related decisions.

---

**Last Updated**: October 2025
**Version**: 1.0.0

