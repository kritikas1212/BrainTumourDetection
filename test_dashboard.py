#!/usr/bin/env python3
"""
Quick test script to verify dashboard can load properly
"""
import sys

print("🔍 Testing Dashboard Components...")
print("=" * 50)

# Test 1: Check imports
print("\n1️⃣ Testing imports...")
try:
    import streamlit as st
    print("   ✅ Streamlit imported successfully")
except ImportError as e:
    print(f"   ❌ Streamlit import failed: {e}")
    sys.exit(1)

try:
    import torch
    print("   ✅ PyTorch imported successfully")
except ImportError as e:
    print(f"   ❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from torchvision import transforms
    print("   ✅ Torchvision imported successfully")
except ImportError as e:
    print(f"   ❌ Torchvision import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print("   ✅ Pillow imported successfully")
except ImportError as e:
    print(f"   ❌ Pillow import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("   ✅ Matplotlib imported successfully")
except ImportError as e:
    print(f"   ❌ Matplotlib import failed: {e}")
    sys.exit(1)

try:
    from models import TransferLearningResNet
    print("   ✅ Model imported successfully")
except ImportError as e:
    print(f"   ❌ Model import failed: {e}")
    sys.exit(1)

# Test 2: Check model file
print("\n2️⃣ Checking model file...")
import os
if os.path.exists("resnet_model.pth"):
    size_mb = os.path.getsize("resnet_model.pth") / (1024 * 1024)
    print(f"   ✅ Model file found ({size_mb:.1f} MB)")
else:
    print("   ❌ Model file 'resnet_model.pth' not found")
    sys.exit(1)

# Test 3: Check if model can load
print("\n3️⃣ Testing model loading...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransferLearningResNet(num_classes=4)
    model.load_state_dict(torch.load("resnet_model.pth", map_location=device))
    model.eval()
    print(f"   ✅ Model loaded successfully on {device}")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    sys.exit(1)

# Test 4: Check test images
print("\n4️⃣ Checking for test images...")
test_dirs = [
    "data/New/Testing/notumor",
    "data/New/Testing/glioma", 
    "data/New/Testing/meningioma",
    "data/New/Testing/pituitary"
]

found_images = []
for test_dir in test_dirs:
    if os.path.exists(test_dir):
        images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if images:
            found_images.append(f"{test_dir}: {len(images)} images")
            
if found_images:
    print(f"   ✅ Found test images:")
    for img_info in found_images:
        print(f"      • {img_info}")
else:
    print("   ⚠️  No test images found (this is optional)")

print("\n" + "=" * 50)
print("✅ All tests passed! Dashboard is ready to run.")
print("\n📝 To start the dashboard, run:")
print("   ./run_dashboard.sh")
print("   OR")
print("   streamlit run app.py")
print("=" * 50)

