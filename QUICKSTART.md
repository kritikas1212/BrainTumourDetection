# 🚀 Quick Start Guide - Brain Tumor Dashboard

## Installation (One-time Setup)

### Step 1: Install Dependencies

**On Mac (you are here):**
```bash
pip install -r requirements-mac.txt
```

**On Windows/Linux:**
```bash
pip install -r requirements.txt
```

### Step 2: Verify Model File

Make sure `resnet_model.pth` exists in this directory. If it doesn't, train the model first:
```bash
python train.py
```

## Running the Dashboard

### Option 1: Quick Launch (Recommended)

**Mac/Linux:**
```bash
./run_dashboard.sh
```

**Windows:**
```bash
run_dashboard.bat
```

### Option 2: Manual Launch

```bash
streamlit run app.py
```

## What to Expect

1. ✅ The terminal will show startup messages
2. 🌐 Your browser will automatically open to `http://localhost:8501`
3. 📤 You'll see an upload interface for MRI scans
4. 🔬 Upload an image and click "Analyze MRI Scan"
5. 📊 View the results with confidence scores and detailed analysis

## Testing the Dashboard

Try uploading sample images from:
- `data/New/Testing/notumor/` (for healthy scans)
- `data/New/Testing/glioma/` (for glioma tumor scans)
- `data/New/Testing/meningioma/` (for meningioma tumor scans)
- `data/New/Testing/pituitary/` (for pituitary tumor scans)

## Stopping the Dashboard

Press `Ctrl+C` in the terminal to stop the server.

## Troubleshooting

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Module not found errors?**
```bash
pip install streamlit torch torchvision pillow matplotlib
```

**Need help?** See the full [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) for detailed documentation.

---

**Ready to start?** Just run: `./run_dashboard.sh`

