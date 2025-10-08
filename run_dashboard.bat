@echo off
REM Brain Tumor Classification Dashboard Launcher for Windows
REM This script activates the virtual environment and launches the Streamlit dashboard

echo 🧠 Brain Tumor Classification Dashboard
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo ✅ Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  Virtual environment not found. Using system Python.
)

REM Check if model file exists
if not exist "resnet_model.pth" (
    echo ❌ Error: Model file 'resnet_model.pth' not found!
    echo Please train the model first by running: python train.py
    pause
    exit /b 1
)

REM Check if streamlit is installed
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit is not installed!
    echo Installing required packages...
    pip install -r requirements.txt
)

echo.
echo 🚀 Starting dashboard...
echo The dashboard will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Launch Streamlit
streamlit run app.py

