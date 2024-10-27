@echo off
SETLOCAL EnableDelayedExpansion

echo Installing BlueStar...

:: Prompt for Hugging Face token first
echo Before proceeding, you need to:
echo 1. Create a Hugging Face account at https://huggingface.co/join
echo 2. Visit https://huggingface.co/mistralai/Mistral-7B-v0.1
echo 3. Click "Access repository" and agree to share your contact information
echo 4. Get your access token from https://huggingface.co/settings/tokens
echo.
set /p HF_TOKEN="Enter your Hugging Face token: "

:: Login to Hugging Face first
echo Logging in to Hugging Face...
python -c "from huggingface_hub import login; login('%HF_TOKEN%')"
if errorlevel 1 (
    echo Failed to login to Hugging Face! Please check your token and try again.
    echo Make sure you've agreed to share contact information at https://huggingface.co/mistralai/Mistral-7B-v0.1
    exit /b 1
)

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    exit /b 1
)

:: Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not installed! Please install pip.
    exit /b 1
)

:: Remove existing virtual environment if it exists
if exist "venv" (
    echo Removing existing virtual environment...
    rmdir /s /q venv
    if errorlevel 1 (
        echo Failed to remove existing virtual environment!
        echo Please close any programs that might be using it and try again.
        exit /b 1
    )
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment!
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment!
    exit /b 1
)

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip!
    exit /b 1
)

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements!
    exit /b 1
)

:: Create necessary directories
echo Creating directories...
if not exist "models\mistral-7b" mkdir models\mistral-7b
if not exist "data\corpus" mkdir data\corpus

:: Download model if not exists
echo Checking for existing model...
if exist "models\mistral-7b\config.json" (
    echo Found existing model files, skipping download...
) else (
    echo Downloading Mistral-7B model...
    cd models\mistral-7b
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mistralai/Mistral-7B-v0.1', local_dir='.')"
    if errorlevel 1 (
        echo Failed to download model!
        echo Please ensure you've agreed to share contact information at https://huggingface.co/mistralai/Mistral-7B-v0.1
        cd ..\..
        exit /b 1
    )
    cd ..\..
)

:: Create corpus and build retrieval index
echo Creating corpus...
python scripts\create_corpus.py
if errorlevel 1 (
    echo Failed to create corpus!
    exit /b 1
)

echo Building retrieval index...
python scripts\build_retrieval.py
if errorlevel 1 (
    echo Failed to build retrieval index!
    exit /b 1
)

:: Quantize model
echo Quantizing model...
python scripts\quantize_model.py
if errorlevel 1 (
    echo Failed to quantize model!
    exit /b 1
)

echo Setup Complete!
echo To activate the virtual environment, run: venv\Scripts\activate.bat
echo To start the CLI, run: python scripts\run_cli.py

ENDLOCAL
