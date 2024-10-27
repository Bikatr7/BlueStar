@echo off
SETLOCAL EnableDelayedExpansion

:: Store the root directory
set ROOT_DIR=%CD%

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

:: Check if virtual environment exists and is up to date
set RECREATE_VENV=0
if exist "venv" (
    echo Found existing virtual environment, checking if it's up to date...
    venv\Scripts\python -m pip check
    if errorlevel 1 (
        echo Virtual environment is outdated, recreating...
        set RECREATE_VENV=1
    ) else (
        venv\Scripts\python -m pip install -r requirements.txt --dry-run
        if errorlevel 1 (
            echo Virtual environment is missing some packages, recreating...
            set RECREATE_VENV=1
        ) else (
            echo Virtual environment is up to date, skipping recreation...
        )
    )
) else (
    set RECREATE_VENV=1
)

:: Create/Recreate virtual environment if needed
if %RECREATE_VENV%==1 (
    if exist "venv" (
        echo Removing existing virtual environment...
        rmdir /s /q venv
        if errorlevel 1 (
            echo Failed to remove existing virtual environment!
            echo Please close any programs that might be using it and try again.
            exit /b 1
        )
    )
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment!
        exit /b 1
    )
    
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo Failed to activate virtual environment!
        exit /b 1
    )
    
    echo Upgrading pip...
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo Failed to upgrade pip!
        exit /b 1
    )
    
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements!
        exit /b 1
    )
) else (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

:: Create necessary directories
echo Creating directories...
if not exist "%ROOT_DIR%\BlueStar\models\mistral-7b" mkdir "%ROOT_DIR%\BlueStar\models\mistral-7b"
if not exist "%ROOT_DIR%\BlueStar\models\quantized-mistral-7b" mkdir "%ROOT_DIR%\BlueStar\models\quantized-mistral-7b"
if not exist "%ROOT_DIR%\BlueStar\data\corpus" mkdir "%ROOT_DIR%\BlueStar\data\corpus"

:: Check for model files
echo Checking for model files...
set MODEL_FILES_MISSING=0
for %%f in (config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors tokenizer.model tokenizer.json) do (
    if not exist "%ROOT_DIR%\BlueStar\models\mistral-7b\%%f" (
        set MODEL_FILES_MISSING=1
    )
)

if %MODEL_FILES_MISSING%==1 (
    echo Some model files are missing. Downloading Mistral-7B...
    cd "%ROOT_DIR%\BlueStar\models\mistral-7b"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mistralai/Mistral-7B-v0.1', local_dir='.')"
    if errorlevel 1 (
        echo Failed to download model!
        echo If this keeps happening, try:
        echo 1. Ensure you've agreed to share contact information at https://huggingface.co/mistralai/Mistral-7B-v0.1
        echo 2. Delete the %ROOT_DIR%\BlueStar\models\mistral-7b directory and run setup again
        cd "%ROOT_DIR%"
        exit /b 1
    )
    cd "%ROOT_DIR%"
) else (
    echo Found existing model files, skipping download...
)

:: Create corpus and build retrieval index only if needed
echo Checking for existing corpus...
if exist "%ROOT_DIR%\BlueStar\data\corpus\*.txt" (
    echo Found existing corpus files, skipping corpus creation...
) else (
    echo Creating corpus...
    python "%ROOT_DIR%\BlueStar\scripts\create_corpus.py"
    if errorlevel 1 (
        echo Failed to create corpus!
        exit /b 1
    )
)

:: Build retrieval index only if needed
if exist "%ROOT_DIR%\BlueStar\data\faiss_index.bin" (
    echo Found existing retrieval index, skipping build...
) else (
    echo Building retrieval index...
    python "%ROOT_DIR%\BlueStar\scripts\build_retrieval.py"
    if errorlevel 1 (
        echo Failed to build retrieval index!
        exit /b 1
    )
)

:: Quantize model
echo Quantizing model...
python "%ROOT_DIR%\BlueStar\scripts\quantize_model.py"
if errorlevel 1 (
    echo Failed to quantize model!
    exit /b 1
)

echo Setup Complete!
echo To activate the virtual environment, run: venv\Scripts\activate.bat
echo To start the CLI, run: python BlueStar\scripts\run_cli.py

ENDLOCAL
