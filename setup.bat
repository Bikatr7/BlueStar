@echo off
setlocal

:: Store the root directory
set ROOT_DIR=%CD%

:: Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

:: Install dependencies first
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Now prompt for Hugging Face token
echo Before proceeding, you need to:
echo 1. Create a Hugging Face account at https://huggingface.co/join
echo 2. Visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
echo 3. Click 'Access repository' and agree to share your contact information
echo 4. Get your access token from https://huggingface.co/settings/tokens
echo.
set /p HF_TOKEN="Enter your Hugging Face token: "

:: Login to Hugging Face
echo Logging in to Hugging Face...
python -c "from huggingface_hub import login; login('%HF_TOKEN%')"
if errorlevel 1 (
    echo Failed to login to Hugging Face! Please check your token and try again.
    exit /b 1
)

:: Create all necessary directories
mkdir "%ROOT_DIR%\BlueStar\data\corpus" 2>nul
mkdir "%ROOT_DIR%\BlueStar\data\raw" 2>nul
mkdir "%ROOT_DIR%\BlueStar\models" 2>nul

:: Create corpus and build retrieval index
echo Creating corpus...
python "%ROOT_DIR%\BlueStar\scripts\create_corpus.py"
if errorlevel 1 (
    echo Error creating corpus. Please check the error message above.
    exit /b 1
)

echo Building retrieval index...
python "%ROOT_DIR%\BlueStar\scripts\build_retrieval.py"
if errorlevel 1 (
    echo Error building retrieval index. Please check the error message above.
    exit /b 1
)

echo Setup Complete!
echo To activate the virtual environment, run: venv\Scripts\activate.bat
echo To start the CLI, run: python BlueStar\scripts\run_cli.py

endlocal
