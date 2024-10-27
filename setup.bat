@echo off
setlocal

:: Store the root directory
set ROOT_DIR=%CD%

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists, skipping creation...
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies first
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Create all necessary directories
mkdir "%ROOT_DIR%\BlueStar\data\corpus" 2>nul
mkdir "%ROOT_DIR%\BlueStar\models" 2>nul

:: Download base model
echo Downloading base model...
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('gpt2-large', local_files_only=False); AutoTokenizer.from_pretrained('gpt2-large', local_files_only=False)"
if errorlevel 1 (
    echo Error downloading model. Please check the error message above.
    exit /b 1
)

:: Quantize model
echo Quantizing model...
python "%ROOT_DIR%\BlueStar\scripts\quantize_model.py"
if errorlevel 1 (
    echo Error quantizing model. Please check the error message above.
    exit /b 1
)

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

:: Run model validation
echo Validating model...
python "%ROOT_DIR%\BlueStar\scripts\validate_model.py"
if errorlevel 1 (
    echo Error validating model. Please check the error message above.
    exit /b 1
)

echo Setup Complete!
echo To activate the virtual environment, run: venv\Scripts\activate.bat
echo To start the CLI, run: python BlueStar\scripts\run_cli.py

endlocal
