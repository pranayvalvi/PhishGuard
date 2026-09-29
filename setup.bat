@echo off
echo ===================================================
echo        PhishGuard Automated Setup Environment
echo ===================================================
echo.

:: 1. Check for the raw dataset
if not exist "data\raw\dataset.csv" (
    echo [ERROR] Dataset not found!
    echo Please download a phishing/spam dataset, rename it to 'dataset.csv', 
    echo and place it in the 'data\raw\' folder before running this setup.
    echo.
    pause
    exit /b
)

:: 2. Create the virtual environment
echo [1/3] Creating Python Virtual Environment (venv)...
if not exist "venv\" (
    python -m venv venv
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists. Skipping creation.
)
echo.

:: 3. Install requirements
echo [2/3] Installing dependencies (This may take several minutes)...
call venv\Scripts\python.exe -m pip install --upgrade pip
call venv\Scripts\python.exe -m pip install -r requirements.txt
echo.

:: 4. Execute the ML Pipeline
echo [3/3] Running the Machine Learning Pipeline...
echo.

echo --- Phase 1: Data Preparation ---
call venv\Scripts\python.exe src\01_data_prep.py
echo.

echo --- Phase 2: NLP Preprocessing ---
call venv\Scripts\python.exe src\02_nlp_preprocessing.py
echo.

echo --- Phase 3: Classical Model Training ---
call venv\Scripts\python.exe src\03_train_classical.py
echo.

echo --- Phase 4: Transformer Fine-Tuning ---
call venv\Scripts\python.exe src\04_train_transformer.py
echo.

echo --- Phase 5: Final Evaluation ---
call venv\Scripts\python.exe src\05_evaluate.py
echo.

echo ===================================================
echo    SETUP COMPLETE! ALL MODELS TRAINED AND SAVED.
echo ===================================================
echo You can now launch the application by double-clicking 'run_phishguard.bat'
pause
