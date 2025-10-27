@echo off
setlocal enabledelayedexpansion

REM 01_clone_and_setup.bat - Clone DIS repository and setup Python environment

echo =====================================
echo CLONE IS-NET AND SETUP ENVIRONMENT
echo =====================================

set "projectRoot=E:\Projects\sprite-isnet"
set "dataRoot=E:\Projects\sprite-data"
set "repoPath=%projectRoot%\DIS"

REM Create project root if it doesn't exist
if not exist "%projectRoot%" (
    mkdir "%projectRoot%"
    if errorlevel 1 (
        echo [ERROR] Failed to create project root directory
        exit /b 1
    )
)

cd /d "%projectRoot%"
if errorlevel 1 (
    echo [ERROR] Failed to navigate to project root
    exit /b 1
)

REM Step 1: Clone repository
if not exist "%repoPath%" (
    echo.
    echo [1/4] Cloning DIS repository...
    git clone https://github.com/xuebinqin/DIS.git
    if errorlevel 1 (
        echo   [ERROR] Failed to clone repository
        exit /b 1
    )
    
    if exist "%repoPath%" (
        echo   [OK] Repository cloned successfully
    ) else (
        echo   [ERROR] Repository directory not found after clone
        exit /b 1
    )
) else (
    echo.
    echo [1/4] Repository already exists
)

cd /d "%repoPath%\IS-Net"
if errorlevel 1 (
    echo [ERROR] Failed to navigate to IS-Net directory
    exit /b 1
)

REM Step 2: Create virtual environment
echo.
echo [2/4] Creating Python virtual environment...

if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo   [ERROR] Failed to create virtual environment
        exit /b 1
    )
    
    if exist "venv" (
        echo   [OK] Virtual environment created
    ) else (
        echo   [ERROR] Virtual environment directory not found after creation
        exit /b 1
    )
) else (
    echo   Virtual environment already exists
)

REM Step 3: Activate and upgrade pip
echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)

echo   Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    exit /b 1
)

REM Step 4: Install dependencies
echo.
echo [4/4] Installing dependencies...

REM Check CUDA availability
echo   Checking for CUDA...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set "cudaAvailable=true"
    echo   [OK] NVIDIA GPU detected - installing PyTorch with CUDA support
    echo   [INFO] Installing latest PyTorch with CUDA 11.8 support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo   [ERROR] Failed to install PyTorch with CUDA
        exit /b 1
    )
) else (
    set "cudaAvailable=false"
    echo   [WARNING] No NVIDIA GPU detected - installing CPU-only PyTorch
    pip install torch torchvision torchaudio
    if errorlevel 1 (
        echo   [ERROR] Failed to install PyTorch
        exit /b 1
    )
)

echo.
echo   Installing other dependencies...

pip install opencv-python
if errorlevel 1 (
    echo   [ERROR] Failed to install opencv-python
    exit /b 1
)

pip install Pillow
if errorlevel 1 (
    echo   [ERROR] Failed to install Pillow
    exit /b 1
)

pip install numpy
if errorlevel 1 (
    echo   [ERROR] Failed to install numpy
    exit /b 1
)

pip install scikit-image
if errorlevel 1 (
    echo   [ERROR] Failed to install scikit-image
    exit /b 1
)

pip install tqdm
if errorlevel 1 (
    echo   [ERROR] Failed to install tqdm
    exit /b 1
)

pip install tensorboard
if errorlevel 1 (
    echo   [ERROR] Failed to install tensorboard
    exit /b 1
)

pip install matplotlib
if errorlevel 1 (
    echo   [ERROR] Failed to install matplotlib
    exit /b 1
)

echo.
echo   Installing ONNX tools...

pip install onnx
if errorlevel 1 (
    echo   [ERROR] Failed to install onnx
    exit /b 1
)

if "%cudaAvailable%"=="true" (
    pip install onnxruntime-gpu
    if errorlevel 1 (
        echo   [ERROR] Failed to install onnxruntime-gpu
        exit /b 1
    )
) else (
    pip install onnxruntime
    if errorlevel 1 (
        echo   [ERROR] Failed to install onnxruntime
        exit /b 1
    )
)

echo.
echo   Installing download utilities...
pip install gdown
if errorlevel 1 (
    echo   [ERROR] Failed to install gdown
    exit /b 1
)

REM Verify installation
echo.
echo [Verification] Testing PyTorch installation...

python -c "import torch; import sys; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None; print(f'GPU device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB') if torch.cuda.is_available() else print('Running on CPU')"
if errorlevel 1 (
    echo [ERROR] PyTorch verification failed
    exit /b 1
)

echo.
echo =====================================
echo SETUP COMPLETE!
echo =====================================

echo.
echo Virtual environment location:
echo   %repoPath%\IS-Net\venv

echo.
echo Data root directory:
echo   %dataRoot%

echo.
echo To activate the environment, run:
echo   cd %repoPath%\IS-Net
echo   venv\Scripts\activate.bat

endlocal
