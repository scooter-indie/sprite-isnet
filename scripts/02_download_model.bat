@echo off
setlocal enabledelayedexpansion

REM 02_download_model.bat - Download pretrained IS-Net model

echo =====================================
echo DOWNLOAD PRETRAINED IS-NET MODEL
echo =====================================

set "projectRoot=E:\Projects\sprite-isnet"
set "dataRoot=E:\Projects\sprite-data"
set "modelDir=%projectRoot%\DIS\saved_models\IS-Net"
set "modelPath=%modelDir%\isnet-general-use.pth"

REM Ensure model directory exists
if not exist "%modelDir%" (
    mkdir "%modelDir%"
    if errorlevel 1 (
        echo [ERROR] Failed to create model directory
        exit /b 1
    )
)

REM Check if model already exists
if exist "%modelPath%" (
    echo.
    echo [OK] Model already exists!
    
    for %%A in ("%modelPath%") do (
        set /a "size=%%~zA / 1048576"
    )
    echo   Path: %modelPath%
    echo   Size: !size! MB
    
    echo.
    set /p "overwrite=Do you want to re-download? (y/n): "
    if /i not "!overwrite!"=="y" (
        echo.
        echo Skipping download. Using existing model.
        exit /b 0
    )
)

echo.
echo Downloading isnet-general-use.pth...
echo   This may take several minutes (file is ~176 MB)

REM Activate virtual environment
cd /d "%projectRoot%\DIS\IS-Net"
if errorlevel 1 (
    echo [ERROR] Failed to navigate to IS-Net directory
    exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)

REM Google Drive file ID for isnet-general-use.pth
set "fileId=1XHIzgTzY5BQHw140EDIgwIb53K659ENH"

REM Method 1: Try gdown
echo.
echo [Method 1] Attempting download with gdown...

gdown %fileId% -O "%modelPath%" --fuzzy
set "gdownResult=%errorlevel%"

set "downloadSuccess=false"

if exist "%modelPath%" (
    for %%A in ("%modelPath%") do (
        set /a "size=%%~zA / 1048576"
    )
    
    REM Verify file size (should be around 176 MB)
    if !size! GTR 150 if !size! LSS 200 (
        echo   [OK] Download successful!
        echo   Size: !size! MB
        set "downloadSuccess=true"
    ) else (
        echo   [ERROR] Downloaded file size unexpected: !size! MB
        del /f "%modelPath%"
        set "downloadSuccess=false"
    )
) else (
    echo   [WARNING] gdown failed to download file
    set "downloadSuccess=false"
)

REM Method 2: Manual download fallback
if "!downloadSuccess!"=="false" (
    echo.
    echo [Method 2] Manual download required
    echo.
    echo Please follow these steps:
    echo   1. Click the link that will open in your browser
    echo   2. Click 'Download' on the Google Drive page
    echo   3. Save the file to:
    echo      %modelPath%
    
    REM Open browser to download page
    set "downloadUrl=https://drive.google.com/file/d/%fileId%/view?usp=sharing"
    echo.
    echo Opening browser...
    timeout /t 2 /nobreak >nul
    start "" "%downloadUrl%"
    
    echo.
    echo Waiting for manual download...
    echo Press Enter after you've saved the file to the location above
    pause >nul
    
    REM Verify manual download
    if exist "%modelPath%" (
        for %%A in ("%modelPath%") do (
            set /a "size=%%~zA / 1048576"
        )
        echo.
        echo [OK] Model file found!
        echo   Size: !size! MB
        
        if !size! LSS 150 (
            echo.
            echo [WARNING] File size seems too small (expected ~176 MB^)
            echo   The file might be incomplete or corrupted
        )
    ) else (
        echo.
        echo [ERROR] Model file not found!
        echo   Expected location: %modelPath%
        echo.
        echo Please download manually and re-run this script
        exit /b 1
    )
)

REM Verify model can be loaded
echo.
echo [Verification] Testing model file...

REM Create temporary Python verification script
set "tempScript=%TEMP%\verify_model.py"

(
echo import torch
echo import sys
echo.
echo model_path = r'%modelPath%'
echo.
echo try:
echo     print^('  Loading model checkpoint...'^)
echo     checkpoint = torch.load^(model_path, map_location='cpu'^)
echo.    
echo     print^(f'  Checkpoint type: {type^(checkpoint^).__name__}'^)
echo.    
echo     if isinstance^(checkpoint, dict^):
echo         print^(f'  Checkpoint keys: {list^(checkpoint.keys^(^)^)[:5]}'^)
echo.        
echo         # Determine state dict
echo         if 'model_state_dict' in checkpoint:
echo             state_dict = checkpoint['model_state_dict']
echo         elif 'state_dict' in checkpoint:
echo             state_dict = checkpoint['state_dict']
echo         else:
echo             state_dict = checkpoint
echo.        
echo         print^(f'  State dict contains {len^(state_dict^)} parameters'^)
echo.        
echo         # Show sample parameters
echo         sample_keys = list^(state_dict.keys^(^)^)[:3]
echo         print^(f'  Sample parameters:'^)
echo         for key in sample_keys:
echo             print^(f'    - {key}: {state_dict[key].shape}'^)
echo.    
echo     print^(''^)
echo     print^('[OK] Model file is valid and loadable!'^)
echo     sys.exit^(0^)
echo.    
echo except Exception as e:
echo     print^(f''^)
echo     print^(f'[ERROR] Error loading model: {e}'^)
echo     print^(f'The file may be corrupted. Please re-download.'^)
echo     sys.exit^(1^)
) > "%tempScript%"

python "%tempScript%"
set "verifyResult=%errorlevel%"

REM Clean up temp script
del /f "%tempScript%" >nul 2>&1

if %verifyResult% equ 0 (
    echo.
    echo =====================================
    echo MODEL DOWNLOAD COMPLETE!
    echo =====================================
    echo.
    echo Model ready for training at:
    echo   %modelPath%
) else (
    echo.
    echo =====================================
    echo MODEL VERIFICATION FAILED
    echo =====================================
    echo.
    echo Please re-download the model and try again
    exit /b 1
)

endlocal
