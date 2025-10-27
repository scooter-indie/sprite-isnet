@echo off
REM 00_setup_project.bat - Initialize project structure

echo =====================================
echo SPRITE IS-NET PROJECT SETUP
echo =====================================
echo.

REM Define project paths
set "projectRoot=E:\Projects\sprite-isnet"
set "dataRoot=E:\Projects\sprite-data"

echo Creating project directory structure...
echo.

REM Create main project directories
if not exist "%projectRoot%" (
    mkdir "%projectRoot%"
    echo   Created: %projectRoot%
) else (
    echo   Exists: %projectRoot%
)

if not exist "%projectRoot%\scripts" (
    mkdir "%projectRoot%\scripts"
    echo   Created: %projectRoot%\scripts
) else (
    echo   Exists: %projectRoot%\scripts
)

if not exist "%projectRoot%\saved_models\sprite-isnet" (
    mkdir "%projectRoot%\saved_models\sprite-isnet"
    echo   Created: %projectRoot%\saved_models\sprite-isnet
) else (
    echo   Exists: %projectRoot%\saved_models\sprite-isnet
)

if not exist "%projectRoot%\logs" (
    mkdir "%projectRoot%\logs"
    echo   Created: %projectRoot%\logs
) else (
    echo   Exists: %projectRoot%\logs
)

if not exist "%projectRoot%\onnx_models" (
    mkdir "%projectRoot%\onnx_models"
    echo   Created: %projectRoot%\onnx_models
) else (
    echo   Exists: %projectRoot%\onnx_models
)

REM Create data directories
if not exist "%dataRoot%\train\images" (
    mkdir "%dataRoot%\train\images"
    echo   Created: %dataRoot%\train\images
) else (
    echo   Exists: %dataRoot%\train\images
)

if not exist "%dataRoot%\train\masks" (
    mkdir "%dataRoot%\train\masks"
    echo   Created: %dataRoot%\train\masks
) else (
    echo   Exists: %dataRoot%\train\masks
)

if not exist "%dataRoot%\valid\images" (
    mkdir "%dataRoot%\valid\images"
    echo   Created: %dataRoot%\valid\images
) else (
    echo   Exists: %dataRoot%\valid\images
)

if not exist "%dataRoot%\valid\masks" (
    mkdir "%dataRoot%\valid\masks"
    echo   Created: %dataRoot%\valid\masks
) else (
    echo   Exists: %dataRoot%\valid\masks
)

if not exist "%dataRoot%\test\images" (
    mkdir "%dataRoot%\test\images"
    echo   Created: %dataRoot%\test\images
) else (
    echo   Exists: %dataRoot%\test\images
)

if not exist "%dataRoot%\test\masks" (
    mkdir "%dataRoot%\test\masks"
    echo   Created: %dataRoot%\test\masks
) else (
    echo   Exists: %dataRoot%\test\masks
)

if not exist "%dataRoot%\test\output" (
    mkdir "%dataRoot%\test\output"
    echo   Created: %dataRoot%\test\output
) else (
    echo   Exists: %dataRoot%\test\output
)

echo.
echo Project structure created successfully!
echo.
echo Project root: %projectRoot%
echo Data root: %dataRoot%

REM Save paths to config file
echo # Project Configuration > "%projectRoot%\project_paths.txt"
echo PROJECT_ROOT=%projectRoot% >> "%projectRoot%\project_paths.txt"
echo DATA_ROOT=%dataRoot% >> "%projectRoot%\project_paths.txt"
echo MODELS_DIR=%projectRoot%\saved_models\sprite-isnet >> "%projectRoot%\project_paths.txt"
echo LOGS_DIR=%projectRoot%\logs >> "%projectRoot%\project_paths.txt"
echo ONNX_DIR=%projectRoot%\onnx_models >> "%projectRoot%\project_paths.txt"

echo.
echo Project paths saved to: %projectRoot%\project_paths.txt
echo.
pause
