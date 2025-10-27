# 04_train_isnet.ps1 - Master script for IS-Net training workflow

param(
    [switch]$SkipValidation = $false,
    [switch]$SkipVisualization = $false,
    [int]$Epochs = 200,
    [int]$BatchSize = 4
)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "SPRITE IS-NET TRAINING WORKFLOW" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

$projectRoot = "E:\Projects\sprite-isnet"
$dataRoot = "E:\sprite-data"
$isnetPath = "$projectRoot\DIS\IS-Net"

# Check if environment is set up
if (-not (Test-Path "$isnetPath\venv")) {
    Write-Host "`n✗ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup scripts first:" -ForegroundColor Yellow
    Write-Host "  .\00_setup_project.ps1" -ForegroundColor White
    Write-Host "  .\01_clone_and_setup.ps1" -ForegroundColor White
    Write-Host "  .\02_download_model.ps1" -ForegroundColor White
    exit 1
}

# Check if pretrained model exists
$pretrainedModel = "$projectRoot\DIS\saved_models\IS-Net\isnet-general-use.pth"
if (-not (Test-Path $pretrainedModel)) {
    Write-Host "`n⚠ Pretrained model not found!" -ForegroundColor Yellow
    Write-Host "Run: .\02_download_model.ps1" -ForegroundColor White
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne 'y') {
        exit 1
    }
}

# Activate environment
Write-Host "`n[Step 1/6] Activating virtual environment..." -ForegroundColor Yellow
Set-Location $isnetPath
.\venv\Scripts\Activate.ps1

# Verify data
if (-not $SkipValidation) {
    Write-Host "`n[Step 2/6] Validating dataset..." -ForegroundColor Yellow
    python "$projectRoot\scripts\03_prepare_data.py"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n✗ Data validation failed!" -ForegroundColor Red
        Write-Host "Please fix issues before training" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "`n[Step 2/6] Skipping validation (--SkipValidation)" -ForegroundColor Gray
}

# Test dataset loading
Write-Host "`n[Step 3/6] Testing dataset loading..." -ForegroundColor Yellow
python sprite_dataset.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Dataset loading failed!" -ForegroundColor Red
    exit 1
}

# Verify configuration
Write-Host "`n[Step 4/6] Verifying configuration..." -ForegroundColor Yellow
python config_sprite.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Configuration verification failed!" -ForegroundColor Red
    exit 1
}

# Start training
Write-Host "`n[Step 5/6] Starting training..." -ForegroundColor Green
Write-Host "  Epochs: $Epochs" -ForegroundColor Cyan
Write-Host "  Batch size: $BatchSize" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop training (state will be saved)" -ForegroundColor Yellow
Write-Host ""

# Update config if parameters provided
if ($Epochs -ne 200 -or $BatchSize -ne 4) {
    Write-Host "Updating configuration..." -ForegroundColor Cyan
    # Note: You'd need to modify config_sprite.py to accept command-line args
    # or create a temporary config file
}

python train_sprite_isnet.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Training failed or was interrupted" -ForegroundColor Red
    exit 1
}

# Training complete
Write-Host "`n[Step 6/6] Training complete!" -ForegroundColor Green

# Check if best model exists
$bestModel = "$projectRoot\saved_models\sprite-isnet\sprite_isnet_best.pth"
if (Test-Path $bestModel) {
    Write-Host "`n✓ Best model saved: $bestModel" -ForegroundColor Green
    
    $convert = Read-Host "`nConvert to ONNX now? (y/n)"
    if ($convert -eq 'y') {
        Write-Host "`nConverting to ONNX..." -ForegroundColor Yellow
        python convert_to_onnx.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n✓ ONNX conversion successful!" -ForegroundColor Green
            
            $test = Read-Host "`nTest with rembg now? (y/n)"
            if ($test -eq 'y') {
                python "$projectRoot\scripts\test_rembg_integration.py"
            }
        }
    }
} else {
    Write-Host "`n⚠ Best model not found - training may not have completed" -ForegroundColor Yellow
}

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "WORKFLOW COMPLETE" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
