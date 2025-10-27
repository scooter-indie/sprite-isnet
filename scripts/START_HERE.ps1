# START_HERE.ps1 - Complete setup and training workflow

Write-Host @"

╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║           SPRITE IS-NET BACKGROUND REMOVAL                    ║
║           Complete Training Pipeline                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

Write-Host "This script will guide you through the complete setup and training process.`n" -ForegroundColor Yellow

# Menu
$choice = Read-Host @"

Choose an option:
  1. Complete setup (first time users)
  2. Prepare training data
  3. Start training
  4. Convert model to ONNX
  5. Test with rembg
  6. Batch process images
  7. Run all steps (automated)
  Q. Quit

Enter choice
"@

$scriptDir = $PSScriptRoot
$scriptsDir = Join-Path $scriptDir "scripts"

switch ($choice) {
    "1" {
        Write-Host "`nRunning complete setup...`n" -ForegroundColor Green
        & "$scriptsDir\00_setup_project.ps1"
        & "$scriptsDir\01_clone_and_setup.ps1"
        & "$scriptsDir\02_download_model.ps1"
        
        Write-Host "`n✓ Setup complete!" -ForegroundColor Green
        Write-Host "Next step: Prepare your training data" -ForegroundColor Yellow
        Write-Host "  Place images in: E:\sprite-data\train\images\" -ForegroundColor White
        Write-Host "  Place masks in: E:\sprite-data\train\masks\" -ForegroundColor White
    }
    
    "2" {
        Write-Host "`nPreparing training data...`n" -ForegroundColor Green
        Set-Location "$scriptDir\DIS\IS-Net"
        .\venv\Scripts\Activate.ps1
        python "$scriptsDir\03_prepare_data.py"
    }
    
    "3" {
        Write-Host "`nStarting training...`n" -ForegroundColor Green
        & "$scriptsDir\04_train_isnet.ps1"
    }
    
    "4" {
        Write-Host "`nConverting to ONNX...`n" -ForegroundColor Green
        Set-Location "$scriptDir\DIS\IS-Net"
        .\venv\Scripts\Activate.ps1
        python convert_to_onnx.py
    }
    
    "5" {
        Write-Host "`nTesting with rembg...`n" -ForegroundColor Green
        Set-Location "$scriptDir\DIS\IS-Net"
        .\venv\Scripts\Activate.ps1
        python "$scriptsDir\test_rembg_integration.py"
    }
    
    "6" {
        Write-Host "`nBatch processing...`n" -ForegroundColor Green
        $inputDir = Read-Host "Input directory"
        $outputDir = Read-Host "Output directory"
        & "$scriptsDir\05_batch_process.ps1" -InputDir $inputDir -OutputDir $outputDir
    }
    
    "7" {
        Write-Host "`nRunning complete automated workflow...`n" -ForegroundColor Green
        Write-Host "⚠ This will take several hours depending on your dataset size`n" -ForegroundColor Yellow
        
        $confirm = Read-Host "Continue? (y/n)"
        if ($confirm -eq 'y') {
            & "$scriptsDir\00_setup_project.ps1"
            & "$scriptsDir\01_clone_and_setup.ps1"
            & "$scriptsDir\02_download_model.ps1"
            
            Write-Host "`n⚠ Please prepare your training data now:" -ForegroundColor Yellow
            Write-Host "  1. Place sprite sheet images in: E:\sprite-data\train\images\" -ForegroundColor White
            Write-Host "  2. Place corresponding masks in: E:\sprite-data\train\masks\" -ForegroundColor White
            Read-Host "`nPress Enter when ready to continue"
            
            Set-Location "$scriptDir\DIS\IS-Net"
            .\venv\Scripts\Activate.ps1
            python "$scriptsDir\03_prepare_data.py"
            
            & "$scriptsDir\04_train_isnet.ps1"
        }
    }
    
    "Q" {
        Write-Host "`nGoodbye!" -ForegroundColor Cyan
        exit 0
    }
    
    default {
        Write-Host "`n✗ Invalid choice" -ForegroundColor Red
    }
}

Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
