# 05_batch_process.ps1 - Batch process sprites with trained model

param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,
    
    [Parameter(Mandatory=$true)]
    [string]$OutputDir,
    
    [string]$ModelPath = "E:\Projects\sprite-isnet\onnx_models\sprite_isnet.onnx",
    
    [switch]$UseCLI = $true,
    [switch]$SaveMasks = $false,
    [switch]$CreateComparisons = $false
)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "BATCH SPRITE BACKGROUND REMOVAL" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Verify paths
if (-not (Test-Path $InputDir)) {
    Write-Host "`n✗ Input directory not found: $InputDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $ModelPath)) {
    Write-Host "`n✗ Model not found: $ModelPath" -ForegroundColor Red
    exit 1
}

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "`nConfiguration:" -ForegroundColor Yellow
Write-Host "  Input: $InputDir" -ForegroundColor White
Write-Host "  Output: $OutputDir" -ForegroundColor White
Write-Host "  Model: $ModelPath" -ForegroundColor White
Write-Host "  Save masks: $SaveMasks" -ForegroundColor White
Write-Host ""

# Get list of images
$images = Get-ChildItem -Path $InputDir -Include *.png,*.jpg,*.jpeg -File

if ($images.Count -eq 0) {
    Write-Host "✗ No images found in input directory" -ForegroundColor Red
    exit 1
}

Write-Host "Found $($images.Count) images to process" -ForegroundColor Cyan
Write-Host ""

if ($UseCLI) {
    # Use rembg CLI (more reliable)
    Write-Host "Using rembg CLI mode..." -ForegroundColor Yellow
    
    # Setup model in rembg directory
    $rembgDir = "$env:USERPROFILE\.u2net"
    $modelName = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)
    $rembgModelPath = "$rembgDir\$modelName.onnx"
    
    if (-not (Test-Path $rembgModelPath)) {
        Write-Host "Copying model to rembg directory..." -ForegroundColor Cyan
        Copy-Item $ModelPath -Destination $rembgModelPath -Force
    }
    
    $successCount = 0
    $failCount = 0
    $totalTime = 0
    
    foreach ($img in $images) {
        $outputPath = Join-Path $OutputDir "$($img.BaseName)_nobg.png"
        
        Write-Host "Processing: $($img.Name)" -ForegroundColor Gray -NoNewline
        
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        
        $result = rembg i -m u2net_custom -x "{`"model_path`": `"~/.u2net/$modelName.onnx`"}" $img.FullName $outputPath 2>&1
        
        $stopwatch.Stop()
        $elapsed = $stopwatch.Elapsed.TotalSeconds
        $totalTime += $elapsed
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host " ✓ ($([math]::Round($elapsed, 2))s)" -ForegroundColor Green
            $successCount++
            
            # Create comparison if requested
            if ($CreateComparisons) {
                $compPath = Join-Path $OutputDir "$($img.BaseName)_comparison.png"
                # You'd use ImageMagick or similar here
                # magick convert $img.FullName $outputPath +append $compPath
            }
        } else {
            Write-Host " ✗ Failed" -ForegroundColor Red
            $failCount++
        }
    }
    
    # Summary
    $avgTime = $totalTime / $images.Count
    
    Write-Host "`n=====================================" -ForegroundColor Cyan
    Write-Host "Batch Processing Complete" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "Successful: $successCount/$($images.Count)" -ForegroundColor Green
    Write-Host "Failed: $failCount/$($images.Count)" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Gray" })
    Write-Host "Average time: $([math]::Round($avgTime, 2)) seconds/image" -ForegroundColor Cyan
    Write-Host "Total time: $([math]::Round($totalTime, 2)) seconds" -ForegroundColor Cyan
    Write-Host "Output: $OutputDir" -ForegroundColor White
    
} else {
    # Use Python inference script
    Write-Host "Using Python inference mode..." -ForegroundColor Yellow
    
    $isnetPath = "E:\Projects\sprite-isnet\DIS\IS-Net"
    Set-Location $isnetPath
    .\venv\Scripts\Activate.ps1
    
    $args = @(
        "inference_sprite.py",
        "--model", $ModelPath,
        "--input", $InputDir,
        "--output", $OutputDir,
        "--batch"
    )
    
    if ($SaveMasks) {
        $args += "--save-masks"
    }
    
    python $args
}
