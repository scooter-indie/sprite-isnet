# Prepare-SpriteData.ps1 - PowerShell wrapper for Ruby data preparation

param(
   [Parameter(Mandatory=$true)]
   [string]$RawDir,
  
   [string]$DataRoot = "C:\sprite-data",
  
   [ValidateSet('auto', 'corner', 'edge', 'threshold')]
   [string]$Method = 'auto',
  
   [int]$Fuzz = 10,
  
   [switch]$Preview,
  
   [double]$TrainRatio = 0.7,
   [double]$ValidRatio = 0.15,
   [double]$TestRatio = 0.15
)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "SPRITE DATA PREPARATION" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Check Ruby is installed
$rubyVersion = ruby --version 2>$null
if (-not $rubyVersion) {
   Write-Host "`n✗ Ruby not found!" -ForegroundColor Red
   Write-Host "Install with: choco install ruby" -ForegroundColor Yellow
   exit 1
}

Write-Host "`nRuby: $rubyVersion" -ForegroundColor Green

# Check ImageMagick is installed
$magickVersion = magick --version 2>$null
if (-not $magickVersion) {
   Write-Host "`n✗ ImageMagick not found!" -ForegroundColor Red
   Write-Host "Install with: choco install imagemagick" -ForegroundColor Yellow
   exit 1
}

Write-Host "ImageMagick: Installed" -ForegroundColor Green

# Verify input directory
if (-not (Test-Path $RawDir)) {
   Write-Host "`n✗ Raw directory not found: $RawDir" -ForegroundColor Red
   exit 1
}

# Build arguments
$args = @(
   'prepare_dataset.rb',
   '-m', $Method,
   '-f', $Fuzz,
   '--train', $TrainRatio,
   '--valid', $ValidRatio,
   '--test', $TestRatio
)

if ($Preview) {
   $args += '-p'
}

$args += @($RawDir, $DataRoot)

# Run Ruby script
Write-Host "`nStarting data preparation..." -ForegroundColor Yellow
Write-Host "  Raw dir: $RawDir" -ForegroundColor Cyan
Write-Host "  Data root: $DataRoot" -ForegroundColor Cyan
Write-Host "  Method: $Method" -ForegroundColor Cyan
Write-Host "  Fuzz: $Fuzz%" -ForegroundColor Cyan
Write-Host ""

ruby $args

if ($LASTEXITCODE -eq 0) {
   Write-Host "`n✓ Data preparation complete!" -ForegroundColor Green
} else {
   Write-Host "`n✗ Data preparation failed" -ForegroundColor Red
   exit 1
}
