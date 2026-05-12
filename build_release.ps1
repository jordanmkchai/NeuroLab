param(
    [string]$Version = "1.0.0",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$ReleaseDir = Join-Path $Root "release"
$DistDir = Join-Path $Root "dist"
$BuildDir = Join-Path $Root "build"
$Spec = Join-Path $Root "eeg_ecg_analyser.spec"
$AppDist = Join-Path $DistDir "EEG ECG Analyzer"
$ZipPath = Join-Path $ReleaseDir "EEG_ECG_Analyzer_Portable_v$Version.zip"
$InstallerScript = Join-Path $Root "installer\EEG_ECG_Analyzer.iss"

if (!(Test-Path -LiteralPath $Python)) {
    throw "Project Python not found: $Python"
}

New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

Write-Host "Installing/updating build dependencies..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $Root "requirement.txt")
& $Python -m pip install pyinstaller

Write-Host "Cleaning old build outputs..."
Remove-Item -LiteralPath $BuildDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $AppDist -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Building application with PyInstaller..."
Push-Location $Root
try {
    & $Python -m PyInstaller --noconfirm --clean $Spec
}
finally {
    Pop-Location
}

if (!(Test-Path -LiteralPath (Join-Path $AppDist "EEG ECG Analyzer.exe"))) {
    throw "Build failed: executable not found in $AppDist"
}

Write-Host "Creating portable zip..."
Remove-Item -LiteralPath $ZipPath -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $AppDist "*") -DestinationPath $ZipPath -Force
Write-Host "Portable zip: $ZipPath"

if (-not $SkipInstaller) {
    $Iscc = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
    $IsccPath = if ($null -ne $Iscc) { $Iscc.Source } else { $null }
    if ($null -eq $IsccPath) {
        $Candidates = @(
            (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
            (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
            (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
        )
        foreach ($Candidate in $Candidates) {
            if ($Candidate -and (Test-Path -LiteralPath $Candidate)) {
                $IsccPath = $Candidate
                break
            }
        }
    }

    if ($null -eq $IsccPath) {
        Write-Warning "Inno Setup compiler not found. Install Inno Setup, then rerun this script without -SkipInstaller."
        Write-Host "Portable zip is still ready to share."
    }
    else {
        Write-Host "Building Windows installer with Inno Setup..."
        & $IsccPath "/DMyAppVersion=$Version" $InstallerScript
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup failed with exit code $LASTEXITCODE"
        }
    }
}

Write-Host "Release build complete."
