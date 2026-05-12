# EEG ECG Analyzer Packaging

This folder can build desktop apps that users run without VS Code or a separate Python install.

## Windows Build

Run from PowerShell:

```powershell
cd D:\eeg_ecg_analyser
.\build_release.ps1 -Version 1.0.0
```

Windows outputs:

- `release\EEG_ECG_Analyzer_Portable_v1.0.0.zip`
- `release\EEG_ECG_Analyzer_Setup_v1.0.0.exe` if Inno Setup is installed

If Inno Setup is not installed, the portable zip is still created.

## Installer Compiler

Install Inno Setup from:

```text
https://jrsoftware.org/isinfo.php
```

Then rerun:

```powershell
.\build_release.ps1 -Version 1.0.0
```

## macOS Build

Windows cannot build a real macOS `.app`. The macOS app must be built on a Mac
or through GitHub Actions macOS runners.

### Option 1: GitHub Actions

Push the repository to GitHub, then open:

```text
GitHub repo -> Actions -> Build macOS App -> Run workflow
```

Enter a version such as `1.0.0`. The workflow builds two artifacts:

- `EEG_ECG_Analyzer_macOS_intel_v1.0.0`
- `EEG_ECG_Analyzer_macOS_apple-silicon_v1.0.0`

Download the artifact matching the user's Mac:

- Intel Mac: use `intel`
- M1/M2/M3/M4 Mac: use `apple-silicon`

### Option 2: Build On A Mac

Copy this project folder to a Mac and run:

```bash
cd ~/Downloads/eeg_ecg_analyser
chmod +x build_macos_release.sh
./build_macos_release.sh 1.0.0
```

macOS output:

```text
release/EEG_ECG_Analyzer_macOS_Portable_v1.0.0.zip
```

Mac users unzip it and double-click:

```text
EEG ECG Analyzer.app
```

If macOS blocks the unsigned app, right-click the app, choose `Open`, then
choose `Open` again. For broad public distribution, sign and notarize the app
with an Apple Developer ID certificate.

## Memory Files

The Windows installer and macOS app bundle current correction memory:

- `ecg_corrections_v2.json`
- `eeg_corrections_v1.json`

On first launch, Windows copies those files to:

```text
%APPDATA%\EEG_ECG_Analyser
```

On first launch, macOS copies those files to:

```text
~/Library/Application Support/EEG_ECG_Analyser
```

This lets the installed app keep learning user corrections without needing write access to the install folder.

## Sharing

For Windows users, share:

```text
release\EEG_ECG_Analyzer_Setup_v1.0.0.exe
```

If Windows SmartScreen warns users, choose `More info` then `Run anyway`. For broad distribution, use a code-signing certificate.

For macOS users, share the GitHub Actions artifact or the zip from a Mac build.
