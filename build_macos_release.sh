#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-1.0.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
RELEASE_DIR="$ROOT/release"
DIST_DIR="$ROOT/dist"
BUILD_DIR="$ROOT/build"
SPEC="$ROOT/eeg_ecg_analyser_macos.spec"
APP_PATH="$DIST_DIR/EEG ECG Analyzer.app"
PACKAGE_DIR="$RELEASE_DIR/EEG ECG Analyzer macOS"
ZIP_PATH="$RELEASE_DIR/EEG_ECG_Analyzer_macOS_Portable_v$VERSION.zip"
REQ_FILE="$ROOT/requirement.txt"

cd "$ROOT"

if [ ! -x "$PY" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV"
fi

mkdir -p "$RELEASE_DIR"

echo "Installing/updating build dependencies..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$REQ_FILE"
"$PY" -m pip install pyinstaller

echo "Cleaning old build outputs..."
rm -rf "$BUILD_DIR" "$APP_PATH" "$PACKAGE_DIR" "$ZIP_PATH"

echo "Building macOS app with PyInstaller..."
"$PY" -m PyInstaller --noconfirm --clean "$SPEC"

if [ ! -d "$APP_PATH" ]; then
    echo "Build failed: app not found at $APP_PATH" >&2
    exit 1
fi

if command -v codesign >/dev/null 2>&1; then
    echo "Applying ad-hoc code signature..."
    codesign --force --deep --sign - "$APP_PATH" || true
fi

echo "Creating portable zip..."
mkdir -p "$PACKAGE_DIR"
cp -R "$APP_PATH" "$PACKAGE_DIR/"
cat > "$PACKAGE_DIR/README_FIRST.txt" <<'README'
EEG ECG Analyzer for macOS

1. Unzip this folder.
2. Double-click EEG ECG Analyzer.app.
3. If macOS blocks it, right-click the app, choose Open, then choose Open again.

Correction memory is stored per user in:
~/Library/Application Support/EEG_ECG_Analyser
README

(
    cd "$RELEASE_DIR"
    ditto -c -k --sequesterRsrc --keepParent "EEG ECG Analyzer macOS" "$ZIP_PATH"
)

echo "Portable macOS zip: $ZIP_PATH"
echo "Release build complete."
