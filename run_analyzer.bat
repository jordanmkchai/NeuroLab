@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "eeg_ecg analyser 2.py"
