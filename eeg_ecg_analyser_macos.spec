# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules("neurokit2")
hiddenimports += collect_submodules("openpyxl")
hiddenimports += [
    "matplotlib.backends.backend_tkagg",
    "PIL._tkinter_finder",
    "scipy.signal",
    "scipy.stats",
    "scipy.ndimage",
    "sklearn",
    "pywt",
]

datas = [
    ("ecg_corrections_v2.json", "."),
    ("eeg_corrections_v1.json", "."),
]

a = Analysis(
    ["eeg_ecg analyser 2.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EEG ECG Analyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EEG ECG Analyzer",
)

app = BUNDLE(
    coll,
    name="EEG ECG Analyzer.app",
    icon=None,
    bundle_identifier="com.sudep.eegecganalyzer",
)
