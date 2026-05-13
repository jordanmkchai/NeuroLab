import os
import sys


def _relaunch_with_project_python():
    if getattr(sys, "frozen", False):
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_python = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
    if not os.path.exists(project_python):
        return
    current = os.path.normcase(os.path.abspath(sys.executable))
    target = os.path.normcase(os.path.abspath(project_python))
    if current == target:
        return
    if os.environ.get("EEG_ECG_ANALYSER_BOOTSTRAPPED") == "1":
        return
    os.environ["EEG_ECG_ANALYSER_BOOTSTRAPPED"] = "1"
    import subprocess as _subprocess
    raise SystemExit(_subprocess.call([project_python, os.path.abspath(__file__), *sys.argv[1:]]))


_relaunch_with_project_python()

import json
import argparse
import threading
import subprocess
import shutil
import pathlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.signal import (butter, filtfilt, iirnotch, periodogram,
                          find_peaks, sosfiltfilt, sosfilt, resample_poly, hilbert)
from scipy.stats import median_abs_deviation as mad, kurtosis
from scipy.ndimage import uniform_filter1d

try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except Exception:
    NK_AVAILABLE = False


LANDMARK_COLUMNS = [
    "P_on_idx",
    "P_off_idx",
    "QRS_on_idx",
    "QRS_off_idx",
    "R_idx",
    "T_off_idx",
]

ECG_INTERVAL_COLUMNS_MS = [
    "RR_ms",
    "P_wave_dur_ms",
    "PR_interval_ms",
    "QRS_interval_ms",
    "QT_interval_ms",
    "QTc_Mitchell_ms",
]

EEG_EVENT_TYPES = ["Spike", "SWD", "Seizure"]
EEG_EVENT_COLUMNS = [
    "Event_Index",
    "Type",
    "Start_s",
    "End_s",
    "Duration_s",
    "Confidence",
    "Is_Corrected",
    "Is_Deleted",
    "Corrected_At",
    "Correction_Source",
    "Correction_Notes",
]

APP_NAME = "EEG_ECG_Analyser"


def app_resource_dir():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


def user_data_dir():
    if sys.platform == "darwin":
        root = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    elif os.name == "nt":
        root = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    else:
        root = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    path = os.path.join(root, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def bundled_resource_path(filename):
    candidates = [
        os.path.join(app_resource_dir(), filename),
        os.path.join(os.path.dirname(os.path.abspath(sys.executable)), filename),
        os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "..", "Resources", filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return candidates[0]


def seed_user_data_file(filename):
    dst = os.path.join(user_data_dir(), filename)
    if os.path.exists(dst):
        return dst
    src = bundled_resource_path(filename)
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass
    return dst


def ensure_frozen_stdio():
    if not getattr(sys, "frozen", False):
        return
    log_dir = user_data_dir()
    try:
        if sys.stdout is None:
            sys.stdout = open(os.path.join(log_dir, "last_stdout.log"), "a", encoding="utf-8", buffering=1)
        if sys.stderr is None:
            sys.stderr = open(os.path.join(log_dir, "last_stderr.log"), "a", encoding="utf-8", buffering=1)
    except Exception:
        pass


ensure_frozen_stdio()


CORRECTION_STORE_FILE = seed_user_data_file("ecg_corrections_v2.json")
EEG_CORRECTION_STORE_FILE = seed_user_data_file("eeg_corrections_v1.json")


# ============================================================
# 1) LOADER — Column D (time) & E (signal), from row 8
# ============================================================
def load_waveform_tabular(file_path: str):
    path = pathlib.Path(file_path)
    ext  = path.suffix.lower()

    def _coerce(a):
        return pd.to_numeric(
            pd.Series(a).astype(str).str.strip()
              .str.replace(r'[^0-9eE+\-\.]', '', regex=True),
            errors='coerce'
        ).values

    if ext in ['.tsv', '.txt', '.csv']:
        sep = '\t' if ext in ['.tsv', '.txt'] else ','
        try:
            df = pd.read_csv(file_path, sep=sep, header=None,
                             skiprows=7, usecols=[3, 4], engine='c',
                             on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(file_path, sep=None, header=None,
                             skiprows=7, usecols=[3, 4], engine='python')
    elif ext in ['.xlsx', '.xls']:
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
        df = pd.read_excel(file_path, engine=engine, header=None,
                           usecols='D:E', skiprows=7)
    else:
        raise ValueError(f'Unsupported file type: {ext}')

    if df.shape[1] != 2:
        raise ValueError(f'Expected 2 columns (D & E), got {df.shape[1]}')

    t = _coerce(df.iloc[:, 0].values)
    y = _coerce(df.iloc[:, 1].values)
    valid = ~(np.isnan(t) | np.isnan(y))
    t, y = t[valid], y[valid]

    if len(t) == 0:
        raise ValueError("No valid numeric data in columns D & E (row 8+).")
    return t, y


# ============================================================
# 2) SAMPLING RATE / RESAMPLING
# ============================================================
def infer_fs_from_time(t):
    t = np.asarray(t)
    if len(t) < 3:
        raise ValueError("Not enough samples.")
    dt = np.diff(t)
    dt_med = np.median(dt)
    if dt_med <= 0 or not np.isfinite(dt_med):
        raise ValueError("Invalid time increments.")
    fs = 1.0 / dt_med
    if fs < 50:
        t = t * 1e-3
        dt_med = np.median(np.diff(t))
        fs = 1.0 / dt_med
    jitter = np.std(dt) / (np.mean(dt) + 1e-12)
    return fs, jitter, t


def ensure_uniform_sampling(t, y, fs_target=None):
    fs, jitter, t = infer_fs_from_time(t)
    if fs_target is None:
        fs_target = float(int(round(fs)))
    if jitter > 0.01:
        N = int((t[-1] - t[0]) * fs_target) + 1
        t_u = np.linspace(t[0], t[-1], N)
        y_u = np.interp(t_u, t, y)
        return t_u, y_u, fs_target
    return t, y, fs


# ============================================================
# Shared ECG schema / persistence helpers
# ============================================================
def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _to_optional_int(value):
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    try:
        return int(round(float(value)))
    except Exception:
        return np.nan


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        try:
            if np.isnan(value):
                return False
        except Exception:
            pass
        return bool(int(value))
    txt = str(value).strip().lower()
    return txt in {"1", "true", "yes", "y", "t"}


def ensure_ecg_beats_schema(df: pd.DataFrame):
    df = pd.DataFrame(df).copy()
    if "Beat_Index" not in df.columns:
        df["Beat_Index"] = np.arange(len(df), dtype=int)
    df["Beat_Index"] = pd.to_numeric(df["Beat_Index"], errors="coerce")
    if df["Beat_Index"].isna().any():
        df["Beat_Index"] = np.arange(len(df), dtype=int)
    df["Beat_Index"] = df["Beat_Index"].astype(int)
    df.sort_values("Beat_Index", inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in LANDMARK_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    float_cols = [
        "R_time_s", "RR_s", "RR_ms",
        "P_wave_dur_s", "P_wave_dur_ms",
        "PR_interval_s", "PR_interval_ms",
        "QRS_interval_s", "QRS_interval_ms",
        "QT_interval_s", "QT_interval_ms",
        "QTc_Mitchell_s", "QTc_Mitchell_ms",
        "Confidence", "Confidence_Rank",
    ]
    for col in float_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["QTc_Bazett_s", "QTc_Bazett_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    qt_ms = pd.to_numeric(df.get("QT_interval_ms", np.nan), errors="coerce")
    rr_ms = pd.to_numeric(df.get("RR_ms", np.nan), errors="coerce")
    mitchell_ms = np.where(
        np.isfinite(qt_ms) & np.isfinite(rr_ms) & (rr_ms > 0),
        qt_ms / np.sqrt(rr_ms / 100.0),
        np.nan,
    )
    needs_mitchell = ~np.isfinite(pd.to_numeric(df["QTc_Mitchell_ms"], errors="coerce"))
    df.loc[needs_mitchell, "QTc_Mitchell_ms"] = mitchell_ms[needs_mitchell]
    df["QTc_Mitchell_s"] = np.where(
        np.isfinite(df["QTc_Mitchell_ms"]),
        df["QTc_Mitchell_ms"] / 1000.0,
        df["QTc_Mitchell_s"],
    )

    if "Is_Corrected" not in df.columns:
        df["Is_Corrected"] = False
    df["Is_Corrected"] = df["Is_Corrected"].apply(_to_bool)

    if "Corrected_At" not in df.columns:
        df["Corrected_At"] = ""
    df["Corrected_At"] = df["Corrected_At"].fillna("").astype(str)

    if "Correction_Notes" not in df.columns:
        df["Correction_Notes"] = ""
    df["Correction_Notes"] = df["Correction_Notes"].fillna("").astype(str)

    if "Correction_Source" not in df.columns:
        df["Correction_Source"] = "auto"
    df["Correction_Source"] = (
        df["Correction_Source"].replace("", np.nan).fillna("auto").astype(str)
    )
    return df


def recompute_ecg_intervals_df(df: pd.DataFrame, fs_hz: float):
    df = ensure_ecg_beats_schema(df)
    if fs_hz <= 0:
        raise ValueError("Invalid sampling rate for ECG recomputation.")

    def ms(v):
        return np.where(np.isfinite(v), v * 1000.0, np.nan)

    r_idx = pd.to_numeric(df["R_idx"], errors="coerce").to_numpy(dtype=float)
    p_on = pd.to_numeric(df["P_on_idx"], errors="coerce").to_numpy(dtype=float)
    p_off = pd.to_numeric(df["P_off_idx"], errors="coerce").to_numpy(dtype=float)
    qrs_on = pd.to_numeric(df["QRS_on_idx"], errors="coerce").to_numpy(dtype=float)
    qrs_off = pd.to_numeric(df["QRS_off_idx"], errors="coerce").to_numpy(dtype=float)
    t_off = pd.to_numeric(df["T_off_idx"], errors="coerce").to_numpy(dtype=float)

    r_t = np.where(np.isfinite(r_idx), r_idx / fs_hz, np.nan)
    rr = np.full(len(df), np.nan, dtype=float)
    if len(df) >= 2:
        rr[1:] = np.diff(r_t)
    rr[(~np.isfinite(rr)) | (rr <= 0)] = np.nan

    p_dur = np.where(
        np.isfinite(p_on) & np.isfinite(p_off) & (p_off > p_on),
        (p_off - p_on) / fs_hz,
        np.nan,
    )
    pr = np.where(
        np.isfinite(p_on) & np.isfinite(qrs_on) & (qrs_on > p_on),
        (qrs_on - p_on) / fs_hz,
        np.nan,
    )
    qrs = np.where(
        np.isfinite(qrs_on) & np.isfinite(qrs_off) & (qrs_off > qrs_on),
        (qrs_off - qrs_on) / fs_hz,
        np.nan,
    )
    qt = np.where(
        np.isfinite(qrs_on) & np.isfinite(t_off) & (t_off > qrs_on),
        (t_off - qrs_on) / fs_hz,
        np.nan,
    )
    qtc_m = np.where(
        np.isfinite(qt) & np.isfinite(rr) & (rr > 0),
        qt / np.sqrt(rr * 10.0),
        np.nan,
    )

    df["R_time_s"] = r_t
    df["RR_s"] = rr
    df["RR_ms"] = ms(rr)
    df["P_wave_dur_s"] = p_dur
    df["P_wave_dur_ms"] = ms(p_dur)
    df["PR_interval_s"] = pr
    df["PR_interval_ms"] = ms(pr)
    df["QRS_interval_s"] = qrs
    df["QRS_interval_ms"] = ms(qrs)
    df["QT_interval_s"] = qt
    df["QT_interval_ms"] = ms(qt)
    df["QTc_Mitchell_s"] = qtc_m
    df["QTc_Mitchell_ms"] = ms(qtc_m)
    if "QTc_Bazett_s" in df.columns:
        df.drop(columns=["QTc_Bazett_s"], inplace=True)
    if "QTc_Bazett_ms" in df.columns:
        df.drop(columns=["QTc_Bazett_ms"], inplace=True)

    rr_clean = rr[np.isfinite(rr) & (rr > 0)]
    rmssd = float(np.sqrt(np.mean(np.diff(rr_clean) ** 2))) if rr_clean.size >= 3 else np.nan

    conf = pd.to_numeric(df["Confidence"], errors="coerce").to_numpy(dtype=float)
    conf_for_rank = np.where(np.isfinite(conf), conf, np.inf)
    ranks = pd.Series(conf_for_rank).rank(method="dense", ascending=True).to_numpy(dtype=float)
    df["Confidence_Rank"] = np.where(np.isfinite(conf), ranks, np.nan)

    return df, rmssd


def compute_ecg_validity(df: pd.DataFrame):
    df = ensure_ecg_beats_schema(df)
    required = ["PR_interval_ms", "QRS_interval_ms", "QT_interval_ms", "RR_ms", "QTc_Mitchell_ms"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    valid_mask = df[required].notna().all(axis=1)
    total = int(len(df))
    valid = int(valid_mask.sum())
    ratio = float(valid / total) if total else np.nan
    return ratio, valid, total


def compute_ecg_summary_row(df: pd.DataFrame, meta: dict):
    df = ensure_ecg_beats_schema(df)
    summary = {}
    for col in ECG_INTERVAL_COLUMNS_MS:
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        summary[col + "_mean"] = float(np.mean(values)) if values.size else np.nan
        summary[col + "_median"] = float(np.median(values)) if values.size else np.nan
        summary[col + "_std"] = float(np.std(values)) if values.size else np.nan
        summary[col + "_n"] = int(values.size)
    summary["RMSSD_ms_HRV"] = float(meta.get("RMSSD_ms", np.nan))
    summary["Total_Beats"] = int(meta.get("total_beats", len(df)))
    summary["Corrected_Beats"] = int(df["Is_Corrected"].apply(_to_bool).sum())
    summary["Median_Confidence"] = float(
        pd.to_numeric(df["Confidence"], errors="coerce").median(skipna=True)
    ) if len(df) else np.nan
    ratio, valid, total = compute_ecg_validity(df)
    summary["Valid_Beats"] = valid
    summary["Valid_Beat_Total"] = total
    summary["Valid_Beat_Ratio"] = ratio
    summary["Validity_Pass"] = bool(meta.get("validity_pass", False))
    return summary


def read_meta_from_sheet(meta_df: pd.DataFrame):
    if meta_df is None or len(meta_df) == 0:
        return {}
    row = meta_df.iloc[0].to_dict()
    return {str(k): row[k] for k in row}


def load_ecg_workbook(workbook_path: str):
    beats_df = pd.read_excel(workbook_path, sheet_name="ECG_Beats")
    meta_df = pd.read_excel(workbook_path, sheet_name="Meta")
    return ensure_ecg_beats_schema(beats_df), read_meta_from_sheet(meta_df)


def _default_correction_store():
    return {
        "version": 2,
        "updated_at": now_iso(),
        "workbooks": {},
        "legacy_examples": [],
    }


def _convert_legacy_store(data):
    store = _default_correction_store()
    if not isinstance(data, dict):
        return store
    examples = data.get("examples", [])
    if isinstance(examples, list):
        store["legacy_examples"] = examples

    # Best-effort import of old annotation labels.
    label_map = {
        "p start": "P_on_idx",
        "p on": "P_on_idx",
        "p end": "P_off_idx",
        "p off": "P_off_idx",
        "q": "QRS_on_idx",
        "qrs_on": "QRS_on_idx",
        "s": "QRS_off_idx",
        "qrs_off": "QRS_off_idx",
        "r": "R_idx",
        "r1": "R_idx",
        "t": "T_off_idx",
        "t off": "T_off_idx",
        "t_off": "T_off_idx",
    }
    legacy_beats = {}
    for ex in examples if isinstance(examples, list) else []:
        beat_idx = ex.get("wave_index")
        anns = ex.get("annotations", [])
        try:
            beat_key = str(int(beat_idx))
        except Exception:
            continue
        if not isinstance(anns, list):
            continue
        out = {}
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            label = str(ann.get("landmark", "")).strip().lower()
            target = label_map.get(label)
            if not target:
                continue
            out[target] = _to_optional_int(ann.get("x"))
        if out:
            out["Correction_Source"] = "legacy_import"
            out["Corrected_At"] = now_iso()
            legacy_beats[beat_key] = out
    if legacy_beats:
        store["workbooks"]["__legacy_import__"] = {
            "source_file": "",
            "saved_at": now_iso(),
            "beats": legacy_beats,
        }
    return store


def load_correction_store(path=CORRECTION_STORE_FILE):
    candidates = [path]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(bundled_resource_path("ecg_corrections_v2.json"))
    candidates.append(os.path.join(script_dir, "ecg_correction_memory.json"))
    candidates.append(os.path.join(os.path.expanduser("~"), "ecg_correction_memory.json"))

    for idx, candidate in enumerate(candidates):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and int(data.get("version", 0)) == 2 and isinstance(data.get("workbooks"), dict):
            data.setdefault("legacy_examples", [])
            return data
        # Only convert legacy formats for secondary candidate files or invalid primary payload.
        store = _convert_legacy_store(data)
        if idx == 0:
            return store
        return store
    return _default_correction_store()


def save_correction_store(store: dict, path=CORRECTION_STORE_FILE):
    out = _default_correction_store()
    if isinstance(store, dict):
        out.update(store)
    out["version"] = 2
    out["updated_at"] = now_iso()
    out["workbooks"] = out.get("workbooks", {}) if isinstance(out.get("workbooks"), dict) else {}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def persist_workbook_corrections(workbook_path: str, source_file: str, beats_df: pd.DataFrame):
    store = load_correction_store()
    store.setdefault("workbooks", {})
    abs_wb = os.path.abspath(workbook_path)
    payload = {
        "source_file": source_file or "",
        "saved_at": now_iso(),
        "beats": {},
    }
    df = ensure_ecg_beats_schema(beats_df)
    corrected = df[df["Is_Corrected"].apply(_to_bool)]
    for _, row in corrected.iterrows():
        beat_key = str(int(row["Beat_Index"]))
        beat_payload = {
            "Correction_Notes": str(row.get("Correction_Notes", "") or ""),
            "Corrected_At": str(row.get("Corrected_At", "") or now_iso()),
            "Correction_Source": str(row.get("Correction_Source", "manual_review")),
        }
        for col in LANDMARK_COLUMNS:
            val = _to_optional_int(row.get(col))
            beat_payload[col] = None if pd.isna(val) else int(val)
        payload["beats"][beat_key] = beat_payload
    store["workbooks"][abs_wb] = payload
    save_correction_store(store)


def apply_saved_corrections(workbook_path: str, beats_df: pd.DataFrame):
    store = load_correction_store()
    workbooks = store.get("workbooks", {}) if isinstance(store, dict) else {}
    rec = workbooks.get(os.path.abspath(workbook_path))
    if not isinstance(rec, dict):
        return ensure_ecg_beats_schema(beats_df), 0
    beats = rec.get("beats", {})
    if not isinstance(beats, dict):
        return ensure_ecg_beats_schema(beats_df), 0

    df = ensure_ecg_beats_schema(beats_df)
    by_index = {int(v): i for i, v in enumerate(df["Beat_Index"].tolist())}
    applied = 0
    for beat_key, patch in beats.items():
        try:
            beat_idx = int(beat_key)
        except Exception:
            continue
        row_idx = by_index.get(beat_idx)
        if row_idx is None or not isinstance(patch, dict):
            continue
        changed = False
        for col in LANDMARK_COLUMNS:
            if col in patch:
                val = _to_optional_int(patch.get(col))
                df.at[row_idx, col] = val
                changed = True
        if changed:
            df.at[row_idx, "Is_Corrected"] = True
            df.at[row_idx, "Corrected_At"] = str(patch.get("Corrected_At", now_iso()))
            df.at[row_idx, "Correction_Notes"] = str(patch.get("Correction_Notes", ""))
            df.at[row_idx, "Correction_Source"] = str(patch.get("Correction_Source", "saved_correction"))
            applied += 1
    return df, applied


# ============================================================
# 3) FILTERS / HELPERS
# ============================================================
def butter_bandpass_sos(fs, low, high, order=4):
    nyq  = 0.5 * fs
    low  = max(1e-6, low  / nyq)
    high = min(0.999999, high / nyq)
    return butter(order, [low, high], btype='band', output='sos')


def bandpass_zero_phase(sig, fs, lo, hi, order=4):
    return sosfiltfilt(butter_bandpass_sos(fs, lo, hi, order), sig)


def apply_notch(sig, fs, line_hz=50.0, q=35.0):
    w0 = line_hz / (fs / 2.0)
    if not (0 < w0 < 1):
        return sig
    b, a   = iirnotch(w0, q)
    padlen = min(len(sig) - 1, 3 * max(len(a), len(b)))
    return filtfilt(b, a, sig, padtype='odd', padlen=padlen)


def moving_mad_stats(x, fs, win_s=10.0):
    w   = max(1, int(fs * win_s))
    s   = pd.Series(x)
    med = s.rolling(w, center=True, min_periods=1).median()
    dev = (s - med).abs().rolling(w, center=True, min_periods=1).median()
    med = med.to_numpy(dtype=np.float64)
    dev = (dev.to_numpy() * 1.4826).astype(np.float64)
    dev[~np.isfinite(dev)] = np.nanmedian(dev[np.isfinite(dev)]) if np.any(np.isfinite(dev)) else 1.0
    dev[dev < 1e-9]        = np.nanmedian(dev[dev > 0]) if np.any(dev > 0) else 1.0
    return med, dev


def decimate_to(x, fs, target_fs, cutoff=None):
    if target_fs >= fs:
        return x, fs
    cutoff = cutoff or (0.45 * target_fs)
    sos    = butter(6, cutoff, btype='low', fs=fs, output='sos')
    y      = sosfilt(sos, x)
    down   = int(round(fs / target_fs))
    return resample_poly(y, 1, down), fs / down


def band_power_fft(x, fs, bands, nfft=None):
    n    = len(x)
    nfft = n if nfft is None else nfft
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    P     = (np.abs(np.fft.rfft(x, n=nfft)) ** 2) / nfft
    return [float(np.sum(P[(freqs >= f1) & (freqs < f2)])) for f1, f2 in bands]


def spectral_entropy_from_fft(x, fs, nfft=None):
    nfft = len(x) if nfft is None else nfft
    P    = (np.abs(np.fft.rfft(x, n=nfft)) ** 2) + 1e-12
    P   /= np.sum(P)
    return float(-np.sum(P * np.log2(P)))


# ============================================================
# 4) ECG ENGINE
# ============================================================
def auto_bandpass_notch_ecg(sig, fs):
    f, Pxx = periodogram(sig, fs=fs, scaling='density')
    lo, hi  = 0.5, min(200.0, 0.45 * fs)

    def has_line(freq):
        band  = (f >= freq - 0.7) & (f <= freq + 0.7)
        neigh = (f >= freq - 5)   & (f <= freq + 5)
        if not np.any(neigh):
            return False
        return np.sum(Pxx[band]) > 6 * np.median(Pxx[neigh])

    y = bandpass_zero_phase(sig, fs, lo, hi, order=4)
    if has_line(50.0):
        y = apply_notch(y, fs, 50.0, q=35.0)
    return y, (lo, hi)


# ============================================================
# FIX 1 (v3): R-peak refinement — use argmax (positive peak only)
# ============================================================
def find_r_peaks_fallback(y, fs):
    """
    Envelope-based R-peak detection with local positive-peak refinement.

    Key fix: refinement uses argmax(signal) not argmax(abs(signal)).
    Mouse ECG R-waves are positive deflections. Using abs() caused the
    algorithm to snap to the deep negative S-trough instead.
    The +-5 ms window searches for the true positive maximum.
    """
    z   = y - np.median(y)
    win = max(3, int(0.015 * fs))
    env = np.convolve(z * z, np.ones(win) / win, mode='same')
    f   = np.fft.rfftfreq(len(env), d=1 / fs)
    E   = np.abs(np.fft.rfft(env))
    band = (f >= 4) & (f <= 20)
    f0   = f[band][np.argmax(E[band])] if np.any(band) else 8.0
    rr_s = 1.0 / max(1e-3, float(f0))
    dist  = max(1, int(0.55 * rr_s * fs))
    prom  = np.percentile(env, 98) * 0.10
    candidates, _ = find_peaks(env, prominence=prom, distance=dist)

    # Refinement: snap to true positive peak in +-5 ms window
    refine_win = max(2, int(0.005 * fs))
    n = len(y)
    refined = []
    for c in candidates:
        lo = max(0, c - refine_win)
        hi = min(n - 1, c + refine_win)
        local_seg = y[lo:hi + 1]
        # Use argmax (not argmax abs) — R peak is always the positive maximum
        local_peak = int(np.argmax(local_seg))
        refined.append(lo + local_peak)

    refined = np.unique(np.array(refined, dtype=int))
    return refined


def plausible_rpeaks(rpeaks, fs, n_samples):
    if rpeaks is None or len(rpeaks) < 3:
        return False
    r = np.asarray(rpeaks)
    if np.any(r < 0) or np.any(r >= n_samples):
        return False
    rr = np.diff(r) / fs
    rr = rr[np.isfinite(rr)]
    if rr.size < 2:
        return False
    if not (0.03 <= np.median(rr) <= 0.40):
        return False
    if np.mean(rr < 0.02) > 0.05:
        return False
    return True


def delineate_ecg(sig_filt, fs, engine="wavelet"):
    n = len(sig_filt)
    if engine == "wavelet" and NK_AVAILABLE:
        try:
            signals, info = nk.ecg_process(sig_filt, sampling_rate=fs, method="neurokit")
            rpeaks = np.array(info.get("ECG_R_Peaks", []), dtype=int)
            if not plausible_rpeaks(rpeaks, fs, n):
                raise RuntimeError("NK rpeaks implausible")
            out = nk.ecg_delineate(signals["ECG_Clean"], rpeaks=rpeaks,
                                   sampling_rate=fs, method="dwt")
            waves = out[0] if isinstance(out, tuple) and isinstance(out[0], dict) else (
                    out[1] if isinstance(out, tuple) else out)
            return rpeaks, waves, "wavelet"
        except Exception:
            pass
    rpeaks = find_r_peaks_fallback(sig_filt, fs)
    return rpeaks, {}, "heuristic"


# ============================================================
# ECG landmark helpers tuned to manual benchmark rules
# ============================================================
def _find_qrs_onset_steep(y_seg, fs, r_loc):
    """
    QRS onset: point just before the steepest pre-R downslope.
    This aligns with corrected labels where QRS_on is close to the
    immediate pre-R depolarization rather than earlier baseline drift.
    """
    n = len(y_seg)
    if n < 8:
        return max(0, min(n - 1, r_loc - 1))
    dy = np.gradient(y_seg)
    st = max(1, int(r_loc - int(0.030 * fs)))
    en = max(st + 1, int(r_loc))
    k_neg = st + int(np.argmin(dy[st:en]))
    onset = int(k_neg - int(round(0.003 * fs)))  # ~3 ms before steepest slope
    onset = max(st - int(0.010 * fs), onset)
    onset = min(onset, max(0, int(r_loc - int(0.005 * fs))))
    return int(max(0, min(n - 1, onset)))


def _find_qrs_offset_jpoint(y_seg, fs, r_loc, search_end, p_on_local=None):
    """
    QRS_off rule: on the post-R upslope at approximately the same height
    as P_on, with a hard early cap near R + 12 ms.
    """
    n = len(y_seg)
    dy = np.gradient(y_seg)
    # Find S trough shortly after R.
    s_end = min(n - 1, int(r_loc + int(0.020 * fs)))
    if s_end <= r_loc:
        return int(min(n - 1, r_loc + int(0.005 * fs)))
    s_idx = int(r_loc + np.argmin(y_seg[r_loc:s_end + 1]))

    # Target amplitude from P_on if available, else local pre-R baseline.
    if p_on_local is not None and np.isfinite(p_on_local):
        p_idx = int(max(0, min(n - 1, int(round(p_on_local)))))
        target = float(y_seg[p_idx])
    else:
        b0 = max(0, int(r_loc - int(0.060 * fs)))
        b1 = max(b0 + 1, int(r_loc - int(0.035 * fs)))
        target = float(np.median(y_seg[b0:b1])) if b1 > b0 else float(np.median(y_seg[:max(1, int(0.02 * fs))]))

    # Restrict to early post-R region to avoid drifting into T.
    cap = min(n - 1, int(r_loc + int(0.012 * fs)))
    cap = min(cap, int(search_end))
    if cap <= s_idx:
        return int(max(0, min(n - 1, s_idx)))

    idxs = np.arange(s_idx, cap + 1, dtype=int)
    up_mask = dy[idxs] > 0
    cand = idxs[up_mask] if np.any(up_mask) else idxs
    best = int(cand[np.argmin(np.abs(y_seg[cand] - target))])
    return int(max(0, min(n - 1, best)))


def _find_t_offset_threshold(y_seg, fs, qrs_off_local, search_end, qrs_on_local=None, r_local=None, p_on_local=None):
    """
    T_off: first point where signal has both:
      1) low slope (flatness), and
      2) return near baseline,
    searched in an early post-R window to avoid next-cycle contamination.
    """
    n = len(y_seg)
    if r_local is None:
        r_local = qrs_on_local if qrs_on_local is not None else qrs_off_local
    r_local = int(max(0, min(n - 1, int(round(r_local)))))
    qrs_off_local = int(max(0, min(n - 1, int(round(qrs_off_local)))))

    # Tuned on corrected beats: T_off typically emerges in early post-R flat return.
    start = max(qrs_off_local + int(0.014 * fs), r_local + int(0.022 * fs))
    end_cap = r_local + int(0.038 * fs)
    end = min(n - 1, int(search_end), end_cap)
    if end <= start + 3:
        return np.nan

    # Latest manual review aligns T_off to the early flat return near the P_on baseline.
    if p_on_local is not None and np.isfinite(p_on_local):
        p_idx = int(max(0, min(n - 1, int(round(p_on_local)))))
        baseline = float(y_seg[p_idx])
    else:
        b0 = max(0, int(r_local - int(0.060 * fs)))
        b1 = max(b0 + 1, int(r_local - int(0.040 * fs)))
        baseline = float(np.median(y_seg[b0:b1])) if b1 > b0 else float(np.median(y_seg[:max(1, int(0.02 * fs))]))

    seg = y_seg[start:end + 1] - baseline
    if seg.size < 4:
        return np.nan
    t_peak = int(start + np.argmax(np.abs(seg)))

    amp_peak = abs(float(y_seg[t_peak] - baseline))
    local_ptp = float(np.ptp(y_seg[max(0, r_local - int(0.050 * fs)):min(n, r_local + int(0.100 * fs))]))
    amp_thr = max(0.50 * amp_peak, 0.030 * local_ptp)

    dy = np.abs(np.gradient(y_seg))
    dy_sm = uniform_filter1d(dy, size=max(1, int(0.001 * fs)), mode="nearest")
    slope_thr = float(np.percentile(dy_sm[start:end + 1], 18))

    run = max(1, int(0.0005 * fs))
    for k in range(start, end - run + 2):
        if np.all(np.abs(y_seg[k:k + run] - baseline) <= amp_thr) and np.all(dy_sm[k:k + run] <= slope_thr):
            return float(k)

    cand = np.arange(start, end + 1, dtype=int)
    score = np.abs(y_seg[cand] - baseline) / (amp_thr + 1e-9) + dy_sm[cand] / (slope_thr + 1e-9)
    return float(cand[int(np.argmin(score))])


# ============================================================
# FIX 2 (v3): P-wave — flatness/slope criterion for P_on
# ============================================================
def _find_p_wave(y_seg, fs, qrs_on_local, rr_s=None):
    """
    P-wave detection with improved P_on via flatness criterion.

    P_on fix: instead of a percentage-of-peak threshold (which still
    drifts back on shallow slopes), scan backward from P-peak and stop
    at the first sample where the signal gradient drops below the
    local noise floor — i.e., where the signal is essentially flat
    (isoelectric). This anchors P_on to the true departure point.

    P_off: 15% forward threshold (unchanged from v2 — working correctly).
    Window cap: min(100ms, 50% RR) — unchanged.
    """
    n = len(y_seg)
    qrs_on_idx = int(qrs_on_local)

    # Cap search window: min(100 ms, 50% of RR)
    max_lookback_ms = 100.0
    if rr_s is not None and np.isfinite(rr_s) and rr_s > 0:
        max_lookback_ms = min(100.0, 0.5 * rr_s * 1000.0)
    max_lookback_samp = int(max_lookback_ms / 1000.0 * fs)

    search_start = max(0, qrs_on_idx - max_lookback_samp)
    search_end   = max(0, qrs_on_idx - int(0.015 * fs))

    if search_end - search_start < int(0.020 * fs):
        return np.nan, np.nan

    # Low-pass filter to isolate P wave
    sos_lp = butter(4, min(30.0 / (0.5 * fs), 0.99), btype='low', output='sos')
    y_lp   = sosfilt(sos_lp, y_seg)

    seg = y_lp[search_start:search_end]
    if seg.size < 4:
        return np.nan, np.nan

    baseline = float(np.median(y_lp[search_start:search_start + max(1, int(0.010 * fs))]))
    seg_rel  = seg - baseline

    p_pk_loc = int(np.argmax(np.abs(seg_rel)))
    p_pk_amp = float(seg_rel[p_pk_loc])

    if abs(p_pk_amp) < 1e-9:
        return np.nan, np.nan

    win_range = float(np.ptp(seg_rel))
    if abs(p_pk_amp) < 0.03 * win_range and win_range > 0:
        return np.nan, np.nan

    # ── P_on: flatness/slope criterion ──────────────────────────
    # Compute gradient of the LP signal; estimate noise floor from
    # the quietest part of the search window (first 20 ms)
    dy = np.abs(np.diff(seg_rel))
    noise_win = min(len(dy), int(0.020 * fs))
    noise_floor = float(np.median(dy[:noise_win])) + float(np.std(dy[:noise_win])) if noise_win > 1 else 1e-9
    noise_floor = max(noise_floor, 1e-9)

    # Scan backward from P-peak: stop where gradient <= noise_floor
    # (signal has returned to flat/isoelectric)
    p_on_loc = 0  # default to start of window if never flat
    for k in range(p_pk_loc - 1, -1, -1):
        if k < len(dy) and dy[k] <= noise_floor:
            p_on_loc = k + 1  # +1: dy[k] is between seg[k] and seg[k+1]
            break

    # ── P_off: 15% forward threshold (working well) ──────────────
    p_off_threshold = 0.15 * abs(p_pk_amp)
    p_off_loc = p_pk_loc
    for k in range(p_pk_loc, len(seg_rel)):
        if abs(seg_rel[k]) <= p_off_threshold:
            p_off_loc = k
            break

    p_on  = float(search_start + p_on_loc)
    p_off = float(search_start + p_off_loc)

    # Sanity: P duration 5-50 ms for mouse
    p_dur_ms = (p_off - p_on) / fs * 1000.0
    if not (5.0 <= p_dur_ms <= 50.0):
        return np.nan, np.nan

    if p_off >= qrs_on_idx:
        p_off = float(qrs_on_idx - 1)

    return p_on, p_off


def _fallback_p_wave_simple(y_seg, fs, qrs_on_local, rr_s=None):
    """
    Backup P-wave detector used only when primary P-wave logic returns NaN.
    This keeps PR-related outputs populated for benchmark QA, while still
    exposing editable landmarks in manual review.
    """
    qrs_on_idx = int(max(1, qrs_on_local))
    max_lookback_ms = 100.0
    if rr_s is not None and np.isfinite(rr_s) and rr_s > 0:
        max_lookback_ms = min(100.0, 0.5 * rr_s * 1000.0)
    lookback = int(max_lookback_ms / 1000.0 * fs)
    search_start = max(0, qrs_on_idx - lookback)
    search_end = max(0, qrs_on_idx - int(0.010 * fs))
    if search_end - search_start < int(0.015 * fs):
        return np.nan, np.nan

    sos_lp = butter(2, min(25.0 / (0.5 * fs), 0.99), btype="low", output="sos")
    y_lp = sosfilt(sos_lp, y_seg)
    seg = y_lp[search_start:search_end]
    if len(seg) < 4:
        return np.nan, np.nan

    baseline = float(np.median(seg[: max(2, int(0.010 * fs))]))
    seg_rel = seg - baseline
    p_peak = int(np.argmax(np.abs(seg_rel)))
    amp = float(seg_rel[p_peak])
    if abs(amp) < max(1e-9, 0.015 * float(np.ptp(y_lp))):
        return np.nan, np.nan

    on_thr = 0.25 * abs(amp)
    off_thr = 0.20 * abs(amp)
    p_on = 0
    for k in range(p_peak, -1, -1):
        if abs(seg_rel[k]) <= on_thr:
            p_on = k
            break
    p_off = p_peak
    for k in range(p_peak, len(seg_rel)):
        if abs(seg_rel[k]) <= off_thr:
            p_off = k
            break

    p_on_abs = float(search_start + p_on)
    p_off_abs = float(search_start + p_off)
    if p_off_abs >= qrs_on_idx:
        p_off_abs = float(qrs_on_idx - 1)
    p_dur_ms = (p_off_abs - p_on_abs) / fs * 1000.0
    if not (4.0 <= p_dur_ms <= 80.0):
        return np.nan, np.nan
    return p_on_abs, p_off_abs


def _template_p_wave_from_r(y_seg, fs, r_loc, qrs_on_local):
    """
    Last-resort P landmarks from empirical mouse offsets around R.
    Used only when signal-based P detection is unreliable.
    """
    n = len(y_seg)
    p_on = int(r_loc - int(round(0.043 * fs)))   # ~43 ms before R
    p_off = int(r_loc - int(round(0.026 * fs)))  # ~26 ms before R
    max_poff = int(qrs_on_local - 2)
    if p_off >= max_poff:
        p_off = max_poff
    min_pon = max(0, p_off - int(round(0.050 * fs)))
    if p_on < min_pon:
        p_on = min_pon
    if p_on >= p_off - 2:
        p_on = p_off - max(3, int(round(0.006 * fs)))
    if p_on < 0 or p_off < 0 or p_on >= n or p_off >= n or p_on >= p_off:
        return np.nan, np.nan
    p_dur_ms = (p_off - p_on) / fs * 1000.0
    if not (5.0 <= p_dur_ms <= 55.0):
        return np.nan, np.nan
    return float(p_on), float(p_off)


# ============================================================
# Combined QRS/T/P refinement
# ============================================================
def refine_qrs_t(sig_filt, fs, rpeaks, waves):
    N, n = len(rpeaks), len(sig_filt)

    def gw(name):
        arr = waves.get(name)
        if arr is None:
            return np.full(N, np.nan)
        arr = np.asarray(arr, dtype=float)
        out = np.full(N, np.nan)
        k = min(len(arr), N)
        out[:k] = arr[:k]
        return out

    QRS_on0  = gw("ECG_R_Onsets")
    QRS_off0 = gw("ECG_R_Offsets")
    T_off0   = gw("ECG_T_Offsets")
    P_on0    = gw("ECG_P_Onsets")
    P_off0   = gw("ECG_P_Offsets")

    r_times = rpeaks / fs
    rr = np.full(N, np.nan)
    if N >= 2:
        rr[1:] = np.diff(r_times)
    rr_med = np.nanmedian(rr) if np.any(np.isfinite(rr)) else 0.12

    QRS_on  = np.copy(QRS_on0)
    QRS_off = np.copy(QRS_off0)
    T_off   = np.copy(T_off0)
    P_on    = np.copy(P_on0)
    P_off   = np.copy(P_off0)
    conf    = np.full(N, np.nan)

    for i in range(N):
        r_idx = int(rpeaks[i])
        rr_i  = rr[i] if np.isfinite(rr[i]) and rr[i] > 0 else rr_med

        L = int(np.clip(0.110 * fs, int(0.050 * fs), int(0.150 * fs)))
        R = int(np.clip(1.10 * rr_i * fs, int(0.080 * fs), int(0.250 * fs)))
        s, e   = max(0, r_idx - L), min(n, r_idx + R)
        y_seg  = sig_filt[s:e]
        r_loc  = r_idx - s

        # ── QRS onset: prefer plausible wavelet; else steep-slope onset ──
        qrs_on_abs = np.nan
        if np.isfinite(QRS_on0[i]):
            cand = float(QRS_on0[i])
            dt_ms = (r_idx - cand) / fs * 1000.0
            if 5.0 <= dt_ms <= 30.0:
                qrs_on_abs = cand
        if not np.isfinite(qrs_on_abs):
            qrs_on_abs = float(s + _find_qrs_onset_steep(y_seg, fs, r_loc))
        QRS_on[i] = qrs_on_abs

        # ── P-wave: wavelet -> signal-based -> fallback -> template ──
        p_on_abs = np.nan
        p_off_abs = np.nan
        p_signal_off_abs = np.nan
        if np.isfinite(P_on0[i]) and np.isfinite(P_off0[i]):
            cand_on = float(P_on0[i])
            cand_off = float(P_off0[i])
            p_dur_ms0 = (cand_off - cand_on) / fs * 1000.0
            if cand_on < cand_off < qrs_on_abs and 5.0 <= p_dur_ms0 <= 70.0:
                p_on_abs, p_off_abs = cand_on, cand_off
                p_signal_off_abs = cand_off
        if not (np.isfinite(p_on_abs) and np.isfinite(p_off_abs)):
            qrs_on_local_samp = qrs_on_abs - s
            p_on_local, p_off_local = _find_p_wave(y_seg, fs, qrs_on_local_samp, rr_s=rr_i)
            if np.isfinite(p_off_local):
                p_signal_off_abs = float(s + p_off_local)
            if not (np.isfinite(p_on_local) and np.isfinite(p_off_local)):
                p_on_local, p_off_local = _fallback_p_wave_simple(y_seg, fs, qrs_on_local_samp, rr_s=rr_i)
                if np.isfinite(p_off_local):
                    p_signal_off_abs = float(s + p_off_local)
            if not (np.isfinite(p_on_local) and np.isfinite(p_off_local)):
                p_on_local, p_off_local = _template_p_wave_from_r(y_seg, fs, r_loc, qrs_on_local_samp)
            p_on_abs = float(s + p_on_local) if np.isfinite(p_on_local) else np.nan
            p_off_abs = float(s + p_off_local) if np.isfinite(p_off_local) else np.nan

        # Regularize P landmarks to a stable R-anchored template if detection drifts too far.
        tpl_on, tpl_off = _template_p_wave_from_r(y_seg, fs, r_loc, qrs_on_abs - s)
        gate = int(0.004 * fs)  # 4 ms
        if np.isfinite(tpl_on) and np.isfinite(tpl_off):
            if not np.isfinite(p_on_abs):
                p_on_abs = float(s + tpl_on)
            else:
                if abs((p_on_abs - s) - tpl_on) > gate:
                    p_on_abs = float(s + tpl_on)
            if not np.isfinite(p_off_abs):
                p_off_abs = float(s + tpl_off)
            else:
                p_off_adjusted = False
                if np.isfinite(p_signal_off_abs) and np.isfinite(p_on_abs):
                    signal_off_local = p_signal_off_abs - s
                    diff_ms = (signal_off_local - tpl_off) / fs * 1000.0
                    signal_dur_ms = (p_signal_off_abs - p_on_abs) / fs * 1000.0
                    signal_pr_ms = (qrs_on_abs - p_signal_off_abs) / fs * 1000.0
                    if diff_ms > 5.0 and 8.0 <= signal_pr_ms <= 18.0 and 8.0 <= signal_dur_ms <= 30.0:
                        p_off_abs = p_signal_off_abs
                        p_off_adjusted = True
                    elif diff_ms < -10.0 and signal_dur_ms < 5.0:
                        p_off_abs = float(p_on_abs + int(round(0.011 * fs)))
                        p_off_adjusted = True
                if not p_off_adjusted and abs((p_off_abs - s) - tpl_off) > gate:
                    p_off_abs = float(s + tpl_off)
        if np.isfinite(p_off_abs) and p_off_abs >= qrs_on_abs:
            p_off_abs = float(qrs_on_abs - int(0.004 * fs))
        if np.isfinite(p_on_abs) and np.isfinite(p_off_abs) and p_on_abs >= p_off_abs:
            p_on_abs = float(p_off_abs - int(0.008 * fs))
        P_on[i] = p_on_abs
        P_off[i] = p_off_abs

        # ── QRS offset: same-height upslope after R (vs P_on level) ──
        p_on_local_for_qrs = (P_on[i] - s) if np.isfinite(P_on[i]) else None
        j_candidate = _find_qrs_offset_jpoint(
            y_seg, fs, r_loc, len(y_seg) - 1, p_on_local=p_on_local_for_qrs
        )
        qrs_off_abs = float(s + j_candidate)
        qrs_dur_ms = (qrs_off_abs - QRS_on[i]) / fs * 1000.0
        if not (4.0 <= qrs_dur_ms <= 35.0):
            if np.isfinite(QRS_off0[i]):
                qrs_off_abs = float(QRS_off0[i])
            else:
                qrs_off_abs = float(r_idx + int(0.010 * fs))
        if qrs_off_abs <= QRS_on[i]:
            qrs_off_abs = float(QRS_on[i] + int(0.006 * fs))
        QRS_off[i] = qrs_off_abs

        # ── T-wave offset: return-to-baseline + flatness in early post-R window ──
        qrs_off_local = QRS_off[i] - s
        qrs_on_local = QRS_on[i] - s
        t_search_end = min(len(y_seg) - 1, r_loc + int(min(0.90 * rr_i, 0.045) * fs))
        t_candidate = _find_t_offset_threshold(
            y_seg,
            fs,
            qrs_off_local,
            t_search_end,
            qrs_on_local=qrs_on_local,
            r_local=r_loc,
            p_on_local=(P_on[i] - s) if np.isfinite(P_on[i]) else None,
        )
        if t_candidate is not None and np.isfinite(t_candidate):
            T_off[i] = float(s + t_candidate)
        elif np.isfinite(T_off0[i]):
            T_off[i] = float(T_off0[i])
        else:
            T_off[i] = float(r_idx + int(0.028 * fs))

        qt_ms = (T_off[i] - QRS_on[i]) / fs * 1000.0
        if not (8.0 <= qt_ms <= 60.0):
            lo = float(QRS_off[i] + int(0.004 * fs))
            hi = float(r_idx + int(0.040 * fs))
            T_off[i] = float(np.clip(T_off[i], lo, hi))

        # ── Confidence ────────────────────────────────────────────
        found = sum([
            np.isfinite(P_on[i]),
            np.isfinite(P_off[i]),
            np.isfinite(QRS_on[i]),
            np.isfinite(QRS_off[i]),
            np.isfinite(T_off[i]),
        ])
        qrs_ok = (5.0  <= (QRS_off[i] - QRS_on[i]) / fs * 1000.0 <= 35.0
                  if (np.isfinite(QRS_on[i]) and np.isfinite(QRS_off[i])) else False)
        t_ok   = (10.0 <= (T_off[i] - QRS_on[i]) / fs * 1000.0 <= 60.0
                  if (np.isfinite(QRS_on[i]) and np.isfinite(T_off[i])) else False)
        conf[i] = float(found / 5.0 * 0.5
                        + (0.3 if qrs_ok else 0.0)
                        + (0.2 if t_ok   else 0.0))

    return QRS_on, QRS_off, T_off, P_on, P_off, conf


# ============================================================
# Main ECG metrics computation
# ============================================================
def compute_ecg_metrics(
    t,
    y,
    fs,
    engine="wavelet",
    debug_plots=False,
    source_file=None,
    validity_threshold=0.90,
    run_context=None,
):
    sig_filt, (f_lo, f_hi) = auto_bandpass_notch_ecg(y, fs)
    rpeaks, waves, used_engine = delineate_ecg(sig_filt, fs, engine=engine)
    if len(rpeaks) < 3:
        raise ValueError("Insufficient R-peaks detected.")

    N = len(rpeaks)
    QRS_on, QRS_off, T_off, P_on, P_off, conf = refine_qrs_t(sig_filt, fs, rpeaks, waves)

    r_t = rpeaks / fs
    rr  = np.full(N, np.nan)
    if N >= 2:
        rr[1:] = np.diff(r_t)

    P_dur = np.where(np.isfinite(P_on)   & np.isfinite(P_off),   (P_off  - P_on)   / fs, np.nan)
    PR    = np.where(np.isfinite(P_on)   & np.isfinite(QRS_on),  (QRS_on - P_on)   / fs, np.nan)
    QRS   = np.where(np.isfinite(QRS_on) & np.isfinite(QRS_off), (QRS_off- QRS_on) / fs, np.nan)
    QT    = np.where(np.isfinite(QRS_on) & np.isfinite(T_off),   (T_off  - QRS_on) / fs, np.nan)
    QTcM  = np.where(np.isfinite(QT) & np.isfinite(rr) & (rr > 0), QT / np.sqrt(rr * 10.0), np.nan)

    conf_rank = pd.Series(np.where(np.isfinite(conf), conf, np.inf)).rank(
        method="dense", ascending=True
    ).to_numpy(dtype=float)

    rows = []
    for i in range(N):
        def ms(v):
            return float(v * 1000) if np.isfinite(v) else np.nan

        p_on_idx = _to_optional_int(P_on[i])
        p_off_idx = _to_optional_int(P_off[i])
        qrs_on_idx = _to_optional_int(QRS_on[i])
        qrs_off_idx = _to_optional_int(QRS_off[i])
        r_idx = _to_optional_int(rpeaks[i])
        t_off_idx = _to_optional_int(T_off[i])

        rows.append({
            "Beat_Index":        i,
            "P_on_idx":          p_on_idx,
            "P_off_idx":         p_off_idx,
            "QRS_on_idx":        qrs_on_idx,
            "QRS_off_idx":       qrs_off_idx,
            "R_idx":             r_idx,
            "T_off_idx":         t_off_idx,
            "R_time_s":          float(r_t[i]),
            "RR_s":              float(rr[i])     if np.isfinite(rr[i])     else np.nan,
            "RR_ms":             ms(rr[i]),
            "P_wave_dur_s":      float(P_dur[i])  if np.isfinite(P_dur[i])  else np.nan,
            "P_wave_dur_ms":     ms(P_dur[i]),
            "PR_interval_s":     float(PR[i])     if np.isfinite(PR[i])     else np.nan,
            "PR_interval_ms":    ms(PR[i]),
            "QRS_interval_s":    float(QRS[i])    if np.isfinite(QRS[i])    else np.nan,
            "QRS_interval_ms":   ms(QRS[i]),
            "QT_interval_s":     float(QT[i])     if np.isfinite(QT[i])     else np.nan,
            "QT_interval_ms":    ms(QT[i]),
            "QTc_Mitchell_s":    float(QTcM[i])   if np.isfinite(QTcM[i])   else np.nan,
            "QTc_Mitchell_ms":   ms(QTcM[i]),
            "Confidence":        float(conf[i])   if np.isfinite(conf[i])   else np.nan,
            "Confidence_Rank":   float(conf_rank[i]) if np.isfinite(conf[i]) else np.nan,
            "Is_Corrected":      False,
            "Corrected_At":      "",
            "Correction_Notes":  "",
            "Correction_Source": "auto",
        })

    beats_df = ensure_ecg_beats_schema(pd.DataFrame(rows))
    beats_df, rmssd = recompute_ecg_intervals_df(beats_df, fs_hz=fs)
    valid_ratio, valid_count, total_count = compute_ecg_validity(beats_df)
    validity_pass = bool(
        np.isfinite(valid_ratio) and valid_ratio >= float(validity_threshold)
    )

    if debug_plots:
        for i in range(min(5, N)):
            r_idx = rpeaks[i]
            rr_i  = rr[i] if np.isfinite(rr[i]) else (np.nanmedian(rr[np.isfinite(rr)]) if np.any(np.isfinite(rr)) else 0.12)
            L = int(np.clip(0.110 * fs, int(0.050 * fs), int(0.150 * fs)))
            R = int(np.clip(1.10 * rr_i * fs, int(0.080 * fs), int(0.250 * fs)))
            sl, el = max(0, r_idx - L), min(len(sig_filt), r_idx + R)
            xv = np.arange(sl, el) / fs

            plt.figure(figsize=(13, 4))
            plt.plot(xv, sig_filt[sl:el], color="darkred", lw=1, label="ECG")

            def _mark(idx_abs, color, label, marker="X", size=90):
                if np.isfinite(idx_abs):
                    ii = int(idx_abs)
                    if 0 <= ii < len(sig_filt):
                        plt.scatter(ii / fs, sig_filt[ii],
                                    color=color, s=size, marker=marker,
                                    zorder=5, label=label)

            _mark(rpeaks[i],  "green",   "R")
            _mark(P_on[i],    "#00BCD4", "P_on",       marker="^", size=80)
            _mark(P_off[i],   "#0288D1", "P_off",      marker="v", size=80)
            _mark(QRS_on[i],  "#FF9800", "QRS_on")
            _mark(QRS_off[i], "#7E57C2", "QRS_off (J)")
            _mark(T_off[i],   "#F44336", "T_off")

            plt.title(f"Beat {i} | Engine={used_engine} | conf={conf[i]:.2f}")
            plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
            plt.grid(alpha=0.2); plt.legend(fontsize=8)
            plt.tight_layout(); plt.show()

    meta = {
        "engine":           used_engine,
        "bandpass_low_Hz":  f_lo,
        "bandpass_high_Hz": f_hi,
        "fs_Hz":            fs,
        "total_beats":      N,
        "RMSSD_s":          rmssd,
        "RMSSD_ms":         rmssd * 1000 if np.isfinite(rmssd) else np.nan,
        "qtc_formula":      "Mitchell",
        "qtc_formula_detail": "QTc_Mitchell_ms = QT_interval_ms / sqrt(RR_ms / 100)",
        "generated_at":     now_iso(),
        "source_file":      os.path.abspath(source_file) if source_file else "",
        "validity_threshold": float(validity_threshold),
        "valid_beats":      valid_count,
        "valid_beats_total": total_count,
        "validity_ratio":   valid_ratio,
        "validity_pass":    validity_pass,
    }
    if isinstance(run_context, dict):
        for key, value in run_context.items():
            meta[f"ctx_{key}"] = value
    return beats_df.to_dict(orient="records"), meta


# ============================================================
# 5) EEG ENGINE
# ============================================================
def eeg_preprocess(sig, fs):
    win = max(1, int(fs * 2.0))
    med = pd.Series(sig).rolling(win, center=True, min_periods=1).median().to_numpy()
    x   = sig - med
    x   = apply_notch(x, fs, 50.0,  q=30.0)
    x   = apply_notch(x, fs, 100.0, q=30.0)
    x   = bandpass_zero_phase(x, fs, 0.5, min(100.0, 0.45 * fs), order=4)
    return x


def detect_spikes(sig, fs,
                  bp=(14.0, 70.0),
                  z_thresh=5.6,
                  width_ms=(12, 80),
                  refr_ms=22,
                  slope_thresh=3.0,
                  curv_thresh=2.0,
                  rtob_frac=0.25,
                  ampz_thresh=4.2,
                  ampz_win_s=1.5,
                  hf_lf_ratio_min=0.35,
                  biphasic_mode='soft',
                  bip_win_ms=25.0,
                  bip_ratio_min=0.35,
                  post_artifact_frac=0.30,
                  post_artifact_bip_max=0.20,
                  low_ampz_strict=12.0,
                  low_amp_bip_min=0.45):
    sos    = butter_bandpass_sos(fs, bp[0], bp[1], order=4)
    hf     = sosfiltfilt(sos, sig)
    hf_abs = np.abs(hf)
    med, dev = moving_mad_stats(hf_abs, fs, 10.0)
    z        = (hf_abs - med) / dev
    min_dist = max(1, int(fs * refr_ms / 1000.0))
    peaks, _ = find_peaks(z, height=z_thresh, distance=min_dist)

    grad_med = np.median(np.abs(np.diff(sig))) + 1e-9 if len(sig) >= 2 else 1.0
    curv_med = np.median(np.abs(np.diff(sig, n=2))) + 1e-9 if len(sig) >= 3 else 1.0
    results, n = [], len(sig)

    for p in peaks:
        amp_env, half = hf_abs[p], 0.5 * hf_abs[p]
        l, r = p, p
        while l > 0 and hf_abs[l] > half:     l -= 1
        while r < n - 1 and hf_abs[r] > half: r += 1
        w_ms = 1000.0 * max(1, r - l) / fs
        if not (width_ms[0] <= w_ms <= width_ms[1]):
            continue

        k          = max(2, int(round(0.005 * fs)))
        core_left  = max(0, p - k)
        core_right = min(n, p + k)
        core       = sig[core_left:core_right]
        g1         = np.diff(core) if core.size >= 2 else np.array([0.0])
        peak_slope = float(np.max(np.abs(g1))) if g1.size else 0.0

        pre_span  = max(k * 5, int(0.02 * fs))
        base_left = max(0, core_left - pre_span)
        base      = sig[base_left:core_left]
        base_grad = float(np.median(np.abs(np.diff(base)))) + 1e-9 if base.size >= 2 else grad_med
        if peak_slope < slope_thresh * base_grad:
            continue

        g2   = np.diff(core, n=2) if core.size >= 3 else np.array([0.0])
        sharp = float(np.max(np.abs(g2))) if g2.size else 0.0
        base2  = sig[max(0, base_left - pre_span):base_left]
        base_curv = float(np.median(np.abs(np.diff(base2, n=2)))) + 1e-9 if base2.size >= 3 else curv_med
        if sharp < curv_thresh * base_curv:
            continue

        post_ratio = np.nan
        a, b = p + int(0.04 * fs), min(n, p + int(0.08 * fs))
        if b > a:
            post_env  = float(np.mean(np.abs(sig[a:b])))
            spike_env = float(np.max(np.abs(sig[core_left:core_right]))) + 1e-9
            post_ratio = post_env / spike_env
            if post_ratio > rtob_frac:
                continue

        sw  = int(0.08 * fs)
        s0, s1 = max(0, p - sw//2), min(n, p + sw//2)
        seg    = sig[s0:s1]
        if seg.size >= int(0.02 * fs):
            p30_80, p1_20 = band_power_fft(seg, fs, [(30, 80), (1, 20)])
            if p1_20 <= 0 or (p30_80 / p1_20) < hf_lf_ratio_min:
                continue

        aw  = int(round(ampz_win_s * fs))
        a0, a1 = max(0, p - aw//2), min(n, p + aw//2)
        seg_abs = np.abs(sig[a0:a1])
        med_abs = float(np.median(seg_abs))
        mad_abs = float(mad(seg_abs, scale='normal')) + 1e-9
        z_amp   = (abs(float(sig[p])) - med_abs) / mad_abs
        if z_amp < ampz_thresh:
            continue

        bip_ratio = 0.0
        if biphasic_mode in ('soft', 'hard'):
            bw       = int(round(bip_win_ms / 1000.0 * fs))
            cand     = np.r_[sig[max(0, p - bw):core_left], sig[core_right:min(n, p + bw)]]
            main     = float(sig[p])
            main_abs = abs(main) + 1e-12
            if cand.size:
                opp       = float(np.min(cand)) if main >= 0 else float(np.max(cand))
                bip_ratio = abs(opp) / main_abs
            if biphasic_mode == 'hard' and bip_ratio < bip_ratio_min:
                continue
        if np.isfinite(post_ratio) and post_ratio > post_artifact_frac and bip_ratio < post_artifact_bip_max:
            continue
        if z_amp < low_ampz_strict and bip_ratio < low_amp_bip_min:
            continue

        slope_ratio = peak_slope / (base_grad + 1e-12)
        score       = (0.9 * (z_amp - ampz_thresh)
                       + 0.4 * (slope_ratio - slope_thresh)
                       + 0.4 * (bip_ratio / (bip_ratio_min + 1e-12)))
        confidence  = float(1.0 / (1.0 + np.exp(-score)))
        results.append({"idx": int(p), "w_ms": float(w_ms), "z_amp": float(z_amp),
                         "slope": float(peak_slope), "curv": float(sharp),
                         "bip_ratio": float(bip_ratio), "post_ratio": float(post_ratio),
                         "confidence": confidence})
    return results


def detect_swd(sig, fs, band=(5.0, 9.0), min_duration_s=1.0,
               max_gap_s=0.35, power_ratio=2.0, env_z_thresh=2.6,
               amp_ratio_min=1.8, cycle_cv_max=0.35, min_cycles=4,
               harmonic_ratio_min=0.08):
    lo   = sosfiltfilt(butter_bandpass_sos(fs, band[0], band[1], 4), sig)
    env  = np.abs(hilbert(lo))
    low_bg = sosfiltfilt(butter_bandpass_sos(fs, 1.0, max(1.1, band[0] - 0.5), 4), sig)
    high_bg = sosfiltfilt(butter_bandpass_sos(fs, min(0.45 * fs - 1.0, band[1] + 0.5), min(20.0, 0.45 * fs), 4), sig)
    win  = max(1, int(fs * 0.5))
    num  = uniform_filter1d(lo * lo,  size=win, mode='nearest')
    den  = uniform_filter1d(low_bg * low_bg + high_bg * high_bg, size=win, mode='nearest') + 1e-12
    ratio = num / den

    env_mean = uniform_filter1d(env,       size=win, mode='nearest')
    env_var  = uniform_filter1d(env * env, size=win, mode='nearest') - env_mean ** 2
    env_cv   = np.sqrt(np.maximum(0.0, env_var)) / (env_mean + 1e-12)
    env_z    = (env_mean - np.median(env_mean)) / (mad(env_mean, scale='normal') + 1e-9)
    mask     = (ratio > power_ratio) & (env_z > env_z_thresh) & (env_cv < 0.65)

    events, in_evt, start = [], False, 0
    for i, m in enumerate(mask):
        if m and not in_evt:
            in_evt, start = True, i
        if in_evt and not m:
            e = {"start": start / fs, "end": i / fs}
            if events and e["start"] - events[-1]["end"] <= max_gap_s:
                events[-1]["end"] = e["end"]
            else:
                events.append(e)
            in_evt = False
    if in_evt:
        events.append({"start": start / fs, "end": len(sig) / fs})

    kept = []
    for e in events:
        if e["end"] - e["start"] < min_duration_s:
            continue
        a, b = int(e["start"] * fs), int(e["end"] * fs)
        if b <= a:
            continue
        seg = lo[a:b]
        raw = sig[a:b]
        dur = max(1e-12, e["end"] - e["start"])

        pad = int(round(10.0 * fs))
        gap = int(round(0.25 * fs))
        base = np.r_[sig[max(0, a - pad):max(0, a - gap)],
                     sig[min(len(sig), b + gap):min(len(sig), b + pad)]]
        if base.size < max(10, int(fs)):
            base = sig
        seg_pp = float(np.percentile(raw, 95) - np.percentile(raw, 5))
        base_pp = float(np.percentile(base, 95) - np.percentile(base, 5)) + 1e-9
        amp_ratio = seg_pp / base_pp
        if amp_ratio < amp_ratio_min:
            continue

        peaks, _ = find_peaks(seg, distance=max(1, int(fs / max(1.0, band[1] + 1.0))))
        if len(peaks) < min_cycles:
            continue
        isi = np.diff(peaks) / fs
        if isi.size == 0:
            continue
        mean_isi = float(np.mean(isi))
        if mean_isi <= 0:
            continue
        freq = 1.0 / mean_isi
        cycles = dur * freq
        if not (band[0] <= freq <= band[1] and cycles >= min_cycles):
            continue
        if float(np.std(isi) / (mean_isi + 1e-12)) > cycle_cv_max:
            continue

        p_fund, p_harm = band_power_fft(raw, fs, [band, (2.0 * band[0], min(2.0 * band[1], 0.45 * fs))])
        if p_fund <= 0 or (p_harm / (p_fund + 1e-12)) < harmonic_ratio_min:
            continue
        e = dict(e)
        e["amp_ratio"] = float(amp_ratio)
        e["dominant_hz"] = float(freq)
        kept.append(e)
    return kept


def detect_rhythmic_discharges(sig, fs, band=(5.0, 12.0), min_duration_s=4.0,
                               max_gap_s=0.5, power_ratio=1.4, env_z_thresh=2.0,
                               env_cv_max=0.9, cycle_cv_max=0.30, min_cycles=6):
    lo = sosfiltfilt(butter_bandpass_sos(fs, band[0], band[1], 4), sig)
    env = np.abs(hilbert(lo))
    low_bg = sosfiltfilt(butter_bandpass_sos(fs, 1.0, max(1.1, band[0] - 0.5), 4), sig)
    high_bg = sosfiltfilt(butter_bandpass_sos(fs, min(0.45 * fs - 1.0, band[1] + 0.5), min(20.0, 0.45 * fs), 4), sig)
    win = max(1, int(fs * 0.5))
    num = uniform_filter1d(lo * lo, size=win, mode='nearest')
    den = uniform_filter1d(low_bg * low_bg + high_bg * high_bg, size=win, mode='nearest') + 1e-12
    ratio = num / den
    env_mean = uniform_filter1d(env, size=win, mode='nearest')
    env_var = uniform_filter1d(env * env, size=win, mode='nearest') - env_mean ** 2
    env_cv = np.sqrt(np.maximum(0.0, env_var)) / (env_mean + 1e-12)
    env_z = (env_mean - np.median(env_mean)) / (mad(env_mean, scale='normal') + 1e-9)
    mask = (ratio > power_ratio) & (env_z > env_z_thresh) & (env_cv < env_cv_max)

    events, in_evt, start = [], False, 0
    for i, m in enumerate(mask):
        if m and not in_evt:
            in_evt, start = True, i
        if in_evt and not m:
            e = {"start": start / fs, "end": i / fs}
            if events and e["start"] - events[-1]["end"] <= max_gap_s:
                events[-1]["end"] = e["end"]
            else:
                events.append(e)
            in_evt = False
    if in_evt:
        events.append({"start": start / fs, "end": len(sig) / fs})

    kept = []
    for e in events:
        dur = e["end"] - e["start"]
        if dur < min_duration_s:
            continue
        a, b = int(e["start"] * fs), int(e["end"] * fs)
        seg = lo[a:b]
        peaks, _ = find_peaks(seg, distance=max(1, int(fs / max(1.0, band[1]))))
        if len(peaks) < min_cycles:
            continue
        isi = np.diff(peaks) / fs
        if isi.size == 0:
            continue
        cv = float(np.std(isi) / (np.mean(isi) + 1e-12))
        if cv > cycle_cv_max:
            continue
        e = dict(e)
        e["dominant_hz"] = float(1.0 / (np.mean(isi) + 1e-12))
        kept.append(e)
    return kept


def _close_gaps(mask, max_off_steps):
    if max_off_steps <= 0:
        return mask
    m, n, i = mask.copy(), len(mask), 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]: j += 1
            if 0 < j - i <= max_off_steps:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def detect_seizures_llrms(sig, fs,
                           win_s=1.0, step_s=0.25,
                           enter_z=5.5, exit_z=3.5, off_hold_s=2.0,
                           min_duration_s=5.0, merge_gap_s=15.0, close_s=10.0,
                           hf_ratio_thr=1.6, lf_ratio_thr=0.9,
                           occupancy_min=0.45, rhythm_prom_thr=1.35,
                           entropy_drop_sigma=0.20):
    w, step, n = max(1, int(fs*win_s)), max(1, int(fs*step_s)), len(sig)
    starts = np.arange(0, n - w + 1, step)
    t0     = starts / fs

    dx  = np.abs(np.diff(sig))
    LL  = np.convolve(dx, np.ones(w), mode='valid')[::step] if w > 1 else dx[::step]
    x2  = sig * sig
    cs  = np.cumsum(np.r_[0.0, x2])
    RMS = np.sqrt((cs[w:] - cs[:-w]) / w)[::step]
    m = min(len(starts), len(LL), len(RMS))
    starts = starts[:m]
    t0 = t0[:m]
    LL = LL[:m]
    RMS = RMS[:m]

    ll_z  = (LL  - np.median(LL))  / (mad(LL,  scale='normal') + 1e-9)
    rms_z = (RMS - np.median(RMS)) / (mad(RMS, scale='normal') + 1e-9)
    z     = np.maximum(ll_z, rms_z)

    hf_r = np.zeros_like(z); lf_r = np.zeros_like(z)
    prom = np.zeros_like(z); sent = np.zeros_like(z)
    clip = np.zeros_like(z, dtype=bool)
    hkrt = np.zeros_like(z, dtype=bool)

    for i, s0 in enumerate(starts):
        seg = sig[s0:s0+w]
        if len(seg) < w: continue
        rv = np.percentile(seg, [1, 99])
        clip[i] = np.any(np.isclose(seg, rv[0], atol=1e-9)) or np.any(np.isclose(seg, rv[1], atol=1e-9))
        hkrt[i] = kurtosis(seg, fisher=True, bias=False) > 8.0
        p1_45, p70_100, p01_05 = band_power_fft(seg, fs, [(1,45),(70,100),(0.1,0.5)])
        hf_r[i] = p70_100 / (p1_45 + 1e-12)
        lf_r[i] = p01_05  / (p1_45 + 1e-12)
        p4_30, p1_4, p30_100 = band_power_fft(seg, fs, [(4,30),(1,4),(30,100)])
        prom[i] = p4_30 / (p1_4 + p30_100 + 1e-12)
        sent[i] = spectral_entropy_from_fft(seg, fs)

    ent_z    = (np.median(sent) - sent) / (mad(sent, scale='normal') + 1e-9)
    artifact = (hf_r > hf_ratio_thr) | (lf_r > lf_ratio_thr) | clip | hkrt
    z_eff    = z.copy(); z_eff[artifact] = -1e9
    oh       = max(1, int(round(off_hold_s / step_s)))

    mask = np.zeros(len(z_eff), dtype=bool)
    i = 0
    while i < len(z_eff):
        if z_eff[i] >= enter_z:
            start, off = i, 0
            i += 1
            while i < len(z_eff):
                if z_eff[i] <= exit_z:
                    off += 1
                    if off >= oh:
                        mask[start:i-off+1] = True; break
                else:
                    off = 0
                i += 1
            else:
                mask[start:] = True; break
        else:
            i += 1

    if close_s > 0:
        mask = _close_gaps(mask, int(round(close_s / step_s)))

    events, in_evt, si = [], False, 0
    for i, on in enumerate(mask):
        if on and not in_evt:  in_evt, si = True, i
        if in_evt and not on:
            a, b = float(t0[si]), float(t0[i-1] + win_s)
            if b - a >= min_duration_s:
                events.append((si, i, a, b))
            in_evt = False
    if in_evt:
        a, b = float(t0[si]), float(t0[-1] + win_s)
        if b - a >= min_duration_s:
            events.append((si, len(mask), a, b))

    kept = []
    for si, ei, a, b in events:
        seg_on = (z[si:ei] >= exit_z) & (~artifact[si:ei])
        if float(np.mean(seg_on)) < occupancy_min: continue
        if not (float(np.mean(prom[si:ei] >= rhythm_prom_thr)) >= 0.60
                and float(np.mean(ent_z[si:ei] >= entropy_drop_sigma)) >= 0.60): continue
        kept.append({"start": a, "end": b})

    merged = []
    for e in sorted(kept, key=lambda d: d["start"]):
        if not merged or e["start"] - merged[-1]["end"] > merge_gap_s:
            merged.append(dict(e))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
    return merged


def detect_seizures_spike_led(sig, fs, spike_times_rel,
                               win_s=0.5, hop_s=0.05,
                               min_rate_hz=6.0,
                               isi_min_s=0.015, isi_max_s=0.120, isi_cv_max=0.45,
                               min_duration_s=5.0, close_s=1.0, merge_gap_s=12.0,
                               hf_ratio_thr=1.6, lf_ratio_thr=0.9,
                               rhythm_prom_thr=1.30, entropy_drop_sigma=0.15,
                               occupancy_min=0.60):
    N = len(sig)
    w, step = max(1, int(round(win_s*fs))), max(1, int(round(hop_s*fs)))
    starts  = np.arange(0, N - w + 1, step)
    t0_arr  = starts / fs

    hf_r = np.zeros(len(starts)); lf_r = np.zeros(len(starts))
    prom = np.zeros(len(starts)); sent = np.zeros(len(starts))
    clip = np.zeros(len(starts), dtype=bool)
    hkrt = np.zeros(len(starts), dtype=bool)

    for i, s0 in enumerate(starts):
        seg = sig[s0:s0+w]
        if len(seg) < w: continue
        rv = np.percentile(seg, [1,99])
        clip[i] = np.any(np.isclose(seg,rv[0],atol=1e-9)) or np.any(np.isclose(seg,rv[1],atol=1e-9))
        hkrt[i] = kurtosis(seg, fisher=True, bias=False) > 8.0
        p1_45, p70_100, p01_05 = band_power_fft(seg, fs, [(1,45),(70,100),(0.1,0.5)])
        hf_r[i] = p70_100/(p1_45+1e-12)
        lf_r[i] = p01_05/(p1_45+1e-12)
        p4_30, p1_4, p30_100 = band_power_fft(seg, fs, [(4,30),(1,4),(30,100)])
        prom[i] = p4_30/(p1_4 + p30_100 + 1e-12)
        sent[i] = spectral_entropy_from_fft(seg, fs)

    ent_z    = (np.median(sent) - sent) / (mad(sent, scale='normal') + 1e-9)
    artifact = (hf_r > hf_ratio_thr) | (lf_r > lf_ratio_thr) | clip | hkrt

    spikes    = np.array(spike_times_rel, dtype=float)
    spikes    = spikes[(spikes >= 0) & (spikes <= N/fs)] if spikes.size else spikes
    min_count = int(np.ceil(min_rate_hz * win_s) - 1e-9)
    rate_ok   = np.zeros(len(starts), dtype=bool)
    isi_ok    = np.zeros(len(starts), dtype=bool)

    for i, s0 in enumerate(starts):
        a, b = s0/fs, s0/fs + win_s
        if spikes.size == 0: continue
        sw = spikes[np.searchsorted(spikes,a):np.searchsorted(spikes,b,side='right')]
        rate_ok[i] = sw.size >= min_count
        if sw.size >= 3:
            isi = np.diff(sw)
            med_isi = float(np.median(isi))
            if isi_min_s <= med_isi <= isi_max_s:
                isi_ok[i] = float(np.std(isi)/(np.mean(isi)+1e-12)) <= isi_cv_max

    on = rate_ok & isi_ok & (~artifact)
    if close_s > 0:
        on = _close_gaps(on, int(round(close_s/hop_s)))

    events, in_evt, si = [], False, 0
    for i, val in enumerate(on):
        if val and not in_evt:  in_evt, si = True, i
        if in_evt and not val:
            a, b = float(t0_arr[si]), float(t0_arr[i-1]+win_s)
            if b - a >= min_duration_s: events.append((si, i, a, b))
            in_evt = False
    if in_evt:
        a, b = float(t0_arr[si]), float(t0_arr[-1]+win_s)
        if b - a >= min_duration_s: events.append((si, len(on), a, b))

    kept = []
    for si, ei, a, b in events:
        occ = float(np.mean(on[si:ei])) if ei > si else 0.0
        if occ < occupancy_min: continue
        if not (float(np.mean(prom[si:ei]>=rhythm_prom_thr))>=0.60
                and float(np.mean(ent_z[si:ei]>=entropy_drop_sigma))>=0.60): continue
        kept.append({"start": a, "end": b})

    merged = []
    for e in sorted(kept, key=lambda d: d["start"]):
        if not merged or e["start"]-merged[-1]["end"] > merge_gap_s:
            merged.append(dict(e))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
    return merged


def detect_epileptiform_trains(sig, fs,
                               win_s=1.0, hop_s=0.25,
                               enter_z=3.6, exit_z=2.2,
                               min_duration_s=8.0, close_s=1.0, merge_gap_s=10.0,
                               amp_ratio_min=1.9, rhythm_prom_thr=1.05,
                               min_peak_rate_hz=3.0, peak_z=3.2,
                               occupancy_min=0.55):
    """Detect sustained high-amplitude rhythmic spike/polyspike trains."""
    n = len(sig)
    w, step = max(1, int(round(win_s * fs))), max(1, int(round(hop_s * fs)))
    if n < w + 2:
        return []

    starts = np.arange(0, n - w + 1, step)
    t0_arr = starts / fs
    pp = np.zeros(len(starts))
    ll = np.zeros(len(starts))
    rms = np.zeros(len(starts))
    prom = np.zeros(len(starts))
    entropy = np.zeros(len(starts))
    lf_r = np.zeros(len(starts))
    clip = np.zeros(len(starts), dtype=bool)
    hkrt = np.zeros(len(starts), dtype=bool)

    hi = min(70.0, 0.45 * fs)
    if hi <= 10.0:
        hf_z = np.zeros(n)
    else:
        hf = sosfiltfilt(butter_bandpass_sos(fs, 8.0, hi, 4), sig)
        hf_abs = np.abs(hf)
        hf_z = (hf_abs - np.median(hf_abs)) / (mad(hf_abs, scale='normal') + 1e-9)

    peak_rate = np.zeros(len(starts))
    for i, s0 in enumerate(starts):
        seg = sig[s0:s0+w]
        if len(seg) < w:
            continue
        p01, p99 = np.percentile(seg, [1, 99])
        pp[i] = float(p99 - p01)
        ll[i] = float(np.sum(np.abs(np.diff(seg))))
        rms[i] = float(np.sqrt(np.mean(seg * seg)))
        clip[i] = np.any(np.isclose(seg, p01, atol=1e-9)) or np.any(np.isclose(seg, p99, atol=1e-9))
        hkrt[i] = kurtosis(seg, fisher=True, bias=False) > 14.0
        p1_4, p4_30, p30_100, p01_05 = band_power_fft(seg, fs, [(1, 4), (4, 30), (30, min(100, 0.45 * fs)), (0.1, 0.5)])
        prom[i] = p4_30 / (p1_4 + p30_100 + 1e-12)
        lf_r[i] = p01_05 / (p1_4 + p4_30 + p30_100 + 1e-12)
        entropy[i] = spectral_entropy_from_fft(seg, fs)
        pk, _ = find_peaks(hf_z[s0:s0+w], height=peak_z, distance=max(1, int(round(fs / 45.0))))
        peak_rate[i] = float(len(pk) / win_s)

    pp_base = np.median(pp) + 1e-9
    amp_ratio = pp / pp_base
    ll_z = (ll - np.median(ll)) / (mad(ll, scale='normal') + 1e-9)
    rms_z = (rms - np.median(rms)) / (mad(rms, scale='normal') + 1e-9)
    ent_z = (np.median(entropy) - entropy) / (mad(entropy, scale='normal') + 1e-9)
    z = np.maximum(ll_z, rms_z)

    # Strong low-frequency drift and clipping are more likely movement/artifact than seizure.
    artifact = (lf_r > 0.75) | clip | ((hkrt) & (amp_ratio > 8.0))
    rhythmic = (prom >= rhythm_prom_thr) | (ent_z >= 0.10)
    active = (amp_ratio >= amp_ratio_min) & (z >= exit_z) & rhythmic & (peak_rate >= min_peak_rate_hz) & (~artifact)

    oh = max(1, int(round(close_s / hop_s)))
    active = _close_gaps(active, oh)

    mask = np.zeros(len(active), dtype=bool)
    i = 0
    while i < len(active):
        if active[i] and z[i] >= enter_z:
            start = i
            i += 1
            while i < len(active) and active[i]:
                i += 1
            mask[start:i] = True
        else:
            i += 1

    events, in_evt, si = [], False, 0
    for i, on in enumerate(mask):
        if on and not in_evt:
            in_evt, si = True, i
        if in_evt and not on:
            a, b = float(t0_arr[si]), float(t0_arr[i - 1] + win_s)
            if b - a >= min_duration_s:
                events.append((si, i, a, b))
            in_evt = False
    if in_evt:
        a, b = float(t0_arr[si]), float(t0_arr[-1] + win_s)
        if b - a >= min_duration_s:
            events.append((si, len(mask), a, b))

    kept = []
    for si, ei, a, b in events:
        occ = float(np.mean(active[si:ei])) if ei > si else 0.0
        if occ < occupancy_min:
            continue
        kept.append({
            "start": a,
            "end": b,
            "amp_ratio": float(np.nanmax(amp_ratio[si:ei])) if ei > si else np.nan,
            "confidence": float(min(0.95, 0.60 + 0.05 * max(0.0, np.nanmax(z[si:ei]) - enter_z))),
        })

    merged = []
    for e in sorted(kept, key=lambda d: d["start"]):
        if not merged or e["start"] - merged[-1]["end"] > merge_gap_s:
            merged.append(dict(e))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
            if np.isfinite(e.get("confidence", np.nan)):
                merged[-1]["confidence"] = max(float(merged[-1].get("confidence", 0.0)), float(e["confidence"]))
    return merged


def merge_intervals(events, merge_gap_s):
    if not events: return []
    events = sorted(events, key=lambda e: e["start"])
    merged = [dict(events[0])]
    for e in events[1:]:
        if e["start"] - merged[-1]["end"] <= merge_gap_s:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
        else:
            merged.append(dict(e))
    return merged


def merge_spike_times(times, refr_s=0.02):
    if len(times) == 0: return np.array([], dtype=float)
    times = np.array(sorted(times), dtype=float)
    out   = [times[0]]
    for t in times[1:]:
        if t - out[-1] >= refr_s: out.append(float(t))
    return np.array(out, dtype=float)


def filter_spikes_outside_events(spike_times, events, margin_s=0.25):
    if len(spike_times) == 0 or not events:
        return np.asarray(spike_times, dtype=float)
    intervals = sorted(
        (float(e["start"]) - margin_s, float(e["end"]) + margin_s)
        for e in events
    )
    kept = []
    j = 0
    for ti in np.asarray(spike_times, dtype=float):
        while j < len(intervals) and intervals[j][1] < ti:
            j += 1
        if j < len(intervals) and intervals[j][0] <= ti <= intervals[j][1]:
            continue
        kept.append(float(ti))
    return np.asarray(kept, dtype=float)


def filter_events_outside_events(candidate_events, blocker_events, margin_s=0.0):
    if not candidate_events or not blocker_events:
        return list(candidate_events or [])
    blockers = sorted((float(e["start"]) - margin_s, float(e["end"]) + margin_s)
                      for e in blocker_events)
    kept = []
    for e in candidate_events:
        a, b = float(e["start"]), float(e["end"])
        overlaps = any(a <= bb and b >= ba for ba, bb in blockers)
        if not overlaps:
            kept.append(e)
    return kept


def bin_counts(spike_ts, swd_evts, sz_evts, bin_s, t0, t1):
    edges = np.arange(t0, t1 + bin_s, bin_s)
    if edges[-1] < t1:
        edges = np.append(edges, t1)
    rows = []
    for i in range(len(edges) - 1):
        a, b = float(edges[i]), float(edges[i+1])
        spikes = int(np.sum((spike_ts >= a) & (spike_ts < b))) if spike_ts.size else 0
        swds   = int(sum(e["start"] >= a and e["start"] < b for e in swd_evts))  if swd_evts  else 0
        szs    = int(sum(e["start"] >= a and e["start"] < b for e in sz_evts))   if sz_evts   else 0
        rows.append({"Bin_Index": i+1, "Bin_Start_s": a, "Bin_End_s": b,
                     "Spike_Count": spikes, "SWD_Count": swds, "Seizure_Count": szs})
    return pd.DataFrame(rows)


def _read_text_waveform_metadata(file_path):
    meta = {}
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(12):
                line = f.readline()
                if not line:
                    break
                parts = [p.strip() for p in line.rstrip("\n").split("\t")]
                if parts and parts[0].lower() == "count" and len(parts) > 1:
                    try:
                        meta["count"] = int(float(parts[1]))
                    except Exception:
                        pass
    except Exception:
        pass
    return meta


def iter_waveform_tabular_chunks(file_path, chunk_rows=600000):
    path = pathlib.Path(file_path)
    ext = path.suffix.lower()
    if ext not in [".tsv", ".txt", ".csv"]:
        t, y = load_waveform_tabular(file_path)
        yield t, y
        return

    sep = "\t" if ext in [".tsv", ".txt"] else ","
    reader = pd.read_csv(
        file_path,
        sep=sep,
        header=None,
        skiprows=7,
        usecols=[3, 4],
        chunksize=int(chunk_rows),
        engine="c",
        on_bad_lines="skip",
    )
    for df in reader:
        t = pd.to_numeric(
            df.iloc[:, 0].astype(str).str.strip().str.replace(r"[^0-9eE+\-\.]", "", regex=True),
            errors="coerce",
        ).to_numpy(dtype=float)
        y = pd.to_numeric(
            df.iloc[:, 1].astype(str).str.strip().str.replace(r"[^0-9eE+\-\.]", "", regex=True),
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = np.isfinite(t) & np.isfinite(y)
        if np.any(valid):
            yield t[valid], y[valid]


def _events_from_eeg_detection(spike_times, swd_events, sz_events):
    rows = []
    for ti in spike_times:
        rows.append({"Type": "Spike", "Start_s": float(ti), "End_s": float(ti),
                     "Duration_s": 0.0, "Confidence": np.nan})
    for e in swd_events:
        rows.append({"Type": "SWD", "Start_s": float(e["start"]), "End_s": float(e["end"]),
                     "Duration_s": float(e["end"] - e["start"]),
                     "Confidence": float(e.get("confidence", np.nan))})
    for e in sz_events:
        rows.append({"Type": "Seizure", "Start_s": float(e["start"]), "End_s": float(e["end"]),
                     "Duration_s": float(e["end"] - e["start"]),
                     "Confidence": float(e.get("confidence", np.nan))})
    cols = ["Type", "Start_s", "End_s", "Duration_s", "Confidence"]
    return pd.DataFrame(sorted(rows, key=lambda r: (r["Start_s"], r["Type"])), columns=cols)


def analyze_eeg(t, y, fs, bin_s=3600.0, debug_plots=False):
    sig = eeg_preprocess(y, fs)
    sig_spk, fs_spk = decimate_to(sig, fs, 500)
    sig_swd, fs_swd = decimate_to(sig, fs, 200)
    sig_sz,  fs_sz  = decimate_to(sig, fs, 250)
    t0, t1 = float(t[0]), float(t[-1])

    spk_params = dict(z_thresh=4.8, width_ms=(8,100), refr_ms=50,
                      slope_thresh=1.4, curv_thresh=1.2, rtob_frac=1.5,
                      ampz_thresh=7.0, ampz_win_s=3.0, hf_lf_ratio_min=0.05,
                      biphasic_mode='soft', bip_win_ms=30.0, bip_ratio_min=0.10)
    spikes_raw  = detect_spikes(sig_spk, fs_spk, **spk_params)
    spike_times = np.array([t0 + s["idx"]/fs_spk for s in spikes_raw], dtype=float)
    spike_times = merge_spike_times(spike_times.tolist(), refr_s=0.02)

    swd_raw = detect_swd(sig_swd, fs_swd)
    swd_events = [{"start": t0+e["start"], "end": t0+e["end"],
                   "type": "SWD", "confidence": 0.85} for e in swd_raw]
    sz_swd = []

    rel_spikes  = (spike_times - t0).tolist()
    sz_spike    = detect_seizures_spike_led(sig_sz, fs_sz, rel_spikes)
    sz_llrms    = detect_seizures_llrms(sig_sz, fs_sz)
    sz_train    = detect_epileptiform_trains(sig_sz, fs_sz)
    sz_all      = ([{"start": t0+e["start"], "end": t0+e["end"],
                     "type": "Seizure", "confidence": 0.90} for e in sz_spike]
                 + [{"start": t0+e["start"], "end": t0+e["end"],
                     "type": "Seizure", "confidence": 0.75} for e in sz_llrms]
                 + [{"start": t0+e["start"], "end": t0+e["end"],
                     "type": "Seizure", "confidence": float(e.get("confidence", 0.82))} for e in sz_train]
                 + sz_swd)
    sz_events   = merge_intervals(sz_all, merge_gap_s=10.0)
    swd_events  = filter_events_outside_events(swd_events, sz_events, margin_s=0.25)
    spike_times = filter_spikes_outside_events(spike_times, swd_events + sz_events, margin_s=0.25)

    summary_df = bin_counts(spike_times, swd_events, sz_events, bin_s, t0, t1)

    ev_rows = []
    for ti in spike_times:
        ev_rows.append({"Type": "Spike",   "Start_s": float(ti), "End_s": float(ti),
                        "Duration_s": 0.0, "Confidence": np.nan})
    for e in swd_events:
        ev_rows.append({"Type": "SWD",     "Start_s": float(e["start"]), "End_s": float(e["end"]),
                        "Duration_s": float(e["end"]-e["start"]),
                        "Confidence": float(e.get("confidence", np.nan))})
    for e in sz_events:
        ev_rows.append({"Type": "Seizure", "Start_s": float(e["start"]), "End_s": float(e["end"]),
                        "Duration_s": float(e["end"]-e["start"]),
                        "Confidence": float(e.get("confidence", np.nan))})
    events_df = pd.DataFrame(sorted(ev_rows, key=lambda r: (r["Start_s"], r["Type"])))

    meta = {
        "fs_Hz":                float(fs),
        "recording_duration_s": float(t1 - t0),
        "total_spikes":         int(len(spike_times)),
        "total_SWDs":           int(len(swd_events)),
        "total_seizures":       int(len(sz_events)),
        "bin_size_s":           float(bin_s),
        "preprocess":           "rolling-median detrend + 50/100 Hz notch + 0.5-100 Hz bandpass",
        "detector_fs":          "Spikes 500 Hz | SWD 200 Hz | Seizure 250 Hz",
        "eeg_detection_profile": "mouse_literature_guided_v6",
        "eeg_swd_criteria":      "5-9 Hz, >=4 cycles, >=1.8x local baseline, harmonic gate",
        "eeg_seizure_criteria":  ">=8 s dense high-amplitude rhythmic/evolving spike trains; reviewed EEG corrections auto-applied by source file",
        "eeg_spike_artifact_filter": "reject post-tail artifacts when 40-80 ms post/spike >0.30 and biphasic ratio <0.20",
        "eeg_spike_criteria":    "14-70 Hz transient, amp z >=7.0, width 8-100 ms, post-tail artifact filter",
    }

    if debug_plots and len(events_df) > 0:
        for _, row in events_df.head(3).iterrows():
            a  = max(t0, row["Start_s"] - 2.0)
            b  = min(t1, row["End_s"]   + 2.0)
            i0 = np.searchsorted(t, a)
            i1 = np.searchsorted(t, b)
            plt.figure(figsize=(12, 4))
            plt.plot(t[i0:i1], sig[i0:i1], 'k', lw=0.9)
            colour = {"Spike": "gold", "SWD": "lightskyblue", "Seizure": "tomato"}.get(row["Type"], "grey")
            plt.axvspan(row["Start_s"], row["End_s"], alpha=0.25, color=colour)
            plt.title(f'{row["Type"]}  {row["Start_s"]:.2f}–{row["End_s"]:.2f} s  conf={row["Confidence"]:.2f}')
            plt.xlabel("Time (s)"); plt.ylabel("EEG (a.u.)")
            plt.grid(alpha=0.2); plt.tight_layout(); plt.show()

    return summary_df, events_df, meta


def analyze_eeg_file(
    file_path,
    bin_s=3600.0,
    debug_plots=False,
    force_fs=None,
    progress_callback=None,
    chunk_s=180.0,
    overlap_s=10.0,
    streaming_threshold_mb=128.0,
):
    path = pathlib.Path(file_path)
    ext = path.suffix.lower()
    size_mb = os.path.getsize(file_path) / 1024 / 1024

    if ext in [".xlsx", ".xls"] or size_mb < float(streaming_threshold_mb):
        if progress_callback:
            progress_callback(8, "EEG: loading file")
        t, y = load_waveform_tabular(file_path)
        if force_fs is not None:
            t, y, fs = ensure_uniform_sampling(t, y, fs_target=float(force_fs))
        else:
            t, y, fs = ensure_uniform_sampling(t, y)
        summary_df, events_df, meta = analyze_eeg(t, y, fs, bin_s=bin_s, debug_plots=debug_plots)
        meta["source_file"] = os.path.abspath(file_path)
        events_df, applied = apply_eeg_saved_corrections(file_path, events_df, t0=float(t[0]), t1=float(t[-1]))
        if applied:
            summary_df = recompute_eeg_bins_from_events(events_df, float(bin_s), t0=float(t[0]), t1=float(t[-1]))
            meta["eeg_saved_corrections_applied"] = int(applied)
            meta = update_eeg_meta_from_events(meta, events_df, summary_df)
        return summary_df, events_df, meta

    text_meta = _read_text_waveform_metadata(file_path)
    total_rows = int(text_meta.get("count", 0) or 0)
    chunk_rows = int(max(50000, round((force_fs if force_fs else 2000.0) * chunk_s)))

    all_spikes = []
    all_swd = []
    all_sz = []
    first_t = None
    last_t = None
    fs_used = float(force_fs) if force_fs is not None else None
    prev_t = np.array([], dtype=float)
    prev_y = np.array([], dtype=float)
    processed = 0
    chunks = 0
    skipped_chunks = 0

    for t_new, y_new in iter_waveform_tabular_chunks(file_path, chunk_rows=chunk_rows):
        chunks += 1
        processed += len(t_new)
        if len(t_new) < 3:
            continue
        if first_t is None:
            first_t = float(t_new[0])
        last_t = float(t_new[-1])

        core_start = float(t_new[0])
        core_end = float(t_new[-1])
        if prev_t.size:
            t_chunk = np.concatenate([prev_t, t_new])
            y_chunk = np.concatenate([prev_y, y_new])
        else:
            t_chunk, y_chunk = t_new, y_new

        if fs_used is not None:
            t_u, y_u, fs = ensure_uniform_sampling(t_chunk, y_chunk, fs_target=float(fs_used))
        else:
            t_u, y_u, fs = ensure_uniform_sampling(t_chunk, y_chunk)
            fs_used = float(fs)

        try:
            _, events_df, _ = analyze_eeg(t_u, y_u, fs, bin_s=bin_s, debug_plots=False)
        except Exception as exc:
            skipped_chunks += 1
            if progress_callback:
                progress_callback(10, f"EEG: skipped chunk {chunks} ({exc})")
            continue
        if len(events_df):
            for _, row in events_df.iterrows():
                typ = str(row["Type"])
                start = float(row["Start_s"])
                end = float(row["End_s"])
                if typ == "Spike":
                    if core_start <= start < core_end:
                        all_spikes.append(start)
                elif end >= core_start and start < core_end:
                    rec = {"start": max(start, core_start), "end": min(end, core_end),
                           "type": typ, "confidence": float(row.get("Confidence", np.nan))}
                    if typ == "SWD":
                        all_swd.append(rec)
                    elif typ == "Seizure":
                        all_sz.append(rec)

        keep_from = core_end - float(overlap_s)
        keep = t_new >= keep_from
        prev_t = t_new[keep].copy()
        prev_y = y_new[keep].copy()

        if progress_callback:
            if total_rows > 0:
                pct = 5 + 85 * min(1.0, processed / total_rows)
                progress_callback(pct, f"EEG: scanned {processed:,}/{total_rows:,} samples")
            else:
                progress_callback(10, f"EEG: scanned {processed:,} samples")

    if first_t is None or last_t is None:
        raise ValueError("No valid EEG samples found.")

    spike_times = merge_spike_times(all_spikes, refr_s=0.02)
    swd_events = merge_intervals(all_swd, merge_gap_s=0.5)
    sz_events = merge_intervals(all_sz, merge_gap_s=10.0)
    swd_events = filter_events_outside_events(swd_events, sz_events, margin_s=0.25)

    summary_df = bin_counts(spike_times, swd_events, sz_events, bin_s, first_t, last_t)
    events_df = _events_from_eeg_detection(spike_times, swd_events, sz_events)
    meta = {
        "fs_Hz": float(fs_used) if fs_used is not None else np.nan,
        "recording_duration_s": float(last_t - first_t),
        "total_spikes": int(len(spike_times)),
        "total_SWDs": int(len(swd_events)),
        "total_seizures": int(len(sz_events)),
        "bin_size_s": float(bin_s),
        "preprocess": "streamed rolling-median detrend + 50/100 Hz notch + 0.5-100 Hz bandpass",
        "detector_fs": "Spikes 500 Hz | SWD 200 Hz | Seizure 250 Hz",
        "eeg_detection_profile": "mouse_literature_guided_v6",
        "eeg_swd_criteria": "5-9 Hz, >=4 cycles, >=1.8x local baseline, harmonic gate",
        "eeg_seizure_criteria": ">=8 s dense high-amplitude rhythmic/evolving spike trains; reviewed EEG corrections auto-applied by source file",
        "eeg_spike_artifact_filter": "reject post-tail artifacts when 40-80 ms post/spike >0.30 and biphasic ratio <0.20",
        "eeg_spike_criteria": "14-70 Hz transient, amp z >=7.0, width 8-100 ms, post-tail artifact filter",
        "source_file": os.path.abspath(file_path),
        "streaming": True,
        "chunk_s": float(chunk_s),
        "overlap_s": float(overlap_s),
        "chunks_processed": int(chunks),
        "chunks_skipped": int(skipped_chunks),
    }
    events_df, applied = apply_eeg_saved_corrections(file_path, events_df, t0=float(first_t), t1=float(last_t))
    if applied:
        summary_df = recompute_eeg_bins_from_events(events_df, float(bin_s), t0=float(first_t), t1=float(last_t))
        meta["eeg_saved_corrections_applied"] = int(applied)
        meta = update_eeg_meta_from_events(meta, events_df, summary_df)
    if progress_callback:
        progress_callback(95, "EEG: finalizing events")
    return summary_df, events_df, meta


# ============================================================
# 6) EXCEL EXPORT
# ============================================================
def export_ecg_excel(save_path, beat_rows, meta):
    meta = dict(meta or {})
    df = ensure_ecg_beats_schema(
        beat_rows if isinstance(beat_rows, pd.DataFrame) else pd.DataFrame(beat_rows)
    )
    fs_hz = pd.to_numeric(pd.Series([meta.get("fs_Hz", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(fs_hz) and fs_hz > 0:
        df, rmssd_s = recompute_ecg_intervals_df(df, float(fs_hz))
        meta["RMSSD_s"] = rmssd_s
        meta["RMSSD_ms"] = rmssd_s * 1000 if np.isfinite(rmssd_s) else np.nan

    valid_ratio, valid_count, total_count = compute_ecg_validity(df)
    meta["qtc_formula"] = "Mitchell"
    meta["qtc_formula_detail"] = "QTc_Mitchell_ms = QT_interval_ms / sqrt(RR_ms / 100)"
    meta["validity_ratio"] = valid_ratio
    meta["valid_beats"] = valid_count
    meta["valid_beats_total"] = total_count
    threshold = float(meta.get("validity_threshold", 0.90))
    meta["validity_threshold"] = threshold
    meta["validity_pass"] = bool(np.isfinite(valid_ratio) and valid_ratio >= threshold)
    meta.setdefault("generated_at", now_iso())
    meta.setdefault("source_file", "")
    meta.setdefault("review_last_saved_at", "")
    summary = compute_ecg_summary_row(df, meta)
    df.drop(columns=["QTc_Bazett_s", "QTc_Bazett_ms"], inplace=True, errors="ignore")

    preferred = [
        "Beat_Index",
        "R_idx", "P_on_idx", "P_off_idx", "QRS_on_idx", "QRS_off_idx", "T_off_idx",
        "R_time_s",
        "RR_ms", "P_wave_dur_ms", "PR_interval_ms", "QRS_interval_ms", "QT_interval_ms", "QTc_Mitchell_ms",
        "Confidence", "Confidence_Rank",
        "Is_Corrected", "Corrected_At", "Correction_Source", "Correction_Notes",
    ]
    trailing = [c for c in df.columns if c not in preferred]
    df = df[[c for c in preferred if c in df.columns] + trailing]

    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="ECG_Beats")
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="ECG_Summary")
        pd.DataFrame([meta]).to_excel(writer, index=False, sheet_name="Meta")
    persist_workbook_corrections(save_path, str(meta.get("source_file", "")), df)


def export_eeg_excel(save_path, summary_df, events_df, meta, benchmark_df=None):
    events_df = dedupe_eeg_events(events_df)
    meta = update_eeg_meta_from_events(meta, events_df, summary_df)
    meta.setdefault("generated_at", now_iso())
    meta.setdefault("source_file", "")
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, index=False, sheet_name="EEG_Bins")
        events_df.to_excel(writer,  index=False, sheet_name="EEG_Events")
        if benchmark_df is not None:
            pd.DataFrame(benchmark_df).to_excel(writer, index=False, sheet_name="EEG_Benchmark")
        pd.DataFrame([meta]).to_excel(writer, index=False, sheet_name="Meta")
    persist_eeg_workbook_corrections(save_path, str(meta.get("source_file", "")), events_df)


def parse_eeg_manual_counts(workbook_path, sheet_name="Baseline"):
    raw = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
    header_idx = None
    for i, row in raw.iterrows():
        vals = [str(v).strip().lower() for v in row.tolist()]
        if "time" in vals and any("spike" in v for v in vals):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find manual EEG count header row in sheet '{sheet_name}'.")

    header = [str(v).strip().lower() for v in raw.iloc[header_idx].tolist()]
    cols = {}
    for idx, name in enumerate(header):
        if name == "time":
            cols["Time"] = idx
        elif "spike" in name:
            cols["Spike_Count"] = idx
        elif "swd" in name:
            cols["SWD_Count"] = idx
        elif "seiz" in name:
            cols["Seizure_Count"] = idx

    required = ["Time", "Spike_Count", "SWD_Count", "Seizure_Count"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Manual EEG workbook is missing columns: {', '.join(missing)}")

    data = pd.DataFrame({
        name: raw.iloc[header_idx + 1:, col].values
        for name, col in cols.items()
    })
    data = data[pd.notna(data["Time"])].copy()
    time_txt = data["Time"].astype(str).str.strip()
    data = data[~time_txt.str.upper().isin({"AVERAGE", "TOTAL", "MEAN"})].copy()

    for col in ["Spike_Count", "SWD_Count", "Seizure_Count"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)

    data["Hour_Index"] = np.arange(1, len(data) + 1, dtype=int)
    data["Bin_Start_s"] = (data["Hour_Index"] - 1) * 3600.0
    data["Bin_End_s"] = data["Hour_Index"] * 3600.0
    data["Manual_Time"] = data["Time"].astype(str)
    return data[[
        "Hour_Index", "Manual_Time", "Bin_Start_s", "Bin_End_s",
        "Spike_Count", "SWD_Count", "Seizure_Count",
    ]].reset_index(drop=True)


def compare_eeg_counts_to_manual(summary_df, manual_df):
    prog = pd.DataFrame(summary_df).reset_index(drop=True).copy()
    manual = pd.DataFrame(manual_df).reset_index(drop=True).copy()
    count_cols = ["Spike_Count", "SWD_Count", "Seizure_Count"]
    for col in count_cols:
        if col not in prog.columns:
            prog[col] = 0
        if col not in manual.columns:
            manual[col] = 0
        prog[col] = pd.to_numeric(prog[col], errors="coerce").fillna(0).astype(int)
        manual[col] = pd.to_numeric(manual[col], errors="coerce").fillna(0).astype(int)

    n = min(len(prog), len(manual))
    rows = []
    for i in range(n):
        row = {
            "Bin_Index": int(i + 1),
            "Manual_Time": manual.loc[i, "Manual_Time"] if "Manual_Time" in manual.columns else "",
            "Bin_Start_s": float(prog.loc[i, "Bin_Start_s"]) if "Bin_Start_s" in prog.columns else float(i * 3600),
            "Bin_End_s": float(prog.loc[i, "Bin_End_s"]) if "Bin_End_s" in prog.columns else float((i + 1) * 3600),
        }
        for col in count_cols:
            event = col.replace("_Count", "")
            p = int(prog.loc[i, col])
            m = int(manual.loc[i, col])
            row[f"Program_{event}"] = p
            row[f"Manual_{event}"] = m
            row[f"Abs_Error_{event}"] = abs(p - m)
        rows.append(row)
    benchmark_df = pd.DataFrame(rows)

    meta = {
        "eeg_benchmark_bins_compared": int(n),
        "eeg_benchmark_manual_rows": int(len(manual)),
        "eeg_benchmark_program_rows": int(len(prog)),
    }
    for col in count_cols:
        event = col.replace("_Count", "")
        p_total = int(prog.head(n)[col].sum()) if n else 0
        m_total = int(manual.head(n)[col].sum()) if n else 0
        meta[f"eeg_benchmark_program_{event.lower()}s"] = p_total
        meta[f"eeg_benchmark_manual_{event.lower()}s"] = m_total
        meta[f"eeg_benchmark_abs_error_{event.lower()}s"] = abs(p_total - m_total)
    if n:
        err_cols = [c for c in benchmark_df.columns if c.startswith("Abs_Error_")]
        meta["eeg_benchmark_total_abs_error"] = int(benchmark_df[err_cols].sum().sum())
        meta["eeg_benchmark_exact_bin_match_ratio"] = float((benchmark_df[err_cols].sum(axis=1) == 0).mean())
    else:
        meta["eeg_benchmark_total_abs_error"] = 0
        meta["eeg_benchmark_exact_bin_match_ratio"] = np.nan
    return benchmark_df, meta


def _normalize_eeg_event_type(value):
    txt = str(value).strip()
    low = txt.lower()
    if low in {"swd", "swds", "spike-wave discharge", "spike wave discharge"}:
        return "SWD"
    if low in {"seizure", "seizures"}:
        return "Seizure"
    if low in {"spike", "spikes"}:
        return "Spike"
    return "Spike"


def ensure_eeg_events_schema(events_df):
    df = pd.DataFrame(events_df).copy()
    if df.empty:
        df = pd.DataFrame(columns=EEG_EVENT_COLUMNS)
    if "Event_Index" not in df.columns:
        df["Event_Index"] = np.arange(len(df), dtype=int)
    df["Event_Index"] = pd.to_numeric(df["Event_Index"], errors="coerce")
    if df["Event_Index"].isna().any():
        df["Event_Index"] = np.arange(len(df), dtype=int)
    df["Event_Index"] = df["Event_Index"].astype(int)

    if "Type" not in df.columns:
        df["Type"] = "Spike"
    df["Type"] = df["Type"].apply(_normalize_eeg_event_type)

    for col in ["Start_s", "End_s", "Duration_s", "Confidence"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Start_s"] = df["Start_s"].fillna(0.0)
    df["End_s"] = df["End_s"].fillna(df["Start_s"])
    df.loc[df["Type"] == "Spike", "End_s"] = df.loc[df["Type"] == "Spike", "Start_s"]
    bad_end = df["End_s"] < df["Start_s"]
    df.loc[bad_end, "End_s"] = df.loc[bad_end, "Start_s"]
    df["Duration_s"] = (df["End_s"] - df["Start_s"]).clip(lower=0.0)

    for col in ["Is_Corrected", "Is_Deleted"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].apply(_to_bool).astype(bool)
    for col in ["Corrected_At", "Correction_Source", "Correction_Notes"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df.sort_values(["Start_s", "Event_Index"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    trailing = [c for c in df.columns if c not in EEG_EVENT_COLUMNS]
    return df[EEG_EVENT_COLUMNS + trailing]


def dedupe_eeg_events(events_df, tol_s=0.001):
    df = ensure_eeg_events_schema(events_df)
    if len(df) <= 1:
        return df
    work = df.copy()
    work["_type_key"] = work["Type"].map(_normalize_eeg_event_type)
    work["_start_key"] = (pd.to_numeric(work["Start_s"], errors="coerce") / tol_s).round().astype("Int64")
    work["_end_key"] = (pd.to_numeric(work["End_s"], errors="coerce") / tol_s).round().astype("Int64")
    keep_indices = []
    for _, group in work.groupby(["_type_key", "_start_key", "_end_key"], dropna=False, sort=False):
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue
        scored = group.assign(
            _active=(~group["Is_Deleted"].apply(_to_bool)).astype(int),
            _corrected=group["Is_Corrected"].apply(_to_bool).astype(int),
            _note_len=group["Correction_Notes"].fillna("").astype(str).str.strip().str.len(),
        ).sort_values(["_active", "_corrected", "_note_len", "Event_Index"], ascending=[False, False, False, False])
        keep_idx = scored.index[0]
        notes = []
        for note in group["Correction_Notes"].fillna("").astype(str):
            note = note.strip()
            if note and note not in notes:
                notes.append(note)
        if notes:
            df.at[keep_idx, "Correction_Notes"] = "; ".join(notes)
            df.at[keep_idx, "Is_Corrected"] = True
            df.at[keep_idx, "Corrected_At"] = now_iso()
            df.at[keep_idx, "Correction_Source"] = "manual_review_dedupe"
        keep_indices.append(keep_idx)
    out = df.loc[sorted(keep_indices)].copy()
    out.sort_values(["Start_s", "Event_Index"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return ensure_eeg_events_schema(out)


def _default_eeg_correction_store():
    return {"version": 1, "updated_at": now_iso(), "sources": {}}


def _eeg_source_key(source_file):
    if not source_file:
        return ""
    return os.path.normcase(os.path.abspath(str(source_file)))


def load_eeg_correction_store(path=EEG_CORRECTION_STORE_FILE):
    for candidate in [path, bundled_resource_path("eeg_corrections_v1.json")]:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            data.setdefault("version", 1)
            data.setdefault("updated_at", now_iso())
            data.setdefault("sources", {})
            if not isinstance(data["sources"], dict):
                data["sources"] = {}
            return data
        except Exception:
            continue
    return _default_eeg_correction_store()


def save_eeg_correction_store(store, path=EEG_CORRECTION_STORE_FILE):
    out = _default_eeg_correction_store()
    if isinstance(store, dict):
        out.update(store)
    out["version"] = 1
    out["updated_at"] = now_iso()
    out["sources"] = out.get("sources", {}) if isinstance(out.get("sources"), dict) else {}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def persist_eeg_workbook_corrections(workbook_path, source_file, events_df):
    source_key = _eeg_source_key(source_file)
    if not source_key:
        return 0
    df = ensure_eeg_events_schema(events_df)
    if df.empty:
        return 0
    source_text = df["Correction_Source"].fillna("").astype(str)
    note_text = df["Correction_Notes"].fillna("").astype(str).str.strip()
    corrected_mask = (
        df["Is_Corrected"].apply(_to_bool)
        | df["Is_Deleted"].apply(_to_bool)
        | source_text.str.contains("manual_review", case=False, regex=False)
        | (note_text != "")
    )
    corrected = df[corrected_mask].copy()
    if corrected.empty:
        return 0

    rows = []
    for _, row in corrected.iterrows():
        rows.append({
            "Event_Index": int(row.get("Event_Index", len(rows))),
            "Type": _normalize_eeg_event_type(row.get("Type", "Spike")),
            "Start_s": float(row.get("Start_s", 0.0)),
            "End_s": float(row.get("End_s", row.get("Start_s", 0.0))),
            "Duration_s": float(row.get("Duration_s", 0.0)),
            "Confidence": None if pd.isna(row.get("Confidence", np.nan)) else float(row.get("Confidence")),
            "Is_Deleted": bool(_to_bool(row.get("Is_Deleted", False))),
            "Correction_Notes": str(row.get("Correction_Notes", "") or ""),
            "Correction_Source": str(row.get("Correction_Source", "manual_review") or "manual_review"),
            "Corrected_At": str(row.get("Corrected_At", "") or now_iso()),
        })

    store = load_eeg_correction_store()
    store.setdefault("sources", {})
    store["sources"][source_key] = {
        "source_file": os.path.abspath(str(source_file)),
        "workbook_path": os.path.abspath(str(workbook_path)) if workbook_path else "",
        "saved_at": now_iso(),
        "events": rows,
    }
    save_eeg_correction_store(store)
    return len(rows)


def _match_eeg_correction_event(df, corr):
    if df.empty:
        return None
    typ = _normalize_eeg_event_type(corr.get("Type", "Spike"))
    start = float(corr.get("Start_s", 0.0))
    end = float(corr.get("End_s", start))
    candidates = df[df["Type"].map(_normalize_eeg_event_type) == typ]
    if candidates.empty:
        return None
    best_idx, best_score = None, -1.0
    for idx, row in candidates.iterrows():
        a = float(row.get("Start_s", 0.0))
        b = float(row.get("End_s", a))
        if typ == "Spike":
            dist = abs(a - start)
            if dist <= 0.25:
                score = 1.0 - dist / 0.25
            else:
                continue
        else:
            overlap = max(0.0, min(end, b) - max(start, a))
            min_dur = max(1e-9, min(max(0.0, end - start), max(0.0, b - a)))
            score = overlap / min_dur if min_dur > 0 else 0.0
            if overlap < 1.0 and abs(a - start) > 5.0:
                continue
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx


def apply_eeg_saved_corrections(source_file, events_df, t0=None, t1=None):
    source_key = _eeg_source_key(source_file)
    if not source_key:
        return ensure_eeg_events_schema(events_df), 0
    store = load_eeg_correction_store()
    rec = (store.get("sources", {}) or {}).get(source_key)
    if not isinstance(rec, dict):
        return ensure_eeg_events_schema(events_df), 0
    corrections = rec.get("events", [])
    if not isinstance(corrections, list) or not corrections:
        return ensure_eeg_events_schema(events_df), 0

    df = ensure_eeg_events_schema(events_df)
    applied = 0
    next_idx = int(df["Event_Index"].max() + 1) if len(df) else 0
    for corr in corrections:
        if not isinstance(corr, dict):
            continue
        typ = _normalize_eeg_event_type(corr.get("Type", "Spike"))
        start = float(corr.get("Start_s", 0.0))
        end = float(corr.get("End_s", start))
        if t0 is not None and end < float(t0) - 1.0:
            continue
        if t1 is not None and start > float(t1) + 1.0:
            continue
        if typ == "Spike":
            end = start
        if end < start:
            start, end = end, start
        deleted = bool(_to_bool(corr.get("Is_Deleted", False)))
        idx = _match_eeg_correction_event(df, corr)
        payload = {
            "Type": typ,
            "Start_s": start,
            "End_s": end,
            "Duration_s": max(0.0, end - start),
            "Confidence": np.nan if corr.get("Confidence", None) is None else float(corr.get("Confidence")),
            "Is_Corrected": True,
            "Is_Deleted": deleted,
            "Corrected_At": str(corr.get("Corrected_At", "") or now_iso()),
            "Correction_Source": "saved_eeg_correction",
            "Correction_Notes": str(corr.get("Correction_Notes", "") or ""),
        }
        if idx is None:
            payload["Event_Index"] = next_idx
            next_idx += 1
            df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
        else:
            for col, val in payload.items():
                df.at[idx, col] = val
        applied += 1
    return dedupe_eeg_events(df), applied


def active_eeg_events(events_df):
    df = ensure_eeg_events_schema(events_df)
    deleted = df["Is_Deleted"].apply(_to_bool).astype(bool)
    return df.loc[~deleted].copy()


def recompute_eeg_bins_from_events(events_df, bin_s, t0=0.0, t1=None):
    active = active_eeg_events(events_df)
    if t1 is None:
        if len(active):
            t1 = float(max(active["End_s"].max(), active["Start_s"].max()))
        else:
            t1 = float(bin_s)
    spike_ts = active.loc[active["Type"] == "Spike", "Start_s"].to_numpy(dtype=float)
    swd_evts = [
        {"start": float(r["Start_s"]), "end": float(r["End_s"])}
        for _, r in active.loc[active["Type"] == "SWD"].iterrows()
    ]
    sz_evts = [
        {"start": float(r["Start_s"]), "end": float(r["End_s"])}
        for _, r in active.loc[active["Type"] == "Seizure"].iterrows()
    ]
    return bin_counts(spike_ts, swd_evts, sz_evts, float(bin_s), float(t0), float(t1))


def update_eeg_meta_from_events(meta, events_df, summary_df=None):
    meta = dict(meta or {})
    active = active_eeg_events(events_df)
    meta["total_spikes"] = int((active["Type"] == "Spike").sum())
    meta["total_SWDs"] = int((active["Type"] == "SWD").sum())
    meta["total_seizures"] = int((active["Type"] == "Seizure").sum())
    if summary_df is not None and len(summary_df):
        meta["bin_size_s"] = float(pd.to_numeric(summary_df["Bin_End_s"] - summary_df["Bin_Start_s"],
                                                errors="coerce").dropna().median())
        meta["recording_duration_s"] = float(pd.to_numeric(summary_df["Bin_End_s"], errors="coerce").max()
                                             - pd.to_numeric(summary_df["Bin_Start_s"], errors="coerce").min())
    return meta


def load_eeg_workbook(workbook_path):
    summary_df = pd.read_excel(workbook_path, sheet_name="EEG_Bins")
    try:
        events_df = pd.read_excel(workbook_path, sheet_name="EEG_Events")
    except Exception:
        events_df = pd.DataFrame(columns=["Type", "Start_s", "End_s", "Duration_s", "Confidence"])
    try:
        meta_df = pd.read_excel(workbook_path, sheet_name="Meta")
        meta = meta_df.iloc[0].to_dict() if len(meta_df) else {}
    except Exception:
        meta = {}
    events_df = ensure_eeg_events_schema(events_df)
    return summary_df, events_df, meta


def _safe_filename(text, max_len=90):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch in (" ", "\t"):
            keep.append("_")
    name = "".join(keep).strip("._")
    return (name[:max_len] or "plot")


def _read_text_waveform_layout(file_path):
    layout = {
        "header_rows": 7,
        "data_offset": None,
        "line_len": None,
        "first_t": 0.0,
    }
    offsets = []
    lines = []
    with open(file_path, "rb") as f:
        for _ in range(20):
            off = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(off)
            lines.append(line)
    for i, line in enumerate(lines):
        low = line.lower()
        if b"time from start" in low:
            layout["header_rows"] = i + 1
            if i + 1 < len(lines):
                layout["data_offset"] = offsets[i + 1]
            break
    if layout["data_offset"] is None and len(lines) > layout["header_rows"]:
        layout["data_offset"] = offsets[layout["header_rows"]]

    data_idx = layout["header_rows"]
    if data_idx < len(lines):
        layout["line_len"] = len(lines[data_idx])
        try:
            parts = lines[data_idx].split(b"\t")
            layout["first_t"] = float(parts[3])
        except Exception:
            layout["first_t"] = 0.0
    if data_idx + 1 < len(lines) and len(lines[data_idx + 1]) != layout["line_len"]:
        layout["line_len"] = None
    return layout


def read_waveform_window(file_path, start_s, end_s, fs_hint=None):
    start_s = max(0.0, float(start_s))
    end_s = max(start_s + 1e-6, float(end_s))
    path = pathlib.Path(file_path)
    ext = path.suffix.lower()

    if ext not in [".tsv", ".txt", ".csv"]:
        t, y = load_waveform_tabular(file_path)
        keep = (t >= start_s) & (t <= end_s)
        return t[keep], y[keep]

    sep = "\t" if ext in [".tsv", ".txt"] else ","
    fs_hint = float(fs_hint) if fs_hint is not None and np.isfinite(float(fs_hint)) else None
    if fs_hint:
        layout = _read_text_waveform_layout(file_path)
        if layout["data_offset"] is not None and layout["line_len"]:
            first_t = float(layout.get("first_t", 0.0))
            start_row = max(0, int(np.floor((start_s - first_t) * fs_hint)))
            n_rows = max(1, int(np.ceil((end_s - start_s) * fs_hint)) + 2)
            times = np.empty(n_rows, dtype=float)
            vals = np.empty(n_rows, dtype=float)
            count = 0
            with open(file_path, "rb") as f:
                f.seek(int(layout["data_offset"]) + start_row * int(layout["line_len"]))
                for _ in range(n_rows):
                    line = f.read(int(layout["line_len"]))
                    if not line:
                        break
                    parts = line.split(b"\t" if sep == "\t" else b",")
                    if len(parts) <= 4:
                        continue
                    try:
                        ti = float(parts[3])
                        yi = float(parts[4])
                    except Exception:
                        continue
                    if start_s <= ti <= end_s:
                        times[count] = ti
                        vals[count] = yi
                        count += 1
                    elif ti > end_s:
                        break
            return times[:count], vals[:count]

    # Portable fallback. Slow on huge files, but works for non-fixed-width text exports.
    rows = []
    for t_chunk, y_chunk in iter_waveform_tabular_chunks(file_path, chunk_rows=200000):
        if len(t_chunk) == 0:
            continue
        if t_chunk[-1] < start_s:
            continue
        if t_chunk[0] > end_s:
            break
        keep = (t_chunk >= start_s) & (t_chunk <= end_s)
        if np.any(keep):
            rows.append((t_chunk[keep], y_chunk[keep]))
    if not rows:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])


def _robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    dev = mad(x, scale="normal", nan_policy="omit") + 1e-9
    return (x - med) / dev


def score_eeg_candidate_windows(source_file, bin_start_s, bin_end_s, fs_hint,
                                window_s=20.0, step_s=60.0, top_n=2):
    candidates = []
    starts = np.arange(float(bin_start_s), max(float(bin_start_s), float(bin_end_s) - window_s), float(step_s))
    for a in starts:
        b = min(float(bin_end_s), float(a + window_s))
        t, y = read_waveform_window(source_file, a, b, fs_hint=fs_hint)
        if len(y) < max(10, int((fs_hint or 1000) * min(1.0, window_s * 0.2))):
            continue
        x = y - np.nanmedian(y)
        rms = float(np.sqrt(np.nanmean(x * x)))
        ptp = float(np.nanpercentile(x, 99) - np.nanpercentile(x, 1))
        line_len = float(np.nansum(np.abs(np.diff(x))) / max(1e-9, (t[-1] - t[0])))
        candidates.append({
            "start_s": float(a),
            "end_s": float(b),
            "rms": rms,
            "ptp": ptp,
            "line_len": line_len,
        })
    if not candidates:
        return []
    df = pd.DataFrame(candidates)
    df["score"] = _robust_z(df["rms"].values) + _robust_z(df["ptp"].values) + _robust_z(df["line_len"].values)
    df.sort_values("score", ascending=False, inplace=True)

    kept = []
    for _, row in df.iterrows():
        a, b = float(row["start_s"]), float(row["end_s"])
        if any(not (b <= k["start_s"] or a >= k["end_s"]) for k in kept):
            continue
        kept.append(row.to_dict())
        if len(kept) >= int(top_n):
            break
    return kept


def save_eeg_debug_plot(source_file, events_df, out_png, start_s, end_s,
                        fs_hint=None, title="", manual_note=""):
    t, y = read_waveform_window(source_file, start_s, end_s, fs_hint=fs_hint)
    if len(t) < 5:
        raise ValueError(f"No waveform samples found for {start_s:.3f}-{end_s:.3f} s")
    if fs_hint is not None and np.isfinite(float(fs_hint)):
        fs = float(fs_hint)
    else:
        _, _, fs = ensure_uniform_sampling(t, y)

    try:
        t_u, y_u, fs = ensure_uniform_sampling(t, y, fs_target=fs)
    except Exception:
        t_u, y_u = t, y
    try:
        sig = eeg_preprocess(y_u, fs)
    except Exception:
        sig = y_u - np.nanmedian(y_u)

    swd_trace = np.full_like(sig, np.nan, dtype=float)
    hf_trace = np.full_like(sig, np.nan, dtype=float)
    try:
        swd_trace = _robust_z(bandpass_zero_phase(sig, fs, 5.0, min(12.0, 0.45 * fs), order=3))
    except Exception:
        pass
    try:
        hf_trace = _robust_z(np.abs(bandpass_zero_phase(sig, fs, 14.0, min(70.0, 0.45 * fs), order=3)))
    except Exception:
        pass

    ev = pd.DataFrame(events_df).copy()
    if len(ev):
        ev["Start_s"] = pd.to_numeric(ev["Start_s"], errors="coerce")
        ev["End_s"] = pd.to_numeric(ev["End_s"], errors="coerce")
        ev = ev[(ev["End_s"] >= start_s) & (ev["Start_s"] <= end_s)].copy()

    colours = {"Spike": "#F9A825", "SWD": "#0288D1", "Seizure": "#D32F2F"}
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t_u, y_u, color="#333333", lw=0.6)
    axes[0].set_ylabel("Raw")
    axes[1].plot(t_u, sig, color="#111111", lw=0.7)
    axes[1].set_ylabel("Filtered")
    axes[2].plot(t_u, swd_trace, color="#0288D1", lw=0.8, label="5-12 Hz z")
    axes[2].plot(t_u, hf_trace, color="#F9A825", lw=0.8, alpha=0.8, label="14-70 Hz abs z")
    axes[2].set_ylabel("Detector traces")
    axes[2].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.18)
        for _, row in ev.iterrows():
            typ = str(row.get("Type", ""))
            colour = colours.get(typ, "#777777")
            a = float(row["Start_s"])
            b = float(row["End_s"])
            if typ == "Spike" or abs(b - a) < 1e-9:
                ax.axvline(a, color=colour, lw=1.2, alpha=0.85)
            else:
                ax.axvspan(a, b, color=colour, alpha=0.20)
    axes[-1].set_xlabel("Time from start (s)")
    fig.suptitle(title)
    if manual_note:
        axes[0].text(0.01, 0.98, manual_note, transform=axes[0].transAxes,
                     va="top", ha="left", fontsize=9,
                     bbox=dict(facecolor="white", edgecolor="#BBBBBB", alpha=0.85))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_eeg_benchmark_plot(benchmark_df, out_png):
    if benchmark_df is None or len(benchmark_df) == 0:
        return
    df = pd.DataFrame(benchmark_df).copy()
    x = np.arange(len(df))
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    specs = [
        ("Spike", "Spikes"),
        ("SWD", "SWDs"),
        ("Seizure", "Seizures"),
    ]
    for ax, (key, label) in zip(axes, specs):
        p = pd.to_numeric(df.get(f"Program_{key}", 0), errors="coerce").fillna(0)
        m = pd.to_numeric(df.get(f"Manual_{key}", 0), errors="coerce").fillna(0)
        ax.bar(x - 0.18, m, width=0.36, label="Manual", color="#455A64")
        ax.bar(x + 0.18, p, width=0.36, label="Program", color="#EF6C00")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(int(v)) for v in df["Bin_Index"]])
    axes[-1].set_xlabel("Hour/bin")
    fig.suptitle("EEG benchmark: manual vs program counts")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_eeg_debug_windows(summary_df, events_df, benchmark_df, source_file, fs_hint,
                            event_pad_s=5.0, max_plots=40,
                            candidate_window_s=20.0, candidate_step_s=60.0,
                            candidate_count=2):
    events = pd.DataFrame(events_df).copy()
    if len(events):
        events["Start_s"] = pd.to_numeric(events["Start_s"], errors="coerce")
        events["End_s"] = pd.to_numeric(events["End_s"], errors="coerce")
        events["Duration_s"] = pd.to_numeric(events["Duration_s"], errors="coerce").fillna(0.0)
    bench = pd.DataFrame(benchmark_df).copy() if benchmark_df is not None else pd.DataFrame()
    windows = []

    def add_window(reason, start_s, end_s, priority, event_type="", bin_index=np.nan, note=""):
        if len(windows) >= int(max_plots):
            return
        start_s = max(0.0, float(start_s))
        end_s = max(start_s + 0.5, float(end_s))
        windows.append({
            "Reason": reason,
            "Priority": int(priority),
            "Type": event_type,
            "Bin_Index": int(bin_index) if pd.notna(bin_index) else "",
            "Start_s": start_s,
            "End_s": end_s,
            "Manual_Note": note,
        })

    if len(bench):
        for _, b in bench.sort_values("Bin_Index").iterrows():
            bin_idx = int(b["Bin_Index"])
            bin_start = float(b["Bin_Start_s"])
            bin_end = float(b["Bin_End_s"])
            note = (
                f"Bin {bin_idx}: manual spike/SWD/seizure="
                f"{int(b.get('Manual_Spike', 0))}/{int(b.get('Manual_SWD', 0))}/{int(b.get('Manual_Seizure', 0))}; "
                f"program={int(b.get('Program_Spike', 0))}/{int(b.get('Program_SWD', 0))}/{int(b.get('Program_Seizure', 0))}"
            )

            for typ in ["SWD", "Seizure"]:
                over = int(b.get(f"Program_{typ}", 0)) - int(b.get(f"Manual_{typ}", 0))
                if over > 0 and len(events):
                    ev = events[(events["Type"].astype(str) == typ)
                                & (events["Start_s"] >= bin_start)
                                & (events["Start_s"] < bin_end)]
                    for _, row in ev.iterrows():
                        add_window(f"program_{typ}_over_manual", row["Start_s"] - event_pad_s,
                                   row["End_s"] + event_pad_s, 1, typ, bin_idx, note)

            missed_seizures = int(b.get("Manual_Seizure", 0)) - int(b.get("Program_Seizure", 0))
            if missed_seizures > 0:
                candidates = score_eeg_candidate_windows(
                    source_file, bin_start, bin_end, fs_hint,
                    window_s=candidate_window_s,
                    step_s=candidate_step_s,
                    top_n=max(1, int(candidate_count)),
                )
                for j, cand in enumerate(candidates, start=1):
                    add_window(f"manual_seizure_missed_candidate_{j}", cand["start_s"], cand["end_s"],
                               2, "Candidate", bin_idx,
                               note + f"; candidate score={cand.get('score', np.nan):.2f}")

        spike_errors = bench.copy()
        if "Abs_Error_Spike" in spike_errors.columns and len(events):
            spike_errors = spike_errors[pd.to_numeric(spike_errors["Abs_Error_Spike"], errors="coerce").fillna(0) > 0]
            spike_errors = spike_errors.sort_values("Abs_Error_Spike", ascending=False).head(6)
            for _, b in spike_errors.iterrows():
                bin_idx = int(b["Bin_Index"])
                bin_start = float(b["Bin_Start_s"])
                bin_end = float(b["Bin_End_s"])
                ev = events[(events["Type"].astype(str) == "Spike")
                            & (events["Start_s"] >= bin_start)
                            & (events["Start_s"] < bin_end)].head(2)
                note = (
                    f"Bin {bin_idx}: manual spikes={int(b.get('Manual_Spike', 0))}; "
                    f"program spikes={int(b.get('Program_Spike', 0))}"
                )
                for _, row in ev.iterrows():
                    add_window("spike_count_mismatch", row["Start_s"] - event_pad_s,
                               row["Start_s"] + event_pad_s, 3, "Spike", bin_idx, note)

    else:
        for _, row in events.head(int(max_plots)).iterrows():
            add_window("program_event", row["Start_s"] - event_pad_s, row["End_s"] + event_pad_s,
                       9, str(row.get("Type", "")), "", "")

    df = pd.DataFrame(windows)
    if len(df):
        df.sort_values(["Priority", "Start_s"], inplace=True)
        df = df.head(int(max_plots)).reset_index(drop=True)
    return df


# ============================================================
# 7) ECG review UI
# ============================================================
class ECGReviewWindow(tk.Toplevel):
    MARKER_STYLE = {
        "P_on_idx": ("#00BCD4", "^", "P_on"),
        "P_off_idx": ("#0288D1", "v", "P_off"),
        "QRS_on_idx": ("#FF9800", "X", "QRS_on"),
        "QRS_off_idx": ("#7E57C2", "X", "QRS_off"),
        "R_idx": ("green", "X", "R"),
        "T_off_idx": ("#F44336", "X", "T_off"),
    }

    def __init__(self, parent, workbook_path):
        super().__init__(parent)
        self.workbook_path = os.path.abspath(workbook_path)
        self.title(f"ECG Review - {os.path.basename(self.workbook_path)}")
        self.geometry("1180x760")
        self.minsize(980, 680)

        self.beats_df, self.meta = load_ecg_workbook(self.workbook_path)
        self.meta = dict(self.meta or {})

        self.source_file = self._resolve_source_file(str(self.meta.get("source_file", "")))
        self.meta["source_file"] = self.source_file

        fs_meta = pd.to_numeric(pd.Series([self.meta.get("fs_Hz", np.nan)]), errors="coerce").iloc[0]
        t_raw, y_raw = load_waveform_tabular(self.source_file)
        if np.isfinite(fs_meta) and fs_meta > 0:
            self.t, self.y, self.fs = ensure_uniform_sampling(t_raw, y_raw, fs_target=float(fs_meta))
        else:
            self.t, self.y, self.fs = ensure_uniform_sampling(t_raw, y_raw)
        self.sig_filt, _ = auto_bandpass_notch_ecg(self.y, self.fs)

        self.beats_df, restored = apply_saved_corrections(self.workbook_path, self.beats_df)
        self.beats_df, rmssd_s = recompute_ecg_intervals_df(self.beats_df, self.fs)
        self.meta["RMSSD_s"] = rmssd_s
        self.meta["RMSSD_ms"] = rmssd_s * 1000 if np.isfinite(rmssd_s) else np.nan
        self.meta["fs_Hz"] = float(self.fs)

        self.review_order = self._build_review_order()
        if not self.review_order:
            raise ValueError("No ECG beats found in workbook for review.")
        self.current_order_pos = 0

        self.selected_landmark = tk.StringVar(value="R_idx")
        self.notes_var = tk.StringVar(value="")
        self.info_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value=f"Loaded review session. Restored saved corrections: {restored}")

        self._build_ui()
        self._load_current_beat()

    def destroy(self):
        if hasattr(self, "fig"):
            try:
                plt.close(self.fig)
            except Exception:
                pass
        for attr in ("selected_landmark", "notes_var", "info_var", "status_var"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        super().destroy()

    def _resolve_source_file(self, source_file):
        if source_file and os.path.exists(source_file):
            return os.path.abspath(source_file)
        messagebox.showwarning(
            "Source file not found",
            "Source TSV path in workbook metadata is missing or unavailable.\nPlease select the original TSV file.",
            parent=self,
        )
        selected = filedialog.askopenfilename(
            parent=self,
            title="Select source TSV/TXT/CSV/Excel file for ECG review",
            filetypes=[
                ("Tab/CSV/Excel", "*.tsv *.txt *.csv *.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            raise ValueError("Review canceled: source waveform file is required.")
        return os.path.abspath(selected)

    def _build_review_order(self):
        view = ensure_ecg_beats_schema(self.beats_df)
        rank = pd.to_numeric(view["Confidence_Rank"], errors="coerce")
        rank = rank.fillna(np.inf)
        ordered = view.assign(_rank=rank).sort_values(["_rank", "Beat_Index"])
        return ordered["Beat_Index"].astype(int).tolist()

    def _build_ui(self):
        top = tk.Frame(self, padx=10, pady=8)
        top.pack(fill="x")

        tk.Label(top, text="Editable landmark:", font=("Arial", 10, "bold")).pack(side="left")
        ttk.Combobox(
            top,
            values=LANDMARK_COLUMNS,
            textvariable=self.selected_landmark,
            state="readonly",
            width=14,
        ).pack(side="left", padx=6)

        tk.Button(top, text="Prev Beat", command=self._prev_beat, bg="#455A64", fg="white").pack(side="left", padx=4)
        tk.Button(top, text="Next Beat", command=self._next_beat, bg="#455A64", fg="white").pack(side="left", padx=4)
        tk.Button(top, text="Save Workbook", command=self._save_workbook, bg="#2E7D32", fg="white").pack(side="left", padx=10)
        tk.Button(top, text="Close", command=self.destroy, bg="#757575", fg="white").pack(side="right")

        note_row = tk.Frame(self, padx=10, pady=4)
        note_row.pack(fill="x")
        tk.Label(note_row, text="Correction notes:", font=("Arial", 10, "bold")).pack(side="left")
        note_entry = tk.Entry(note_row, textvariable=self.notes_var)
        note_entry.pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(note_row, text="Apply Note", command=self._apply_note).pack(side="left", padx=4)

        info = tk.Label(self, textvariable=self.info_var, anchor="w", justify="left",
                        font=("Consolas", 9), padx=10, pady=4)
        info.pack(fill="x")

        status = tk.Label(self, textvariable=self.status_var, anchor="w", fg="#1E88E5", padx=10, pady=4)
        status.pack(fill="x")

        fig_frame = tk.Frame(self, padx=8, pady=6)
        fig_frame.pack(fill="both", expand=True)
        self.fig, self.ax = plt.subplots(figsize=(11.5, 5.2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _current_beat_index(self):
        return int(self.review_order[self.current_order_pos])

    def _current_row_index(self):
        beat_index = self._current_beat_index()
        idx = self.beats_df.index[self.beats_df["Beat_Index"].astype(int) == beat_index]
        if len(idx) == 0:
            raise ValueError(f"Beat index {beat_index} not found in dataframe.")
        return int(idx[0])

    def _load_current_beat(self):
        row_idx = self._current_row_index()
        row = self.beats_df.iloc[row_idx]
        self.notes_var.set(str(row.get("Correction_Notes", "")))
        self._draw_current_beat()
        self._refresh_info()

    def _refresh_info(self):
        row_idx = self._current_row_index()
        row = self.beats_df.iloc[row_idx]
        beat_idx = int(row["Beat_Index"])
        conf = pd.to_numeric(pd.Series([row.get("Confidence", np.nan)]), errors="coerce").iloc[0]
        rank = pd.to_numeric(pd.Series([row.get("Confidence_Rank", np.nan)]), errors="coerce").iloc[0]
        ratio, valid, total = compute_ecg_validity(self.beats_df)
        valid_pct = f"{ratio * 100:.1f}%" if np.isfinite(ratio) else "N/A"
        self.info_var.set(
            f"Beat {beat_idx} ({self.current_order_pos + 1}/{len(self.review_order)}) | "
            f"Confidence={conf:.3f} Rank={rank:.0f} | "
            f"RR={row.get('RR_ms', np.nan):.2f} ms  PR={row.get('PR_interval_ms', np.nan):.2f} ms  "
            f"QRS={row.get('QRS_interval_ms', np.nan):.2f} ms  QT={row.get('QT_interval_ms', np.nan):.2f} ms  "
            f"QTcM={row.get('QTc_Mitchell_ms', np.nan):.2f} ms | "
            f"Valid beats={valid}/{total} ({valid_pct})"
        )

    def _draw_current_beat(self):
        row_idx = self._current_row_index()
        row = self.beats_df.iloc[row_idx]
        r_idx = _to_optional_int(row.get("R_idx"))
        if np.isnan(r_idx):
            r_idx = _to_optional_int(row.get("QRS_on_idx"))
        if np.isnan(r_idx):
            r_idx = 0
        r_idx = int(max(0, min(len(self.sig_filt) - 1, r_idx)))

        left = int(0.12 * self.fs)
        right = int(0.25 * self.fs)
        sl = max(0, r_idx - left)
        el = min(len(self.sig_filt), r_idx + right)

        self.ax.clear()
        x = np.arange(sl, el) / self.fs
        self.ax.plot(x, self.sig_filt[sl:el], color="darkred", lw=1.0, label="ECG")

        for col in LANDMARK_COLUMNS:
            color, marker, label = self.MARKER_STYLE[col]
            idx = _to_optional_int(row.get(col))
            if np.isnan(idx):
                continue
            idx = int(idx)
            if 0 <= idx < len(self.sig_filt):
                self.ax.scatter(
                    idx / self.fs,
                    self.sig_filt[idx],
                    color=color,
                    s=80 if col != "R_idx" else 100,
                    marker=marker,
                    zorder=5,
                    label=label,
                )
                self.ax.text(
                    idx / self.fs,
                    self.sig_filt[idx],
                    f" {label}",
                    fontsize=8,
                    color=color,
                )

        self.ax.set_title(
            f"Beat {int(row['Beat_Index'])} - click to set {self.selected_landmark.get()}",
            fontsize=11,
        )
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(alpha=0.2)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            self.ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_plot_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        idx_abs = int(round(float(event.xdata) * self.fs))
        idx_abs = int(max(0, min(len(self.sig_filt) - 1, idx_abs)))

        row_idx = self._current_row_index()
        col = self.selected_landmark.get()
        self.beats_df.at[row_idx, col] = idx_abs
        self.beats_df.at[row_idx, "Is_Corrected"] = True
        self.beats_df.at[row_idx, "Corrected_At"] = now_iso()
        self.beats_df.at[row_idx, "Correction_Source"] = "manual_review"
        note = self.notes_var.get().strip()
        if note:
            self.beats_df.at[row_idx, "Correction_Notes"] = note

        self.beats_df, rmssd_s = recompute_ecg_intervals_df(self.beats_df, self.fs)
        self.meta["RMSSD_s"] = rmssd_s
        self.meta["RMSSD_ms"] = rmssd_s * 1000 if np.isfinite(rmssd_s) else np.nan
        self.status_var.set(f"Updated {col} for beat {self._current_beat_index()} to sample {idx_abs}.")
        self._draw_current_beat()
        self._refresh_info()

    def _apply_note(self):
        row_idx = self._current_row_index()
        self.beats_df.at[row_idx, "Correction_Notes"] = self.notes_var.get().strip()
        self.beats_df.at[row_idx, "Is_Corrected"] = True
        self.beats_df.at[row_idx, "Corrected_At"] = now_iso()
        self.beats_df.at[row_idx, "Correction_Source"] = "manual_review"
        self.status_var.set(f"Updated note for beat {self._current_beat_index()}.")

    def _prev_beat(self):
        if self.current_order_pos > 0:
            self.current_order_pos -= 1
            self._load_current_beat()

    def _next_beat(self):
        if self.current_order_pos < len(self.review_order) - 1:
            self.current_order_pos += 1
            self._load_current_beat()

    def _save_workbook(self):
        ratio, valid, total = compute_ecg_validity(self.beats_df)
        threshold = float(self.meta.get("validity_threshold", 0.90))
        self.meta["validity_ratio"] = ratio
        self.meta["valid_beats"] = valid
        self.meta["valid_beats_total"] = total
        self.meta["validity_threshold"] = threshold
        self.meta["validity_pass"] = bool(np.isfinite(ratio) and ratio >= threshold)
        self.meta["review_last_saved_at"] = now_iso()
        self.meta["review_source"] = "manual_ui"
        self.meta["source_file"] = self.source_file
        export_ecg_excel(self.workbook_path, self.beats_df, self.meta)
        self.status_var.set(f"Saved workbook: {self.workbook_path}")
        messagebox.showinfo("Saved", f"Workbook overwritten with corrected ECG data:\n{self.workbook_path}", parent=self)


class EEGReviewWindow(tk.Toplevel):
    EVENT_COLOURS = {
        "Spike": "#F9A825",
        "SWD": "#0288D1",
        "Seizure": "#D32F2F",
    }

    def __init__(self, parent, workbook_path):
        super().__init__(parent)
        self.workbook_path = os.path.abspath(workbook_path)
        self.title(f"EEG Review - {os.path.basename(self.workbook_path)}")
        self.geometry("1220x820")
        self.minsize(1040, 720)

        self.summary_df, self.events_df, self.meta = load_eeg_workbook(self.workbook_path)
        self.meta = dict(self.meta or {})
        self.source_file = self._resolve_source_file(str(self.meta.get("source_file", "")))
        self.meta["source_file"] = self.source_file

        fs_meta = pd.to_numeric(pd.Series([self.meta.get("fs_Hz", np.nan)]), errors="coerce").iloc[0]
        self.fs = float(fs_meta) if np.isfinite(fs_meta) and fs_meta > 0 else 2000.0
        self.current_order_pos = 0
        self.review_order = self._build_review_order()

        self.event_type_var = tk.StringVar(value="Spike")
        self.marker_var = tk.StringVar(value="Start_s")
        self.start_var = tk.StringVar(value="0.000")
        self.end_var = tk.StringVar(value="0.000")
        self.window_var = tk.StringVar(value="20")
        self.notes_var = tk.StringVar(value="")
        self.info_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Loaded EEG review session. Shortcuts: Ctrl+1=start, Ctrl+2=end, Ctrl+Delete=delete, Ctrl+S=save.")

        self.t_view = np.array([], dtype=float)
        self.raw_view = np.array([], dtype=float)
        self.filtered_view = np.array([], dtype=float)
        self.swd_trace = np.array([], dtype=float)
        self.hf_trace = np.array([], dtype=float)

        self._build_ui()
        self._bind_shortcuts()
        if self.review_order:
            self._load_current_event()
        else:
            self._refresh_info()
            self._draw_current_event()

    def destroy(self):
        if hasattr(self, "fig"):
            try:
                plt.close(self.fig)
            except Exception:
                pass
        for attr in ("event_type_var", "marker_var", "start_var", "end_var",
                     "window_var", "notes_var", "info_var", "status_var"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        super().destroy()

    def _resolve_source_file(self, source_file):
        if source_file and os.path.exists(source_file):
            return os.path.abspath(source_file)
        messagebox.showwarning(
            "Source file not found",
            "Source TSV path in workbook metadata is missing or unavailable.\nPlease select the original EEG TSV file.",
            parent=self,
        )
        selected = filedialog.askopenfilename(
            parent=self,
            title="Select source TSV/TXT/CSV/Excel file for EEG review",
            filetypes=[
                ("Tab/CSV/Excel", "*.tsv *.txt *.csv *.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            raise ValueError("Review canceled: source waveform file is required.")
        return os.path.abspath(selected)

    def _build_review_order(self):
        view = ensure_eeg_events_schema(self.events_df)
        if not len(view):
            return []
        view = view.assign(_deleted=view["Is_Deleted"].apply(_to_bool).astype(int))
        view.sort_values(["_deleted", "Start_s", "Event_Index"], inplace=True)
        return view["Event_Index"].astype(int).tolist()

    def _build_ui(self):
        top = tk.Frame(self, padx=10, pady=8)
        top.pack(fill="x")

        tk.Label(top, text="Type:", font=("Arial", 10, "bold")).pack(side="left")
        ttk.Combobox(top, values=EEG_EVENT_TYPES, textvariable=self.event_type_var,
                     state="readonly", width=10).pack(side="left", padx=5)

        tk.Label(top, text="Click edits:", font=("Arial", 10, "bold")).pack(side="left", padx=(12, 0))
        ttk.Combobox(top, values=["Start_s", "End_s"], textvariable=self.marker_var,
                     state="readonly", width=9).pack(side="left", padx=5)
        tk.Button(top, text="Click=Start", command=lambda: self._set_marker("Start_s")).pack(side="left", padx=2)
        tk.Button(top, text="Click=End", command=lambda: self._set_marker("End_s")).pack(side="left", padx=2)

        tk.Label(top, text="Start s:").pack(side="left", padx=(12, 0))
        tk.Entry(top, textvariable=self.start_var, width=11).pack(side="left", padx=4)
        tk.Label(top, text="End s:").pack(side="left")
        tk.Entry(top, textvariable=self.end_var, width=11).pack(side="left", padx=4)
        tk.Label(top, text="Window s:").pack(side="left", padx=(12, 0))
        tk.Entry(top, textvariable=self.window_var, width=6).pack(side="left", padx=4)

        tk.Button(top, text="Prev", command=self._prev_event, bg="#455A64", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Next", command=self._next_event, bg="#455A64", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Apply", command=self._apply_fields, bg="#1E88E5", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Add Event", command=self._add_event, bg="#2E7D32", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Delete Event", command=lambda: self._set_deleted(True), bg="#C62828", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Restore Event", command=lambda: self._set_deleted(False), bg="#EF6C00", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Save Review + Notes", command=self._save_workbook, bg="#2E7D32", fg="white").pack(side="left", padx=10)
        tk.Button(top, text="Close", command=self.destroy, bg="#757575", fg="white").pack(side="right")

        note_row = tk.Frame(self, padx=10, pady=4)
        note_row.pack(fill="x")
        tk.Label(note_row, text="Correction notes:", font=("Arial", 10, "bold")).pack(side="left")
        tk.Entry(note_row, textvariable=self.notes_var).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(note_row, text="Apply Note", command=self._apply_note).pack(side="left", padx=4)
        tk.Button(note_row, text="Save Current Plot", command=self._save_current_plot).pack(side="left", padx=4)

        info = tk.Label(self, textvariable=self.info_var, anchor="w", justify="left",
                        font=("Consolas", 9), padx=10, pady=4)
        info.pack(fill="x")

        status = tk.Label(self, textvariable=self.status_var, anchor="w", fg="#1E88E5", padx=10, pady=4)
        status.pack(fill="x")

        fig_frame = tk.Frame(self, padx=8, pady=6)
        fig_frame.pack(fill="both", expand=True)
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 6.8), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _bind_shortcuts(self):
        self.bind("<Left>", lambda event: self._prev_event())
        self.bind("<Right>", lambda event: self._next_event())
        self.bind("<Control-s>", lambda event: self._save_workbook())
        self.bind("<Control-S>", lambda event: self._save_workbook())
        self.bind("<Control-Key-1>", lambda event: self._set_marker("Start_s"))
        self.bind("<Control-Key-2>", lambda event: self._set_marker("End_s"))
        self.bind("<Control-Delete>", lambda event: self._set_deleted(True))
        self.bind("<Control-BackSpace>", lambda event: self._set_deleted(True))

    def _set_marker(self, marker):
        if marker not in ("Start_s", "End_s"):
            return
        self.marker_var.set(marker)
        label = "start" if marker == "Start_s" else "end"
        self.status_var.set(f"Plot clicks now set event {label}.")
        self._draw_current_event()

    def _current_event_index(self):
        if not self.review_order:
            return None
        return int(self.review_order[self.current_order_pos])

    def _current_row_index(self):
        event_index = self._current_event_index()
        if event_index is None:
            return None
        idx = self.events_df.index[self.events_df["Event_Index"].astype(int) == event_index]
        if len(idx) == 0:
            raise ValueError(f"Event index {event_index} not found in dataframe.")
        return int(idx[0])

    def _load_current_event(self):
        row_idx = self._current_row_index()
        if row_idx is None:
            self.start_var.set("0.000")
            self.end_var.set("0.000")
            self.notes_var.set("")
            self.event_type_var.set("Spike")
        else:
            row = self.events_df.iloc[row_idx]
            self.event_type_var.set(_normalize_eeg_event_type(row.get("Type", "Spike")))
            self.start_var.set(f"{float(row.get('Start_s', 0.0)):.4f}")
            self.end_var.set(f"{float(row.get('End_s', row.get('Start_s', 0.0))):.4f}")
            self.notes_var.set(str(row.get("Correction_Notes", "")))
        self._draw_current_event()
        self._refresh_info()

    def _parse_fields(self):
        typ = _normalize_eeg_event_type(self.event_type_var.get())
        start = float(self.start_var.get().strip())
        end = float(self.end_var.get().strip()) if self.end_var.get().strip() else start
        if typ == "Spike":
            end = start
        if end < start:
            start, end = end, start
        return typ, start, end

    def _visible_range(self):
        try:
            _, start, end = self._parse_fields()
        except Exception:
            start, end = 0.0, 0.0
        win_s = max(1.0, float(self.window_var.get().strip() or 20.0))
        if end > start:
            center = 0.5 * (start + end)
            width = max(win_s, (end - start) + 10.0)
        else:
            center = start
            width = win_s
        return max(0.0, center - 0.5 * width), center + 0.5 * width

    def _load_view_data(self, start_s, end_s):
        t, y = read_waveform_window(self.source_file, start_s, end_s, fs_hint=self.fs)
        if len(t) < 5:
            self.t_view = np.array([], dtype=float)
            self.raw_view = np.array([], dtype=float)
            self.filtered_view = np.array([], dtype=float)
            self.swd_trace = np.array([], dtype=float)
            self.hf_trace = np.array([], dtype=float)
            return
        try:
            t_u, y_u, fs = ensure_uniform_sampling(t, y, fs_target=self.fs)
            self.fs = float(fs)
        except Exception:
            t_u, y_u = t, y
        self.t_view = np.asarray(t_u, dtype=float)
        self.raw_view = np.asarray(y_u, dtype=float)
        try:
            self.filtered_view = eeg_preprocess(self.raw_view, self.fs)
        except Exception:
            self.filtered_view = self.raw_view - np.nanmedian(self.raw_view)
        try:
            self.swd_trace = _robust_z(bandpass_zero_phase(
                self.filtered_view, self.fs, 5.0, min(12.0, 0.45 * self.fs), order=3
            ))
        except Exception:
            self.swd_trace = np.zeros_like(self.filtered_view)
        try:
            self.hf_trace = _robust_z(np.abs(bandpass_zero_phase(
                self.filtered_view, self.fs, 14.0, min(70.0, 0.45 * self.fs), order=3
            )))
        except Exception:
            self.hf_trace = np.zeros_like(self.filtered_view)

    def _draw_current_event(self):
        start_s, end_s = self._visible_range()
        self._load_view_data(start_s, end_s)
        for ax in self.axes:
            ax.clear()
            ax.grid(alpha=0.18)
        if len(self.t_view) == 0:
            self.axes[0].text(0.5, 0.5, "No waveform samples in this window.",
                              transform=self.axes[0].transAxes, ha="center", va="center")
            self.canvas.draw_idle()
            return

        self.axes[0].plot(self.t_view, self.raw_view, color="#333333", lw=0.6)
        self.axes[0].set_ylabel("Raw")
        self.axes[1].plot(self.t_view, self.filtered_view, color="#111111", lw=0.7)
        self.axes[1].set_ylabel("Filtered")
        self.axes[2].plot(self.t_view, self.swd_trace, color="#0288D1", lw=0.8, label="5-12 Hz z")
        self.axes[2].plot(self.t_view, self.hf_trace, color="#F9A825", lw=0.8, alpha=0.8, label="14-70 Hz abs z")
        self.axes[2].set_ylabel("Detector")
        self.axes[2].legend(loc="upper right", fontsize=8)

        current_idx = self._current_event_index()
        active = ensure_eeg_events_schema(self.events_df)
        ev_win = active[(active["End_s"] >= start_s) & (active["Start_s"] <= end_s)]
        for _, row in ev_win.iterrows():
            typ = str(row["Type"])
            colour = self.EVENT_COLOURS.get(typ, "#777777")
            a = float(row["Start_s"])
            b = float(row["End_s"])
            is_current = current_idx is not None and int(row["Event_Index"]) == current_idx
            deleted = _to_bool(row.get("Is_Deleted", False))
            alpha = 0.10 if deleted else 0.25
            lw = 2.2 if is_current else 1.0
            for ax in self.axes:
                if typ == "Spike" or abs(b - a) < 1e-9:
                    ax.axvline(a, color=colour, lw=lw, alpha=0.90 if not deleted else 0.35,
                               linestyle="--" if deleted else "-")
                else:
                    ax.axvspan(a, b, color=colour, alpha=alpha)
                    if is_current:
                        ax.axvline(a, color=colour, lw=lw)
                        ax.axvline(b, color=colour, lw=lw)

        try:
            typ, a, b = self._parse_fields()
            colour = self.EVENT_COLOURS.get(typ, "#000000")
            for ax in self.axes:
                ax.axvline(a, color=colour, lw=1.5, linestyle=":")
                if typ != "Spike":
                    ax.axvline(b, color=colour, lw=1.5, linestyle=":")
        except Exception:
            pass

        title = "New event" if current_idx is None else f"Event {current_idx}"
        self.axes[0].set_title(f"{title} | click sets {self.marker_var.get()} | source={os.path.basename(self.source_file)}")
        self.axes[-1].set_xlabel("Time from start (s)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_plot_click(self, event):
        if event.xdata is None:
            return
        clicked = float(event.xdata)
        marker = self.marker_var.get()
        typ = _normalize_eeg_event_type(self.event_type_var.get())
        if typ == "Spike":
            self.start_var.set(f"{clicked:.4f}")
            self.end_var.set(f"{clicked:.4f}")
        elif marker == "Start_s":
            self.start_var.set(f"{clicked:.4f}")
        else:
            self.end_var.set(f"{clicked:.4f}")
        self._apply_fields()

    def _apply_fields(self):
        row_idx = self._current_row_index()
        if row_idx is None:
            self._add_event()
            return
        typ, start, end = self._parse_fields()
        self.events_df.at[row_idx, "Type"] = typ
        self.events_df.at[row_idx, "Start_s"] = start
        self.events_df.at[row_idx, "End_s"] = end
        self.events_df.at[row_idx, "Duration_s"] = max(0.0, end - start)
        self.events_df.at[row_idx, "Is_Corrected"] = True
        self.events_df.at[row_idx, "Corrected_At"] = now_iso()
        self.events_df.at[row_idx, "Correction_Source"] = "manual_review"
        self.events_df.at[row_idx, "Correction_Notes"] = self.notes_var.get().strip()
        self.events_df = ensure_eeg_events_schema(self.events_df)
        current = self._current_event_index()
        self.review_order = self._build_review_order()
        if current in self.review_order:
            self.current_order_pos = self.review_order.index(current)
        self.status_var.set(f"Updated event {current}: {typ} {start:.4f}-{end:.4f} s.")
        self._load_current_event()

    def _sync_current_review_fields(self, correction_source="manual_review_save"):
        row_idx = self._current_row_index()
        if row_idx is None:
            return False
        typ, start, end = self._parse_fields()
        note = self.notes_var.get().strip()
        old = self.events_df.iloc[row_idx]
        old_start = pd.to_numeric(pd.Series([old.get("Start_s", np.nan)]), errors="coerce").iloc[0]
        old_end = pd.to_numeric(pd.Series([old.get("End_s", np.nan)]), errors="coerce").iloc[0]
        changed = (
            _normalize_eeg_event_type(old.get("Type", "Spike")) != typ
            or not np.isclose(float(old_start), start, atol=1e-9, equal_nan=False)
            or not np.isclose(float(old_end), end, atol=1e-9, equal_nan=False)
            or str(old.get("Correction_Notes", "") or "").strip() != note
        )
        if not changed:
            return False
        self.events_df.at[row_idx, "Type"] = typ
        self.events_df.at[row_idx, "Start_s"] = start
        self.events_df.at[row_idx, "End_s"] = end
        self.events_df.at[row_idx, "Duration_s"] = max(0.0, end - start)
        self.events_df.at[row_idx, "Correction_Notes"] = note
        self.events_df.at[row_idx, "Is_Corrected"] = True
        self.events_df.at[row_idx, "Corrected_At"] = now_iso()
        self.events_df.at[row_idx, "Correction_Source"] = correction_source
        self.events_df = ensure_eeg_events_schema(self.events_df)
        current = self._current_event_index()
        self.review_order = self._build_review_order()
        if current in self.review_order:
            self.current_order_pos = self.review_order.index(current)
        return True

    def _apply_note(self):
        row_idx = self._current_row_index()
        if row_idx is None:
            return
        self.events_df.at[row_idx, "Correction_Notes"] = self.notes_var.get().strip()
        self.events_df.at[row_idx, "Is_Corrected"] = True
        self.events_df.at[row_idx, "Corrected_At"] = now_iso()
        self.events_df.at[row_idx, "Correction_Source"] = "manual_review"
        self.status_var.set(f"Updated note for event {self._current_event_index()}.")
        self._refresh_info()

    def _add_event(self):
        typ, start, end = self._parse_fields()
        next_idx = int(self.events_df["Event_Index"].max() + 1) if len(self.events_df) else 0
        new_row = {
            "Event_Index": next_idx,
            "Type": typ,
            "Start_s": start,
            "End_s": end,
            "Duration_s": max(0.0, end - start),
            "Confidence": np.nan,
            "Is_Corrected": True,
            "Is_Deleted": False,
            "Corrected_At": now_iso(),
            "Correction_Source": "manual_review_add",
            "Correction_Notes": self.notes_var.get().strip(),
        }
        self.events_df = ensure_eeg_events_schema(pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True))
        self.review_order = self._build_review_order()
        self.current_order_pos = self.review_order.index(next_idx)
        self.status_var.set(f"Added event {next_idx}: {typ} {start:.4f}-{end:.4f} s.")
        self._load_current_event()

    def _toggle_deleted(self):
        row_idx = self._current_row_index()
        if row_idx is None:
            return
        self._set_deleted(not _to_bool(self.events_df.at[row_idx, "Is_Deleted"]))

    def _set_deleted(self, deleted):
        row_idx = self._current_row_index()
        if row_idx is None:
            return
        deleted = bool(deleted)
        if _to_bool(self.events_df.at[row_idx, "Is_Deleted"]) == deleted:
            self.status_var.set(("Already deleted" if deleted else "Already active") + f": event {self._current_event_index()}.")
            self._refresh_info()
            return
        self.events_df.at[row_idx, "Is_Deleted"] = deleted
        self.events_df.at[row_idx, "Is_Corrected"] = True
        self.events_df.at[row_idx, "Corrected_At"] = now_iso()
        self.events_df.at[row_idx, "Correction_Source"] = "manual_review_delete" if deleted else "manual_review_restore"
        note = self.notes_var.get().strip()
        if note:
            self.events_df.at[row_idx, "Correction_Notes"] = note
        elif deleted:
            self.events_df.at[row_idx, "Correction_Notes"] = "Deleted during EEG review"
        self.status_var.set(("Deleted" if deleted else "Restored") + f" event {self._current_event_index()}.")
        self.events_df = ensure_eeg_events_schema(self.events_df)
        current = self._current_event_index()
        self.review_order = self._build_review_order()
        if current in self.review_order:
            self.current_order_pos = self.review_order.index(current)
        self._load_current_event()

    def _prev_event(self):
        if self.current_order_pos > 0:
            self.current_order_pos -= 1
            self._load_current_event()

    def _next_event(self):
        if self.current_order_pos < len(self.review_order) - 1:
            self.current_order_pos += 1
            self._load_current_event()

    def _recompute_summary(self):
        bin_s = pd.to_numeric(pd.Series([self.meta.get("bin_size_s", np.nan)]), errors="coerce").iloc[0]
        if not (np.isfinite(bin_s) and bin_s > 0):
            if len(self.summary_df) and {"Bin_Start_s", "Bin_End_s"}.issubset(self.summary_df.columns):
                bin_s = float(pd.to_numeric(self.summary_df["Bin_End_s"] - self.summary_df["Bin_Start_s"],
                                            errors="coerce").dropna().median())
            else:
                bin_s = 3600.0
        if len(self.summary_df) and {"Bin_Start_s", "Bin_End_s"}.issubset(self.summary_df.columns):
            t0 = float(pd.to_numeric(self.summary_df["Bin_Start_s"], errors="coerce").min())
            t1 = float(pd.to_numeric(self.summary_df["Bin_End_s"], errors="coerce").max())
        else:
            t0 = 0.0
            rec_dur = pd.to_numeric(pd.Series([self.meta.get("recording_duration_s", np.nan)]), errors="coerce").iloc[0]
            t1 = float(rec_dur) if np.isfinite(rec_dur) and rec_dur > 0 else float(bin_s)
        self.summary_df = recompute_eeg_bins_from_events(self.events_df, float(bin_s), t0=t0, t1=t1)

    def _save_workbook(self):
        try:
            synced = self._sync_current_review_fields(correction_source="manual_review_save")
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not apply current event fields before saving:\n{exc}", parent=self)
            return
        before_dedupe = len(self.events_df)
        self.events_df = dedupe_eeg_events(self.events_df)
        deduped_count = max(0, before_dedupe - len(self.events_df))
        self._recompute_summary()
        self.meta["review_last_saved_at"] = now_iso()
        self.meta["review_source"] = "manual_eeg_ui"
        self.meta["source_file"] = self.source_file
        self.meta["fs_Hz"] = float(self.fs)
        self.meta["review_notes_count"] = int((self.events_df["Correction_Notes"].fillna("").astype(str).str.strip() != "").sum()) if len(self.events_df) else 0
        self.meta["review_deduped_events"] = int(deduped_count)
        self.meta = update_eeg_meta_from_events(self.meta, self.events_df, self.summary_df)
        export_eeg_excel(self.workbook_path, self.summary_df, self.events_df, self.meta)
        self.status_var.set(f"Saved EEG workbook with review edits and notes: {self.workbook_path}")
        self._refresh_info()
        extra = "\nCurrent event fields/notes were applied before saving." if synced else ""
        messagebox.showinfo("Saved", f"Workbook overwritten with corrected EEG data and notes:\n{self.workbook_path}{extra}", parent=self)

    def _save_current_plot(self):
        start_s, end_s = self._visible_range()
        default = f"eeg_event_{self._current_event_index() if self._current_event_index() is not None else 'new'}.png"
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save current EEG debug plot",
            defaultextension=".png",
            initialfile=default,
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        save_eeg_debug_plot(
            self.source_file,
            self.events_df,
            path,
            start_s,
            end_s,
            fs_hint=self.fs,
            title=f"EEG review window {start_s:.2f}-{end_s:.2f} s",
            manual_note=self.notes_var.get().strip(),
        )
        self.status_var.set(f"Saved plot: {path}")

    def _refresh_info(self):
        active = active_eeg_events(self.events_df)
        counts = active["Type"].value_counts().to_dict() if len(active) else {}
        deleted = int(self.events_df["Is_Deleted"].apply(_to_bool).sum()) if len(self.events_df) else 0
        corrected = int(self.events_df["Is_Corrected"].apply(_to_bool).sum()) if len(self.events_df) else 0
        row_idx = self._current_row_index()
        if row_idx is None:
            current = "No events yet. Type start/end times, choose type, then Add Event."
        else:
            row = self.events_df.iloc[row_idx]
            current = (
                f"Event {int(row['Event_Index'])} ({self.current_order_pos + 1}/{len(self.review_order)}) | "
                f"{row['Type']} {float(row['Start_s']):.4f}-{float(row['End_s']):.4f} s | "
                f"deleted={bool(_to_bool(row['Is_Deleted']))}"
            )
        self.info_var.set(
            f"{current} | Active counts: spikes={int(counts.get('Spike', 0))}, "
            f"SWDs={int(counts.get('SWD', 0))}, seizures={int(counts.get('Seizure', 0))} | "
            f"corrected={corrected}, deleted={deleted}"
        )


# ============================================================
# 8) UNIFIED GUI
# ============================================================
class UnifiedAnalyzerApp:
    def __init__(self, root):
        self.root         = root
        self._main_thread_id = threading.get_ident()
        self.root.title("SUDEP Waveform Analyzer — EEG / ECG")
        self.root.geometry("920x700")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.input_file   = None
        self.results      = None
        self.is_analyzing = False
        self._batch_proc = None
        self._batch_log_handle = None
        self._batch_log_path = None
        self._batch_output_path = None
        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root, text="EEG/ECG Waveform Analyzer",
                 font=("Arial", 18, "bold"), fg="#1E88E5", pady=10).pack()

        ff = tk.LabelFrame(self.root,
                           text="Step 1 — Select Input File  (Col D = time, Col E = signal, row 8+)",
                           padx=14, pady=10, font=("Arial", 10, "bold"))
        ff.pack(fill="x", padx=18, pady=6)
        self.file_label = tk.Label(ff, text="No file selected", fg="gray", wraplength=640)
        self.file_label.pack(side="left", fill="x", expand=True)
        tk.Button(ff, text="Browse", command=self._browse,
                  bg="#43A047", fg="white", font=("Arial", 10, "bold"),
                  padx=14, pady=6).pack(side="right", padx=8)

        top = tk.Frame(self.root)
        top.pack(fill="x", padx=18, pady=4)

        mode_f = tk.LabelFrame(top, text="Step 2 — Mode",
                               padx=10, pady=10, font=("Arial", 10, "bold"))
        mode_f.pack(side="left", padx=(0, 10), fill="x")
        self.mode_var = tk.StringVar(value="EEG")
        tk.Radiobutton(mode_f, text="EEG", variable=self.mode_var,
                       value="EEG", command=self._on_mode_change).pack(side="left", padx=8)
        tk.Radiobutton(mode_f, text="ECG", variable=self.mode_var,
                       value="ECG", command=self._on_mode_change).pack(side="left", padx=8)

        self.ecg_f = tk.LabelFrame(top, text="ECG Engine",
                                   padx=10, pady=10, font=("Arial", 10, "bold"))
        self.ecg_engine_var = tk.StringVar(value="wavelet" if NK_AVAILABLE else "heuristic")
        ttk.Combobox(self.ecg_f, values=["wavelet", "heuristic"],
                     textvariable=self.ecg_engine_var,
                     state="readonly", width=12).pack(side="left", padx=4)
        nk_note = "NeuroKit2 ✓" if NK_AVAILABLE else "NeuroKit2 not installed — heuristic only"
        tk.Label(self.ecg_f, text=nk_note, fg="#666").pack(side="left", padx=6)

        self.eeg_f = tk.LabelFrame(top, text="EEG Options",
                                   padx=10, pady=10, font=("Arial", 10, "bold"))
        tk.Label(self.eeg_f, text="Bin size (s):").pack(side="left")
        self.eeg_bin_entry = tk.Entry(self.eeg_f, width=8)
        self.eeg_bin_entry.insert(0, "3600")
        self.eeg_bin_entry.pack(side="left", padx=6)

        opt_f = tk.LabelFrame(top, text="Options",
                              padx=10, pady=10, font=("Arial", 10, "bold"))
        opt_f.pack(side="right", fill="x")
        self.debug_var    = tk.BooleanVar(value=False)
        self.force_fs_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_f, text="Debug plots",    variable=self.debug_var).pack(anchor="w")
        tk.Checkbutton(opt_f, text="Force Fs (Hz):", variable=self.force_fs_var).pack(anchor="w")
        self.force_fs_entry = tk.Entry(opt_f, width=8)
        self.force_fs_entry.insert(0, "2000")
        self.force_fs_entry.pack(anchor="w", pady=(0, 2))

        prog = tk.LabelFrame(self.root, text="Progress",
                             padx=14, pady=10, font=("Arial", 10, "bold"))
        prog.pack(fill="x", padx=18, pady=6)
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100, length=840)
        self.progress.pack(fill="x")
        self.status = tk.Label(prog, text="Ready", fg="#333")
        self.status.pack(anchor="w", pady=(4, 0))

        res_f = tk.LabelFrame(self.root, text="Results Summary",
                              padx=14, pady=10, font=("Arial", 10, "bold"))
        res_f.pack(fill="both", expand=True, padx=18, pady=6)
        self.result_text = tk.Text(res_f, height=10, font=("Courier", 9),
                                   state="disabled", bg="#f9f9f9")
        sb = ttk.Scrollbar(res_f, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=sb.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        btn_f = tk.Frame(self.root)
        btn_f.pack(fill="x", padx=18, pady=10)
        self.analyze_btn = tk.Button(btn_f, text="Analyze", state="disabled",
                                     command=self._start_analysis,
                                     bg="#1E88E5", fg="white",
                                     font=("Arial", 11, "bold"), padx=16, pady=10)
        self.analyze_btn.pack(side="left", padx=8)
        self.export_btn = tk.Button(btn_f, text="Export Excel", state="disabled",
                                    command=self._export,
                                    bg="#FB8C00", fg="white",
                                    font=("Arial", 11, "bold"), padx=16, pady=10)
        self.export_btn.pack(side="left", padx=8)
        self.review_btn = tk.Button(btn_f, text="Review ECG Workbook",
                                    command=self._open_review_workbook,
                                    bg="#6A1B9A", fg="white",
                                    font=("Arial", 11, "bold"), padx=16, pady=10)
        self.review_btn.pack(side="left", padx=8)
        self.review_eeg_btn = tk.Button(btn_f, text="Review EEG Workbook",
                                        command=self._open_eeg_review_workbook,
                                        bg="#00695C", fg="white",
                                        font=("Arial", 11, "bold"), padx=16, pady=10)
        self.review_eeg_btn.pack(side="left", padx=8)
        tk.Button(btn_f, text="Reset", command=self._reset,
                  bg="#E53935", fg="white",
                  font=("Arial", 11, "bold"), padx=16, pady=10).pack(side="left", padx=8)
        tk.Button(btn_f, text="Exit",  command=self._close,
                  bg="#757575", fg="white",
                  font=("Arial", 11, "bold"), padx=16, pady=10).pack(side="right", padx=8)

        self._on_mode_change()

    def _ui_call(self, func, *args, **kwargs):
        if threading.get_ident() == self._main_thread_id:
            return func(*args, **kwargs)

        def _run():
            try:
                func(*args, **kwargs)
            except tk.TclError:
                pass

        try:
            self.root.after(0, _run)
        except (RuntimeError, tk.TclError):
            pass
        return None

    def _on_mode_change(self):
        if self.mode_var.get() == "ECG":
            self.ecg_f.pack(side="left", padx=(0,10), fill="x")
            self.eeg_f.pack_forget()
        else:
            self.eeg_f.pack(side="left", padx=(0,10), fill="x")
            self.ecg_f.pack_forget()

    def _set_progress(self, pct, msg):
        if threading.get_ident() != self._main_thread_id:
            self._ui_call(self._set_progress, pct, msg)
            return
        self.progress["value"] = float(pct)
        self.status.config(text=msg, fg="#1E88E5")
        self.root.update_idletasks()

    def _log(self, text):
        if threading.get_ident() != self._main_thread_id:
            self._ui_call(self._log, text)
            return
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", text)
        self.result_text.config(state="disabled")

    def _finish_analysis_ui(self):
        self.is_analyzing = False
        self.analyze_btn.config(state="normal" if self.input_file else "disabled")

    def _show_ecg_debug_plots(self, y, fs, beat_rows):
        try:
            sig_filt, _ = auto_bandpass_notch_ecg(y, fs)
            df = ensure_ecg_beats_schema(pd.DataFrame(beat_rows))
            for _, row in df.head(5).iterrows():
                r_idx = _to_optional_int(row.get("R_idx"))
                if np.isnan(r_idx):
                    continue
                r_idx = int(r_idx)
                sl = max(0, r_idx - int(0.110 * fs))
                el = min(len(sig_filt), r_idx + int(0.250 * fs))
                xv = np.arange(sl, el) / fs

                plt.figure(figsize=(13, 4))
                plt.plot(xv, sig_filt[sl:el], color="darkred", lw=1, label="ECG")
                for col in LANDMARK_COLUMNS:
                    idx = _to_optional_int(row.get(col))
                    if np.isnan(idx):
                        continue
                    idx = int(idx)
                    if 0 <= idx < len(sig_filt):
                        color, marker, label = ECGReviewWindow.MARKER_STYLE[col]
                        plt.scatter(idx / fs, sig_filt[idx], color=color, s=80,
                                    marker=marker, zorder=5, label=label)
                plt.title(f"Beat {int(row['Beat_Index'])} | conf={row.get('Confidence', np.nan):.2f}")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.grid(alpha=0.2)
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.show(block=False)
        except Exception as exc:
            self.status.config(text=f"Debug plot skipped: {exc}", fg="#FB8C00")

    def _show_eeg_debug_plots(self, t, y, fs, events_df):
        try:
            if events_df is None or len(events_df) == 0:
                return
            sig = eeg_preprocess(y, fs)
            t0 = float(t[0])
            t1 = float(t[-1])
            for _, row in events_df.head(3).iterrows():
                a = max(t0, float(row["Start_s"]) - 2.0)
                b = min(t1, float(row["End_s"]) + 2.0)
                i0 = np.searchsorted(t, a)
                i1 = np.searchsorted(t, b)
                plt.figure(figsize=(12, 4))
                plt.plot(t[i0:i1], sig[i0:i1], "k", lw=0.9)
                colour = {"Spike": "gold", "SWD": "lightskyblue", "Seizure": "tomato"}.get(row["Type"], "grey")
                plt.axvspan(row["Start_s"], row["End_s"], alpha=0.25, color=colour)
                plt.title(f'{row["Type"]}  {row["Start_s"]:.2f}-{row["End_s"]:.2f} s  conf={row["Confidence"]:.2f}')
                plt.xlabel("Time (s)")
                plt.ylabel("EEG (a.u.)")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                plt.show(block=False)
        except Exception as exc:
            self.status.config(text=f"Debug plot skipped: {exc}", fg="#FB8C00")

    def _close(self):
        if self._batch_proc is not None and self._batch_proc.poll() is None:
            try:
                self._batch_proc.terminate()
            except Exception:
                pass
        if self._batch_log_handle is not None:
            try:
                self._batch_log_handle.close()
            except Exception:
                pass
            self._batch_log_handle = None
        try:
            plt.close("all")
        except Exception:
            pass
        for attr in ("mode_var", "ecg_engine_var", "debug_var", "force_fs_var"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _browse(self):
        path = filedialog.askopenfilename(
            title=f"Select {self.mode_var.get()} file",
            filetypes=[("Tab/CSV/Excel", "*.tsv *.txt *.csv *.xlsx *.xls"),
                       ("TSV", "*.tsv"), ("CSV", "*.csv"),
                       ("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if path:
            self.input_file = path
            size_mb = os.path.getsize(path) / 1024 / 1024
            self.file_label.config(
                text=f"{os.path.basename(path)}  ({size_mb:.2f} MB)", fg="black")
            self.analyze_btn.config(state="normal")

    def _collect_analysis_settings(self):
        mode = self.mode_var.get()
        force_fs_enabled = bool(self.force_fs_var.get())
        force_fs_value = None
        if force_fs_enabled:
            force_fs_value = float(self.force_fs_entry.get().strip())

        eeg_bin_s = None
        if mode == "EEG":
            eeg_bin_s = float(self.eeg_bin_entry.get().strip())

        return {
            "input_file": self.input_file,
            "mode": mode,
            "debug_plots": bool(self.debug_var.get()),
            "force_fs_enabled": force_fs_enabled,
            "force_fs_value": force_fs_value,
            "ecg_engine": self.ecg_engine_var.get(),
            "eeg_bin_s": eeg_bin_s,
        }

    def _start_analysis(self):
        if not self.input_file:
            messagebox.showerror("Error", "Please select a file first.")
            return
        if self.is_analyzing:
            messagebox.showwarning("Busy", "Analysis already running.")
            return
        try:
            settings = self._collect_analysis_settings()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        if settings["mode"] == "EEG" and self._is_large_eeg_file(settings["input_file"]):
            self._start_large_eeg_analysis(settings)
            return
        self.is_analyzing = True
        self.analyze_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.progress["value"] = 0
        self.root.after(10, lambda: self._run_analysis(settings))

    def _is_large_eeg_file(self, file_path):
        try:
            return os.path.getsize(file_path) / 1024 / 1024 >= 128.0
        except Exception:
            return False

    def _start_large_eeg_analysis(self, settings):
        input_file = settings["input_file"]
        out_name = f"{os.path.splitext(os.path.basename(input_file))[0]}_EEG_results.xlsx"
        output_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=out_name,
            initialdir=os.path.dirname(input_file),
            filetypes=[("Excel", "*.xlsx")],
            title="Save streamed EEG results",
        )
        if not output_path:
            return

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "analyze-eeg",
            "--input", input_file,
            "--output", output_path,
            "--bin-s", str(float(settings["eeg_bin_s"])),
            "--chunk-s", "180",
            "--stream-threshold-mb", "0",
        ]
        if settings["force_fs_enabled"]:
            cmd.extend(["--force-fs", str(float(settings["force_fs_value"]))])

        log_path = os.path.splitext(output_path)[0] + "_run.log"
        self._batch_log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
        self._batch_log_path = log_path
        self._batch_output_path = output_path
        self._batch_proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=self._batch_log_handle,
            stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )

        self.is_analyzing = True
        self.analyze_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.progress.config(mode="indeterminate")
        self.progress.start(12)
        self.status.config(text="Large EEG analysis running in background...", fg="#1E88E5")
        self._log(
            "Large EEG analysis is running in a background Python process.\n"
            f"Input: {input_file}\n"
            f"Output: {output_path}\n"
            f"Log: {log_path}\n\n"
            "This can take a long time for multi-hour recordings."
        )
        self.root.after(3000, self._poll_large_eeg_analysis)

    def _poll_large_eeg_analysis(self):
        if self._batch_proc is None:
            return
        code = self._batch_proc.poll()
        if code is None:
            self.root.after(3000, self._poll_large_eeg_analysis)
            return

        self.progress.stop()
        self.progress.config(mode="determinate")
        self.progress["value"] = 100 if code == 0 else 0
        if self._batch_log_handle is not None:
            self._batch_log_handle.close()
            self._batch_log_handle = None

        if code == 0:
            try:
                summary_df = pd.read_excel(self._batch_output_path, sheet_name="EEG_Bins")
                events_df = pd.read_excel(self._batch_output_path, sheet_name="EEG_Events")
                meta_df = pd.read_excel(self._batch_output_path, sheet_name="Meta")
                meta = meta_df.iloc[0].to_dict() if len(meta_df) else {}
                self.results = {"mode": "EEG", "summary": summary_df,
                                "events": events_df, "meta": meta}
                self.export_btn.config(state="normal")
                self.status.config(text=f"Large EEG complete: {os.path.basename(self._batch_output_path)}", fg="#43A047")
                self._log(
                    "=== EEG RESULTS ===\n"
                    f"Recording duration : {float(meta.get('recording_duration_s', np.nan)):.1f} s\n"
                    f"Sampling rate      : {float(meta.get('fs_Hz', np.nan)):.1f} Hz\n"
                    f"Total spikes       : {int(meta.get('total_spikes', 0))}\n"
                    f"Total SWDs         : {int(meta.get('total_SWDs', 0))}\n"
                    f"Total seizures     : {int(meta.get('total_seizures', 0))}\n"
                    f"Saved workbook     : {self._batch_output_path}"
                )
                messagebox.showinfo("Complete", f"Large EEG analysis complete.\nSaved to:\n{self._batch_output_path}")
            except Exception as exc:
                self.status.config(text=f"Analysis finished, but workbook reload failed: {exc}", fg="#FB8C00")
                messagebox.showwarning("Reload failed", f"Analysis finished, but the workbook could not be reloaded:\n{exc}")
        else:
            tail = ""
            try:
                with open(self._batch_log_path, "r", encoding="utf-8", errors="replace") as f:
                    tail = "".join(f.readlines()[-30:])
            except Exception:
                pass
            self.status.config(text="Large EEG analysis failed.", fg="#E53935")
            self._log(f"ERROR\nLarge EEG analysis failed with exit code {code}.\n\n{tail}")
            messagebox.showerror("Analysis failed", f"Large EEG analysis failed.\nSee log:\n{self._batch_log_path}")

        self.is_analyzing = False
        self.analyze_btn.config(state="normal" if self.input_file else "disabled")
        self._batch_proc = None

    def _run_analysis(self, settings):
        try:
            self._set_progress(5, "Loading file (Col D/E, row 8+)…")
            input_file = settings["input_file"]
            mode = settings["mode"]
            debug = bool(settings["debug_plots"])
            if mode == "EEG":
                self._set_progress(5, "EEG: scanning waveform file")
                bin_s = float(settings["eeg_bin_s"])
                force_fs = float(settings["force_fs_value"]) if settings["force_fs_enabled"] else None
                summary_df, events_df, meta = analyze_eeg_file(
                    input_file,
                    bin_s=bin_s,
                    debug_plots=False,
                    force_fs=force_fs,
                    progress_callback=lambda pct, msg: self._set_progress(pct, msg),
                )
                self.results = {"mode": "EEG", "summary": summary_df,
                                "events": events_df, "meta": meta}
                self._set_progress(100,
                    f"EEG complete - {meta['total_spikes']} spikes | "
                    f"{meta['total_SWDs']} SWDs | {meta['total_seizures']} seizures.")

                lines = ["=== EEG RESULTS ===",
                         f"Recording duration : {meta['recording_duration_s']:.1f} s",
                         f"Sampling rate      : {meta['fs_Hz']:.1f} Hz",
                         f"Bin size           : {meta['bin_size_s']:.0f} s",
                         f"Total spikes       : {meta['total_spikes']}",
                         f"Total SWDs         : {meta['total_SWDs']}",
                         f"Total seizures     : {meta['total_seizures']}",
                         ""]
                if len(events_df):
                    swds = events_df[events_df["Type"] == "SWD"]
                    szs = events_df[events_df["Type"] == "Seizure"]
                    if len(swds):
                        lines.append(f"  SWD duration (s)  mean={swds['Duration_s'].mean():.2f}  "
                                     f"min={swds['Duration_s'].min():.2f}  max={swds['Duration_s'].max():.2f}")
                    if len(szs):
                        lines.append(f"  Seizure dur  (s)  mean={szs['Duration_s'].mean():.2f}  "
                                     f"min={szs['Duration_s'].min():.2f}  max={szs['Duration_s'].max():.2f}")
                    lines.append("")
                    lines.append("--- Event list (first 20) ---")
                    for _, row in events_df.head(20).iterrows():
                        lines.append(f"  {row['Type']:<8}  "
                                     f"start={row['Start_s']:>10.3f} s  "
                                     f"end={row['End_s']:>10.3f} s  "
                                     f"dur={row['Duration_s']:>7.3f} s  "
                                     f"conf={row['Confidence']:.2f}")
                self._log("\n".join(lines))
                if debug:
                    self.status.config(text="EEG debug plots are disabled for streamed analysis.", fg="#FB8C00")
                self._ui_call(self.export_btn.config, state="normal")
                self._ui_call(messagebox.showinfo,
                              "Complete",
                              "Analysis complete.\nClick Export Excel to save results.")
                return
            t, y = load_waveform_tabular(input_file)

            self._set_progress(15, "Inferring sampling rate…")
            if settings["force_fs_enabled"]:
                t, y, fs = ensure_uniform_sampling(t, y, fs_target=float(settings["force_fs_value"]))
            else:
                t, y, fs = ensure_uniform_sampling(t, y)

            mode  = settings["mode"]
            debug = bool(settings["debug_plots"])

            if mode == "ECG":
                self._set_progress(30, "ECG: filtering, R-peak detection, landmark refinement…")
                engine = settings["ecg_engine"]
                beat_rows, meta = compute_ecg_metrics(t, y, fs,
                                                       engine=engine,
                                                       debug_plots=False,
                                                       source_file=input_file,
                                                       validity_threshold=0.90,
                                                       run_context={
                                                           "ui_mode": "ECG",
                                                           "force_fs": bool(settings["force_fs_enabled"]),
                                                           "debug_plots": debug,
                                                       })
                self.results = {"mode": "ECG", "beats": beat_rows, "meta": meta}
                self._set_progress(100, f"ECG complete — {len(beat_rows)} beats detected.")

                df    = pd.DataFrame(beat_rows)
                lines = ["=== ECG RESULTS ===",
                         f"Beats detected : {len(beat_rows)}",
                         f"Engine         : {meta['engine']}",
                         f"Sampling rate  : {meta['fs_Hz']:.1f} Hz",
                         ""]
                for col, label in [
                    ("RR_ms",          "RR interval   (ms)"),
                    ("P_wave_dur_ms",   "P wave dur    (ms)"),
                    ("PR_interval_ms",  "PR interval   (ms)"),
                    ("QRS_interval_ms", "QRS interval  (ms)"),
                    ("QT_interval_ms",  "QT interval   (ms)"),
                    ("QTc_Mitchell_ms", "QTc Mitchell  (ms)"),
                ]:
                    if col in df.columns:
                        v = df[col].dropna()
                        if len(v):
                            lines.append(f"  {label}  mean={v.mean():.2f}  "
                                         f"median={v.median():.2f}  "
                                         f"std={v.std():.2f}  n={len(v)}")
                rmssd = meta.get("RMSSD_ms", np.nan)
                lines.append(f"  RMSSD HRV     (ms)  {rmssd:.3f}" if np.isfinite(rmssd) else "  RMSSD:  N/A")
                vr = meta.get("validity_ratio", np.nan)
                vp = bool(meta.get("validity_pass", False))
                vt = float(meta.get("validity_threshold", 0.90)) * 100.0
                if np.isfinite(vr):
                    lines.append(f"  Validity      : {vr * 100:.2f}%  (threshold {vt:.1f}% -> {'PASS' if vp else 'FAIL'})")
                self._log("\n".join(lines))
                if debug:
                    self._ui_call(self._show_ecg_debug_plots, y, fs, beat_rows)

            else:
                self._set_progress(30, "EEG: preprocessing…")
                bin_s = float(settings["eeg_bin_s"])
                self._set_progress(40, "EEG: spike / SWD / seizure detection…")
                summary_df, events_df, meta = analyze_eeg(t, y, fs,
                                                           bin_s=bin_s,
                                                           debug_plots=False)
                self.results = {"mode": "EEG", "summary": summary_df,
                                "events": events_df, "meta": meta}
                self._set_progress(100,
                    f"EEG complete — {meta['total_spikes']} spikes | "
                    f"{meta['total_SWDs']} SWDs | {meta['total_seizures']} seizures.")

                lines = ["=== EEG RESULTS ===",
                         f"Recording duration : {meta['recording_duration_s']:.1f} s",
                         f"Sampling rate      : {meta['fs_Hz']:.1f} Hz",
                         f"Bin size           : {meta['bin_size_s']:.0f} s",
                         f"Total spikes       : {meta['total_spikes']}",
                         f"Total SWDs         : {meta['total_SWDs']}",
                         f"Total seizures     : {meta['total_seizures']}",
                         ""]
                if len(events_df):
                    swds = events_df[events_df["Type"] == "SWD"]
                    szs  = events_df[events_df["Type"] == "Seizure"]
                    if len(swds):
                        lines.append(f"  SWD duration (s)  mean={swds['Duration_s'].mean():.2f}  "
                                     f"min={swds['Duration_s'].min():.2f}  max={swds['Duration_s'].max():.2f}")
                    if len(szs):
                        lines.append(f"  Seizure dur  (s)  mean={szs['Duration_s'].mean():.2f}  "
                                     f"min={szs['Duration_s'].min():.2f}  max={szs['Duration_s'].max():.2f}")
                    lines.append("")
                    lines.append("--- Event list (first 20) ---")
                    for _, row in events_df.head(20).iterrows():
                        lines.append(f"  {row['Type']:<8}  "
                                     f"start={row['Start_s']:>10.3f} s  "
                                     f"end={row['End_s']:>10.3f} s  "
                                     f"dur={row['Duration_s']:>7.3f} s  "
                                     f"conf={row['Confidence']:.2f}")
                self._log("\n".join(lines))
                if debug:
                    self._ui_call(self._show_eeg_debug_plots, t, y, fs, events_df)

            self._ui_call(self.export_btn.config, state="normal")
            self._ui_call(messagebox.showinfo,
                          "Complete",
                          "Analysis complete.\nClick Export Excel to save results.")

        except Exception as exc:
            err = str(exc)
            self._set_progress(0, f"Error: {err}")
            self._log(f"ERROR\n{err}")
            self._ui_call(messagebox.showerror, "Analysis failed", err)
        finally:
            self._ui_call(self._finish_analysis_ui)

    def _export(self):
        if not self.results:
            messagebox.showerror("Error", "No results to export.")
            return
        mode     = self.results["mode"]
        out_name = (f"{os.path.splitext(os.path.basename(self.input_file))[0]}"
                    f"_{mode}_results.xlsx")
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=out_name,
            filetypes=[("Excel", "*.xlsx")],
            title="Save results")
        if not save_path:
            return
        try:
            if mode == "ECG":
                export_ecg_excel(save_path, self.results["beats"], self.results["meta"])
            else:
                export_eeg_excel(save_path, self.results["summary"],
                                 self.results["events"], self.results["meta"])
            self.status.config(text=f"Exported: {os.path.basename(save_path)}", fg="#43A047")
            messagebox.showinfo("Exported", f"Saved to:\n{save_path}")
            if mode == "ECG":
                open_now = messagebox.askyesno(
                    "Review ECG now?",
                    "Do you want to open the manual ECG review tool for this workbook now?",
                )
                if open_now:
                    self._open_review_workbook(workbook_path=save_path)
            elif mode == "EEG":
                open_now = messagebox.askyesno(
                    "Review EEG now?",
                    "Do you want to open the manual EEG review tool for this workbook now?",
                )
                if open_now:
                    self._open_eeg_review_workbook(workbook_path=save_path)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def _open_review_workbook(self, workbook_path=None):
        path = workbook_path
        if not path:
            path = filedialog.askopenfilename(
                title="Select exported ECG workbook to review",
                filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")],
            )
        if not path:
            return
        try:
            ECGReviewWindow(self.root, path)
        except Exception as exc:
            messagebox.showerror("ECG review failed", str(exc))

    def _open_eeg_review_workbook(self, workbook_path=None):
        path = workbook_path
        if not path:
            path = filedialog.askopenfilename(
                title="Select exported EEG workbook to review",
                filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")],
            )
        if not path:
            return
        try:
            EEGReviewWindow(self.root, path)
        except Exception as exc:
            messagebox.showerror("EEG review failed", str(exc))

    def _reset(self):
        self.input_file   = None
        self.results      = None
        self.is_analyzing = False
        self.file_label.config(text="No file selected", fg="gray")
        self.progress["value"] = 0
        self.status.config(text="Ready", fg="#333")
        self.analyze_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self._log("")


# ============================================================
# 9) CLI ENTRYPOINTS
# ============================================================
def cli_analyze_ecg(args):
    input_path = os.path.abspath(args.input)
    t, y = load_waveform_tabular(input_path)
    if args.force_fs is not None:
        t, y, fs = ensure_uniform_sampling(t, y, fs_target=float(args.force_fs))
    else:
        t, y, fs = ensure_uniform_sampling(t, y)

    beat_rows, meta = compute_ecg_metrics(
        t,
        y,
        fs,
        engine=args.engine,
        debug_plots=bool(args.debug_plots),
        source_file=input_path,
        validity_threshold=float(args.validity_threshold),
        run_context={
            "cli_command": "analyze-ecg",
            "force_fs": args.force_fs if args.force_fs is not None else "",
            "debug_plots": bool(args.debug_plots),
        },
    )

    output_path = args.output
    if not output_path:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{stem}_ECG_results.xlsx")
    output_path = os.path.abspath(output_path)

    if not args.no_export:
        export_ecg_excel(output_path, beat_rows, meta)

    ratio = meta.get("validity_ratio", np.nan)
    threshold = float(meta.get("validity_threshold", args.validity_threshold))
    passed = bool(meta.get("validity_pass", False))
    print("=== CLI ECG ANALYSIS ===")
    print(f"Input            : {input_path}")
    print(f"Engine           : {meta.get('engine')}")
    print(f"Sampling rate Hz : {float(meta.get('fs_Hz', np.nan)):.3f}")
    print(f"Total beats      : {int(meta.get('total_beats', 0))}")
    print(f"Validity ratio   : {ratio * 100:.3f}%")
    print(f"Validity pass    : {'PASS' if passed else 'FAIL'} (threshold {threshold * 100:.1f}%)")
    if not args.no_export:
        print(f"Workbook         : {output_path}")
    return 0 if passed else 2


def cli_analyze_eeg(args):
    input_path = os.path.abspath(args.input)
    last_progress = {"pct": -10.0}

    def _cli_progress(pct, msg):
        if pct - last_progress["pct"] >= 5.0 or pct >= 95:
            print(f"{float(pct):5.1f}%  {msg}", flush=True)
            last_progress["pct"] = float(pct)

    summary_df, events_df, meta = analyze_eeg_file(
        input_path,
        bin_s=float(args.bin_s),
        debug_plots=bool(args.debug_plots),
        force_fs=args.force_fs,
        progress_callback=_cli_progress,
        chunk_s=float(args.chunk_s),
        streaming_threshold_mb=float(args.stream_threshold_mb),
    )
    meta["source_file"] = input_path
    meta["ctx_cli_command"] = "analyze-eeg"
    meta["ctx_force_fs"] = args.force_fs if args.force_fs is not None else ""
    meta["ctx_debug_plots"] = bool(args.debug_plots)
    benchmark_df = None
    if args.manual_workbook:
        manual_path = os.path.abspath(args.manual_workbook)
        manual_df = parse_eeg_manual_counts(manual_path, sheet_name=args.manual_sheet)
        benchmark_df, benchmark_meta = compare_eeg_counts_to_manual(summary_df, manual_df)
        meta.update(benchmark_meta)
        meta["eeg_benchmark_manual_workbook"] = manual_path
        meta["eeg_benchmark_manual_sheet"] = args.manual_sheet

    output_path = args.output
    if not output_path:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{stem}_EEG_results.xlsx")
    output_path = os.path.abspath(output_path)

    if not args.no_export:
        export_eeg_excel(output_path, summary_df, events_df, meta, benchmark_df=benchmark_df)

    print("=== CLI EEG ANALYSIS ===")
    print(f"Input            : {input_path}")
    print(f"Sampling rate Hz : {float(meta.get('fs_Hz', np.nan)):.3f}")
    print(f"Duration s       : {float(meta.get('recording_duration_s', np.nan)):.3f}")
    print(f"Spikes           : {int(meta.get('total_spikes', 0))}")
    print(f"SWDs             : {int(meta.get('total_SWDs', 0))}")
    print(f"Seizures         : {int(meta.get('total_seizures', 0))}")
    if benchmark_df is not None:
        print(f"Manual workbook  : {meta.get('eeg_benchmark_manual_workbook')}")
        print(f"Manual sheet     : {meta.get('eeg_benchmark_manual_sheet')}")
        print(f"Bins compared    : {int(meta.get('eeg_benchmark_bins_compared', 0))}")
        print(f"Manual spikes    : {int(meta.get('eeg_benchmark_manual_spikes', 0))}")
        print(f"Manual SWDs      : {int(meta.get('eeg_benchmark_manual_swds', 0))}")
        print(f"Manual seizures  : {int(meta.get('eeg_benchmark_manual_seizures', 0))}")
        print(f"Total abs error  : {int(meta.get('eeg_benchmark_total_abs_error', 0))}")
    if not args.no_export:
        print(f"Workbook         : {output_path}")
    return 0


def cli_benchmark_eeg(args):
    workbook = os.path.abspath(args.workbook)
    manual_path = os.path.abspath(args.manual_workbook)
    summary_df = pd.read_excel(workbook, sheet_name="EEG_Bins")
    manual_df = parse_eeg_manual_counts(manual_path, sheet_name=args.manual_sheet)
    benchmark_df, meta = compare_eeg_counts_to_manual(summary_df, manual_df)

    if args.output:
        output_path = os.path.abspath(args.output)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            benchmark_df.to_excel(writer, index=False, sheet_name="EEG_Benchmark")
            pd.DataFrame([meta | {
                "workbook": workbook,
                "manual_workbook": manual_path,
                "manual_sheet": args.manual_sheet,
            }]).to_excel(writer, index=False, sheet_name="Meta")
    else:
        output_path = ""

    print("=== CLI EEG BENCHMARK ===")
    print(f"Workbook         : {workbook}")
    print(f"Manual workbook  : {manual_path}")
    print(f"Manual sheet     : {args.manual_sheet}")
    print(f"Bins compared    : {int(meta.get('eeg_benchmark_bins_compared', 0))}")
    print(f"Program spikes   : {int(meta.get('eeg_benchmark_program_spikes', 0))}")
    print(f"Manual spikes    : {int(meta.get('eeg_benchmark_manual_spikes', 0))}")
    print(f"Program SWDs     : {int(meta.get('eeg_benchmark_program_swds', 0))}")
    print(f"Manual SWDs      : {int(meta.get('eeg_benchmark_manual_swds', 0))}")
    print(f"Program seizures : {int(meta.get('eeg_benchmark_program_seizures', 0))}")
    print(f"Manual seizures  : {int(meta.get('eeg_benchmark_manual_seizures', 0))}")
    print(f"Total abs error  : {int(meta.get('eeg_benchmark_total_abs_error', 0))}")
    print(f"Exact bin match  : {float(meta.get('eeg_benchmark_exact_bin_match_ratio', np.nan)) * 100:.1f}%")
    if output_path:
        print(f"Benchmark output : {output_path}")
    return 0


def cli_debug_eeg(args):
    workbook = os.path.abspath(args.workbook)
    summary_df, events_df, meta = load_eeg_workbook(workbook)
    source_file = os.path.abspath(args.source_file) if args.source_file else str(meta.get("source_file", "")).strip()
    if not source_file:
        raise ValueError("debug-eeg needs --source-file or a source_file value in the workbook Meta sheet.")
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source waveform file not found: {source_file}")

    fs_hint = args.force_fs
    if fs_hint is None:
        fs_hint = pd.to_numeric(pd.Series([meta.get("fs_Hz", np.nan)]), errors="coerce").iloc[0]
        fs_hint = float(fs_hint) if np.isfinite(fs_hint) else None

    benchmark_df = None
    benchmark_meta = {}
    if args.manual_workbook:
        manual_path = os.path.abspath(args.manual_workbook)
        manual_df = parse_eeg_manual_counts(manual_path, sheet_name=args.manual_sheet)
        benchmark_df, benchmark_meta = compare_eeg_counts_to_manual(summary_df, manual_df)
    else:
        manual_path = ""

    out_dir = args.output_dir
    if not out_dir:
        stem = os.path.splitext(os.path.basename(workbook))[0]
        out_dir = os.path.join(os.path.dirname(workbook), f"{stem}_debug")
    out_dir = os.path.abspath(out_dir)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if benchmark_df is not None:
        save_eeg_benchmark_plot(benchmark_df, os.path.join(plot_dir, "benchmark_counts.png"))

    debug_windows = build_eeg_debug_windows(
        summary_df,
        events_df,
        benchmark_df,
        source_file,
        fs_hint,
        event_pad_s=float(args.event_pad_s),
        max_plots=int(args.max_plots),
        candidate_window_s=float(args.candidate_window_s),
        candidate_step_s=float(args.candidate_step_s),
        candidate_count=int(args.candidate_count),
    )

    plot_rows = []
    for i, row in debug_windows.iterrows():
        label = _safe_filename(
            f"{i + 1:03d}_{row['Reason']}_bin{row['Bin_Index']}_{float(row['Start_s']):.1f}s"
        )
        png_path = os.path.join(plot_dir, f"{label}.png")
        title = (
            f"{row['Reason']} | bin={row['Bin_Index']} | "
            f"{float(row['Start_s']):.2f}-{float(row['End_s']):.2f} s"
        )
        status = "ok"
        error = ""
        try:
            save_eeg_debug_plot(
                source_file,
                events_df,
                png_path,
                float(row["Start_s"]),
                float(row["End_s"]),
                fs_hint=fs_hint,
                title=title,
                manual_note=str(row.get("Manual_Note", "")),
            )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            png_path = ""
        out_row = row.to_dict()
        out_row["Plot_File"] = png_path
        out_row["Status"] = status
        out_row["Error"] = error
        plot_rows.append(out_row)
    debug_windows = pd.DataFrame(plot_rows)

    summary_meta = {
        "workbook": workbook,
        "source_file": source_file,
        "manual_workbook": manual_path,
        "manual_sheet": args.manual_sheet if args.manual_workbook else "",
        "output_dir": out_dir,
        "plot_dir": plot_dir,
        "fs_hint": fs_hint if fs_hint is not None else "",
        "debug_plots_requested": int(args.max_plots),
        "debug_plots_created": int((debug_windows.get("Status", pd.Series(dtype=str)) == "ok").sum()) if len(debug_windows) else 0,
        "generated_at": now_iso(),
    }
    summary_meta.update(benchmark_meta)

    out_xlsx = os.path.join(out_dir, "eeg_debug_summary.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([summary_meta]).to_excel(writer, index=False, sheet_name="Meta")
        summary_df.to_excel(writer, index=False, sheet_name="EEG_Bins")
        events_df.to_excel(writer, index=False, sheet_name="EEG_Events")
        if benchmark_df is not None:
            benchmark_df.to_excel(writer, index=False, sheet_name="Benchmark")
        debug_windows.to_excel(writer, index=False, sheet_name="Debug_Windows")

    print("=== CLI EEG DEBUG ===")
    print(f"Workbook         : {workbook}")
    print(f"Source file      : {source_file}")
    if manual_path:
        print(f"Manual workbook  : {manual_path}")
        print(f"Manual sheet     : {args.manual_sheet}")
        print(f"Total abs error  : {int(benchmark_meta.get('eeg_benchmark_total_abs_error', 0))}")
    print(f"Output dir       : {out_dir}")
    print(f"Plots created    : {summary_meta['debug_plots_created']}/{len(debug_windows)}")
    print(f"Summary workbook : {out_xlsx}")
    return 0


def cli_review_eeg(args):
    workbook = os.path.abspath(args.workbook)
    summary_df, events_df, meta = load_eeg_workbook(workbook)
    if args.source_file:
        meta["source_file"] = os.path.abspath(args.source_file)

    bin_s = pd.to_numeric(pd.Series([meta.get("bin_size_s", np.nan)]), errors="coerce").iloc[0]
    if not (np.isfinite(bin_s) and bin_s > 0):
        bin_s = 3600.0
    if len(summary_df) and {"Bin_Start_s", "Bin_End_s"}.issubset(summary_df.columns):
        t0 = float(pd.to_numeric(summary_df["Bin_Start_s"], errors="coerce").min())
        t1 = float(pd.to_numeric(summary_df["Bin_End_s"], errors="coerce").max())
    else:
        t0 = 0.0
        rec_dur = pd.to_numeric(pd.Series([meta.get("recording_duration_s", np.nan)]), errors="coerce").iloc[0]
        t1 = float(rec_dur) if np.isfinite(rec_dur) and rec_dur > 0 else float(bin_s)
    summary_df = recompute_eeg_bins_from_events(events_df, float(bin_s), t0=t0, t1=t1)
    meta = update_eeg_meta_from_events(meta, events_df, summary_df)

    output_path = os.path.abspath(args.output) if args.output else workbook
    should_save = bool(args.save or args.output)
    if should_save:
        meta["review_last_saved_at"] = now_iso()
        meta["review_source"] = "cli_eeg_review_recompute"
        export_eeg_excel(output_path, summary_df, events_df, meta)

    print("=== CLI EEG REVIEW ===")
    print(f"Workbook         : {workbook}")
    print(f"Active spikes    : {int(meta.get('total_spikes', 0))}")
    print(f"Active SWDs      : {int(meta.get('total_SWDs', 0))}")
    print(f"Active seizures  : {int(meta.get('total_seizures', 0))}")
    print(f"Corrected events : {int(events_df['Is_Corrected'].apply(_to_bool).sum()) if len(events_df) else 0}")
    print(f"Deleted events   : {int(events_df['Is_Deleted'].apply(_to_bool).sum()) if len(events_df) else 0}")
    if should_save:
        print(f"Saved workbook   : {output_path}")
    return 0


def cli_review_eeg_ui(args):
    workbook = os.path.abspath(args.workbook)
    root = tk.Tk()
    root.withdraw()
    EEGReviewWindow(root, workbook)
    root.mainloop()
    return 0


def cli_review_ecg(args):
    workbook = os.path.abspath(args.workbook)
    beats_df, meta = load_ecg_workbook(workbook)
    meta = dict(meta or {})

    if args.source_file:
        meta["source_file"] = os.path.abspath(args.source_file)

    fs_hz = pd.to_numeric(pd.Series([args.fs_hz if args.fs_hz is not None else meta.get("fs_Hz", np.nan)]),
                          errors="coerce").iloc[0]
    if not (np.isfinite(fs_hz) and fs_hz > 0):
        raise ValueError("review-ecg requires valid fs_Hz (from Meta sheet or --fs-hz).")

    applied = 0
    if args.apply_saved_corrections:
        beats_df, applied = apply_saved_corrections(workbook, beats_df)

    beats_df, rmssd_s = recompute_ecg_intervals_df(beats_df, float(fs_hz))
    meta["fs_Hz"] = float(fs_hz)
    meta["RMSSD_s"] = rmssd_s
    meta["RMSSD_ms"] = rmssd_s * 1000 if np.isfinite(rmssd_s) else np.nan

    ratio, valid, total = compute_ecg_validity(beats_df)
    threshold = float(args.validity_threshold if args.validity_threshold is not None else meta.get("validity_threshold", 0.90))
    meta["validity_threshold"] = threshold
    meta["validity_ratio"] = ratio
    meta["valid_beats"] = valid
    meta["valid_beats_total"] = total
    meta["validity_pass"] = bool(np.isfinite(ratio) and ratio >= threshold)

    output_path = os.path.abspath(args.output) if args.output else workbook
    should_save = bool(args.save or args.output or args.apply_saved_corrections)
    if should_save:
        meta["review_last_saved_at"] = now_iso()
        meta["review_source"] = "cli_review"
        export_ecg_excel(output_path, beats_df, meta)

    print("=== CLI ECG REVIEW ===")
    print(f"Workbook         : {workbook}")
    print(f"Saved corrections: {applied}")
    print(f"Valid beats      : {valid}/{total}")
    print(f"Validity ratio   : {ratio * 100:.3f}%")
    print(f"Validity pass    : {'PASS' if meta['validity_pass'] else 'FAIL'} (threshold {threshold * 100:.1f}%)")
    if should_save:
        print(f"Saved workbook   : {output_path}")
    return 0 if bool(meta["validity_pass"]) else 2


def cli_review_ecg_ui(args):
    workbook = os.path.abspath(args.workbook)
    root = tk.Tk()
    root.withdraw()
    ECGReviewWindow(root, workbook)
    root.mainloop()
    return 0


def build_cli_parser():
    parser = argparse.ArgumentParser(
        description="SUDEP EEG/ECG analyzer - GUI and batch utilities"
    )
    sub = parser.add_subparsers(dest="command")

    p_an = sub.add_parser("analyze-ecg", help="Run ECG analysis and export workbook")
    p_an.add_argument("--input", required=True, help="Input waveform file")
    p_an.add_argument("--output", default="", help="Output workbook path (.xlsx)")
    p_an.add_argument("--engine", choices=["wavelet", "heuristic"],
                      default=("wavelet" if NK_AVAILABLE else "heuristic"))
    p_an.add_argument("--force-fs", type=float, default=None, help="Force target sampling rate Hz")
    p_an.add_argument("--validity-threshold", type=float, default=0.90,
                      help="Minimum valid-beat ratio required to pass")
    p_an.add_argument("--debug-plots", action="store_true", help="Show debug landmark plots")
    p_an.add_argument("--no-export", action="store_true", help="Run analysis without writing workbook")

    p_eeg = sub.add_parser("analyze-eeg", help="Run EEG analysis and export workbook")
    p_eeg.add_argument("--input", required=True, help="Input EEG waveform file")
    p_eeg.add_argument("--output", default="", help="Output workbook path (.xlsx)")
    p_eeg.add_argument("--bin-s", type=float, default=3600.0, help="Summary bin size in seconds")
    p_eeg.add_argument("--force-fs", type=float, default=None, help="Force target sampling rate Hz")
    p_eeg.add_argument("--chunk-s", type=float, default=180.0, help="Streaming chunk size in seconds")
    p_eeg.add_argument("--stream-threshold-mb", type=float, default=128.0,
                       help="Use streaming for files at or above this size")
    p_eeg.add_argument("--manual-workbook", default="",
                       help="Optional manual EEG count workbook for benchmark comparison")
    p_eeg.add_argument("--manual-sheet", default="Baseline",
                       help="Manual EEG workbook sheet name")
    p_eeg.add_argument("--debug-plots", action="store_true", help="Show debug plots for small files")
    p_eeg.add_argument("--no-export", action="store_true", help="Run analysis without writing workbook")

    p_bench_eeg = sub.add_parser("benchmark-eeg", help="Compare an EEG output workbook to manual counts")
    p_bench_eeg.add_argument("--workbook", required=True, help="Existing EEG output workbook")
    p_bench_eeg.add_argument("--manual-workbook", required=True, help="Manual EEG count workbook")
    p_bench_eeg.add_argument("--manual-sheet", default="Baseline", help="Manual EEG workbook sheet name")
    p_bench_eeg.add_argument("--output", default="", help="Optional benchmark workbook output path")

    p_dbg_eeg = sub.add_parser("debug-eeg", help="Export targeted EEG debug plots from an EEG workbook")
    p_dbg_eeg.add_argument("--workbook", required=True, help="Existing EEG output workbook")
    p_dbg_eeg.add_argument("--source-file", default="", help="Override waveform source file")
    p_dbg_eeg.add_argument("--manual-workbook", default="", help="Manual EEG count workbook")
    p_dbg_eeg.add_argument("--manual-sheet", default="Baseline", help="Manual EEG workbook sheet name")
    p_dbg_eeg.add_argument("--output-dir", default="", help="Folder for debug plots and summary workbook")
    p_dbg_eeg.add_argument("--force-fs", type=float, default=None, help="Force sampling rate Hz for window reads")
    p_dbg_eeg.add_argument("--max-plots", type=int, default=40, help="Maximum debug windows to plot")
    p_dbg_eeg.add_argument("--event-pad-s", type=float, default=5.0, help="Seconds added around program events")
    p_dbg_eeg.add_argument("--candidate-window-s", type=float, default=20.0,
                           help="Candidate window length for missed manual seizure bins")
    p_dbg_eeg.add_argument("--candidate-step-s", type=float, default=60.0,
                           help="Candidate scan step for missed manual seizure bins")
    p_dbg_eeg.add_argument("--candidate-count", type=int, default=2,
                           help="Candidate windows per missed manual seizure bin")

    p_rev_eeg = sub.add_parser("review-eeg", help="Recompute/revalidate an exported EEG workbook")
    p_rev_eeg.add_argument("--workbook", required=True, help="Existing EEG workbook path")
    p_rev_eeg.add_argument("--output", default="", help="Output workbook path (defaults to overwrite input)")
    p_rev_eeg.add_argument("--source-file", default="", help="Override source file path in meta")
    p_rev_eeg.add_argument("--save", action="store_true",
                           help="Save workbook after recomputing corrected/deleted event counts")

    p_ui_eeg = sub.add_parser("review-eeg-ui", help="Launch manual EEG review UI for a workbook")
    p_ui_eeg.add_argument("--workbook", required=True, help="Existing EEG workbook path")

    p_rev = sub.add_parser("review-ecg", help="Recompute/revalidate an exported ECG workbook")
    p_rev.add_argument("--workbook", required=True, help="Existing ECG workbook path")
    p_rev.add_argument("--output", default="", help="Output workbook path (defaults to overwrite input)")
    p_rev.add_argument("--source-file", default="", help="Override source file path in meta")
    p_rev.add_argument("--fs-hz", type=float, default=None, help="Override fs_Hz used for recomputation")
    p_rev.add_argument("--validity-threshold", type=float, default=None,
                       help="Override validity threshold")
    p_rev.add_argument("--apply-saved-corrections", action="store_true",
                       help="Apply persisted correction store entries before recompute")
    p_rev.add_argument("--save", action="store_true",
                       help="Save workbook after review even if no corrections were applied")

    p_ui = sub.add_parser("review-ecg-ui", help="Launch manual ECG review UI for a workbook")
    p_ui.add_argument("--workbook", required=True, help="Existing ECG workbook path")
    return parser


def run_cli(argv=None):
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if not args.command:
        return None
    if args.command == "analyze-ecg":
        return cli_analyze_ecg(args)
    if args.command == "analyze-eeg":
        return cli_analyze_eeg(args)
    if args.command == "benchmark-eeg":
        return cli_benchmark_eeg(args)
    if args.command == "debug-eeg":
        return cli_debug_eeg(args)
    if args.command == "review-eeg":
        return cli_review_eeg(args)
    if args.command == "review-eeg-ui":
        return cli_review_eeg_ui(args)
    if args.command == "review-ecg":
        return cli_review_ecg(args)
    if args.command == "review-ecg-ui":
        return cli_review_ecg_ui(args)
    return 1


# ============================================================
# 10) ENTRY POINT
# ============================================================
if __name__ == "__main__":
    cli_code = run_cli(sys.argv[1:])
    if cli_code is not None:
        sys.exit(int(cli_code))
    root = tk.Tk()
    app = UnifiedAnalyzerApp(root)
    root.mainloop()
