# NeuroLab

Python desktop software for reviewing brain (EEG) and heart (ECG) recordings. The project brings waveform inspection, assisted analysis, manual corrections and spreadsheet reporting into one research workflow.

## Project highlights

- Interactive waveform browsing and keyboard-driven review.
- ECG landmark editing and interval measurements.
- EEG event review with manual annotations.
- Chunked processing for long recordings and Excel reporting.

## Engineering focus

Translating research workflows into usable desktop tools: signal processing, event-driven interfaces, reliable correction state, file handling and regression testing.

**Full application stack:** Python, Tkinter, NumPy, SciPy, pandas and Matplotlib.

## Code sample

This portfolio edition contains a small, standalone scrolling utility from the application and synthetic unit tests. It demonstrates local event routing, cross-platform wheel handling and cleanup without removing unrelated widget bindings.

It is **not the complete analyzer**. Research data and analysis internals are not distributed here. No application installer is included.

With Python 3.10+ and pytest installed, run from this folder:

```sh
python -m pytest tests -q
```

Research software, not a clinical diagnostic tool. Detection and measurement accuracy remain under evaluation; this code sample makes no clinical or accuracy claims.
