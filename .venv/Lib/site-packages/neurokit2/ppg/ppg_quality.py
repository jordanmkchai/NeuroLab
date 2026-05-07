import numpy as np

from ..signal.signal_interpolate import signal_interpolate
from ..signal.signal_power import signal_power
from ..signal.signal_quality import signal_quality
from .ppg_peaks import ppg_peaks


def ppg_quality(
    ppg_cleaned,
    peaks=None,
    sampling_rate=1000,
    method="templatematch",
    window_sec=None,
    overlap_sec=None,
    no_bins=16,
    ppg_raw=None,
):
    """**PPG Signal Quality Assessment**

    Assess the quality of the PPG Signal using various methods:

    * The ``"templatematch"`` method (loosely based on Orphanidou et al., 2015) computes a continuous
      index of quality of the PPG signal, by calculating the correlation coefficient between each
      individual pulse wave and an average (template) pulse wave shape. This index is therefore
      relative: 1 corresponds to pulse waves that are closest to the average pulse wave shape (i.e.
      correlate exactly with it) and 0 corresponds to there being no correlation with the average
      pulse wave shape.

    * The ``"dissimilarity"`` method (loosely based on Sabeti et al., 2019) computes a continuous index
      of quality of the PPG signal, by calculating the level of dissimilarity between each individual
      pulse wave and an average (template) pulse wave shape (after they are normalised). A value of
      zero indicates no dissimilarity (i.e. equivalent pulse wave shapes), whereas values above or below
      indicate increasing dissimilarity. The original method used dynamic time-warping to align the pulse
      waves prior to calculating the level of dsimilarity, whereas this implementation does not currently
      include this step.

    * The ``"ho2025"`` method (Ho et al., 2025) assesses PPG quality on a beat-by-beat basis by predicting
      whether each interbeat-interval (IBI) is accurate. To do so, beats are detected using a primary beat detector,
      and each IBI is predicted to be accurate only if a secondary beat detector detects beats
      in the same positions (within a tolerance). In this implementation, all signal samples within an
      IBI are rated as high quality (1) if that IBI is predicted to be accurate, or low
      quality (0) if that IBI is predicted to be inaccurate. Ho et al. proposed this approach for the ECG, and here
      it has been applied to the PPG. The general approach was derived by Ho et al from the previously proposed bSQI
      approach.

    * The ``"skewness"`` method (based on Selveraj, 2011 and Elgendi, 2016) computes the skewness of the PPG signal.
      The skewness is a measure of the asymmetry of the probability distribution of the signal's amplitude values.
      In Elgendi (2016), higher quality signals were generally found to have higher skewness values.

    * The ``"kurtosis"`` method (based on Selveraj, 2011 and Elgendi, 2016) computes the kurtosis of the PPG signal.
      The kurtosis is a measure of the "tailedness" of the probability distribution of the signal's amplitude values.
      In Elgendi (2016), higher quality signals were generally found to have higher kurtosis values.

    * The ``"entropy"`` method (based on Selvaraj et al., 2011, and inspired by Elgendi, 2016) computes the entropy of the
      signal in moving windows. The entropy is a measure of the randomness in the signal's amplitude values.

    * The ``"perfusion"`` method (based on Elgendi, 2016) computes the perfusion index of the PPG signal.
      The perfusion index is the ratio of the amplitude of the pulsatile (AC) component of the PPG to its baseline (DC)
      amplitude, expressed as a percentage. It is calculated over moving windows (default: 3 seconds window, 2 seconds
      overlap). Requires raw PPG signal.

    * The ``"relative_power"`` method (based on Elgendi, 2016) computes the relative power of the PPG signal.
      The relative power is the ratio of the power in the 1-2.25 Hz band to the power in the 0-8 Hz band, giving a value
      between 0 and 1. It is calculated over moving windows (default: 60 seconds window, 30 seconds overlap). Requires
      raw PPG signal.

    Parameters
    ----------
    ppg_cleaned : Union[list, np.array, pd.Series]
        The cleaned PPG signal in the form of a vector of values.
    peaks : tuple or list
        The list of PPG pulse wave peak samples returned by ``ppg_peaks()``. If None, peaks is computed from
        the signal input.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    method : str
        The method for computing PPG signal quality, can be ``"templatematch"`` (default), ``"dissimilarity"``,
        ``"ho2025"``, ``"skewness"``, ``"kurtosis"``, ``"entropy"``, ``"perfusion"``, or ``"relative_power"``.
    window_sec : float, optional
        Window length in seconds for windowed metrics. Default is 3 seconds for ``"skewness"``, ``"kurtosis"``,
        ``"perfusion"``, and 60 seconds for ``"relative_power"``.
    overlap_sec : float, optional
        Overlap between windows in seconds for windowed metrics. Default is 2 seconds for ``"skewness"``, ``"kurtosis"``,
        ``"perfusion"``, and 30 seconds for ``"relative_power"``.
    no_bins : int, optional
        Number of bins for ``"entropy"`` calculation (default: 16).
    ppg_raw : Union[list, np.array, pd.Series], optional
        The raw PPG signal: used for the ``"perfusion"`` and ``"relative_power"`` methods.

    Returns
    -------
    quality : array
        Vector containing the quality index ranging from 0 to 1 for ``"templatematch"`` method,
        or an unbounded value (where 0 indicates high quality) for ``"dissimilarity"`` method,
        or zeros and ones (where 1 indicates high quality) for ``"ho2025"`` method,
        or an unbounded value for ``"skewness"``, ``"kurtosis"``, or ``"entropy"``,
        or a value between 0 and 100% for ``"perfusion"`` method (100% being high quality),
        or a value between 0 and 1 for ``"relative_power"`` method.

    See Also
    --------
    signal_quality, ppg_clean

    References
    ----------
    * Orphanidou, C. et al. (2015). "Signal-quality indices for the electrocardiogram and photoplethysmogram:
      derivation and applications to wireless monitoring". IEEE Journal of Biomedical and Health Informatics, 19(3), 832-8.
    * Sabeti E. et al. (2019). Signal quality measure for pulsatile physiological signals using morphological features:
      Applications in reliability measure for pulse oximetry. Informatics in Medicine Unlocked, 16, 100222.
    * Ho, S.Y.S et al. (2025). "Accurate RR-interval extraction from single-lead, telehealth electrocardiogram signals.
      medRxiv, 2025.03.10.25323655. https://doi.org/10.1101/2025.03.10.25323655
    * Elgendi, M. et al. (2016). "Optimal signal quality index for photoplethysmogram signals".
      Bioengineering, 3(4), 1–15. doi: https://doi.org/10.3390/bioengineering3040021
    * Selvaraj, N. et al. (2011). "Statistical approach for the detection of motion/noise artifacts in Photoplethysmogram".
      Proc IEEE EMBC; pp. 4972–4975.

    Examples
    --------
    * **Example 1:** 'templatematch' method

    .. ipython:: python

      import neurokit2 as nk

      sampling_rate = 100
      ppg = nk.ppg_simulate(duration=30, sampling_rate=sampling_rate, heart_rate=80)
      ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)
      quality = nk.ppg_quality(ppg_cleaned, sampling_rate=sampling_rate, method="templatematch")

      @savefig p_ppg_quality.png scale=100%
      nk.signal_plot([ppg_cleaned, quality], standardize=True)
      @suppress
      plt.close()

    * **Example 2:** 'skewness' method

    .. ipython:: python

      import neurokit2 as nk

      sampling_rate = 100
      ppg = nk.ppg_simulate(duration=30, sampling_rate=sampling_rate, heart_rate=80)
      ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)
      quality = nk.ppg_quality(ppg_cleaned, sampling_rate=sampling_rate, method="skewness", window_sec=3, overlap_sec=2)

      nk.signal_plot([ppg_cleaned, quality], standardize=True)
      plt.close()

    """

    method = method.lower()  # remove capitalised letters

    # Sanitise method name
    if method in ["templatematch", "orphanidou2015"]:
        method = "templatematch"
    elif method in ["dissimilarity", "sabeti2019"]:
        method = "dissimilarity"
    elif method in ["ho2025", "ho", "ibi", "ici"]:
        method = "ici"
    elif method in ["skewness"]:
        method = "skewness"
    elif method in ["kurtosis"]:
        method = "kurtosis"
    elif method in ["entropy"]:
        method = "entropy"
    elif method in ["perfusion"]:
        method = "perfusion"
    elif method in ["relative_power"]:
        method = "relative_power"
    else:
        raise ValueError(
            f"Method '{method}' not recognised. Please use 'templatematch', 'dissimilarity', 'ici', 'skewness',"
            " 'kurtosis', 'entropy', 'perfusion', or 'relative_power'."
        )

    # Check that raw PPG signal has been provided if required
    if (method in ["perfusion", "relative_power"]) and (ppg_raw is None):
        raise ValueError(f"ppg_raw must be provided for the {method} method.")

    # Set default window values based on method if not provided
    if method == "relative_power":
        if window_sec is None:
            window_sec = 60
        if overlap_sec is None:
            overlap_sec = 30
    elif method in ["skewness", "kurtosis", "perfusion", "entropy"]:
        if window_sec is None:
            window_sec = 3
        if overlap_sec is None:
            overlap_sec = 2

    # Detect PPG peaks (if not done already, and if required for the specified quality-assessment method)
    if peaks is None:
        if method in ["templatematch", "dissimilarity", "ici"]:
            _, peaks = ppg_peaks(ppg_cleaned, sampling_rate=sampling_rate)
            peaks = peaks["PPG_Peaks"]

    # Run 'templatematch' and 'dissimilarity' methods
    if method in ["templatematch", "dissimilarity"]:
        quality = signal_quality(
            ppg_cleaned,
            cycle_inds=peaks,
            signal_type="ppg",
            sampling_rate=sampling_rate,
            method=method,
        )
    elif method == "ici":
        # Assess quality using Ho2025 method (IBI accuracy prediction)
        quality = signal_quality(
            ppg_cleaned,
            signal_type="ppg",
            primary_detector="charlton",
            secondary_detector="elgendi",
            sampling_rate=sampling_rate,
            method="ici",
        )
    elif method == "skewness":
        quality = signal_quality(
            ppg_cleaned,
            sampling_rate=sampling_rate,
            signal_type="ppg",
            method="skewness",
            window_sec=window_sec,
            overlap_sec=overlap_sec,
        )
    elif method == "kurtosis":
        quality = signal_quality(
            ppg_cleaned,
            sampling_rate=sampling_rate,
            signal_type="ppg",
            method="kurtosis",
            window_sec=window_sec,
            overlap_sec=overlap_sec,
        )
    elif method == "entropy":
        quality = signal_quality(
            ppg_cleaned,
            sampling_rate=sampling_rate,
            signal_type="ppg",
            method="entropy",
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            no_bins=no_bins,
        )
    elif method == "perfusion":
        quality = _windowed_metric(
            ppg_cleaned,
            ppg_raw,
            _perfusion_func,
            sampling_rate=sampling_rate,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
        )
    elif method == "relative_power":
        quality = _windowed_metric(
            ppg_cleaned,
            ppg_raw,
            _rel_power_func,
            sampling_rate=sampling_rate,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
        )

    return quality


# Common window calculation for perfusion and relative_power
def _windowed_metric(clean_signal, raw_signal, func, sampling_rate, window_sec, overlap_sec, **kwargs):
    # check that clean_signal and raw_signal have the same length
    if len(clean_signal) != len(raw_signal):
        raise ValueError("Clean and raw signals must have the same length.")

    # setup windows
    window_size = int(window_sec * sampling_rate)
    step_size = int((window_sec - overlap_sec) * sampling_rate)
    n_samples = len(clean_signal)
    if n_samples < window_size:
        raise ValueError(f"Signal length ({n_samples} samples) is shorter than window size ({window_size} samples).")

    # calculate metric for each window
    metric_values = []
    for start in range(0, n_samples - window_size + 1, step_size):
        if func == _rel_power_func:
            metric_values.append(func(raw_signal[start : start + window_size], sampling_rate=sampling_rate))
        else:
            metric_values.append(
                func(
                    raw_signal[start : start + window_size],
                    clean_signal[start : start + window_size],
                )
            )

    # interpolate window to provide a continuous output (same length as input signal)
    window_centers = np.arange(0, n_samples - window_size + 1, step_size) + window_size // 2
    output = signal_interpolate(
        x_values=window_centers,
        y_values=metric_values,
        x_new=np.arange(n_samples),
        method="previous",
    )
    if np.isnan(output[0]):
        output[: window_centers[0]] = metric_values[0]

    return output


def _perfusion_func(raw_ppg, cleaned_ppg):
    """
    Compute perfusion index for PPG signal quality in moving windows.

    Parameters
    ----------
    raw_ppg : array-like
        Raw PPG signal
    cleaned_ppg : array-like
        Cleaned PPG signal

    Returns
    -------
    perfusion : np.ndarray
        Perfusion index values for each window, interpolated to signal length.
    """

    if raw_ppg is None:
        raise ValueError("raw_ppg must be provided for perfusion calculation.")

    # calculate baseline
    x_bar = np.mean(raw_ppg)

    # avoid dividing by zero
    if x_bar == 0:
        return 0

    # calculate perfusion
    perfusion = ((np.max(cleaned_ppg) - np.min(cleaned_ppg)) / abs(x_bar)) * 100

    return perfusion


# Define relative power function for a window
def _rel_power_func(raw_ppg, sampling_rate):
    """
    Compute relative power for PPG signal quality.

    Parameters
    ----------
    raw_ppg : array-like
        Raw PPG signal.
    sampling_rate : int
        Sampling frequency (Hz).

    Returns
    -------
    relative_power : float
        Relative power: power in 1-2.25 Hz band divided by power in 0-8 Hz band.
    """

    # Compute power in both bands
    power = signal_power(
        raw_ppg,
        frequency_band=[(1, 2.25), (0, 8)],
        sampling_rate=sampling_rate,
        continuous=False,
        normalize=False,
    )
    power_1_2_25 = power.loc[0, "Hz_1_2.25"]
    power_0_8 = power.loc[0, "Hz_0_8"]

    # Avoid division by zero
    if power_0_8 == 0:
        return 0.0

    # calculate relative power
    rel_power = power_1_2_25 / power_0_8

    return rel_power
