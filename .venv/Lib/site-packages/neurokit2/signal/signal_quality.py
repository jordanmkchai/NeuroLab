import numpy as np
from scipy.stats import entropy as scipy_entropy

from ..epochs import epochs_to_df
from ..signal import signal_cyclesegment, signal_interpolate


def signal_quality(
    signal,
    sampling_rate=1000,
    cycle_inds=None,
    signal_type=None,
    method="templatematch",
    primary_detector=None,
    secondary_detector=None,
    tolerance_window_ms=50,
    window_sec=3,
    overlap_sec=2,
    no_bins=16,
):
    """**Assess quality of signal using various metrics**

    Assess the quality of a quasi-periodic signal (e.g. PPG, ECG or RSP) using the specified method. You can pass an
    unfiltered signal as an input, but typically a filtered signal (e.g. cleaned using ``ppg_clean()``, ``ecg_clean()`` or
    ``rsp_clean()``) will result in more reliable results. The following methods are available:

    * The ``"templatematch"`` method (loosely based on Orphanidou et al., 2015) computes a continuous
      index of quality of the PPG or ECG signal, by calculating the correlation coefficient between each
      individual cycle's (i.e. beat's) morphology and an average (template) cycle morphology. This index is therefore
      relative: 1 corresponds to a signal where each individual cycle's morphology is closest to the average cycle morphology
      (i.e. correlate exactly with it) and 0 corresponds to there being no correlation with the average cycle morphology.

    * The ``"dissimilarity"`` method (loosely based on Sabeti et al., 2019) computes a continuous index
      of quality of the PPG or ECG signal, by calculating the level of dissimilarity between each individual
      cycle's (i.e. beat's) morphology and an average (template) cycle morphology (after they are normalised). A value of
      zero indicates no dissimilarity (i.e. equivalent cycle morphologies), whereas values above or below
      indicate increasing dissimilarity. The original method used dynamic time-warping to align the pulse
      waves prior to calculating the level of dsimilarity, whereas this implementation does not currently
      include this step.

    * The ``"ici"`` method (based on Ho et al., 2025) assesses signal quality on a cycle-by-cycle (e.g. beat-by-beat)
      basis by predicting whether each intercycle-interval (ICI) (e.g. interbeat-intervals, IBIs) is accurate.
      To do so, cycles (e.g. beats) are detected using a primary detector, and each ICI is predicted to be accurate
      only if a secondary detector detects cycles in the same positions (within a tolerance). In this implementation,
      all signal samples within an ICI are rated as high quality (1) if that ICI is predicted to be accurate, or low
      quality (0) if that ICI is predicted to be inaccurate. This approach was derived from the previously
      proposed bSQI approach. The resulting quality signal is 1-padded, with all samples before the first cycle
      and after the last cycle set to 1.

    * The ``"skewness"`` method (based on Selveraj, 2011 and Elgendi, 2016) computes the skewness of the signal in moving
      windows. The skewness is a measure of the asymmetry of the probability distribution of the signal's amplitude values.
      For the PPG, higher quality signals were generally found to have higher skewness values (Elgendi, 2016).
      The window length and overlap can be adjusted using the ``window_sec`` and ``overlap_sec`` parameters.

    * The ``"kurtosis"`` method (based on Selveraj, 2011 and Elgendi, 2016) computes the kurtosis of the signal in moving
      windows. The kurtosis is a measure of the "tailedness" of the probability distribution of the signal's amplitude values.
      For the PPG, higher quality signals are generally found to have higher kurtosis values (Elgendi, 2016).
      The window length and overlap can be adjusted using the ``window_sec`` and ``overlap_sec`` parameters.

    * The ``"entropy"`` method (based on Selvaraj et al., 2011, and inspired by Elgendi, 2016) computes the entropy of the
      signal in moving windows. The entropy is a measure of the randomness in the signal's amplitude values.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The cleaned signal, such as that returned by ``ppg_clean()`` or ``ecg_clean()``.
    sampling_rate : int
        The sampling frequency of ``signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    cycle_inds : tuple or list
        The list of cycle samples (e.g. beat or breath samples, such as PPG or ECG peaks returned by ``ppg_peaks()``
        or ``ecg_peaks()``, or RSP peaks returned by ``rsp_peaks()``).
    signal_type : str
        The signal type (e.g. 'ppg', 'ecg', or 'rsp').
    method : str
        The processing pipeline to apply. Can be one of ``"templatematch"``, ``"dissimilarity"``, ``"ici"``,
        ``"skewness"``, ``"kurtosis"``, or ``"entropy"``. The default is ``"templatematch"``.
    primary_detector : str
        The name of the primary cycle (i.e. beat or breath) detector (e.g. the defaults are ``"unsw"`` for the ECG, and
        ``"charlton"`` for the PPG).
    secondary_detector : str
        The name of the secondary cycle detector (e.g. the defaults are ``"neurokit"`` for the ECG, and ``"elgendi"``
        for the PPG).
    tolerance_window_ms : int
        The tolerance window size (in milliseconds) for use with the "ici" method when assessing agreement between
        primary and secondary cycle detectors.
    window_sec : float, optional
        Window length in seconds for windowed metrics (default: 3). Used for the ``"skewness"`` and ``"kurtosis"`` methods.
    overlap_sec : float, optional
        Overlap between windows in seconds for windowed metrics (default: 2). Used for the ``"skewness"`` and ``"kurtosis"``
        methods.
    no_bins : int
        Number of bins for ``"entropy"`` calculation (default: 16).
    **kwargs
        Additional keyword arguments, usually specific for each method.

    Returns
    -------
    quality : array
        Vector containing the quality index ranging from 0 to 1 for ``"templatematch"`` method,
        or an unbounded value (where 0 indicates high quality) for ``"dissimilarity"`` method,
        or zeros and ones (where 1 indicates high quality) for ``"ici"`` method,
        or an unbounded value for ``"skewness"``, ``"kurtosis"``, or ``"entropy"`` methods.

    See Also
    --------
    ppg_quality, ecg_quality, rsp_quality

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
    * **Example 1:** Using ICI method to assess PPG signal quality

    .. ipython:: python

      import neurokit2 as nk

      sampling_rate = 100
      ppg = nk.ppg_simulate(
          duration=30, sampling_rate=sampling_rate, heart_rate=70, motion_amplitude=1, burst_number=10, random_state=12
      )
      quality = nk.ppg_quality(ppg, sampling_rate=sampling_rate, method="ici")
      nk.signal_plot([ppg, quality], standardize=True)
      plt.close()

    * **Example 2:** Using template-matching method to assess ECG signal quality

    .. ipython:: python

      import neurokit2 as nk

      sampling_rate = 100
      duration = 20
      ecg = nk.ecg_simulate(
          duration=duration, sampling_rate=sampling_rate, heart_rate=70, noise=0.5
      )
      quality = nk.ecg_quality(ecg, sampling_rate=sampling_rate, method="templatematch")
      nk.signal_plot([ecg, quality], standardize=True)
      plt.close()

    * **Example 3:** Using template-matching method to assess RSP signal quality

    .. ipython:: python

      import neurokit2 as nk

      sampling_rate = 50
      duration = 30
      rsp = nk.rsp_simulate(duration=30, sampling_rate=sampling_rate, method="breathmetrics")
      rsp_cleaned = nk.rsp_clean(rsp, sampling_rate=sampling_rate, method="charlton")
      quality = nk.rsp_quality(rsp_cleaned, sampling_rate=sampling_rate, method="templatematch")
      nk.signal_plot([rsp_cleaned, quality], standardize=True)
      plt.close()

    * **Example 4:** Using skewness method to assess PPG signal quality

    .. ipython:: python

      import neurokit2 as nk

      sampling_rate = 100
      ppg = nk.ppg_simulate(duration=30, sampling_rate=sampling_rate, heart_rate=80)
      ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)
      quality = nk.signal_quality(ppg_cleaned, sampling_rate=sampling_rate, signal_type="ppg", method="skewness")
      nk.signal_plot([ppg_cleaned, quality], standardize=True)
      plt.close()

    """

    # Check inputs
    if signal_type is None:
        raise ValueError("`signal_type` must be specified (e.g. 'ppg', 'ecg', or 'rsp').")

    # Standardize inputs first so all checks are case-insensitive
    signal_type = signal_type.lower()
    method = method.lower()

    if method == "ici" and signal_type not in ["ecg", "ppg"]:
        raise ValueError("`method` 'ici' is only supported for 'ppg' and 'ecg' signal types.")
    if method not in ["ici", "skewness", "kurtosis", "entropy"] and (cycle_inds is None or len(cycle_inds) == 0):
        raise ValueError("`templatematch` and `dissimilarity` require at least one detected peak.")

    # Run selected quality assessment method
    if method in ["templatematch"]:  # Based on the approach in Orphanidou et al. (2015) and Charlton et al. (2021)
        quality = _quality_templatematch(
            signal,
            cycle_inds=cycle_inds,
            signal_type=signal_type,
            sampling_rate=sampling_rate,
        )
    elif method in ["dissimilarity"]:  # Based on the approach in Sabeti et al. (2019)
        quality = _quality_dissimilarity(
            signal,
            cycle_inds=cycle_inds,
            signal_type=signal_type,
            sampling_rate=sampling_rate,
        )
    elif method in ["ici", "ho2025"]:  # Based on the approach in Ho et al. (2025)
        quality = _quality_ici(
            signal,
            signal_type=signal_type,
            primary_detector=primary_detector,
            secondary_detector=secondary_detector,
            sampling_rate=sampling_rate,
            tolerance_window_ms=tolerance_window_ms,
        )
    elif method == "skewness":
        quality = _quality_windowed_metric(
            signal,
            sampling_rate=sampling_rate,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            metric="moment",
            moment=3,
        )
    elif method == "kurtosis":
        quality = _quality_windowed_metric(
            signal,
            sampling_rate=sampling_rate,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            metric="moment",
            moment=4,
        )
    elif method == "entropy":
        quality = _quality_windowed_metric(
            signal,
            sampling_rate=sampling_rate,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            metric="entropy",
            no_bins=no_bins,
        )
    else:
        raise ValueError(
            f"The `{method}` method does not exist in signal_quality. "
            "Please choose one of: `templatematch`, `dissimilarity`, `ici`, `skewness`, `kurtosis`, or `entropy`."
        )

    return quality


# =============================================================================
# Calculate template morphology
# =============================================================================
def _calc_template_morph(signal, cycle_inds, signal_type, sampling_rate=1000):
    # Segment to get individual cycle morphologies
    cycles, average_cycle_rate = signal_cyclesegment(signal, cycle_inds, sampling_rate=sampling_rate)

    # convert these to dataframe
    ind_morph = epochs_to_df(cycles).pivot(index="Label", columns="Time", values="Signal")
    ind_morph.index = ind_morph.index.astype(int)
    ind_morph = ind_morph.sort_index()

    # Filter Nans
    valid_cycles_mask = ~ind_morph.isnull().any(axis=1)
    ind_morph = ind_morph[valid_cycles_mask]
    cycle_inds = np.array(cycle_inds)[valid_cycles_mask.values]

    # Find template pulse wave as the average pulse wave shape
    templ_pw = ind_morph.mean()

    return templ_pw, ind_morph, cycle_inds


# =============================================================================
# Quality assessment using template-matching method
# =============================================================================
def _quality_templatematch(signal, cycle_inds=None, signal_type="ppg", sampling_rate=1000):
    # Obtain individual cycle morphologies and template cycle morphology
    templ_morph, ind_morph, cycle_inds = _calc_template_morph(
        signal,
        cycle_inds=cycle_inds,
        signal_type=signal_type,
        sampling_rate=sampling_rate,
    )

    # Find correlation coefficients (CCs) between individual cycle morphologies and the template
    cc = np.zeros(len(cycle_inds) - 1)
    for cycle_no in range(0, len(cycle_inds) - 1):
        temp = np.corrcoef(ind_morph.iloc[cycle_no], templ_morph)
        cc[cycle_no] = temp[0, 1]

    # Interpolate cycle-by-cycle CCs
    quality = signal_interpolate(cycle_inds[0:-1], cc, x_new=np.arange(len(signal)), method="previous")

    return quality


# =============================================================================
# Disimilarity measure method
# =============================================================================
def _norm_sum_one(pw):
    # ensure all values are positive
    pw = pw - pw.min() + 1

    # normalise pulse wave to sum to one
    pw = pw / np.sum(pw)

    return pw


def _calc_dis(pw1, pw2):
    # following the methodology in https://doi.org/10.1016/j.imu.2019.100222 (Sec. 3.1.2.5)

    # convert to numpy arrays
    pw1 = np.array(pw1)
    pw2 = np.array(pw2)

    # normalise to sum to one
    pw1 = _norm_sum_one(pw1)
    pw2 = _norm_sum_one(pw2)

    # ignore any elements which are zero because log(0) is -inf
    rel_els = (pw1 != 0) & (pw2 != 0)

    # calculate dissimilarity measure (using pw2 as the template)
    dis = np.sum(pw2[rel_els] * np.log(pw2[rel_els] / pw1[rel_els]))

    return dis


# =============================================================================
# Quality assessment using dissimilarity method
# =============================================================================
def _quality_dissimilarity(signal, cycle_inds=None, signal_type="ppg", sampling_rate=1000):
    # Obtain individual cycle morphologies and template cycle morphology
    templ_morph, ind_morph, cycle_inds = _calc_template_morph(
        signal,
        cycle_inds=cycle_inds,
        signal_type=signal_type,
        sampling_rate=sampling_rate,
    )

    # Find individual dissimilarity measures
    dis = np.zeros(len(cycle_inds) - 1)
    for cycle_no in range(0, len(cycle_inds) - 1):
        dis[cycle_no] = _calc_dis(ind_morph.iloc[cycle_no], templ_morph)

    # Interpolate cycle-by-cycle dis's
    quality = signal_interpolate(cycle_inds[0:-1], dis, x_new=np.arange(len(signal)), method="previous")

    return quality


# =============================================================================
# Quality assessment using ICI method
# =============================================================================
def _quality_ici(
    signal,
    signal_type,
    primary_detector,
    secondary_detector,
    sampling_rate,
    tolerance_window_ms=50,
):
    # Specify default cycle (e.g. beat) detectors
    if primary_detector is None:
        if signal_type == "ecg":
            primary_detector = "unsw"
        elif signal_type == "ppg":
            primary_detector = "charlton"
        else:
            raise Exception("default ICI quality assessment detectors only available for ECG and PPG signals.")
    if secondary_detector is None:
        if signal_type == "ecg":
            secondary_detector = "neurokit"
        elif signal_type == "ppg":
            secondary_detector = "elgendi"
        else:
            raise Exception("default ICI quality assessment detectors only available for ECG and PPG signals.")

    # Sanitize inputs
    signal_type = signal_type.lower()  # remove capitalised letters
    primary_detector = primary_detector.lower()  # remove capitalised letters
    secondary_detector = secondary_detector.lower()  # remove capitalised letters
    signal = np.asarray(signal)

    # Specify constants - tolerance_window_ms is tolerance window size, in milliseconds
    tolerance_samps = int((tolerance_window_ms / 1000) * sampling_rate)

    # Detect cycles using each cycle detector in turn
    cycles_primary = _signal_cycles(
        signal,
        signal_type=signal_type,
        cycle_detector=primary_detector,
        sampling_rate=sampling_rate,
    )
    cycles_secondary = _signal_cycles(
        signal,
        signal_type=signal_type,
        cycle_detector=secondary_detector,
        sampling_rate=sampling_rate,
    )

    # Filter closely spaced cycles to keep only the highest amplitude cycle
    cycles_primary = _filter_close_cycles(cycles_primary, signal, tolerance_samps)
    cycles_secondary = _filter_close_cycles(cycles_secondary, signal, tolerance_samps)

    # Build quality array
    quality = np.zeros(len(signal), dtype=int)
    for i in range(len(cycles_primary) - 1):
        # identify start and end of current ICI
        start = cycles_primary[i]
        end = cycles_primary[i + 1]

        # check if secondary detector has detected cycles within tolerance at both the start and end of the ICI
        match_start = any(abs(start - s) <= tolerance_samps for s in cycles_secondary)
        match_end = any(abs(end - s) <= tolerance_samps for s in cycles_secondary)

        # check whether the secondary detector has detected any additional cycles within the ICI
        cycle_within_IBI = any((start + tolerance_samps) < s < (end - tolerance_samps) for s in cycles_secondary)

        # if they have both detected cycles within the tolerance, and there are not additional cycles within the ICI
        if match_start and match_end and not cycle_within_IBI:
            # then assign quality = 1 (high quality) to the period within the ICI
            quality[start:end] = 1
            # and if this is the first or last ICI, then extend the quality assignment to the start or end of the signal
            if i == 0:
                quality[0:start] = 1
            if i == len(cycles_primary) - 2:
                quality[end:] = 1

    return quality


def _filter_close_cycles(cycles, signal, tolerance_samps):
    if len(cycles) == 0:
        return []

    filtered = [cycles[0]]
    for i in range(1, len(cycles)):
        if cycles[i] - filtered[-1] > tolerance_samps:
            filtered.append(cycles[i])
        # Keep the higher amplitude peak
        elif signal[cycles[i]] > signal[filtered[-1]]:
            filtered[-1] = cycles[i]

    return filtered


def _signal_cycles(signal, signal_type, cycle_detector, sampling_rate):
    # Import peak-detection functions (placed here to avoid circular imports)
    from ..ecg import ecg_peaks
    from ..ppg import ppg_peaks

    if signal_type == "ecg":
        # Detect beats in ECG signal
        signals, info = ecg_peaks(
            signal,
            sampling_rate=sampling_rate,
            method=cycle_detector,
        )

        # Extract cycles
        cycles = info["ECG_R_Peaks"]

    elif signal_type == "ppg":
        # Detect beats in PPG signal
        signals, info = ppg_peaks(
            signal,
            sampling_rate=sampling_rate,
            method=cycle_detector,
        )

        # Extract cycles
        cycles = info["PPG_Peaks"]

    return cycles


# =============================================================================
# Generic code for calculating windowed metric (skewness, kurtosis, entropy)
# =============================================================================
def _quality_windowed_metric(
    signal,
    sampling_rate=1000,
    window_sec=3,
    overlap_sec=2,
    metric="moment",
    moment=3,
    no_bins=16,
):
    """
    Compute a windowed metric (moment-based or entropy) for signal quality.

    Parameters
    ----------
    signal : array-like
        Cleaned signal.
    sampling_rate : int
        Sampling frequency (Hz).
    window_sec : float
        Window length in seconds (default: 3).
    overlap_sec : float
        Overlap between windows in seconds (default: 2).
    metric : str
        Metric to compute ("moment" for skewness/kurtosis, "entropy" for entropy).
    moment : int
        The moment to compute (used if metric="moment"; 3 for skewness, 4 for kurtosis).
    no_bins : int
        Number of bins for entropy calculation (used if metric="entropy"; default: 16).

    Returns
    -------
    metric_values : np.ndarray
        Metric values for each window, interpolated to signal length.
    """
    window_size = int(window_sec * sampling_rate)
    step_size = int((window_sec - overlap_sec) * sampling_rate)
    n_samples = len(signal)
    metric_values = []

    for start in range(0, n_samples - window_size + 1, step_size):
        window = signal[start : start + window_size]
        if metric == "moment":
            mu_x = np.mean(window)
            omega = np.std(window)
            if omega == 0:
                val = 0
            else:
                val = np.mean(((window - mu_x) / omega) ** moment)
            metric_values.append(val)
        elif metric == "entropy":
            # Bin the data
            bin_counts, _ = np.histogram(window, bins=no_bins, range=(np.min(window), np.max(window)))
            bin_p = bin_counts / window_size
            # Use scipy's entropy function (base e), normalize by log(1/no_bins) for consistency with previous code
            mask = bin_p > 0
            ent = scipy_entropy(bin_p[mask], base=np.e)
            metric_values.append(ent)

    # interpolate
    window_centers = np.arange(0, n_samples - window_size + 1, step_size) + window_size // 2
    output = signal_interpolate(
        x_values=window_centers,
        y_values=metric_values,
        x_new=np.arange(n_samples),
        method="previous",
    )

    # add interpolation for start
    if np.isnan(output[0]):
        output[: window_centers[0]] = metric_values[0]

    return output
