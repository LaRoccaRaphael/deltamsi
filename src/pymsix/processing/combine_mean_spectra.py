"""
Spectral Combination Utilities
==============================

This module provides tools for merging multiple mean mass spectra into a single 
representative "mean-of-mean" spectrum. 

Because different MSI acquisitions may have slightly different m/z calibrations 
ou ranges, this module uses linear interpolation to project all spectra onto a 
common, high-resolution binned grid before averaging.


"""

import numpy as np
from typing import Iterable, Tuple, List
from scipy.interpolate import interp1d

from pymsix.params.options import GlobalMeanSpectrumOptions

# Define Spectrum type
Spectrum = Tuple[np.ndarray, np.ndarray]


def combine_mean_spectra(
    spectra: Iterable[Spectrum],
    options: GlobalMeanSpectrumOptions,
) -> Spectrum:
    """
    Combine several mean spectra into a single mean-of-mean spectrum.

    This function aligns multiple spectra by interpolating them onto a common 
    m/z axis. It supports global normalization (TIC) to ensure that samples 
    with different absolute intensities contribute equally to the final 
    consensus spectrum.

    Parameters
    ----------
    spectra : Iterable[Tuple[np.ndarray, np.ndarray]]
        An iterable of spectra, where each spectrum is a tuple of (m/z, intensity) 
        1D arrays. Arrays must have the same length within each tuple.
    options : GlobalMeanSpectrumOptions
        A configuration object containing:
        
        * **binning_p**: The step size (resolution) of the common m/z grid in Da.
        * **use_intersection**: If True, only the m/z range common to all 
          spectra is kept. If False, the union of all ranges is used.
        * **tic_normalize**: Whether to normalize each spectrum to unit 
          total intensity on the common grid before averaging.
        * **compress_axis**: If True, removes m/z bins where the final 
          mean intensity is exactly zero.

    Returns
    -------
    mz_axis : np.ndarray
        The generated common m/z axis.
    mean_intensity : np.ndarray
        The averaged intensity values corresponding to `mz_axis`.

    Raises
    ------
    ValueError
        If the input spectra are empty, have inconsistent array shapes, 
        or if `use_intersection=True` is requested for non-overlapping spectra.
    RuntimeError
        If no valid spectra remain after the processing steps.

    Notes
    -----
    The interpolation is performed using a linear method via ``scipy.interpolate.interp1d``. 
    Values outside the original m/z range of a spectrum are filled with zeros.
    
    [Image showing the difference between intersection and union m/z range strategies]

    This process is typically used to generate a reference spectrum for 
    global peak picking in multi-sample MSI studies.

    Examples
    --------
    >>> from pymsix.params.options import GlobalMeanSpectrumOptions
    >>> from pymsix.processing.spectra import combine_mean_spectra
    >>> 
    >>> # Configure averaging options
    >>> opts = GlobalMeanSpectrumOptions(binning_p=0.0001, tic_normalize=True)
    >>> 
    >>> # spectra = [(mz1, int1), (mz2, int2), ...]
    >>> mz_avg, int_avg = combine_mean_spectra(spectra, options=opts)
    >>> print(f"Combined spectrum has {len(mz_avg)} bins.")
    """

    options.validate()

    # Materialize and ensure float arrays
    spec_list: List[Spectrum] = []
    for mzs, ints in spectra:
        mzs = np.asarray(mzs, dtype=float)
        ints = np.asarray(ints, dtype=float)
        if mzs.ndim != 1 or ints.ndim != 1:
            raise ValueError("Each mzs and ints must be 1D arrays.")
        if mzs.size != ints.size:
            raise ValueError("mzs and ints must have the same length.")
        spec_list.append((mzs, ints))

    if not spec_list:
        raise ValueError("No spectra provided.")

    # Infer global m/z range
    mins = [mzs.min() for mzs, _ in spec_list]
    maxs = [mzs.max() for mzs, _ in spec_list]

    if options.use_intersection:
        min_mz = max(mins)
        max_mz = min(maxs)
        if min_mz >= max_mz:
            raise ValueError(
                "No overlapping m/z range between spectra; "
                "set use_intersection=False to use the union."
            )
    else:
        min_mz = min(mins)
        max_mz = max(maxs)

    # Check if the range is valid for numpy arange
    if min_mz >= max_mz:
        # Should not happen if use_intersection is True and passed the check,
        # but protects against floating point issues or edge cases with False.
        return np.array([]), np.array([])

    mz_axis = np.arange(min_mz, max_mz, options.binning_p, dtype=float)
    combined = np.zeros_like(mz_axis)
    n_spec = 0

    for mzs, ints in spec_list:
        # Ensure sorted m/z (just in case)
        order = np.argsort(mzs)
        mzs_sorted = mzs[order]
        ints_sorted = ints[order]

        # Interpolate onto common axis; outside range -> 0
        f = interp1d(
            mzs_sorted,
            ints_sorted,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )
        ints_interp = f(mz_axis)

        # Clean negatives from interpolation (if any)
        ints_interp[ints_interp < 0] = 0.0

        # TIC-normalize on the common grid if requested
        if options.tic_normalize:
            tic = ints_interp.sum()
            if tic > 0:
                ints_interp /= tic

        combined += ints_interp
        n_spec += 1

    if n_spec == 0:
        raise RuntimeError("No valid spectra after processing.")

    # Mean of mean spectra
    mean_intensity = combined / n_spec

    if options.compress_axis:
        nz = mean_intensity > 0
        return mz_axis[nz], mean_intensity[nz]

    return mz_axis, mean_intensity
