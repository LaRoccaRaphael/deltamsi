"""
Mean Spectrum Generation Utilities
==================================

This module provides efficient methods for aggregating thousands of individual 
pixel spectra into a single representative mean spectrum. 

It handles:
1. **Binning**: Projecting high-resolution m/z values onto a fixed grid.
2. **Normalization**: TIC-normalizing each pixel before aggregation.
3. **Centroid Smoothing**: Converting discrete centroid peaks into Gaussian 
   shapes to better estimate the underlying profile.
"""

from typing import Tuple
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from scipy.stats import norm
from scipy.signal import fftconvolve

from deltamsi.params.options import MeanSpectrumOptions


def _smooth_centroid_constant_da(
    spike: np.ndarray,
    binning_p: float,
    tolerance_da: float,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a centroided spectrum using a fixed width in Da.

    This function uses FFT convolution for high performance. The standard 
    deviation of the Gaussian kernel is approximately ``tolerance_da / 4``.

    Parameters
    ----------
    spike : np.ndarray
        1D array containing intensities at binned positions.
    binning_p : float
        The width of each bin in Dalton (precision).
    tolerance_da : float
        The full width in Dalton used to define the smoothing kernel.

    Returns
    -------
    np.ndarray
        The smoothed spectrum intensities.
    """
    if tolerance_da <= 0:
        return spike.copy()

    tol_bins = max(1, int(round(tolerance_da / binning_p)))
    offsets = np.arange(-tol_bins, tol_bins + 1, dtype=int)

    # Same shape logic as original: sigma ~ tol_bins / 4
    sigma_bins = tol_bins / 4.0
    kernel = norm(loc=0.0, scale=sigma_bins).pdf(offsets)

    # No explicit normalization: behaviour matches your original script
    smoothed = fftconvolve(spike, kernel, mode="same")
    return smoothed


def _smooth_centroid_ppm(
    spike: np.ndarray,
    axis_mz: np.ndarray,
    binning_p: float,
    mass_accuracy_ppm: float,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """
    Apply Gaussian smoothing where the width scales with m/z (PPM).

    The standard deviation (sigma) is calculated for each peak individually as:
    $$\sigma_{Da}(m/z) = m/z \times \text{accuracy\_ppm} \times 10^{-6}$$

    Parameters
    ----------
    spike : np.ndarray
        1D array of binned intensities.
    axis_mz : np.ndarray
        1D array of m/z values corresponding to each bin center.
    binning_p : float
        Bin size in Dalton.
    mass_accuracy_ppm : float
        Instrumental accuracy in parts-per-million.
    n_sigma : float, default 3.0
        The number of standard deviations to include in the Gaussian tail.

    Returns
    -------
    np.ndarray
        The smoothed intensities.
    """
    out = np.zeros_like(spike, dtype=np.float64)

    # Only bins that actually have signal can contribute
    nz = np.nonzero(spike)[0]
    if nz.size == 0:
        return out

    mz_nz = axis_mz[nz]
    intens_nz = spike[nz]

    # 1σ in Da and in bins for each bin center
    sigma_da = mz_nz * mass_accuracy_ppm * 1e-6
    sigma_bins = sigma_da / binning_p

    # If everything is almost delta-like, just return original
    if np.all(sigma_bins <= 0):
        return spike.copy()

    max_radius = int(np.ceil(n_sigma * sigma_bins.max()))
    if max_radius == 0:
        return spike.copy()

    offsets = np.arange(-max_radius, max_radius + 1, dtype=int)  # (n_offsets,)

    # weights: shape (n_nz, n_offsets)
    # ratios[i, j] = offsets[j] / sigma_bins[i]
    ratios = offsets[None, :] / sigma_bins[:, None]
    weights = np.exp(-0.5 * ratios**2)

    # Optional: drop tiny weights to save work
    weights[weights < 1e-6] = 0.0

    values = intens_nz[:, None] * weights  # (n_nz, n_offsets)
    idx2d = nz[:, None] + offsets[None, :]  # (n_nz, n_offsets)

    valid = (idx2d >= 0) & (idx2d < out.size) & (values != 0.0)

    np.add.at(out, idx2d[valid], values[valid])

    return out


def compute_mean_spectrum(
    imzml_path: str,
    options: MeanSpectrumOptions,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a mean spectrum from an imzML file using predefined options.

    This function iterates through all pixels in an MSI dataset, normalizes the 
    intensities, and accumulates them into a binned m/z axis. If the input 
    data is in centroid mode, it applies Gaussian smoothing to reconstruct 
    a pseudo-profile spectrum.

    Parameters
    ----------
    imzml_path : str
        Path to the ``.imzML`` file on disk.
    options : MeanSpectrumOptions
        Configuration object containing parameters for:
        
        * **min_mz / max_mz**: The spectral range to consider.
        * **binning_p**: The width of each m/z bin (in Daltons).
        * **mode**: Either "profile" or "centroid".
        * **mass_accuracy_ppm**: Used for smoothing where the Gaussian 
          width increases with m/z (standard for TOF and Orbitrap).
        * **tolerance_da**: Used for smoothing with a fixed Gaussian width.

    Returns
    -------
    axis_mz : np.ndarray
        The m/z values of the generated bins (filtered for non-zero intensities).
    mean_intensity : np.ndarray
        The mean intensity values corresponding to each bin.

    Raises
    ------
    FileNotFoundError
        If the imzML file is missing or corrupted.
    ValueError
        If the options fail validation (e.g., negative bin width).

    Notes
    -----
    **Smoothing Algorithms**
    
    * **Constant Da**: Applies a uniform smoothing kernel across the whole spectrum 
      via FFT convolution.
    * **PPM-based**: Adjusts the smoothing kernel width $\sigma$ dynamically as: 
      $$\sigma(m/z) = \frac{m/z \times \text{ppm}}{10^6}$$
      This accurately reflects the decreasing mass resolution at higher m/z 
      values in most mass analyzers.

    

    Examples
    --------
    >>> from deltamsi.params.options import MeanSpectrumOptions
    >>> from deltamsi.processing.mean_spectrum import compute_mean_spectrum
    >>> 
    >>> opts = MeanSpectrumOptions(min_mz=100, max_mz=1000, binning_p=0.001)
    >>> mz, intensities = compute_mean_spectrum("sample.imzML", options=opts)
    >>> print(f"Mean spectrum computed over {len(mz)} bins.")
    """
    options.validate()

    # 1) Build axis and spike accumulator
    axis_mz = np.arange(
        options.min_mz, options.max_mz, options.binning_p, dtype=np.float64
    )
    spike = np.zeros(axis_mz.size, dtype=np.float64)

    try:
        p = ImzMLParser(imzml_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not parse imzML file at '{imzml_path}'. Error: {e}"
        ) from e

    # 2) Loop over spectra: TIC-normalize + simple binning (profile mode)
    for idx, _coord in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        if mzs is None or intensities is None or intensities.size == 0:
            continue

        mask = (mzs >= options.min_mz) & (mzs < options.max_mz)
        if not np.any(mask):
            continue

        mzs = mzs[mask]
        intensities = intensities[mask]

        tic = intensities.sum()
        if tic <= 0:
            continue

        intens_norm = intensities / tic
        intens_norm[intens_norm < 0] = 0.0

        indices = np.rint((mzs - options.min_mz) / options.binning_p).astype(int)
        valid = (indices >= 0) & (indices < spike.size)
        if not np.any(valid):
            continue

        np.add.at(spike, indices[valid], intens_norm[valid])

    if options.mode == "profile":
        mean_spec = spike
    else:
        if options.mass_accuracy_ppm is not None:
            mean_spec = _smooth_centroid_ppm(
                spike,
                axis_mz,
                options.binning_p,
                options.mass_accuracy_ppm,
                n_sigma=options.n_sigma,
            )
        else:
            mean_spec = _smooth_centroid_constant_da(
                spike, options.binning_p, tolerance_da=options.tolerance_da or 0.0
            )
    axis_mz = axis_mz[mean_spec >0]
    mean_spec = mean_spec[mean_spec >0]
    return axis_mz, mean_spec
