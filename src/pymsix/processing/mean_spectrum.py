from typing import Tuple
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from scipy.stats import norm
from scipy.signal import fftconvolve

from pymsix.params.options import MeanSpectrumOptions


def _smooth_centroid_constant_da(
    spike: np.ndarray,
    binning_p: float,
    tolerance_da: float,
) -> np.ndarray:
    """
    Centroid smoothing with a constant Gaussian width in Da.
    Equivalent to smearing each peak individually, but done as a
    single convolution (much faster).
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
    Centroid smoothing where Gaussian width is derived from instrument
    accuracy in ppm: sigma_da(m/z) = m/z * mass_accuracy_ppm * 1e-6.
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

    Parameters
    ----------
    imzml_path : str
        Path to the imzML file.
    options : MeanSpectrumOptions
        Object containing all parameters for spectrum aggregation and smoothing.

    Returns
    -------
    axis_mz : np.ndarray
        m/z values of the mean spectrum (full axis).
    mean_intensity : np.ndarray
        Mean intensity values on that axis.
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
