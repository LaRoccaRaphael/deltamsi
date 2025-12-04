import numpy as np
from typing import Iterable, Tuple, List
from scipy.interpolate import interp1d

from msix.params.options import GlobalMeanSpectrumOptions

# Define Spectrum type
Spectrum = Tuple[np.ndarray, np.ndarray]


def combine_mean_spectra(
    spectra: Iterable[Spectrum],
    options: GlobalMeanSpectrumOptions,
) -> Spectrum:
    """
    Combine several mean spectra into a single mean-of-mean spectrum.

    Parameters
    ----------
    spectra
        Iterable of (mzs, intensities) arrays.
        Each mzs and intensities must be 1D and same length.
    binning_p : float
        Bin width in Da for the common m/z axis (default: 1e-4).
    use_intersection : bool
        If True, the common axis is built on the overlapping m/z range:
            [max(min(mzs)), min(max(mzs))]
        If False, it uses the union:
            [min(min(mzs)), max(max(mzs))] and extrapolated regions are zero.
    tic_normalize : bool
        If True, TIC-normalize each mean spectrum on the common axis
        (divide by sum of intensities) before averaging.
    compress_axis : bool
        If True, drop bins where the final mean intensity is zero.

    Returns
    -------
    mz_axis : np.ndarray
        Common m/z axis.
    mean_intensity : np.ndarray
        Mean of mean spectra on that axis.
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
