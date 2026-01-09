"""
Kendrick Mass Defect (KMD) Utilities
====================================

This module provides tools for transforming standard m/z values into Kendrick 
coordinates. It supports custom chemical bases, formula parsing, and 
integration with AnnData objects.

The transformation follows the standard formula:
    Kendrick Mass (KM) = observed_mass * (nominal_base_mass / exact_base_mass)
    Kendrick Mass Defect (KMD) = integer_mass - exact_mass (depending on mode)
"""

from __future__ import annotations

import logging
import re
from typing import Literal, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)

# Monoisotopic & nominal (integer) masses for base parsing
MONO = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "S": 31.9720711744,
    "P": 30.97376199842,
    "F": 18.99840316273,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.9044719,
}
NOMI = {
    "H": 1,
    "C": 12,
    "N": 14,
    "O": 16,
    "S": 32,
    "P": 31,
    "F": 19,
    "Cl": 35,
    "Br": 79,
    "I": 127,
}

# Capture element symbol and optional count separately to avoid mixing the digits
# into the element name (e.g. "CH2" -> ("C", ""), ("H", "2"))
_FORM_RX = re.compile(r"([A-Z][a-z]?)(\d*)")


def _parse_formula_to_mass(formula: str) -> Tuple[float, float]:
    """
    Parse a simple empirical formula into its exact and nominal masses.

    Parameters
    ----------
    formula : str
        Chemical formula string (e.g., "CH2", "C18H34O2"). Dots and mid-dots 
        are ignored.

    Returns
    -------
    exact_mass : float
        The calculated monoisotopic exact mass.
    nominal_mass : float
        The calculated nominal (integer) mass.

    Raises
    ------
    ValueError
        If the formula is empty or contains unknown chemical elements.
    """

    s = str(formula).replace("·", "").replace(".", "").strip()
    if not s:
        raise ValueError("Empty base formula.")
    m_exact = 0.0
    m_nom = 0.0
    for el, num in _FORM_RX.findall(s):
        n = int(num) if num else 1
        if el not in MONO or el not in NOMI:
            raise ValueError(f"Unknown element in base: {el}")
        m_exact += MONO[el] * n
        m_nom += NOMI[el] * n
    return float(m_exact), float(m_nom)


def _base_masses(base: Union[str, float, Tuple[float, float]]) -> Tuple[float, float]:
    """
    Normalize a Kendrick base description into a numerical mass tuple.

    Parameters
    ----------
    base : Union[str, float, Tuple[float, float]]
        Description of the Kendrick base:
        - str: An empirical formula (e.g., "CH2").
        - float: The exact mass (nominal mass will be inferred by rounding).
        - tuple: Explicit (exact, nominal) masses.

    Returns
    -------
    Tuple[float, float]
        A tuple of (exact_mass, nominal_mass).

    Raises
    ------
    ValueError
        If the input type or format is unsupported.
    """

    if isinstance(base, tuple) and len(base) == 2:
        return (float(base[0]), float(base[1]))
    if isinstance(base, (int, float)):
        m_exact = float(base)
        return (m_exact, float(round(m_exact)))
    if isinstance(base, str):
        return _parse_formula_to_mass(base)
    raise ValueError("Unsupported base type. Use 'CH2', a float, or (exact, nominal).")


def kendrick_coords(
    masses: Sequence[float],
    base: Union[str, float, Tuple[float, float]],
    kmd_mode: Literal["fraction", "defect"] = "fraction",
) -> dict[str, np.ndarray]:
    """
    Compute Kendrick mass and defect coordinates for a set of masses.

    This is a standalone utility that calculates rescaled masses based on 
    a specified chemical unit (the "base").

    Parameters
    ----------
    masses : Sequence[float]
        Array or list of m/z values to transform.
    base : str, float, or Tuple[float, float]
        The repeating unit used for rescaling. 
        - If ``str``: An empirical formula like "CH2" or "H2O".
        - If ``float``: The exact mass of the unit.
        - If ``tuple``: Explicit (exact_mass, nominal_mass).
    kmd_mode : {"fraction", "defect"}, default "fraction"
        - ``"fraction"``: $KM - \lfloor KM \rfloor$ (range [0, 1]).
        - ``"defect"``: $round(KM) - KM$ (range [-0.5, 0.5]).

    Returns
    -------
    dict of str to np.ndarray
        A dictionary containing:
        - ``"KM"``: The rescaled Kendrick Mass.
        - ``"KMD_fraction"``: The fractional mass defect.
        - ``"KMD_defect"``: The centered mass defect.
        - ``"scale"``: The scaling factor used ($nominal / exact$).
        - ``"base_exact"``: The exact mass of the base unit.

    Examples
    --------
    >>> coords = kendrick_coords([300.21, 314.23], base="CH2")
    >>> print(coords["KM"])
    [299.89... 313.91...]
    """

    masses_arr = np.asarray(masses, dtype=float)
    base_exact, base_nom = _base_masses(base)
    scale = base_nom / base_exact

    km = masses_arr * scale
    km_floor = np.floor(km)
    km_round = np.round(km)

    kmd_fraction = km - km_floor
    kmd_defect = km_round - km

    return {
        "KM": km,
        "KMD_fraction": kmd_fraction,
        "KMD_defect": kmd_defect,
        "KM_int_floor": km_floor.astype(int),
        "KM_int_round": km_round.astype(int),
        "scale": scale,
        "base_exact": base_exact,
        "base_nominal": base_nom,
    }


def default_kendrick_varm_key(
    base: Union[str, float, Tuple[float, float]], kmd_mode: Literal["fraction", "defect"]
) -> str:
    """
    Generate a deterministic varm key for storing Kendrick coordinates.

    Parameters
    ----------
    base : Union[str, float, Tuple[float, float]]
        The Kendrick base used for the calculation.
    kmd_mode : {"fraction", "defect"}
        The type of Kendrick analysis performed.

    Returns
    -------
    str
        A formatted string key (e.g., "X_kendrick_CH2_defect").
    """

    if isinstance(base, str):
        base_str = base
    elif isinstance(base, (int, float)):
        base_str = "mass"
    else:
        base_str = "tuple"
    return f"X_kendrick_{base_str}_{kmd_mode}"


def compute_kendrick_varm(
    adata: ad.AnnData,
    *,
    mz_key: str = "mz",
    base: Union[str, float, Tuple[float, float]] = "CH2",
    kmd_mode: Literal["fraction", "defect"] = "fraction",
    varm_key: Optional[str] = None,
    store_1d_in_var: bool = False,
    var_prefix: str = "kendrick",
) -> str:
    """
    Compute Kendrick coordinates and store them in ``adata.varm``.

    This function is the primary entry point for integrating KMD analysis 
    into an AnnData workflow. It populates ``varm`` with a 2D array 
    representing the Kendrick "latent space".

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing variable metadata.
    mz_key : str, default "mz"
        Column in ``adata.var`` used as the raw mass input.
    base : str, float, or Tuple[float, float], default "CH2"
        The chemical base for the Kendrick transformation.
    kmd_mode : {"fraction", "defect"}, default "fraction"
        The calculation mode for the defect (Y-axis).
    varm_key : str, optional
        The key in ``adata.varm`` where the $(N_{vars}, 2)$ array is stored.
        If None, a default key like ``"X_kendrick_CH2_fraction"`` is used.
    store_1d_in_var : bool, default False
        If True, also saves KM and KMD as individual columns in ``adata.var``.
    var_prefix : str, default "kendrick"
        Prefix for the columns if `store_1d_in_var` is True.

    Returns
    -------
    str
        The key used to store the coordinates in ``adata.varm``.

    Notes
    -----
    The Kendrick transformation effectively "rotates" the mass spectrum 
    data so that homologous series (e.g., polymers or lipids with different 
    chain lengths) appear as horizontal lines.

    

    Metadata, including the scaling factor and exact mass of the base, 
    is stored in ``adata.uns[f"{varm_key}_info"]``.
    """

    if mz_key not in adata.var.columns:
        raise KeyError(f"adata.var does not contain column '{mz_key}'")

    masses = np.asarray(adata.var[mz_key], dtype=float)
    if np.any(~np.isfinite(masses)):
        raise ValueError(
            f"adata.var['{mz_key}'] contains NaN/inf; fix or filter before computing Kendrick coords."
        )

    base_exact, base_nom = _base_masses(base)
    scale = base_nom / base_exact

    km = masses * scale
    if kmd_mode == "fraction":
        kmd = km - np.floor(km)
    elif kmd_mode == "defect":
        kmd = np.round(km) - km
    else:
        raise ValueError("kmd_mode must be 'fraction' or 'defect'")

    if varm_key is None:
        varm_key = default_kendrick_varm_key(base, kmd_mode)

    coords = np.column_stack([km, kmd]).astype(np.float32)
    adata.varm[varm_key] = coords

    adata.uns[f"{varm_key}_info"] = {
        "mz_key": mz_key,
        "kmd_mode": kmd_mode,
        "base": base,
        "base_exact": float(base_exact),
        "base_nominal": float(base_nom),
        "scale": float(scale),
        "columns": ["kendrick_mass", f"kmd_{kmd_mode}"],
    }

    if store_1d_in_var:
        adata.var[f"{var_prefix}_mass_{kmd_mode}"] = km
        adata.var[f"{var_prefix}_kmd_{kmd_mode}"] = kmd

    logger.info("Stored Kendrick coordinates in adata.varm['%s']", varm_key)
    return varm_key
