"""Kendrick mass utilities."""
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

_FORM_RX = re.compile(r"([A-Z][a-z]?(\d*))")


def _parse_formula_to_mass(formula: str) -> Tuple[float, float]:
    """
    Return ``(exact_mass, nominal_mass)`` for a simple empirical formula like ``"CH2"``.
    Parentheses/hydrates are not supported. ``'.'`` and ``'·'`` are ignored.
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
    Normalize a Kendrick base description into ``(exact_mass, nominal_mass)``.

    Parameters
    ----------
    base:
        Either an empirical formula (``"CH2"``), a float for the exact mass (nominal
        inferred by rounding) or an explicit ``(exact, nominal)`` tuple.
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
    """Compute Kendrick mass and defect coordinates for the provided masses."""

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
    """Generate a deterministic varm key for Kendrick coordinates."""

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
    Compute Kendrick coordinates from ``adata.var[mz_key]`` and store them in ``adata.varm``.

    The generated varm entry contains two columns: Kendrick mass (KM) and the corresponding
    Kendrick mass defect (KMD) for the requested mode. Metadata about the computation is
    stored in ``adata.uns`` under ``f"{varm_key}_info"``.

    Returns the varm key used for storage.
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
