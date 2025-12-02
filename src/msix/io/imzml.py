# msix/io/imzml.py
"""
Thin, streaming-first imzML access built on pyimzML.

Goals
-----
- Never load the whole file; stream spectra on demand.
- Provide order-agnostic pixel indexing: n_obs follows the imzML spectrum order;
  `.obsm["spatial"][i] == (x_i, y_i)` records the coordinate for row i (0-based).
- Works for regular and irregular grids.

This module DOES NOT build the `/X` matrix. That belongs to later phases.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser

from ..utils.validate import mode as _validate_mode


# ----------------------------- public datatypes -----------------------------


@dataclass(frozen=True)
class ImzmlProbe:
    path: str
    ibd_path: Optional[str]
    mode: str  # "profile" | "centroid"
    n_pixels: int
    width: int  # inferred grid width  (>= max x + 1; 0-based coords)
    height: int  # inferred grid height (>= max y + 1; 0-based coords)
    mz_range: Tuple[float, float]  # (min_mz, max_mz) sampled across a few spectra
    metadata: Dict[str, Any]  # small header subset (best-effort, may be partial)


# ------------------------------- open/close --------------------------------


def open_parser(path: str) -> ImzMLParser:
    """
    Open an imzML file and return a live `ImzMLParser`.

    Notes
    -----
    - We don't pass extra kwargs to keep compatibility across pyimzML versions.
    - The returned parser owns the .ibd handle; no explicit close API is required.
    """
    return ImzMLParser(path)


def close_parser(p: ImzMLParser) -> None:
    """No-op for pyimzML (included for symmetry and future backends)."""
    # pyimzML closes underlying files on GC; nothing to do here.
    return None


def _ensure_parser(path_or_parser: str | ImzMLParser) -> tuple[ImzMLParser, bool]:
    """Return (parser, created_here) for convenience."""
    if isinstance(path_or_parser, ImzMLParser):
        return path_or_parser, False
    return open_parser(str(path_or_parser)), True


# ------------------------------ probe helpers ------------------------------


def _guess_ibd_path(imzml_path: str) -> Optional[str]:
    p = Path(imzml_path)
    if p.suffix.lower() != ".imzml":
        return None
    ibd = p.with_suffix(".ibd")
    return str(ibd) if ibd.exists() else None


def scan_coords(p: ImzMLParser) -> tuple[np.ndarray, int, int]:
    """
    Scan (x,y) for all spectra; return coords array (n_obs,2) in **0-based** pixels
    and inferred (width, height) = (max_x+1, max_y+1).
    """
    coords = getattr(p, "coordinates", None)
    if not coords:
        # pyimzML always has coordinates; this is a very defensive fallback.
        n = len(p.mzLengths) if hasattr(p, "mzLengths") else 0
        xy = np.zeros((n, 2), dtype=np.float32)
        return xy, 0, 0

    # pyimzML coordinates are typically 1-based tuples: (x, y) or (x, y, z)
    xy = np.empty((len(coords), 2), dtype=np.float32)
    max_x = 0
    max_y = 0
    for i, c in enumerate(coords):
        try:
            x, y = int(c[0]), int(c[1])
        except Exception:
            # fallback for odd records
            x, y = int(c), 0
        x0 = max(0, x - 1)
        y0 = max(0, y - 1)
        xy[i, 0] = x0
        xy[i, 1] = y0
        if x0 > max_x:
            max_x = x0
        if y0 > max_y:
            max_y = y0

    width = int(max_x) + 1
    height = int(max_y) + 1
    return xy, width, height


def quick_mz_span(p: ImzMLParser, k: int = 16) -> tuple[float, float]:
    """Sample up to k spectra to estimate (min_mz, max_mz)."""
    n = len(getattr(p, "coordinates", []))
    if n == 0:
        return (np.inf, -np.inf)
    idx = np.linspace(0, n - 1, num=min(k, n), dtype=int)
    lo = np.inf
    hi = -np.inf
    for i in idx:
        mz, _ = p.getspectrum(int(i))
        if mz is None or len(mz) == 0:
            continue
        m0 = float(np.min(mz))
        m1 = float(np.max(mz))
        if m0 < lo:
            lo = m0
        if m1 > hi:
            hi = m1
    return (lo, hi)


def _collect_header_metadata(p: ImzMLParser) -> Dict[str, Any]:
    """Small, safe subset of header metadata (best-effort)."""
    out: Dict[str, Any] = {}
    d = getattr(p, "imzmldict", {}) or {}
    # Common keys (vary by writer); we keep a tiny set to stay JSON-safe and portable.
    for k in (
        "instrument model",
        "instrument source",
        "analyzer",
        "software name",
        "software version",
        "userParam",
        "ms level",
    ):
        if k in d:
            try:
                out[k] = str(d[k])
            except Exception:
                pass
    return out


def probe(path: str, mode: str) -> ImzmlProbe:
    """
    Open and probe an imzML quickly.

    The caller MUST provide the acquisition mode: "profile" or "centroid".
    """
    # NEW: validate/normalize the user-provided mode
    mode = _validate_mode(mode)

    p = open_parser(path)
    try:
        n = len(getattr(p, "coordinates", []))
        coords, width, height = scan_coords(p)
        md = _collect_header_metadata(p)
        mz_lo, mz_hi = quick_mz_span(p, k=16)
        return ImzmlProbe(
            path=str(path),
            ibd_path=_guess_ibd_path(path),
            mode=mode,
            n_pixels=int(n),
            width=int(width),
            height=int(height),
            mz_range=(float(mz_lo), float(mz_hi)),
            metadata=md,
        )
    finally:
        close_parser(p)


# ------------------------------ spectrum I/O -------------------------------


def read_spectrum(
    path_or_parser: str | ImzMLParser, i: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the i-th spectrum (0-based).

    Returns
    -------
    (mz, intens)
      mz     : np.ndarray, float64, shape (n_peaks,)
      intens : np.ndarray, float32, shape (n_peaks,)
    """
    p, created = _ensure_parser(path_or_parser)
    try:
        mz, intens = p.getspectrum(int(i))
        mz = np.asarray(mz, dtype=np.float64)
        intens = np.asarray(intens, dtype=np.float32)
        return mz, intens
    finally:
        if created:
            close_parser(p)


def iter_spectra(
    path_or_parser: str | ImzMLParser,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """
    Stream spectra from the file.

    Yields
    ------
    (i, mz, intens) with:
      - i      : int, 0-based index in file order
      - mz     : np.ndarray, float64
      - intens : np.ndarray, float32
    """
    p, created = _ensure_parser(path_or_parser)
    try:
        n = len(getattr(p, "coordinates", []))
        for i in range(n):
            mz, intens = p.getspectrum(i)
            yield i, np.asarray(mz, np.float64), np.asarray(intens, np.float32)
    finally:
        if created:
            close_parser(p)


def compute_tic(path_or_parser: str | ImzMLParser) -> np.ndarray:
    """
    Compute TIC per spectrum (sum of intensities). Returns float32 of length n_pixels.
    """
    p, created = _ensure_parser(path_or_parser)
    try:
        n = len(getattr(p, "coordinates", []))
        out = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            _, intens = p.getspectrum(i)
            if intens is not None and len(intens) > 0:
                out[i] = float(np.sum(intens, dtype=np.float64))
        return out
    finally:
        if created:
            close_parser(p)
