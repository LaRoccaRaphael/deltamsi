"""
Centralized exception types for msix.

All package-specific errors should inherit from `MsixError` so callers can
`except MsixError` to catch any msix failure, while still allowing fine-grained
handling of specific subclasses.

Design goals:
- Human-readable messages with key context (names, shapes, paths, ppm, etc.).
- Lightweight: no heavy imports; pure stdlib + type hints.
"""

from __future__ import annotations


__all__ = [
    "MsixError",
    "StorageError",
    "ShapeMismatchError",
    "MissingLayerError",
    "DuplicateLayerError",
    "InvalidROIError",
    "AnnotationJoinError",
]


class MsixError(Exception):
    """Base class for all msix errors."""


class StorageError(MsixError):
    """Raised for storage/backend I/O failures (Zarr/HDF5/Parquet, etc.)."""

    def __init__(self, message: str, *, path: str | None = None, op: str | None = None):
        ctx = []
        if op:
            ctx.append(f"op={op}")
        if path:
            ctx.append(f"path={path}")
        suffix = f" ({', '.join(ctx)})" if ctx else ""
        super().__init__(f"{message}{suffix}")
        self.path = path
        self.op = op


class ShapeMismatchError(MsixError):
    """Raised when an array/table does not match the expected (n_obs, n_vars) semantics."""

    def __init__(
        self,
        *,
        expected: tuple[int, ...] | None = None,
        actual: tuple[int, ...] | None = None,
        context: str | None = None,
        message: str | None = None,
    ):
        if message is None:
            parts = ["Shape mismatch"]
            if context:
                parts.append(f"in {context}")
            if expected is not None and actual is not None:
                parts.append(f"(expected {expected}, got {actual})")
            message = " ".join(parts)
        super().__init__(message)
        self.expected = expected
        self.actual = actual
        self.context = context


class MissingLayerError(MsixError):
    """Raised when a requested layer is not present in the cube."""

    def __init__(self, layer: str, *, available: list[str] | None = None):
        msg = f"Layer '{layer}' not found"
        if available:
            msg += f". Available layers: {sorted(available)}"
        super().__init__(msg)
        self.layer = layer
        self.available = available or []


class DuplicateLayerError(MsixError):
    """Raised when attempting to create a layer that already exists."""

    def __init__(self, layer: str):
        super().__init__(f"Layer '{layer}' already exists")
        self.layer = layer


class InvalidROIError(MsixError):
    """Raised for malformed or out-of-bounds ROIs (mask, polygon, or bbox)."""

    def __init__(
        self,
        *,
        reason: str,
        shape: tuple[int, int] | None = None,
        bounds: tuple[int, int] | None = None,
    ):
        # shape = (height, width) of the target canvas
        # bounds = (y, x) max indices or similar context
        ctx = []
        if shape is not None:
            ctx.append(f"canvas_shape={shape}")
        if bounds is not None:
            ctx.append(f"bounds={bounds}")
        suffix = f" ({', '.join(ctx)})" if ctx else ""
        super().__init__(f"Invalid ROI: {reason}{suffix}")
        self.reason = reason
        self.shape = shape
        self.bounds = bounds


class AnnotationJoinError(MsixError):
    """
    Raised when joining external annotations (e.g., METASPACE) to `.var`
    fails due to ambiguity or tolerance violations.
    """

    def __init__(
        self,
        *,
        mz: float | None = None,
        matches: int | None = None,
        ppm_tol: float | None = None,
        message: str | None = None,
    ):
        if message is None:
            parts = ["Annotation join failed"]
            if mz is not None:
                parts.append(f"for m/z={mz:g}")
            if matches is not None:
                parts.append(f"(matches={matches})")
            if ppm_tol is not None:
                parts.append(f"within ±{ppm_tol:g} ppm")
            message = " ".join(parts)
        super().__init__(message)
        self.mz = mz
        self.matches = matches
        self.ppm_tol = ppm_tol
