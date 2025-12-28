"""Interactive manual labeling in Kendrick space for :class:`~pymsix.core.msicube.MSICube`.

This module exposes a Plotly + ipywidgets helper that mirrors the workflow described in
the user request. It reads Kendrick coordinates stored in ``adata.varm`` (created via
``compute_kendrick_varm``), lets the user select a region with a lasso/box tool, and
writes the chosen label back into ``adata.var``. The returned widget updates colors live
and provides the selected indices and variable names for export.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets


if TYPE_CHECKING:
    from pymsix.core.msicube import MSICube


@dataclass
class KendrickManualLabelState:
    """Holds the current manual selection in Kendrick space."""

    selected_var_pos: List[int]
    selected_var_names: List[str]


def manual_label_vars_from_kendrick(
    msicube: "MSICube",
    *,
    varm_key: str,
    label_key: str = "manual_label",
    default_label: str = "unlabeled",
    mz_key: Optional[str] = "mz",
    coord_cols: Tuple[int, int] = (0, 1),
    dragmode: str = "lasso",
    point_size: int = 6,
    height: int = 650,
    max_points_warn: int = 120_000,
) -> Tuple[widgets.VBox, KendrickManualLabelState]:
    """
    Interactive manual labeling of ``adata.var`` based on a region selected in Kendrick space.

    Args:
        msicube: The :class:`~pymsix.core.msicube.MSICube` instance holding the AnnData object.
        varm_key: The key in ``adata.varm`` that contains Kendrick coordinates
            (as produced by :func:`pymsix.processing.kendrick.compute_kendrick_varm`).
        label_key: Column name to write in ``adata.var`` for the manual labels.
        default_label: Value to use for unannotated variables.
        mz_key: Optional column in ``adata.var`` containing m/z values for hover display.
        coord_cols: Column indices inside ``adata.varm[varm_key]`` to use as (KM, KMD).
        dragmode: Plotly drag mode ("lasso" or "select").
        point_size: Marker size in the scatter plot.
        height: Plot height in pixels.
        max_points_warn: Threshold above which a warning is printed about interactivity speed.

    Returns:
        A tuple ``(ui, state)`` where ``ui`` is a :class:`ipywidgets.VBox` to display in a
        notebook cell and ``state`` holds the last selection (positions and var names).

    Notes:
        * Requires the optional ``viz`` dependencies (``plotly`` + ``ipywidgets`` + ``anywidget``).
        * Best used in Jupyter Notebook/Lab due to reliance on ``FigureWidget`` callbacks.
    """

    if msicube.adata is None:
        raise ValueError("MSICube.adata is empty. Run peak picking before manual labeling.")

    adata = msicube.adata

    if varm_key not in adata.varm:
        raise KeyError(
            f"adata.varm['{varm_key}'] not found. Run compute_kendrick_varm(...) first."
        )

    coords = np.asarray(adata.varm[varm_key])
    if coords.ndim != 2 or coords.shape[0] != adata.n_vars:
        raise ValueError(
            f"adata.varm['{varm_key}'] must be shape (n_vars, k). Got {coords.shape}."
        )

    xcol, ycol = coord_cols
    if not (0 <= xcol < coords.shape[1] and 0 <= ycol < coords.shape[1]):
        raise ValueError(f"coord_cols {coord_cols} out of range for varm shape {coords.shape}.")

    KM = coords[:, xcol].astype(float)
    KMD = coords[:, ycol].astype(float)

    n = adata.n_vars
    if n > max_points_warn:
        print(
            f"Warning: {n:,} points. Interactive selection may be slow. "
            "Consider filtering variables before labeling or increasing max_points_warn."
        )

    finite_mask = np.isfinite(KM) & np.isfinite(KMD)
    if not finite_mask.any():
        raise ValueError(
            "Kendrick coordinates contain no finite points to plot. "
            "Check adata.varm entries for NaN/inf or recompute with compute_kendrick_varm()."
        )
    if not finite_mask.all():
        warnings.warn(
            "Some Kendrick coordinates are non-finite and will be hidden in the plot.",
            RuntimeWarning,
        )
        KM = KM[finite_mask]
        KMD = KMD[finite_mask]
        var_names = var_names[finite_mask]
        var_pos = var_pos[finite_mask]

    if label_key not in adata.var.columns:
        adata.var[label_key] = default_label
    else:
        adata.var[label_key] = adata.var[label_key].astype("string").fillna(default_label)

    if mz_key is not None and mz_key in adata.var.columns:
        mz = np.asarray(adata.var[mz_key], dtype=float)
    else:
        mz = np.full(n, np.nan, dtype=float)

    var_names = adata.var_names.astype(str).to_numpy()
    var_pos = np.arange(n, dtype=int)

    palette = (
        [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        + [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )

    def _label_to_color(labels: np.ndarray) -> np.ndarray:
        cats = pd.Series(labels).astype("string").fillna(default_label)
        uniq = sorted(cats.unique(), key=lambda z: str(z))
        lut = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
        return cats.map(lut).to_numpy()

    def _make_hover(labels: np.ndarray) -> np.ndarray:
        labels = pd.Series(labels).astype("string").fillna(default_label).to_numpy()
        out = []
        for i in range(n):
            out.append(
                f"var_pos={i}"
                f"<br>var_name={var_names[i]}"
                + (f"<br>mz={mz[i]:.6f}" if np.isfinite(mz[i]) else "")
                + f"<br>KM={KM[i]:.6f}"
                + f"<br>KMD={KMD[i]:.6f}"
                + f"<br>{label_key}={labels[i]}"
            )
        return np.asarray(out, dtype=object)

    colors = _label_to_color(adata.var[label_key].to_numpy())
    hover = _make_hover(adata.var[label_key].to_numpy())
    if not finite_mask.all():
        colors = colors[finite_mask]
        hover = hover[finite_mask]

    try:
        import anywidget  # noqa: F401
    except ImportError as e:  # pragma: no cover - optional dependency import guard
        raise ImportError(
            "manual_label_vars_from_kendrick requires the optional 'anywidget' dependency. "
            "Install it directly with `pip install anywidget` or via the `pymsix[viz]` extras."
        ) from e

    try:
        fig = go.FigureWidget()
    except ImportError as e:  # pragma: no cover - forwarded from plotly
        raise ImportError(
            "Plotly FigureWidget support is unavailable. Ensure `anywidget` is installed and that "
            "the optional visualization dependencies are present."
        ) from e
    fig.add_trace(
        go.Scatter(
            x=KM,
            y=KMD,
            mode="markers",
            marker=dict(size=point_size, color=colors, opacity=0.9, line=dict(width=0)),
            customdata=var_pos,
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=height,
        dragmode=dragmode,
        xaxis_title="Kendrick mass (KM)",
        yaxis_title="Kendrick mass defect (KMD)",
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Kendrick plot → {varm_key}",
    )

    label_text = widgets.Text(
        value="region_1",
        description="Label:",
        placeholder="e.g. lipid_series_A",
        layout=widgets.Layout(width="340px"),
    )
    apply_btn = widgets.Button(description="Apply label to selection", button_style="success")
    clear_btn = widgets.Button(description="Clear selection", button_style="")
    set_default_btn = widgets.Button(description=f"Set selected to '{default_label}'", button_style="warning")
    mode_dd = widgets.Dropdown(
        options=[("Lasso", "lasso"), ("Box", "select")],
        value=dragmode,
        description="Select:",
        layout=widgets.Layout(width="220px"),
    )
    append_cb = widgets.Checkbox(value=False, description="Append selection")

    out = widgets.Output()

    state = KendrickManualLabelState(selected_var_pos=[], selected_var_names=[])
    selected_set: set[int] = set()

    def _refresh_state_and_print() -> None:
        state.selected_var_pos = sorted(selected_set)
        state.selected_var_names = [var_names[i] for i in state.selected_var_pos]
        with out:
            out.clear_output()
            print(f"Selected: {len(state.selected_var_pos)} vars")
            if len(state.selected_var_pos) <= 30:
                print("var_pos:", state.selected_var_pos)
                print("var_names:", state.selected_var_names)

    def _on_select(trace: go.Scatter, points: Any, selector: Any) -> None:  # type: ignore[override]
        nonlocal selected_set
        inds = [int(trace.customdata[i]) for i in points.point_inds]
        if append_cb.value:
            selected_set |= set(inds)
        else:
            selected_set = set(inds)
        _refresh_state_and_print()

    fig.data[0].on_selection(_on_select)

    def _apply_label(value: str) -> None:
        nonlocal colors, hover
        if not value or value.strip() == "":
            raise ValueError("Label text is empty.")
        if not selected_set:
            return

        idx = np.array(sorted(selected_set), dtype=int)
        adata.var.loc[adata.var.index[idx], label_key] = value

        colors = _label_to_color(adata.var[label_key].to_numpy())
        hover = _make_hover(adata.var[label_key].to_numpy())
        fig.data[0].marker.color = colors
        fig.data[0].hovertext = hover

    def _set_default() -> None:
        nonlocal colors, hover
        if not selected_set:
            return
        idx = np.array(sorted(selected_set), dtype=int)
        adata.var.loc[adata.var.index[idx], label_key] = default_label

        colors = _label_to_color(adata.var[label_key].to_numpy())
        hover = _make_hover(adata.var[label_key].to_numpy())
        fig.data[0].marker.color = colors
        fig.data[0].hovertext = hover

    def _clear() -> None:
        nonlocal selected_set
        selected_set = set()
        fig.data[0].selectedpoints = None
        _refresh_state_and_print()

    def _apply_clicked(_: widgets.Button) -> None:
        try:
            _apply_label(label_text.value.strip())
            _refresh_state_and_print()
        except Exception as e:  # noqa: BLE001
            with out:
                out.clear_output()
                print("Error:", e)

    def _default_clicked(_: widgets.Button) -> None:
        try:
            _set_default()
            _refresh_state_and_print()
        except Exception as e:  # noqa: BLE001
            with out:
                out.clear_output()
                print("Error:", e)

    def _clear_clicked(_: widgets.Button) -> None:
        _clear()

    def _mode_changed(change: dict[str, Any]) -> None:
        fig.layout.dragmode = change["new"]

    apply_btn.on_click(_apply_clicked)
    set_default_btn.on_click(_default_clicked)
    clear_btn.on_click(_clear_clicked)
    mode_dd.observe(_mode_changed, names="value")

    _refresh_state_and_print()

    controls = widgets.HBox([label_text, apply_btn, set_default_btn, clear_btn])
    controls2 = widgets.HBox([mode_dd, append_cb])
    ui = widgets.VBox([controls2, controls, fig, out])

    return ui, state

