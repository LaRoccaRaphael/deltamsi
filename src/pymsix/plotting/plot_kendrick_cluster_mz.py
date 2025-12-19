# --- Kendrick plotting from cluster_masses_with_candidates result ---

from __future__ import annotations
import re
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Mapping, Optional, Tuple, List, TypedDict

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

# ---- base utilities ----
_FORM_RX = re.compile(r"([A-Z][a-z]?)(\d*)")


class KendrickCoords(TypedDict):
    KM: NDArray[np.floating]
    KMD_fraction: NDArray[np.floating]
    KMD_defect: NDArray[np.floating]
    KM_int_floor: NDArray[np.integer]
    KM_int_round: NDArray[np.integer]
    scale: float
    base_exact: float
    base_nominal: float


def _parse_formula_to_mass(formula: str) -> Tuple[float, float]:
    """
    Return (exact_mass, nominal_mass) for a simple empirical formula like 'CH2' or 'C2H3N1O1'.
    No parentheses/hydrates; '.' is ignored.
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


def _base_masses(base: str | float | Tuple[float, float]) -> Tuple[float, float]:
    """
    Accept:
      - string formula (e.g., 'CH2')
      - float (exact base mass) -> nominal inferred by rounding
      - (exact, nominal) tuple
    Returns (exact, nominal).
    """
    if isinstance(base, tuple) and len(base) == 2:
        return (float(base[0]), float(base[1]))
    if isinstance(base, (int, float)):
        m_exact = float(base)
        return (m_exact, round(m_exact))
    if isinstance(base, str):
        return _parse_formula_to_mass(base)
    raise ValueError(
        "Unsupported 'base' type. Use 'CH2', a float, or (exact, nominal)."
    )


def kendrick_coords(
    masses: Sequence[float],
    base: str | float | Tuple[float, float],
    kmd_mode: str = "fraction",
) -> KendrickCoords:
    """
    Compute Kendrick Mass (KM) and KMD for given masses and base.
    kmd_mode:
      - 'fraction' : KMD_f = KM - floor(KM) in [0,1)
      - 'defect'   : KMD_d = round(KM) - KM  in [-0.5,0.5]
    Returns dict with arrays: KM, KMD_fraction, KMD_defect, KM_int_floor, KM_int_round
    """
    masses_arr: NDArray[np.floating] = np.asarray(masses, dtype=float)

    m_exact, m_nom = _base_masses(base)
    scale = m_nom / m_exact

    KM = masses_arr * scale
    KM_floor = np.floor(KM)
    KM_round = np.round(KM)

    KMD_fraction = KM - KM_floor
    KMD_defect = KM_round - KM

    out: KendrickCoords = {
        "KM": KM,
        "KMD_fraction": KMD_fraction,
        "KMD_defect": KMD_defect,
        "KM_int_floor": KM_floor.astype(int),
        "KM_int_round": KM_round.astype(int),
        "scale": scale,
        "base_exact": m_exact,
        "base_nominal": m_nom,
    }
    return out


# ---- dataframe builder from clustering result ----
def kendrick_df_from_clustering(
    masses: Sequence[float],
    clustering_result: Mapping[str, object],
    *,
    family: Optional[Sequence[object]] = None,
    base: str | float | Tuple[float, float] = "CH2",
    mass_col: str = "exact_molecular_weight",
) -> pd.DataFrame:
    masses_arr: NDArray[np.floating] = np.asarray(masses, dtype=float)
    labels: NDArray[np.integer] = np.asarray(clustering_result["labels"], dtype=int)

    if masses_arr.size != labels.size:
        raise ValueError("masses and clustering_result['labels'] must have same length")

    df = pd.DataFrame({mass_col: masses_arr, "cluster": labels})

    if family is not None:
        fam_arr: NDArray[np.object_] = np.asarray(family, dtype=object)
        if fam_arr.size != masses_arr.size:
            raise ValueError("'family' must have same length as masses")
        df["family"] = fam_arr

    # add Kendrick columns (both modes; you can choose which to plot)
    kc = kendrick_coords(masses, base=base)

    df["kendrick_mass"] = kc["KM"]
    df["kmd_fraction"] = kc["KMD_fraction"]
    df["kmd_defect"] = kc["KMD_defect"]

    df.attrs["kendrick_info"] = {
        "scale": kc["scale"],
        "base_exact": kc["base_exact"],
        "base_nominal": kc["base_nominal"],
    }
    return df


# ---- plotting (with cluster filters) ----
def plot_kendrick_from_clustering(
    masses: Sequence[float],
    clustering_result: Mapping[str, object],
    *,
    family: Optional[Sequence[object]] = None,
    base: str | float | Tuple[float, float] = "CH2",
    mass_col: str = "exact_molecular_weight",
    # axes & style
    x_axis: str = "kendrick_mass",  # 'kendrick_mass' or 'm_over_z'
    kmd_mode: str = "fraction",  # 'fraction' or 'defect'
    point_size: float = 24,
    alpha: float = 0.9,
    hgrid_step: float = 1.0,  # KMD grid step (in 'fraction' units). For 'defect', use e.g. 0.1
    jitter: float = 0.0,  # horizontal jitter on x-axis
    annotate: bool = False,
    max_ann_per_group: int = 0,  # annotate up to N points per color group (by label index)
    # cluster filters
    top_k_clusters: Optional[int] = None,  # keep only K largest clusters
    selected_clusters: Optional[
        List[int]
    ] = None,  # explicit cluster IDs to keep (e.g. [-1,0,3])
    include_minus1_in_top: bool = True,
    min_cluster_size: int = 1,
    # layout
    two_panels: bool = True,  # second panel colored by 'family' (if provided)
    figsize: Tuple[float, float] = (9, 4.5),
) -> tuple[plt.Figure, list[plt.Axes] | plt.Axes, pd.DataFrame]:
    """
    Build Kendrick plots directly from cluster_masses_with_candidates result.
    Returns (fig, axes, df_used), where df_used is the filtered table with Kendrick columns.
    """
    # Build full DF
    df = kendrick_df_from_clustering(
        masses, clustering_result, family=family, base=base, mass_col=mass_col
    )

    # Apply cluster filters
    if selected_clusters is not None:
        allow = set(int(c) for c in selected_clusters)
    elif top_k_clusters is not None:
        vc = df["cluster"].value_counts().sort_values(ascending=False)
        if not include_minus1_in_top:
            vc = vc[vc.index != -1]
        if min_cluster_size > 1:
            vc = vc[vc >= int(min_cluster_size)]
        allow = set(int(c) for c in vc.head(int(top_k_clusters)).index.tolist())
    else:
        allow = None
    if allow is not None:
        df = df[df["cluster"].astype(int).isin(allow)].reset_index(drop=True)
        if df.empty:
            raise ValueError("No points remain after cluster filtering.")

    # Choose coordinates
    x = (
        df["kendrick_mass"].to_numpy(float)
        if x_axis == "kendrick_mass"
        else df[mass_col].to_numpy(float)
    )
    y = (
        df["kmd_fraction"].to_numpy(float)
        if kmd_mode == "fraction"
        else df["kmd_defect"].to_numpy(float)
    )
    if jitter:
        rng = np.random.default_rng(0)
        x = x + rng.normal(0.0, float(jitter), size=len(x))

    # Set up figure(s)
    if two_panels and ("family" in df.columns):
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=140, sharey=True)
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=140)
        axes = [ax1]

    # Colormaps
    cmap = plt.get_cmap("tab20")
    palette = list(getattr(cmap, "colors", [])) or [
        cmap(i / max(1, cmap.N - 1)) for i in range(cmap.N)
    ]

    # ---- Panel A: color by cluster ----
    df["_x"] = x
    df["_y"] = y
    groups = sorted(df["cluster"].astype(int).unique())
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    for g in groups:
        gd = df[df["cluster"].astype(int) == g]
        ax1.scatter(
            gd["_x"],
            gd["_y"],
            s=point_size,
            alpha=alpha,
            color=color_map[g],
            label=str(g),
        )
        if annotate and max_ann_per_group > 0:
            # annotate first N points of each group (by row order)
            for idx in gd.head(max_ann_per_group).index:
                ax1.annotate(
                    str(idx),
                    (gd.loc[idx, "_x"], gd.loc[idx, "_y"]),
                    fontsize=8,
                    alpha=0.8,
                )

    # grid lines for KMD
    if kmd_mode == "fraction":
        ymin, ymax = 0.0, 1.0
        ax1.set_ylim(ymin, ymax)
        if hgrid_step and hgrid_step > 0:
            for k in np.arange(0, 1 + 1e-9, hgrid_step):
                ax1.axhline(k, ls=":", lw=0.8, c="grey", alpha=0.4, zorder=0)
    else:  # defect in [-0.5,0.5]
        ax1.set_ylim(-0.5, 0.5)
        if hgrid_step and hgrid_step > 0:
            for k in np.arange(-0.5, 0.5 + 1e-9, hgrid_step):
                ax1.axhline(k, ls=":", lw=0.8, c="grey", alpha=0.4, zorder=0)

    ax1.set_xlabel(
        "Kendrick mass (base {})".format(base) if x_axis == "kendrick_mass" else "m/z"
    )
    ax1.set_ylabel("KMD ({})".format(kmd_mode))
    ax1.set_title("Kendrick by CLUSTER")
    ax1.grid(True, ls=":", alpha=0.25)
    ax1.legend(
        title="cluster", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False
    )

    # ---- Panel B: color by family (optional) ----
    if len(axes) == 2:
        ax2 = axes[1]
        fams = sorted(df["family"].astype(str).unique())
        color_map_f = {f: palette[i % len(palette)] for i, f in enumerate(fams)}
        for f in fams:
            gd = df[df["family"].astype(str) == f]
            ax2.scatter(
                gd["_x"],
                gd["_y"],
                s=point_size,
                alpha=alpha,
                color=color_map_f[f],
                label=str(f),
            )
            if annotate and max_ann_per_group > 0:
                for idx in gd.head(max_ann_per_group).index:
                    ax2.annotate(
                        str(idx),
                        (gd.loc[idx, "_x"], gd.loc[idx, "_y"]),
                        fontsize=8,
                        alpha=0.8,
                    )
        ax2.set_xlabel(
            "Kendrick mass (base {})".format(base)
            if x_axis == "kendrick_mass"
            else "m/z"
        )
        ax2.set_title("Kendrick by FAMILY")
        ax2.grid(True, ls=":", alpha=0.25)
        ax2.legend(
            title="family", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False
        )

    # housekeeping
    for a in axes:
        a.margins(x=0.02, y=0.02)
    fig.tight_layout()
    # drop helper cols before returning
    df = df.drop(columns=["_x", "_y"])
    return fig, axes, df
