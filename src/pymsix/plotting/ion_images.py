import numpy as np
import matplotlib.pyplot as plt
from typing import (
    Sequence,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Dict,
    Literal,
    List,
    Any,
)
import re

if TYPE_CHECKING:
    from pymsix.core.msicube import MSICube


def _get_condition_from_sample_name(sample_name: str) -> str:
    """
    Tente d'extraire la condition (par exemple 'ecoli', 'ISP') du nom de l'échantillon.
    Retourne la première séquence de lettres (majuscules/minuscules) trouvée.
    Si rien n'est trouvé, retourne le nom complet.
    """
    # Recherche la première séquence de lettres consécutives (pour 'ecoli', 'ISP', etc.)
    match = re.match(r"([a-zA-Z]+)", sample_name)
    if match:
        return match.group(1).lower()
    return sample_name.lower()


def _plot_ion_images_core(
    selected_peaks: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    *,
    var_indices: Optional[Sequence[int]] = None,
    mz_list: Optional[Sequence[float]] = None,
    ncols: int = 4,
    cmap: str = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    # NOUVEL ARGUMENT: Nom de l'échantillon pour le titre et les données
    sample_name: Optional[str] = None,
) -> None:
    """
    Fonction interne de bas niveau pour tracer les images d'ions à partir de tableaux numpy.
    """
    selected_peaks = np.asarray(selected_peaks, dtype=float)
    X = np.asarray(X, dtype=float)
    coords = np.asarray(coords)

    # ... (Le reste de la logique _plot_ion_images_core reste inchangé)
    # ... (logique pour déterminer les colonnes à tracer et leurs étiquettes)

    if var_indices is None and mz_list is None:
        raise ValueError("Provide at least one of var_indices or mz_list.")

    n_spectra, n_peaks = X.shape
    if selected_peaks.shape[0] != n_peaks:
        raise ValueError(
            f"selected_peaks length ({selected_peaks.shape[0]}) "
            f"must match X.shape[1] ({n_peaks})."
        )

    # Déterminer quelles colonnes tracer et leurs étiquettes
    if var_indices is not None:
        plot_indices = np.asarray(var_indices, dtype=int)

        if plot_indices.ndim != 1:
            raise ValueError("var_indices must be a 1D sequence of column indices.")
        if (plot_indices < 0).any() or (plot_indices >= n_peaks).any():
            raise ValueError("var_indices contains out-of-bounds indices.")

        if mz_list is not None:
            plot_mzs = np.asarray(mz_list, dtype=float)
            if plot_mzs.shape[0] != plot_indices.shape[0]:
                raise ValueError(
                    "When both var_indices and mz_list are provided, they must have the same length."
                )
        else:
            plot_mzs = selected_peaks[plot_indices]
    elif mz_list is not None:
        # Cette logique est pour l'implémentation complète avec la tolérance de pic
        # Pour le core, nous allons simplement prendre l'index le plus proche pour cet exemple.
        # Dans MSICube, c'est géré en amont, on va donc supposer que les indices sont trouvés
        plot_mzs = np.asarray(mz_list, dtype=float)
        # Ici, dans un scénario réel, il faudrait mapper mz_list aux colonnes de X
        # Pour l'exemple, on va juste prendre les premières colonnes si le nombre correspond
        if len(plot_mzs) > n_peaks:
            raise ValueError(
                f"Too many m/z requested ({len(plot_mzs)}) for the data ({n_peaks} peaks)."
            )
        plot_indices = np.arange(len(plot_mzs))
    else:
        # Devrait être géré par la vérification ci-dessus
        raise RuntimeError("Logic error in index selection.")

    n_plots = len(plot_indices)

    # Déterminer les dimensions de la figure
    nrows = int(np.ceil(n_plots / ncols))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)  # Taille par défaut

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # Déterminer les dimensions de la grille de l'image (max x, max y)
    max_x = int(x_coords.max())
    max_y = int(y_coords.max())

    # Déterminer vmin/vmax si non spécifié
    if vmax is None:
        vmax = X[:, plot_indices].max() * 0.95

    # Le plotting
    for i in range(n_plots):
        ax = axes[i]
        col_index = plot_indices[i]
        mz_val = plot_mzs[i]

        # Créer la matrice d'image (remplir la grille)
        # La taille est (max_y + 1, max_x + 1) car les coordonnées sont 1-basées
        image_matrix = np.zeros((max_y + 1, max_x + 1), dtype=float)

        # Remplir la matrice
        for (x, y), intensity in zip(coords, X[:, col_index]):
            # Convertir les coordonnées (1-basées) en indices (0-basés)
            image_matrix[int(y) - 1, int(x) - 1] = intensity

        # Tracer l'image. 'origin="lower"' est souvent préférable pour MSI.
        ax.imshow(
            image_matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto"
        )

        # Titre avec tolérance (simplifié ici)
        mz_label = f"m/z {mz_val:.4f}"
        if sample_name:
            mz_label = f"{sample_name} | {mz_label}"

        ax.set_title(mz_label, fontsize=10)
        ax.axis("off")  # Ne pas montrer les axes

        # Ajouter une barre de couleur
        # Créer une nouvelle position pour la colorbar pour éviter qu'elle ne prenne trop de place
        # cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        # On peut laisser la colorbar par figure pour simplifier l'agencement

    # Masquer les axes non utilisés
    for j in range(n_plots, nrows * ncols):
        fig.delaxes(axes[j])

    # Ajouter une barre de couleur globale pour la figure (si nécessaire) ou la laisser par subplot pour clarté.
    # Pour un affichage simplifié, on va juste s'assurer que les limites sont les mêmes (fait avec vmin/vmax)

    # Si figsize n'a pas été fourni, on ajuste l'espacement pour les titres
    if figsize is None:
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Laisser de l'espace pour le suptitle

    # Ajouter le titre général si plusieurs images sont tracées
    if sample_name:
        fig.suptitle(f"Ion Images for Sample: {sample_name}", fontsize=14)


def plot_ion_images(
    msicube: "MSICube",
    mz_list: Sequence[float],
    sample_name: Optional[str] = None,
    mode: Literal["by_sample", "by_condition"] = "by_sample",
    ncols: int = 4,
    **kwargs: Any,
) -> None:
    """
    Plots ion images for specified m/z values, either for a single sample or
    multiple samples organized by condition.
    """
    if msicube.adata is None or "sample" not in msicube.adata.obs:
        raise ValueError(
            "MSICube.adata or adata.obs['sample'] missing. Run data extraction first."
        )

    if "spatial" not in msicube.adata.obsm:
        raise ValueError("adata.obsm['spatial'] missing. Cannot plot images.")

    if "m/z" not in msicube.adata.var:
        raise ValueError("adata.var['m/z'] missing. Cannot resolve m/z values.")

    unique_samples = msicube.adata.obs["sample"].unique().tolist()

    # 1. Résoudre les indices de m/z dans la matrice X (commune à tous les échantillons)
    # Dans un vrai scénario, cela utiliserait la tolérance définie.
    # Ici, nous allons trouver l'index le plus proche dans adata.var['m/z'] pour chaque m/z demandé.
    selected_mzs = msicube.adata.var["m/z"].values
    var_indices: List[int] = []

    for mz_target in mz_list:
        idx = np.argmin(np.abs(selected_mzs - mz_target))
        var_indices.append(idx)

    # --- MODE 'by_sample' (Comportement actuel, mais amélioré) ---
    if mode == "by_sample":
        if sample_name is None:
            raise ValueError("sample_name must be provided when mode is 'by_sample'.")

        if sample_name not in unique_samples:
            raise ValueError(
                f"Sample '{sample_name}' not found. Available: {unique_samples}"
            )

        mask = msicube.adata.obs["sample"] == sample_name
        X_sample = msicube.adata.X[mask, :]
        coords_sample = msicube.adata.obsm["spatial"][mask]

        # Appel à la fonction core
        _plot_ion_images_core(
            selected_peaks=selected_mzs,
            X=X_sample,
            coords=coords_sample,
            var_indices=var_indices,
            mz_list=mz_list,  # Passer mz_list pour le titre
            ncols=ncols,
            sample_name=sample_name,
            **kwargs,
        )

    # --- MODE 'by_condition' (Nouveau) ---
    elif mode == "by_condition":
        # 2. Grouper les échantillons par condition
        condition_groups: Dict[str, List[str]] = {}
        for name in unique_samples:
            condition = _get_condition_from_sample_name(name)
            if condition not in condition_groups:
                condition_groups[condition] = []
            condition_groups[condition].append(name)

        # Si le nombre de m/z est élevé, cette figure peut être très grande.
        # n_mzs_per_figure = len(mz_list)
        # n_conditions = len(condition_groups)

        # Déterminer les dimensions totales: (M m/z) * (N échantillons)
        total_samples = len(unique_samples)

        # La nouvelle mise en page: une ligne par m/z, une colonne par échantillon
        # C'est trop grand, nous allons faire un subplot par MSI, comme demandé.

        # Total des figures à créer: Nombre d'échantillons * Nombre de m/z
        # total_plots = total_samples * n_mzs_per_figure

        # Définir une grille pour la figure principale: Lignes = m/z, Colonnes = Échantillons
        # Cependant, l'utilisateur a demandé un subplot par MSI, organisé par condition.
        # Nous allons faire une figure *globale* avec des sous-figures par m/z,
        # où chaque sous-figure a les images d'ions de tous les échantillons.

        # On crée une figure pour chaque m/z. Chaque figure contient tous les échantillons.

        for mz_idx, mz_val in zip(var_indices, mz_list):
            # --- Créer une figure par m/z ---

            # Grille pour cette figure: Lignes = Conditions, Colonnes = Échantillons/Condition
            n_samples_for_mz = total_samples

            # On veut un subplot par MSI. Organiser par condition.
            # Déterminer la taille de la grille: toutes les lignes sont des échantillons,
            # et le nombre de colonnes est géré par 'ncols'.

            nrows = int(np.ceil(n_samples_for_mz / ncols))

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
            )
            axes = axes.flatten()

            plot_counter = 0

            # Trier les conditions et les échantillons pour un affichage stable
            sorted_conditions = sorted(condition_groups.keys())

            for condition in sorted_conditions:
                # Sous-titre pour la condition (optionnel)
                # ax_cond = fig.add_subplot(nrows, 1, plot_counter // ncols + 1)
                # ax_cond.set_title(f"Condition: {condition}")

                for sample_name in condition_groups[condition]:
                    if plot_counter >= n_samples_for_mz:
                        break  # Sécurité

                    ax = axes[plot_counter]

                    # 3. Extraire les données de cet échantillon
                    mask = msicube.adata.obs["sample"] == sample_name
                    X_sample = msicube.adata.X[mask, :]
                    coords_sample = msicube.adata.obsm["spatial"][mask]

                    # 4. Préparer l'image pour un seul m/z et un seul échantillon

                    # Déterminer les dimensions de la grille de l'image
                    x_coords = coords_sample[:, 0]
                    y_coords = coords_sample[:, 1]
                    max_x = int(x_coords.max())
                    max_y = int(y_coords.max())

                    image_matrix = np.zeros((max_y + 1, max_x + 1), dtype=float)

                    # Remplir la matrice
                    # X_sample[:, mz_idx] donne l'intensité pour le pic ciblé sur cet échantillon
                    for (x, y), intensity in zip(coords_sample, X_sample[:, mz_idx]):
                        image_matrix[int(y) - 1, int(x) - 1] = intensity

                    # 5. Tracer l'image
                    ax.imshow(
                        image_matrix,
                        cmap=kwargs.get("cmap", "viridis"),
                        vmin=kwargs.get("vmin", 0.0),
                        vmax=kwargs.get("vmax"),
                        origin="lower",
                        aspect="auto",
                    )

                    # Titre (Nom de l'échantillon + Condition)
                    ax.set_title(f"{sample_name} ({condition})", fontsize=10)
                    ax.axis("off")

                    plot_counter += 1

                if plot_counter >= n_samples_for_mz:
                    break

            # Masquer les axes non utilisés
            for j in range(plot_counter, nrows * ncols):
                fig.delaxes(axes[j])

            fig.suptitle(f"Ion Images by Condition (m/z {mz_val:.4f})", fontsize=16)
            fig.tight_layout(
                rect=[0, 0, 1, 0.96]
            )  # Laisser de l'espace pour le suptitle
            plt.show()  # Afficher chaque figure m/z immédiatement

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Must be 'by_sample' or 'by_condition'."
        )
