import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, Union, List
import pandas as pd

# Note: We don't import MSICube directly to avoid circular imports,
# but we assume the object passed has the structure of MSICube.

def plot_ion_images(
   msicube,  # Type: MSICube
   mz: Union[float, Sequence[float]],
   samples: Union[str, Sequence[str]],
   *,
   share_intensity_scale: bool = False,
   ncols: int = 4,
   cmap: str = "viridis",
   figsize: Optional[Tuple[float, float]] = None,
   show_axes: bool = True,
) -> None:
   """
   Plots ion images with high flexibility and ergonomic defaults.
  
   Can plot:
   - One m/z for multiple samples (comparison).
   - Multiple m/z for one sample (exploration).
   - Multiple m/z for multiple samples (matrix).

   Parameters
   ----------
   msicube : MSICube
       The MSICube object.
   mz : float or list of float
       The m/z value(s) to visualize. automatically finds the closest peaks.
   samples : str or list of str
       The sample name(s) to visualize.
   share_intensity_scale : bool
       If True, all images will share the same intensity scale (vmax).
       Useful to compare absolute intensities between samples.
   ncols : int
       Number of columns in the grid.
   cmap : str
       Matplotlib colormap (e.g., 'viridis', 'magma', 'inferno').
   figsize : tuple, optional
       Custom figure size. If None, it is calculated automatically.
   show_axes : bool
       If True, prints X/Y coordinates on the image borders.
   """
   if msicube.adata is None:
       raise ValueError("MSICube.adata is None. Run extract_peak_matrix first.")

   # --- 1. Normalize Inputs (make them lists) ---
   if isinstance(mz, (float, int)):
       target_mzs = [float(mz)]
   else:
       target_mzs = [float(m) for m in mz]

   if isinstance(samples, str):
       target_samples = [samples]
   else:
       target_samples = list(samples)

   # --- 2. Resolve m/z indices ---
   # We find the closest actual peak in adata.var for each requested mz
   available_mzs = msicube.adata.var["mz"].values
   col_indices = []
   actual_mzs = []
  
   for m in target_mzs:
       idx = int(np.argmin(np.abs(available_mzs - m)))
       col_indices.append(idx)
       actual_mzs.append(available_mzs[idx])

   # --- 3. Prepare Data Extraction Loop ---
   # We need to extract data for every (sample, mz) combination
   plot_data = []
  
   global_max = 0.0
  
   for sample in target_samples:
       # Filter for the sample
       mask = msicube.adata.obs["sample"] == sample
       if not np.any(mask):
           print(f"WARNING: Sample '{sample}' not found. Skipping.")
           continue
          
       # Get coordinates
       coords = msicube.adata.obsm["spatial"][mask]
       if hasattr(coords, "values"):
           coords = coords.values
          
       # Get intensities for selected columns
       # Slicing: [rows (pixels), cols (features)]
       intensities = msicube.adata.X[mask, :][:, col_indices]
      
       # Determine Grid Dimensions for this sample
       x = coords[:, 0].astype(int)
       y = coords[:, 1].astype(int)
       x_min, x_max = x.min(), x.max()
       y_min, y_max = y.min(), y.max()
       width = x_max - x_min + 1
       height = y_max - y_min + 1

       # Process each m/z for this sample
       for i, mz_val in enumerate(actual_mzs):
           # Construct dense image
           img = np.full((height, width), np.nan, dtype=float)
           # Normalize coords to 0-based
           img[y - y_min, x - x_min] = intensities[:, i]
          
           local_max = np.nanmax(img)
           if not np.isnan(local_max) and local_max > global_max:
               global_max = local_max
              
           plot_data.append({
               "img": img,
               "sample": sample,
               "mz": mz_val,
               "extent": (x_min, x_max, y_min, y_max),
               "local_max": local_max
           })

   if not plot_data:
       raise ValueError("No data found for the provided samples/mz.")

   # --- 4. Plotting Setup ---
   n_plots = len(plot_data)
   nrows = int(np.ceil(n_plots / ncols))
  
   if figsize is None:
       # Auto-scale figure: ~4 inches per column, ~3.5 per row
       figsize = (4 * ncols, 3.5 * nrows)

   fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
  
   # Handle single ax case
   if n_plots == 1:
       axes = [axes]
   else:
       axes = axes.flatten()

   # --- 5. Draw Images ---
   for i, ax in enumerate(axes):
       if i < n_plots:
           data = plot_data[i]
           img = data["img"]
          
           # Determine Scaling
           if share_intensity_scale:
               vmax = global_max
           else:
               vmax = data["local_max"] if not np.isnan(data["local_max"]) else 1.0
              
           # Plot
           im = ax.imshow(
               img,
               origin="lower",
               cmap=cmap,
               interpolation="nearest",
               extent=data["extent"], # This sets the axes ticks to real coordinates
               vmin=0,
               vmax=vmax
           )
          
           # Aesthetics
           ax.set_aspect('equal') # Prevents distortion
          
           # Smart Title Logic
           # If multiple samples & 1 mz -> Title = Sample
           # If 1 sample & multiple mz -> Title = m/z
           # Else -> Title = Sample | m/z
           if len(target_samples) > 1 and len(target_mzs) == 1:
               title = f"{data['sample']}"
               fig.suptitle(f"m/z {data['mz']:.4f}", fontsize=14)
           elif len(target_samples) == 1 and len(target_mzs) > 1:
               title = f"{data['mz']:.4f} m/z"
               fig.suptitle(f"Sample: {data['sample']}", fontsize=14)
           else:
               title = f"{data['sample']}\n{data['mz']:.4f} m/z"

           ax.set_title(title, fontsize=11)
          
           if show_axes:
               ax.tick_params(axis='both', which='major', labelsize=8)
               ax.set_xlabel("x", fontsize=8)
               ax.set_ylabel("y", fontsize=8)
           else:
               ax.axis("off")

           # Colorbar (Right size magic)
           # Create an axes divider to place colorbar exactly beside the plot
           from mpl_toolkits.axes_grid1 import make_axes_locatable
           divider = make_axes_locatable(ax)
           cax = divider.append_axes("right", size="5%", pad=0.1)
           cbar = fig.colorbar(im, cax=cax)
           cbar.ax.tick_params(labelsize=8)
           # Scientific notation for intensity if numbers are huge
           cbar.formatter.set_powerlimits((0, 0))
          
       else:
           # Hide unused axes
           ax.axis("off")
           ax.set_visible(False)

   plt.show()
