# deltamsi

This repository contains a python package for High-Resolution Mass Spectrometry Imaging (MSI) analysis.

Developed during the PhD thesis *"High-Resolution Mass Spectrometry Imaging: From Mass-Shift Correction to Graph-Based Molecular Interpretation"* by Raphaël La Rocca, `deltamsi` provides a unified analysis framework for spatial metabolomics and lipidomics built on top of [AnnData](https://anndata.readthedocs.io/). The central `MSICube` object handles everything from loading raw `.imzML` files to preprocessing, peak picking, spatial analysis, and visualization.

<p align="center"><img src="Overview_deltamsi.png" width="680"></p>

Overview of the main functionalities implemented in deltamsi for high-resolution MSI processing and interpretation.(a) Raw MSI processing and data representation: conversion of raw MSI data into a datacube-like representation in Python, organized in an AnnData-backed structure for efficient access and metadata handling.(b) Recalibration: pixel-wise (or spectrum-wise) mass-shift correction to improve mass accuracy, illustrated by the reduction of m/z error and improved peak alignment, leading to cleaner and more consistent ion images after recalibration.(c) MSI preprocessing and feature selection: typical preprocessing steps (e.g., TIC normalization, scaling, image filtering, hot-spot removal) followed by intensity thresholding and spatial-structure scoring (chaos score) to support ion/feature selection.(d) Kendrick analysis: Kendrick mass defect visualization used for exploratory analysis and interactive selection/labeling of ions of interest, with corresponding ion images.(e) Graph-based m/z clustering: construction of an ion graph from expected mass differences (Δm) and/or spatial similarity (colocalization), followed by clustering to obtain molecular groups and cluster-level aggregated ion images.(f) Differential analysis between conditions: comparison of control versus experimental groups using fold-change metrics (intensity and/or spatial-structure/chaos-derived criteria), illustrated by ion images and fold-change direction.

## Features

- **Data loading** - import one or more `.imzML`/`.ibd` files into a single multi-sample object
- **Preprocessing** - TIC normalization, log-transformation, hotspot capping, median filtering, quantile thresholding
- **Spectral processing** - mean spectrum computation, peak picking, intensity matrix extraction
- **Spatial analysis** - cosine colocalization, spatial chaos scoring, Kendrick mass defect analysis
- **Mass recalibration** - internal recalibration against a reference database
- **Discriminant analysis** - rank ions between groups
- **Visualization** - ion images, mean spectrum plots, Kendrick diagrams
- **Persistence** - save/load in `h5ad` (HDF5) or `zarr` format

## Requirements

- Python `>=3.11, <3.13`
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation with uv

### Clone and set up the environment

```bash
git clone https://github.com/LaRoccaRaphael/deltamsi.git
cd deltamsi
uv sync
```

This creates a `.venv` and installs all core dependencies from `uv.lock`.

With optional extras:

```bash
# Visualization tools (matplotlib, seaborn, scanpy, plotly, jupyterlab, ...)
uv sync --extra viz

# Network analysis utilities
uv sync --extra analysis

# Multiple extras at once
uv sync --extra viz --extra analysis
```

For development (includes linting, testing, and docs tools):

```bash
uv sync --all-extras --dev
```

### Use in another project (once published to PyPI)

```bash
uv add deltamsi
uv add "deltamsi[viz]"          # with visualization extras
uv add "deltamsi[viz,analysis]" # multiple extras
```

### Install from a local checkout into another project

```bash
uv add /path/to/deltamsi
uv add "/path/to/deltamsi[viz]"
```

## Quick start

```python
import deltamsi as dm

# 1. Point to the directory containing your .imzML and .ibd files
cube = dm.MSICube("./data/my_experiment/")
# INFO: MSICube initialized with 3 samples found.

# 2. Load spectra into an AnnData object
cube.load_mean_spectrum()      # compute per-sample mean spectra
cube.combine_mean_spectra()    # align and merge into a global mean spectrum
cube.pick_peaks()              # detect peaks on the combined spectrum
cube.extract_peak_matrix()     # fill the intensity matrix (pixels × peaks)

# 3. Preprocess
cube.tic_normalize()
cube.log1p_intensity()

# 4. Explore
dm.plot_ion_images(cube.adata, mz_values=[756.5, 885.5], ncols=2)
dm.plot_mean_spectrum_windows(cube.adata)

# 5. Save
cube.save()                        # saves to ./data/my_experiment/adata.h5ad
cube.save(file_format="zarr")      # or zarr for large datasets
```

## Running tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src/deltamsi
```

## Building documentation

```bash
uv run --group dev sphinx-build docs/source docs/_build/html
```

## Project structure

```
src/deltamsi/
├── core/           # MSICube - the main analysis object
├── processing/     # Normalization, peak picking, colocalization, recalibration, ...
├── plotting/       # Ion images, spectra, Kendrick plots
├── params/         # Typed parameter / options dataclasses
└── utils/          # Validation helpers
```

## How to cite

If you use `deltamsi` in your research, please cite:

```bibtex
@software{deltamsi,
  author  = {La Rocca, Raphaël},
  title   = {deltamsi: A Scanpy-like Python package for High-Resolution Mass Spectrometry Imaging analysis},
  year    = {2026},
  url     = {https://github.com/LaRoccaRaphael/deltamsi},
}
```

## License

BSD 3-Clause - see [LICENSE](LICENSE).
