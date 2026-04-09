# deltamsi

A Scanpy-like Python package for High-Resolution Mass Spectrometry Imaging (MSI) analysis.

`deltamsi` provides a unified analysis framework for spatial metabolomics and proteomics built on top of [AnnData](https://anndata.readthedocs.io/). The central `MSICube` object handles everything from loading raw `.imzML` files to preprocessing, peak picking, spatial analysis, and visualization.

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
git clone https://github.com/your-org/deltamsi.git
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

## License

BSD 3-Clause - see [LICENSE](LICENSE).
