# Developer Guide

This guide is intended for developers who want to understand the architecture of SOLWEIG-GPU and contribute to its development.

## Reporting issues and getting support

- **Bug reports and feature requests**: Please open an [issue on GitHub](https://github.com/nvnsudharsan/SOLWEIG-GPU/issues). Use the [bug report](https://github.com/nvnsudharsan/SOLWEIG-GPU/issues/new?template=bug_report.md) or [feature request](https://github.com/nvnsudharsan/SOLWEIG-GPU/issues/new?template=feature_request.md) templates when applicable so we can help you faster.
- **Questions and discussion**: For usage questions, ideas, or general discussion, you can open a [GitHub Discussion](https://github.com/nvnsudharsan/SOLWEIG-GPU/discussions) or use the repository’s issue tracker with the “Question” label.

## Architecture Overview

SOLWEIG-GPU is a modular package that separates the different stages of the thermal comfort modeling process. The main components are:

-   **Input Data Builder** (`create_inputs.py`, new in v2): Downloads and builds the required input rasters and meteorological data from near-globally available datasets via `build_inputs()`.
-   **Wind Extension Coefficients** (`wind_ext_coeff.py`, new in v2): Computes direction-based wind-extension coefficient rasters (GLIDE-SOL scheme) via `build_wind_ext_coeff()`.
-   **Data Preprocessing**: Handles the validation, tiling, and extraction of input data.
-   **Geometry Processing**: Calculates wall heights and aspects from the input rasters.
-   **Core SOLWEIG Model**: The main radiation and thermal comfort calculation engine (including UTCI and WBGT), accelerated with PyTorch.
-   **Interfaces**: Provides a command-line interface (CLI) for user interaction. (The graphical user interface was removed in Version 2.)

## Pipeline Stages

The workflow is implemented as four callable stages in `solweig_gpu.solweig_gpu`:

1. **`preprocess(...)`** – Validates rasters, creates tiles (including optional wind coefficient tiles via `windcoeff_folder`), and prepares metfiles (from your met file or ERA5/WRF; with `use_uhi=True` the ERA5 path also writes the diagnostic UHII). Writes to `{base_path}/processed_inputs/` (or a custom `preprocess_dir`). Returns the path to the preprocessing directory.

2. **`run_walls_aspect(preprocess_dir)`** – Computes wall heights and aspects for all tiles. Writes to `{preprocess_dir}/walls/` and `{preprocess_dir}/aspect/`. Call this after `preprocess()`.

3. **`calculate_svf(base_path, patch_option=2, overwrite=False)`** – Computes standalone Sky View Factor outputs for all tiles in the preprocessing directory. Writes to `{preprocess_dir}/SVF/`. Existing outputs are skipped unless `overwrite=True`.

4. **`run_utci_tiles(base_path, preprocess_dir, selected_date_str, ...)`** – Runs the SOLWEIG/UTCI (and optional WBGT) computation per tile and writes GeoTIFFs to `{base_path}/output_folder/{tile_key}/`. Optional argument `tile_keys` lets you run only specific tiles (e.g. `["0_0", "1000_0"]`). Call this after the previous stages.

The high-level **`thermal_comfort(...)`** function calls these stages in order (optionally preceded by `build_wind_ext_coeff()` when `ERA_5_z0_find=True`). You can use `thermal_comfort()` for one-shot runs, or call the stage functions separately when you need to:

- Run preprocessing once and then run UTCI for a subset of tiles.
- Reuse the same preprocessed tiles with different parameters.

See the [API Reference](api_reference.md) for full parameter lists and examples of calling stages separately.

## Module Interactions

The `solweig_gpu.py` module provides the public entry points (`thermal_comfort`, `preprocess`, `build_inputs`, `build_wind_ext_coeff`, `run_walls_aspect`, `calculate_svf`, `run_utci_tiles`) and orchestrates the pipeline. The `utci_process.py` module is the core of the computation, bringing together the various components to calculate the final UTCI (and WBGT, via `calculate_wbgt.py`) values.

## Contributing to SOLWEIG-GPU

We welcome contributions to the SOLWEIG-GPU project. To contribute, please follow these steps:

1.  **Fork the Repository**: Create your own fork of the [SOLWEIG-GPU repository](https://github.com/nvnsudharsan/SOLWEIG-GPU) on GitHub.
2.  **Create a Feature Branch**: Create a new branch for your feature or bug fix.
3.  **Set up for development**: Clone your fork and install with the test extra: `pip install -e ".[test]"`. See [Installation](installation.md).
4.  **Run the test suite** before submitting: `pytest tests/ -q`. See the [Testing Guide](testing.md).
5.  **Commit Your Changes**: Make your changes and commit them with clear and concise commit messages.
6.  **Push to Your Branch**: Push your changes to your forked repository.
7.  **Create a Pull Request**: Open a pull request from your branch to the `main` branch of the original repository.

Please ensure that your pull requests are small and focused on a single feature or bug fix. For the full contributor guide including reporting issues, see [CONTRIBUTING](https://github.com/nvnsudharsan/SOLWEIG-GPU/blob/main/CONTRIBUTING.md) in the repository root.

