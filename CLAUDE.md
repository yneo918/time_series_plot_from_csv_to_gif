# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for creating animated time series visualizations. The project reads CSV data and generates animated GIF plots showing how data changes over time.

## Repository Structure

- `src/` - Main source code directory
- `data/` - Input CSV files for processing
- `output/` - Generated GIF animations and plots
- `LICENSE` - MIT license
- `.gitignore` - Python-specific gitignore with comprehensive exclusions

## Development Environment

This project uses Python and is configured for modern Python development practices:

- Uses standard Python .gitignore with support for multiple package managers (pip, poetry, pdm, uv)
- Configured to ignore common Python artifacts (\_\_pycache\_\_, .pyc files, virtual environments)
- Excludes IDE-specific files (.vscode/, .idea/, etc.)
- Supports Ruff for linting/formatting

## Expected Dependencies

Based on the project purpose, the codebase likely uses:
- `pandas` for CSV data manipulation
- `matplotlib` or `plotly` for plotting
- `imageio` or `pillow` for GIF creation
- `numpy` for numerical operations

## Development Commands

When implementing features, verify dependencies and check for existing configuration files like:
- `requirements.txt` or `pyproject.toml` for dependencies
- `ruff.toml` or similar for linting configuration
- Test configuration files (pytest.ini, tox.ini, etc.)

## Code Architecture

The project follows a typical data visualization pipeline:
1. CSV data ingestion and preprocessing
2. Time series data analysis and preparation
3. Frame-by-frame plot generation
4. GIF assembly and output

When adding new functionality, maintain separation between data processing, visualization, and file I/O operations.