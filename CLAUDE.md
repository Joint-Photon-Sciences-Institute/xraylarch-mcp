# CLAUDE.md — xraylarch-mcp Builder

## Mission

Build a production-quality MCP (Model Context Protocol) server that wraps **xraylarch** (the headless X-ray spectroscopy analysis library) so that Claude can load, process, analyze, and plot X-ray spectra through natural tool calls. The package should be pip-installable and usable via Claude Desktop or Claude Code with zero configuration beyond `pip install`.

## Repository

```
git remote: https://github.com/Joint-Photon-Sciences-Institute/xraylarch-mcp.git
branch: main
```

---

## Phase 0: Probe xraylarch (DO THIS FIRST)

Before writing any MCP code, you must understand the xraylarch API by introspecting it directly. Do NOT rely on your training data — larch's API evolves and your knowledge may be stale.

### 0.1 Install xraylarch (headless)

```bash
pip install xraylarch
# Do NOT install GUI dependencies — we only need the computational core
```

### 0.2 Discover the API surface

Run Python introspection to catalog functions, their signatures, and docstrings. Focus on these critical modules:

```python
import larch
from larch import Interpreter
session = Interpreter()

# Core I/O — discover all readers
import larch.io
# List all public functions: dir(larch.io)
# For each reader (read_ascii, read_xdi, read_athena, read_gsexdi, etc.):
#   - inspect.signature(func)
#   - func.__doc__

# XAFS analysis — the most important module
from larch.xafs import (
    pre_edge, autobk, xftf, xftr, xftf_fast,
    feffit, feffit_dataset, feffit_transform, feffpath,
    fluo_corr, over_absorption, mback, mback_norm,
    estimate_noise, pre_edge_baseline
)
# For each: get signature, docstring, required/optional params, defaults

# XRF analysis
from larch.xrf import xrf_background, xrf_calib_energy, xrf_calib_compute

# Fitting
from larch.fitting import param, param_group, guess, minimize, confidence_intervals

# XES (X-ray emission spectroscopy)  
# Check if larch.xes exists, catalog it

# Math/utilities
from larch.math import (
    index_of, index_nearest, interp, smooth, deriv,
    pca_train, pca_fit, nmf_train
)

# FEFF interface
from larch.xafs import feffpath, feffit, feffit_dataset, feffit_transform
```

### 0.3 Catalog the Group object

The `Group` is larch's central data container. After operations, results are stored as attributes on the Group. You MUST understand what attributes each function adds:

```python
# Example: after pre_edge(group), group gains these attributes:
# .e0, .edge_step, .pre_edge, .post_edge, .norm, .flat, .pre_edge_details
# 
# After autobk(group): .chi, .k, .kwin, .rbkg, .bkg, etc.
# After xftf(group): .r, .chir, .chir_mag, .chir_re, .chir_im, .kwin, etc.
#
# Document ALL attribute additions for each major function.
```

### 0.4 Catalog supported file formats

Probe `larch.io` thoroughly:
- `.xdi` files (XAS Data Interchange standard)
- Generic ASCII columnar data (most beamline data)
- Athena `.prj` project files (contain multiple groups + parameters)
- GSECARS formats
- SpecFile formats  
- Any other readers you discover

For each reader, document: accepted extensions, return type, key column-name conventions.

### 0.5 Test canonical workflows end-to-end

Before building the MCP, verify these workflows actually run:

```python
# Workflow 1: XANES normalization
import numpy as np
from larch import Interpreter
from larch.io import read_ascii  # or whichever reader works
from larch.xafs import pre_edge

session = Interpreter()
# Try loading a test/example file if larch ships any
# Or create synthetic data for testing:
group = session.create_group()
group.energy = np.linspace(7000, 7300, 500)
group.mu = np.arctan((group.energy - 7112) / 10)  # synthetic Fe K-edge
pre_edge(group, _larch=session)
# Verify: group.norm, group.e0, group.edge_step exist

# Workflow 2: EXAFS extraction  
from larch.xafs import autobk, xftf
autobk(group, rbkg=1.0, _larch=session)
xftf(group, kmin=2, kmax=12, dk=2, window='kaiser', _larch=session)
# Verify: group.k, group.chi, group.r, group.chir_mag exist

# Workflow 3: Plotting (matplotlib, headless)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(group.energy, group.norm)
fig.savefig('/tmp/test_xanes.png', dpi=150, bbox_inches='tight')
```

Record any issues, deprecation warnings, or API changes you discover. Use this information to build the MCP tools correctly.

---

## Phase 1: Project Structure

Create this exact structure:

```
xraylarch-mcp/
├── README.md                    # Installation + usage + Claude Desktop config
├── LICENSE                      # BSD-3 (matching xraylarch's license)
├── pyproject.toml               # Modern Python packaging
├── src/
│   └── xraylarch_mcp/
│       ├── __init__.py
│       ├── __main__.py          # Entry point: python -m xraylarch_mcp
│       ├── server.py            # FastMCP server definition + tool registration
│       ├── session.py           # Larch Interpreter session + Group state manager
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── io_tools.py      # load_spectrum, list_groups, inspect_group, read_athena_project
│       │   ├── xafs_tools.py    # normalize, extract_exafs, fourier_transform, back_transform
│       │   ├── fitting_tools.py # feffit, lcf, peak_fit
│       │   ├── plot_tools.py    # plot_spectrum (returns base64 PNG or saves file)
│       │   ├── math_tools.py    # smooth, derivative, interpolate, rebin, merge_groups
│       │   └── xes_tools.py     # XES-specific tools (if larch supports it)
│       ├── resources/
│       │   ├── __init__.py
│       │   └── docs.py          # MCP resources serving larch documentation
│       └── util.py              # Shared helpers: error formatting, group serialization
├── tests/
│   ├── test_io.py
│   ├── test_xafs.py
│   ├── test_plot.py
│   └── conftest.py              # Shared fixtures (synthetic spectra, temp dirs)
└── examples/
    ├── claude_desktop_config.json
    └── example_prompts.md       # Example user prompts + expected behavior
```

---

## Phase 2: Implementation Requirements

### 2.1 Session Manager (`session.py`)

This is the most critical piece. It manages a persistent larch `Interpreter` and a dictionary of named `Group` objects.

```python
# Key design:
# - Single Interpreter instance created at server startup via lifespan
# - Groups stored in dict[str, Group], keyed by user-friendly ID
# - Group IDs auto-generated from filename (e.g., "fe_foil" from "fe_foil.xdi")
# - Collision handling: append _1, _2, etc.
# - Methods: add_group, get_group, list_groups, remove_group
# - Serialization: method to summarize a Group's attributes for tool responses
#   (list available arrays with their shapes, scalar values, etc.)
```

### 2.2 I/O Tools (`io_tools.py`)

```
larch_load_spectrum(filepath, format="auto", group_id=None)
  - Auto-detect format from extension (.xdi, .dat, .chi, .prj, etc.)
  - Return: group_id, detected columns, energy range, number of points
  - If .prj (Athena project): list contained groups, let user pick or load all

larch_read_athena_project(filepath)
  - List all groups in an Athena .prj file with their labels and parameters

larch_list_groups()
  - Return all loaded group IDs with summary info (what's been computed)

larch_inspect_group(group_id)
  - Return detailed attribute listing: arrays (name, shape, min, max),
    scalars (e0, edge_step, rbkg, etc.), and what processing has been done

larch_remove_group(group_id)
  - Free memory
```

### 2.3 XAFS Tools (`xafs_tools.py`)

```
larch_normalize(group_id, e0=None, pre1=None, pre2=None, norm1=None, norm2=None, nnorm=None, nvict=None)
  - Wraps pre_edge()
  - Return: e0, edge_step, pre_edge_details summary
  - IMPORTANT: Expose all pre_edge kwargs that the probed API reveals

larch_autobk(group_id, rbkg=1.0, kweight=2, kmin=0, kmax=None, ...)
  - Wraps autobk()
  - Return: k-range achieved, rbkg used, chi statistics

larch_xftf(group_id, kmin=2, kmax=None, dk=2, kweight=2, window="kaiser", ...)
  - Forward Fourier transform
  - Return: R-range, first-shell peak position estimate

larch_xftr(group_id, rmin, rmax, dr=0.1, window="hanning", ...)
  - Back Fourier transform

larch_fluo_corr(group_id, formula, elem, edge, ...)
  - Fluorescence self-absorption correction

larch_mback_norm(group_id, ...)
  - MBACK normalization (alternative to pre_edge for problematic data)

larch_estimate_noise(group_id, ...)
  - Noise estimation in chi(k)
```

### 2.4 Plot Tools (`plot_tools.py`)

```
larch_plot(group_id, plot_type, options={})
  plot_type: "mu" | "norm" | "flat" | "dmude" | "chi_k" | "chi_r" | "chi_r_mag" | "chi_q"
  options:
    - kweight: int (for chi plots)
    - xmin, xmax: float
    - ymin, ymax: float  
    - title: str
    - overlay_group_ids: list[str]  # plot multiple groups together
    - show_e0: bool
    - show_pre_post_edge: bool (show fit lines on mu plot)
    - figsize: tuple
    - dpi: int

  Returns: base64-encoded PNG image OR filepath to saved PNG
  MUST use matplotlib with Agg backend (headless)
```

### 2.5 Math Tools (`math_tools.py`)

```
larch_smooth(group_id, array_name, sigma=None, ...)
larch_deriv(group_id, array_name)
larch_interpolate(group_id, new_energy_grid)  
larch_merge_groups(group_ids, master_energy_grid=None)
  - Merge/average multiple spectra
larch_deglitch(group_id, ...)
  - Remove glitch points
larch_rebin(group_id, ...)
  - Rebin to uniform energy grid
```

### 2.6 Fitting Tools (`fitting_tools.py`)

```
larch_lcf(group_id, standard_group_ids, emin, emax, ...)
  - Linear combination fitting
  - Return: weights, R-factor, fitted spectrum

larch_pca(group_ids, ...)
  - PCA on a set of spectra
  - Return: eigenvalues, component spectra, scree plot data

larch_feffit(group_id, paths_config, transform_params, ...)
  - Full FEFFIT shell fitting
  - This is complex — expose the high-level interface
  - Return: fit statistics, best-fit parameters, R-factor
```

### 2.7 Resources (`resources/docs.py`)

Expose larch documentation as MCP resources that Claude can read on demand:

```python
@mcp.resource("larch://docs/xafs")
@mcp.resource("larch://docs/io")  
@mcp.resource("larch://docs/fitting")
@mcp.resource("larch://docs/feff")
@mcp.resource("larch://docs/examples/xanes")
@mcp.resource("larch://docs/examples/exafs")
```

**Generate these resource contents from the actual docstrings and signatures you probed in Phase 0.** This is the "skill" layer baked into the MCP — Claude reads these to understand how to use the tools correctly.

### 2.8 Run Code Escape Hatch

```
larch_run_code(code: str)
  - Execute arbitrary Python code in the larch session context
  - The session interpreter, all loaded groups, and numpy/matplotlib are available
  - Return: stdout capture + any result
  - This is the power-user tool for anything not covered by structured tools
```

---

## Phase 3: Key Implementation Details

### 3.1 The `_larch` parameter

Most larch functions require `_larch=session` where `session` is an `Interpreter()`. Your session manager must pass this to every call. Probe whether newer larch versions have changed this pattern.

### 3.2 Error handling

Larch functions can fail silently or raise obscure errors. Wrap every tool call in try/except and return actionable error messages:
- "Column 'mu' not found in data. Available columns: energy, i0, it, ir. You may need to calculate mu = log(i0/it)."
- "E0 auto-detection failed — the data may not contain an absorption edge in the provided energy range. Try setting e0 manually."
- "rbkg=1.0 may be too large for this system — nearest-neighbor distance is ~2.0 Å, so try rbkg ≤ 1.0."

### 3.3 Group serialization for responses

When returning group info, don't dump raw numpy arrays. Return structured summaries:
```json
{
  "group_id": "fe_foil",
  "arrays": {
    "energy": {"shape": [500], "min": 7000.0, "max": 7300.0, "unit": "eV"},
    "mu": {"shape": [500], "min": 0.1, "max": 1.8},
    "norm": {"shape": [500], "min": -0.05, "max": 1.3}
  },
  "scalars": {"e0": 7112.0, "edge_step": 1.2},
  "processing": ["pre_edge", "autobk", "xftf"]
}
```

### 3.4 Plotting

- ALWAYS use `matplotlib.use('Agg')` before importing pyplot
- Return images as base64 data URIs so Claude can display them inline
- Use sensible defaults: publication-quality font sizes, axis labels with units
- Energy axis: label as "Energy (eV)" 
- k-axis: label as "k (Å⁻¹)"
- R-axis: label as "R (Å)"
- chi axis: label as "χ(k)·k² (Å⁻²)" (adjust for kweight)
- Add proper legends when overlaying multiple spectra

### 3.5 Lifespan management

Use FastMCP's lifespan to create the Interpreter once and share it:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def larch_lifespan():
    from larch import Interpreter
    session = Interpreter()
    state = SessionManager(session)
    yield {"session": state}
    # cleanup

mcp = FastMCP("xraylarch_mcp", lifespan=larch_lifespan)
```

---

## Phase 4: Packaging

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[project]
name = "xraylarch-mcp"
version = "0.1.0"
description = "MCP server for X-ray spectroscopy analysis via xraylarch"
readme = "README.md"
license = {text = "BSD-3-Clause"}
requires-python = ">=3.9"
dependencies = [
    "xraylarch",
    "mcp[cli]",
    "numpy",
    "matplotlib",
]

[project.scripts]
xraylarch-mcp = "xraylarch_mcp.__main__:main"
```

### README.md must include

1. One-paragraph description
2. `pip install xraylarch-mcp`
3. Claude Desktop config JSON (copy-paste ready):
   ```json
   {
     "mcpServers": {
       "xraylarch": {
         "command": "python",
         "args": ["-m", "xraylarch_mcp"]
       }
     }
   }
   ```
4. Claude Code config: `claude mcp add xraylarch -- python -m xraylarch_mcp`
5. Quick example: "Ask Claude: 'Load the spectrum at ~/data/fe_foil.xdi and show me the normalized XANES'"
6. Full tool reference table
7. Link to xraylarch docs

---

## Phase 5: Testing

Write tests using pytest. Use synthetic spectra (arctangent edge, known E0) to test:
- Loading and group management
- Normalization produces expected e0 and edge_step
- autobk produces chi(k) of expected length
- xftf produces R-space data
- Plotting doesn't crash and produces valid PNG bytes
- Error cases: missing file, wrong column names, invalid parameters

---

## Phase 6: Git & Push

```bash
cd xraylarch-mcp
git init
git add .
git commit -m "Initial implementation: xraylarch MCP server with XAFS/XES/plotting tools"
git remote add origin https://github.com/Joint-Photon-Sciences-Institute/xraylarch-mcp.git
git branch -M main
git push -u origin main
```

---

## Critical Reminders

1. **Probe first, code second.** The entire Phase 0 must complete before you write any tool code. Your training data about larch may be wrong.
2. **Every tool must work.** Run each tool at least once with synthetic data before considering it done.
3. **The `_larch` parameter is non-negotiable.** If you forget it, larch functions silently fail or produce garbage.
4. **Group state is the whole point.** The session must persist groups across tool calls. Without this, users have to reload data every time.
5. **Headless only.** No wxPython, no GUI imports. If larch tries to import wx on startup, find the headless entry point.
6. **Return actionable information.** Don't just return "success" — return the scientifically meaningful values (e0, edge_step, k-range, R-factor, etc.)
7. **Naming convention**: All tool names start with `larch_` prefix.
8. **Transport**: stdio only (local execution). No HTTP server needed.
9. **Python >= 3.9** compatibility.
10. **Do not hallucinate larch API calls.** If you can't verify a function exists via introspection, don't wrap it.
