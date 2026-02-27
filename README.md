# xraylarch-mcp

MCP (Model Context Protocol) server that wraps [xraylarch](https://xraypy.github.io/xraylarch/) for X-ray spectroscopy analysis. Load, process, analyze, and plot XAS/XAFS spectra through Claude Desktop or Claude Code with natural language tool calls. Supports XANES normalization, EXAFS extraction, Fourier transforms, linear combination fitting, PCA, FEFFIT shell fitting, and publication-quality plotting.

## Installation

```bash
pip install xraylarch-mcp
```

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

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

### Claude Code

```bash
claude mcp add xraylarch -- python -m xraylarch_mcp
```

## Quick Start

Ask Claude:

> "Load the spectrum at ~/data/fe_foil.xdi and show me the normalized XANES"

Claude will:
1. Call `larch_load_spectrum` to read your data
2. Call `larch_normalize` to perform pre-edge subtraction and normalization
3. Call `larch_plot` with `plot_type="norm"` to generate a publication-quality plot

## Tool Reference

### I/O Tools

| Tool | Description |
|------|-------------|
| `larch_load_spectrum` | Load a spectrum from file (.xdi, .dat, .csv, .prj, etc.) |
| `larch_read_athena_project` | Read/inspect Athena project files |
| `larch_list_groups` | List all loaded spectrum groups |
| `larch_inspect_group` | Detailed view of a group's arrays and scalars |
| `larch_remove_group` | Remove a group to free memory |

### XAFS Analysis Tools

| Tool | Description |
|------|-------------|
| `larch_normalize` | Pre-edge subtraction and normalization (wraps `pre_edge`) |
| `larch_autobk` | EXAFS background removal (AUTOBK algorithm) |
| `larch_xftf` | Forward Fourier transform: chi(k) -> chi(R) |
| `larch_xftr` | Back Fourier transform: chi(R) -> chi(q) |
| `larch_fluo_corr` | Fluorescence self-absorption correction |
| `larch_mback_norm` | MBACK normalization (alternative for problematic data) |
| `larch_estimate_noise` | Noise estimation in chi(k) and chi(R) |
| `larch_cauchy_wavelet` | Cauchy wavelet transform for simultaneous k- and R-resolution |
| `larch_rebin` | Rebin to standard 3-region XAFS energy grid |

### Plot Tools

| Tool | Description |
|------|-------------|
| `larch_plot` | Publication-quality plots (mu, norm, flat, dmude, chi_k, chi_r, chi_q, cauchy_wavelet) |

### Math Tools

| Tool | Description |
|------|-------------|
| `larch_smooth` | Smooth arrays with Lorentzian/Gaussian/Voigt kernel |
| `larch_deriv` | Numerical derivative |
| `larch_interpolate` | Interpolate onto new energy grid |
| `larch_merge_groups` | Merge/average multiple spectra |
| `larch_deglitch` | Remove glitch points |

### Fitting Tools

| Tool | Description |
|------|-------------|
| `larch_lcf` | Linear combination fitting against standards |
| `larch_pca` | Principal Component Analysis |
| `larch_feffit` | FEFFIT EXAFS shell fitting |
| `larch_peak_fit` | Peak fitting (Gaussian, Voigt, etc.) |

### Interactive Tools

| Tool | Description |
|------|-------------|
| `larch_interactive_norm` | Open an interactive HTML page to tune normalization parameters |
| `larch_apply_norm_params` | Apply normalization parameters from a saved JSON file |

#### Interactive normalization workflow

1. Ask Claude to interactively normalize your spectrum
2. An HTML page opens in your browser with Plotly.js plots and sliders
3. Adjust E0, pre-edge range, post-edge range, and polynomial degree — plots update in real time
4. Click **Save Parameters** to download a JSON file, or **Copy to Clipboard**
5. Tell Claude to apply the saved parameters

> "Load Pdfoil.prj and let me interactively choose the normalization parameters"

### Advanced

| Tool | Description |
|------|-------------|
| `larch_run_code` | Execute arbitrary Python code in the larch session |

## Resources

The server exposes documentation resources that Claude can read on demand:

- `larch://docs/xafs` — XAFS analysis reference
- `larch://docs/io` — I/O functions reference
- `larch://docs/fitting` — Fitting reference
- `larch://docs/feff` — FEFF interface reference
- `larch://docs/interactive` — Interactive tools reference
- `larch://docs/examples/xanes` — XANES workflow example
- `larch://docs/examples/exafs` — EXAFS workflow example

## Links

- [xraylarch documentation](https://xraypy.github.io/xraylarch/)
- [MCP specification](https://modelcontextprotocol.io/)
- [Source code](https://github.com/Joint-Photon-Sciences-Institute/xraylarch-mcp)

## License

BSD-3-Clause (matching xraylarch)
