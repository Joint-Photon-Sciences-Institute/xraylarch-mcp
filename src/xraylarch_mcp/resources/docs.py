"""MCP resources serving larch documentation generated from API introspection."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.resource("larch://docs/xafs")
    def xafs_docs() -> str:
        """XAFS analysis functions reference."""
        return """# xraylarch XAFS Analysis Reference

## pre_edge(group, e0=None, pre1=None, pre2=None, norm1=None, norm2=None, nnorm=None, nvict=0)
Pre-edge subtraction and normalization. Determines E0, fits pre-edge line and
post-edge polynomial, computes normalized and flattened mu(E).

**Adds to group:** e0, edge_step, norm, flat, pre_edge, post_edge, dmude, d2mude,
atsym, edge, pre_edge_details (with pre1, pre2, norm1, norm2, nnorm, nvict)

**Parameters:**
- e0: Edge energy (eV). Auto-detected from max of d(mu)/dE if None.
- pre1, pre2: Pre-edge fit range relative to E0 (eV). Defaults: ~(E_start - E0) to ~pre1/3.
- norm1, norm2: Post-edge fit range relative to E0 (eV).
- nnorm: Post-edge polynomial degree (0-3). Auto: 2 if range>300eV, 1 if >30eV, else 0.
- nvict: Energy exponent for pre-edge. Default 0 (no victoreen).

## autobk(group, rbkg=1.0, kweight=2, kmin=0, kmax=None, nknots=None, clamp_lo=1, clamp_hi=1)
AUTOBK background removal. Extracts chi(k) from mu(E).

**Adds to group:** k, chi, bkg, chie, rbkg, autobk_details

**Parameters:**
- rbkg: R-space cutoff for background (Angstroms). Should be < nearest-neighbor distance.
- kweight: k-weighting for the fit. Default 2.
- kmin, kmax: k-range for fitting.
- clamp_lo, clamp_hi: Clamp values at low/high k (0=none, 1=moderate, 2=strong).

## xftf(group, kmin=0, kmax=20, kweight=2, dk=1, dk2=None, window='kaiser', nfft=2048, kstep=0.05)
Forward Fourier transform: chi(k) -> chi(R).

**Adds to group:** r, chir, chir_mag, chir_re, chir_im, kwin

**Parameters:**
- kmin, kmax: k-range for the FT window.
- dk: Window sill width.
- kweight: k-weighting power.
- window: Window function (kaiser, hanning, parzen, welch, gaussian, sine).

## xftr(group, rmin=0, rmax=10, dr=0.1, window='hanning')
Reverse Fourier transform: chi(R) -> chi(q). Back-transforms filtered R-range.

**Adds to group:** q, chiq, chiq_mag, chiq_re, chiq_im, rwin

## fluo_corr(group, formula, elem, edge='K', anginp=45, angout=45)
Fluorescence self-absorption correction (FLUO/Haskel algorithm).

## mback_norm(group, z=None, edge='K', e0=None, order=3)
MBACK normalization - matches post-edge to tabulated cross-sections.
Alternative to pre_edge() for problematic data.

## estimate_noise(group)
Estimates noise in chi(k) and chi(R).
**Adds:** epsilon_k, epsilon_r

## rebin_xafs(group, e0=None, pre1=None, pre2=None, pre_step=None, xanes_step=None, exafs_kstep=0.05)
Rebin to standard 3-region XAFS grid (pre-edge/XANES/EXAFS).
"""

    @mcp.resource("larch://docs/io")
    def io_docs() -> str:
        """I/O functions reference."""
        return """# xraylarch I/O Reference

## File Readers

### read_ascii(filename, labels=None, simple_labels=False, sort=False)
Read column ASCII file. Returns a Group with array_labels and data columns.
Works for most beamline data (.dat, .txt files).

### read_xdi(filename, labels=None)
Read XAS Data Interchange (XDI) format. Parses standard XDI headers.

### read_athena(filename, match=None, do_preedge=True, do_bkg=False, do_fft=False)
Read Athena project file (.prj). Returns a Group containing sub-Groups.
- do_preedge=True: auto-run pre_edge on each group
- do_bkg=True: also run autobk
- do_fft=True: also run xftf
- match: regex pattern to select specific groups

### read_csv(filename)
Read CSV file. Returns Group with columns as arrays.

### read_gsexdi(fname, nmca=128, bad=None)
Read GSECARS XDI fluorescence scan data.

### read_specfile(filename, scan=None)
Read Spec/BLISS file. Specify scan number to load specific scan.

## File Writers

### write_ascii(filename, *args, commentchar='#', label=None, header=None)
Write arrays to ASCII column file.

### groups2csv(grouplist, filename, x='energy', y='norm')
Export multiple groups to CSV.

## Utility

### guess_filereader(path)
Auto-detect the appropriate reader for a file.

### merge_groups(grouplist, xarray='energy', yarray='mu', kind='cubic', trim=True)
Merge/average spectra. Interpolates onto common grid and averages.

### is_athena_project(filename)
Test if file is an Athena project.
"""

    @mcp.resource("larch://docs/fitting")
    def fitting_docs() -> str:
        """Fitting functions reference."""
        return """# xraylarch Fitting Reference

## Linear Combination Fitting

### lincombo_fit(group, components, arrayname='norm', xmin=-inf, xmax=inf, sum_to_one=True, vary_e0=False)
Fit spectrum as weighted sum of standard spectra.
Returns result with weights, rfactor, chisqr, redchi.

### lincombo_fitall(group, components, arrayname='norm', xmin=-inf, xmax=inf, max_ncomps=None, sum_to_one=True)
Try all combinations of standards and rank by R-factor.

## PCA

### pca_train(groups, arrayname='norm', xmin=-inf, xmax=inf)
Train PCA model. Returns model with eigenvalues, variances, components.

### pca_fit(group, pca_model, ncomps=None)
Fit a spectrum to a PCA model.

## Peak Fitting

### fit_peak(x, y, model, background=None, form=None)
Fit 1D peak. Models: gaussian, lorentzian, voigt, pvoigt, pearson7, students_t.
Background: linear, constant, None.

## FEFFIT (EXAFS Shell Fitting)

### FeffPathGroup(filename, label='', s02=None, sigma2=None, deltar=None, e0=None, degen=None)
Create a Feff scattering path from a feffNNNN.dat file.

### TransformGroup(kmin=0, kmax=20, kweight=2, dk=4, window='kaiser', rmin=0, rmax=10, fitspace='r')
FT transform parameters for feffit.

### FeffitDataSet(data=None, paths=None, transform=None)
Dataset container for feffit.

### feffit(paramgroup, datasets)
Perform the fit. Returns result with chi_square, chi_reduced, rfactor, nvarys, n_independent.

### feffit_report(result)
Generate a printable fit report.

## Parameters

### param(value, vary=True, min=None, max=None, name=None)
Create a fitting parameter.

### param_group(**kws)
Create a group of parameters.

### minimize(fcn, paramgroup, method='leastsq')
General-purpose minimization wrapper.
"""

    @mcp.resource("larch://docs/feff")
    def feff_docs() -> str:
        """FEFF interface reference."""
        return """# xraylarch FEFF Interface Reference

## Running FEFF calculations

### feff6l(feffinp='feff.inp', folder='.', verbose=True)
Run Feff6l calculation.

### feff8l(feffinp='feff.inp', folder='.', module=None, verbose=True)
Run Feff8l calculation.

## Workflow

1. Prepare feff.inp (atomic cluster, potentials, etc.)
2. Run feff6l() or feff8l() to generate feffNNNN.dat files
3. Create FeffPathGroup objects from .dat files
4. Set up TransformGroup with k/R ranges
5. Create FeffitDataSet with data, paths, transform
6. Create param_group with guess parameters
7. Run feffit(params, [dataset])
8. Inspect results with feffit_report(result)

## Key concepts

- Each feffNNNN.dat file describes one scattering path
- Parameters: S0^2 (amplitude), sigma2 (Debye-Waller), deltar (path length change), e0 (energy shift)
- Fit in R-space (default), k-space, or q-space
- The number of independent points limits how many parameters you can fit
  (Nidp = 2*dk*dR/pi, where dk=kmax-kmin and dR=rmax-rmin)
"""

    @mcp.resource("larch://docs/examples/xanes")
    def xanes_example() -> str:
        """XANES analysis workflow example."""
        return """# XANES Analysis Workflow

## Step 1: Load data
Use larch_load_spectrum to load your XAS data file.
Supported formats: .xdi, .dat, .txt, .csv, .prj

## Step 2: Normalize
Use larch_normalize to perform pre-edge subtraction and normalization.
This determines E0, edge_step, and produces the normalized spectrum.

Key parameters:
- e0: set manually if auto-detection fails
- pre1, pre2: adjust pre-edge fit range (relative to E0)
- norm1, norm2: adjust post-edge fit range (relative to E0)

## Step 3: Inspect results
Use larch_inspect_group to see all computed arrays and scalars.

## Step 4: Plot
Use larch_plot with plot_type="norm" to visualize the normalized XANES.
Use plot_type="dmude" for the derivative, which shows edge features more clearly.
Use show_e0=True to mark the edge energy.

## Step 5: Compare
Use overlay_group_ids in larch_plot to compare multiple spectra.
Use larch_lcf for linear combination fitting against standards.

## Example prompt
"Load the spectrum at ~/data/fe_foil.xdi, normalize it, and show me the
XANES with the derivative overlaid."
"""

    @mcp.resource("larch://docs/interactive")
    def interactive_docs() -> str:
        """Interactive GUI tools reference."""
        return """# xraylarch Interactive Tools Reference

Interactive tools generate self-contained HTML pages that open in the user's
browser. This avoids blocking the MCP server and provides a responsive UI
using Plotly.js with real-time parameter adjustment.

## larch_interactive_norm

Generates an HTML page with interactive Plotly.js plots and sliders for
tuning normalization parameters. The page opens automatically in the
default browser.

**Parameters:**
- group_id: ID of the loaded spectrum (must have energy and mu arrays)
- e0: Initial edge energy (auto-detected if omitted)
- pre1, pre2: Initial pre-edge fit range relative to E0
- norm1, norm2: Initial post-edge fit range relative to E0
- nnorm: Initial post-edge polynomial degree (1-3)
- html_path: Optional path for the HTML file (temp file if omitted)

**Workflow:**
1. Load a spectrum with larch_load_spectrum
2. Call larch_interactive_norm with the group_id
3. An HTML page opens in the browser with Plotly plots and sliders
4. Adjust sliders (E0, pre1, pre2, norm1, norm2, nnorm) - plots update in real time
5. Click "Save Parameters" to download a JSON file, or "Copy to Clipboard"
6. Call larch_apply_norm_params with the downloaded JSON to apply the parameters

**What the page shows:**
- Left plot: raw mu(E) with pre-edge line, post-edge curve, and shaded fit regions
- Right plot: normalized and flattened spectrum with edge_step displayed
- Six sliders: E0, pre1, pre2, norm1, norm2, nnorm (with linked number inputs)
- Parameter display panel showing current values
- Save (JSON download), Copy to Clipboard, and Reset buttons

## larch_apply_norm_params

Reads normalization parameters from a JSON file (saved from the interactive
HTML tool) and applies them to a group via pre_edge().

**Parameters:**
- group_id: ID of the spectrum group
- params_path: Path to the JSON file from the interactive tool
- e0, pre1, pre2, norm1, norm2, nnorm: Optional overrides for file values
"""

    @mcp.resource("larch://docs/examples/exafs")
    def exafs_example() -> str:
        """EXAFS analysis workflow example."""
        return """# EXAFS Analysis Workflow

## Step 1: Load and normalize
Same as XANES workflow steps 1-2.

## Step 2: Extract EXAFS
Use larch_autobk to remove background and extract chi(k).
Key parameter: rbkg (should be < nearest-neighbor distance, typically 0.8-1.2 A).

## Step 3: Fourier transform
Use larch_xftf to transform chi(k) to chi(R).
Key parameters: kmin, kmax, dk, kweight, window.

## Step 4: Back transform (optional)
Use larch_xftr to filter specific shells and back-transform.

## Step 5: Plot
Use larch_plot with:
- plot_type="chi_k" for k-space (adjust kweight)
- plot_type="chi_r" for R-space (radial distribution)
- plot_type="chi_q" for back-transform

## Step 6: Noise estimation
Use larch_estimate_noise to assess data quality.

## Step 7: Fitting (optional)
Use larch_feffit with FEFF-calculated scattering paths.

## Example prompt
"Load the spectrum, normalize it, extract the EXAFS with rbkg=1.0,
do a Fourier transform from k=2 to k=12 with k-weight 2,
and show me chi(k) and chi(R) plots."
"""
