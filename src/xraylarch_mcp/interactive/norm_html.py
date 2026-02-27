"""Generate interactive HTML for XANES normalization parameter selection.

Produces a self-contained HTML file that uses Plotly.js for interactive
plots and implements the pre_edge normalization algorithm in JavaScript
for real-time slider updates.

Workflow:
    1. MCP tool generates the HTML file with embedded spectrum data
    2. User opens the HTML in a browser
    3. User adjusts sliders to optimize normalization parameters
    4. User clicks "Save Parameters" → downloads a JSON file
    5. User tells Claude to apply the saved parameters
"""

from __future__ import annotations

import json

import numpy as np


def generate_norm_html(
    energy: np.ndarray,
    mu: np.ndarray,
    e0: float,
    pre1: float,
    pre2: float,
    norm1: float,
    norm2: float,
    nnorm: int,
    edge_step: float,
    label: str,
    group_id: str,
) -> str:
    """Generate a self-contained HTML file for interactive normalization.

    Parameters
    ----------
    energy : array
        Energy array in eV.
    mu : array
        Absorption coefficient array.
    e0 : float
        Initial edge energy (eV).
    pre1, pre2 : float
        Initial pre-edge fit bounds relative to E0 (eV).
    norm1, norm2 : float
        Initial post-edge fit bounds relative to E0 (eV).
    nnorm : int
        Post-edge polynomial degree (1-3).
    edge_step : float
        Initial edge step from larch's pre_edge.
    label : str
        Spectrum label for the plot title.
    group_id : str
        Group ID to embed in the saved parameters.

    Returns
    -------
    str
        Complete HTML document as a string.
    """
    energy_json = json.dumps(energy.tolist())
    mu_json = json.dumps(mu.tolist())

    # Compute slider ranges from data
    e_min = float(energy.min())
    e_max = float(energy.max())
    e_min_rel = e_min - e0
    e_max_rel = e_max - e0

    return _HTML_TEMPLATE.format(
        label=label,
        group_id=group_id,
        energy_json=energy_json,
        mu_json=mu_json,
        e0=e0,
        pre1=pre1,
        pre2=pre2,
        norm1=norm1,
        norm2=norm2,
        nnorm=nnorm,
        edge_step=edge_step,
        e0_min=e0 - 30,
        e0_max=e0 + 30,
        pre1_min=max(e_min_rel, -1000),
        pre1_max=-5,
        pre2_min=max(e_min_rel, -200),
        pre2_max=-2,
        norm1_min=5,
        norm1_max=min(e_max_rel, 800),
        norm2_min=50,
        norm2_max=e_max_rel,
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interactive XANES Normalization: {label}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f0f2f5; padding: 16px; color: #333;
  }}
  h1 {{
    font-size: 1.3em; margin-bottom: 12px; color: #1a1a2e;
    border-bottom: 2px solid #4361ee; padding-bottom: 6px;
  }}
  .plots {{
    display: flex; gap: 8px; width: 100%;
    background: white; border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1); padding: 8px;
  }}
  .plot {{ flex: 1; min-height: 350px; }}
  .controls {{
    background: white; padding: 16px 20px; border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-top: 12px;
  }}
  .controls h2 {{ font-size: 1em; margin-bottom: 10px; color: #4361ee; }}
  .slider-row {{
    display: flex; align-items: center; margin: 6px 0; gap: 8px;
  }}
  .slider-row label {{
    width: 70px; font-weight: 600; font-size: 0.9em; color: #555;
  }}
  .slider-row input[type=range] {{ flex: 1; cursor: pointer; }}
  .slider-row input[type=number] {{
    width: 100px; padding: 3px 6px; border: 1px solid #ccc;
    border-radius: 4px; font-family: 'Consolas', monospace; font-size: 0.85em;
    text-align: right;
  }}
  .slider-row .unit {{ font-size: 0.8em; color: #888; width: 30px; }}
  .bottom-row {{
    display: flex; gap: 12px; margin-top: 12px; align-items: flex-start;
  }}
  .params-box {{
    flex: 1; background: #1e1e2e; color: #a6e3a1; padding: 14px;
    border-radius: 8px; font-family: 'Consolas', monospace; font-size: 0.85em;
    white-space: pre; line-height: 1.6; min-height: 140px;
  }}
  .actions {{ display: flex; flex-direction: column; gap: 8px; }}
  .btn {{
    padding: 10px 24px; border: none; border-radius: 6px;
    font-size: 0.95em; font-weight: 600; cursor: pointer; transition: all 0.15s;
  }}
  .btn-save {{
    background: #4361ee; color: white;
  }}
  .btn-save:hover {{ background: #3451d1; }}
  .btn-reset {{
    background: #e0e0e0; color: #333;
  }}
  .btn-reset:hover {{ background: #ccc; }}
  .btn-copy {{
    background: #2d6a4f; color: white;
  }}
  .btn-copy:hover {{ background: #1b4332; }}
  .status {{
    font-size: 0.8em; color: #888; margin-top: 4px; min-height: 1.2em;
  }}
</style>
</head>
<body>

<h1>Interactive XANES Normalization: {label}</h1>

<div class="plots">
  <div class="plot" id="plot-mu"></div>
  <div class="plot" id="plot-norm"></div>
</div>

<div class="controls">
  <h2>Normalization Parameters</h2>

  <div class="slider-row">
    <label>E0</label>
    <input type="range" id="sl-e0" min="{e0_min}" max="{e0_max}" value="{e0}" step="0.1">
    <input type="number" id="num-e0" value="{e0}" step="0.1">
    <span class="unit">eV</span>
  </div>
  <div class="slider-row">
    <label>pre1</label>
    <input type="range" id="sl-pre1" min="{pre1_min}" max="{pre1_max}" value="{pre1}" step="1">
    <input type="number" id="num-pre1" value="{pre1}" step="1">
    <span class="unit">eV</span>
  </div>
  <div class="slider-row">
    <label>pre2</label>
    <input type="range" id="sl-pre2" min="{pre2_min}" max="{pre2_max}" value="{pre2}" step="1">
    <input type="number" id="num-pre2" value="{pre2}" step="1">
    <span class="unit">eV</span>
  </div>
  <div class="slider-row">
    <label>norm1</label>
    <input type="range" id="sl-norm1" min="{norm1_min}" max="{norm1_max}" value="{norm1}" step="1">
    <input type="number" id="num-norm1" value="{norm1}" step="1">
    <span class="unit">eV</span>
  </div>
  <div class="slider-row">
    <label>norm2</label>
    <input type="range" id="sl-norm2" min="{norm2_min}" max="{norm2_max}" value="{norm2}" step="5">
    <input type="number" id="num-norm2" value="{norm2}" step="5">
    <span class="unit">eV</span>
  </div>
  <div class="slider-row">
    <label>nnorm</label>
    <input type="range" id="sl-nnorm" min="1" max="3" value="{nnorm}" step="1">
    <input type="number" id="num-nnorm" value="{nnorm}" step="1" min="1" max="3">
    <span class="unit"></span>
  </div>
</div>

<div class="bottom-row">
  <div class="params-box" id="params-display"></div>
  <div class="actions">
    <button class="btn btn-save" onclick="saveParams()">Save Parameters (JSON)</button>
    <button class="btn btn-copy" onclick="copyParams()">Copy to Clipboard</button>
    <button class="btn btn-reset" onclick="resetParams()">Reset to Defaults</button>
    <div class="status" id="status-msg"></div>
  </div>
</div>

<script>
// ============================================================
// Embedded spectrum data
// ============================================================
const energy = {energy_json};
const mu = {mu_json};
const GROUP_ID = "{group_id}";

// Default parameter values (from larch pre_edge auto-detection)
const DEFAULTS = {{
  e0: {e0}, pre1: {pre1}, pre2: {pre2},
  norm1: {norm1}, norm2: {norm2}, nnorm: {nnorm}
}};

// ============================================================
// Polynomial least-squares fitting (normal equations + pivoting)
// ============================================================
function polyfit(x, y, degree) {{
  const n = x.length;
  const m = degree + 1;
  if (n < m) return null;

  // Build normal equations: (X^T X) a = X^T y
  const XTX = Array.from({{length: m}}, () => Array(m).fill(0));
  const XTy = Array(m).fill(0);

  for (let k = 0; k < n; k++) {{
    let xi = 1;
    for (let i = 0; i < m; i++) {{
      XTy[i] += xi * y[k];
      let xj = 1;
      for (let j = 0; j < m; j++) {{
        XTX[i][j] += xi * xj;
        xj *= x[k];
      }}
      xi *= x[k];
    }}
  }}

  // Gaussian elimination with partial pivoting
  const aug = XTX.map((row, i) => [...row, XTy[i]]);
  for (let col = 0; col < m; col++) {{
    let maxVal = Math.abs(aug[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < m; row++) {{
      if (Math.abs(aug[row][col]) > maxVal) {{
        maxVal = Math.abs(aug[row][col]);
        maxRow = row;
      }}
    }}
    if (maxVal < 1e-15) return null; // singular
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    for (let row = col + 1; row < m; row++) {{
      const f = aug[row][col] / aug[col][col];
      for (let j = col; j <= m; j++) aug[row][j] -= f * aug[col][j];
    }}
  }}

  // Back substitution
  const c = Array(m).fill(0);
  for (let i = m - 1; i >= 0; i--) {{
    c[i] = aug[i][m];
    for (let j = i + 1; j < m; j++) c[i] -= aug[i][j] * c[j];
    c[i] /= aug[i][i];
  }}
  return c; // c[0] + c[1]*x + c[2]*x^2 + ...
}}

function polyeval(coeffs, x) {{
  let val = 0, xp = 1;
  for (let i = 0; i < coeffs.length; i++) {{
    val += coeffs[i] * xp;
    xp *= x;
  }}
  return val;
}}

function polyevalArr(coeffs, xArr) {{
  return xArr.map(x => polyeval(coeffs, x));
}}

// ============================================================
// Pre-edge normalization (mirrors larch's pre_edge algorithm)
// ============================================================
function computeNorm(e0, pre1, pre2, norm1, norm2, nnorm) {{
  // Work in (E - E0) coordinates for numerical stability
  const erel = energy.map(e => e - e0);

  // Pre-edge: linear fit in [pre1, pre2]
  const preIdx = [];
  for (let i = 0; i < erel.length; i++) {{
    if (erel[i] >= pre1 && erel[i] <= pre2) preIdx.push(i);
  }}
  if (preIdx.length < 2) return null;
  const preX = preIdx.map(i => erel[i]);
  const preY = preIdx.map(i => mu[i]);
  const preC = polyfit(preX, preY, 1);
  if (!preC) return null;

  // Post-edge: polynomial of degree nnorm in [norm1, norm2]
  const postIdx = [];
  for (let i = 0; i < erel.length; i++) {{
    if (erel[i] >= norm1 && erel[i] <= norm2) postIdx.push(i);
  }}
  if (postIdx.length < nnorm + 1) return null;
  const postX = postIdx.map(i => erel[i]);
  const postY = postIdx.map(i => mu[i]);
  const postC = polyfit(postX, postY, nnorm);
  if (!postC) return null;

  // Edge step = post(E0) - pre(E0), where E-E0 = 0
  const edge_step = postC[0] - preC[0];
  if (Math.abs(edge_step) < 1e-10) return null;

  // Evaluate pre/post at all energies
  const pre_line = erel.map(er => polyeval(preC, er));
  const post_line = erel.map(er => polyeval(postC, er));

  // Normalized mu
  const norm = mu.map((m, i) => (m - pre_line[i]) / edge_step);

  // Flattened mu
  const flat = norm.map((n, i) => {{
    if (erel[i] <= 0) return n;
    const denom = (post_line[i] - pre_line[i]) / edge_step;
    if (Math.abs(denom) < 1e-10) return n;
    return n / denom;
  }});

  return {{ pre_line, post_line, norm, flat, edge_step }};
}}

// ============================================================
// Slider / input sync
// ============================================================
const params = ['e0', 'pre1', 'pre2', 'norm1', 'norm2', 'nnorm'];

function getVal(name) {{
  return parseFloat(document.getElementById('num-' + name).value);
}}

function getAllParams() {{
  const p = {{}};
  params.forEach(name => p[name] = getVal(name));
  return p;
}}

params.forEach(name => {{
  const sl = document.getElementById('sl-' + name);
  const num = document.getElementById('num-' + name);
  sl.addEventListener('input', () => {{
    num.value = sl.value;
    update();
  }});
  num.addEventListener('change', () => {{
    sl.value = num.value;
    update();
  }});
}});

// ============================================================
// Plotly initialization
// ============================================================
const muTrace = {{ x: energy, y: mu, name: '\u03BC(E)', line: {{ color: '#4361ee', width: 1.8 }} }};
const preTrace = {{ x: energy, y: [], name: 'pre-edge', line: {{ color: '#2d6a4f', width: 1.5, dash: 'dash' }} }};
const postTrace = {{ x: energy, y: [], name: 'post-edge', line: {{ color: '#c1121f', width: 1.5, dash: 'dash' }} }};

const normTrace = {{ x: energy, y: [], name: 'normalized', line: {{ color: '#4361ee', width: 1.8 }} }};
const flatTrace = {{ x: energy, y: [], name: 'flattened', line: {{ color: '#e76f51', width: 1.5 }} }};

const muLayout = {{
  title: {{ text: 'Raw \u03BC(E)', font: {{ size: 14 }} }},
  xaxis: {{ title: 'Energy (eV)' }},
  yaxis: {{ title: '\u03BC(E)' }},
  shapes: [],
  margin: {{ t: 40, b: 50, l: 60, r: 20 }},
  showlegend: true,
  legend: {{ x: 0.02, y: 0.98, font: {{ size: 10 }} }},
}};

const normLayout = {{
  title: {{ text: 'Normalized', font: {{ size: 14 }} }},
  xaxis: {{ title: 'Energy (eV)' }},
  yaxis: {{ title: 'Normalized \u03BC(E)' }},
  shapes: [],
  margin: {{ t: 40, b: 50, l: 60, r: 20 }},
  showlegend: true,
  legend: {{ x: 0.02, y: 0.98, font: {{ size: 10 }} }},
}};

const plotConfig = {{ responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }};

Plotly.newPlot('plot-mu', [muTrace, preTrace, postTrace], muLayout, plotConfig);
Plotly.newPlot('plot-norm', [normTrace, flatTrace], normLayout, plotConfig);

// ============================================================
// Update function
// ============================================================
function update() {{
  const p = getAllParams();
  const result = computeNorm(p.e0, p.pre1, p.pre2, p.norm1, p.norm2, p.nnorm);

  if (!result) {{
    document.getElementById('params-display').textContent =
      'Error: could not compute normalization.\nCheck parameter ranges.';
    return;
  }}

  // Update mu plot traces
  Plotly.restyle('plot-mu', {{ y: [result.pre_line] }}, [1]);
  Plotly.restyle('plot-mu', {{ y: [result.post_line] }}, [2]);

  // Update shapes: E0 line + pre/post shading
  const yMin = Math.min(...mu);
  const yMax = Math.max(...mu);
  muLayout.shapes = [
    {{ type: 'line', x0: p.e0, x1: p.e0, y0: yMin, y1: yMax,
      line: {{ color: '#888', width: 1.5, dash: 'dot' }} }},
    {{ type: 'rect', x0: p.e0 + p.pre1, x1: p.e0 + p.pre2, y0: yMin, y1: yMax,
      fillcolor: 'rgba(45,106,79,0.08)', line: {{ width: 0 }} }},
    {{ type: 'rect', x0: p.e0 + p.norm1, x1: p.e0 + p.norm2, y0: yMin, y1: yMax,
      fillcolor: 'rgba(193,18,31,0.08)', line: {{ width: 0 }} }},
  ];
  Plotly.relayout('plot-mu', {{ shapes: muLayout.shapes }});

  // Update norm plot
  Plotly.restyle('plot-norm', {{ y: [result.norm] }}, [0]);
  Plotly.restyle('plot-norm', {{ y: [result.flat] }}, [1]);

  normLayout.title.text = `Normalized (edge_step = ${{result.edge_step.toFixed(4)}})`;
  normLayout.shapes = [
    {{ type: 'line', x0: p.e0, x1: p.e0, y0: -0.1, y1: 1.5,
      line: {{ color: '#888', width: 1.5, dash: 'dot' }} }},
    {{ type: 'line', x0: energy[0], x1: energy[energy.length-1], y0: 0, y1: 0,
      line: {{ color: '#aaa', width: 0.8, dash: 'dot' }} }},
    {{ type: 'line', x0: energy[0], x1: energy[energy.length-1], y0: 1, y1: 1,
      line: {{ color: '#aaa', width: 0.8, dash: 'dot' }} }},
  ];
  Plotly.relayout('plot-norm', {{ shapes: normLayout.shapes, title: normLayout.title }});

  // Update parameter display
  const paramText = [
    `group_id  : "${{GROUP_ID}}"`,
    `e0        : ${{p.e0.toFixed(2)}}`,
    `pre1      : ${{p.pre1.toFixed(1)}}`,
    `pre2      : ${{p.pre2.toFixed(1)}}`,
    `norm1     : ${{p.norm1.toFixed(1)}}`,
    `norm2     : ${{p.norm2.toFixed(1)}}`,
    `nnorm     : ${{p.nnorm}}`,
    `edge_step : ${{result.edge_step.toFixed(6)}}`,
  ].join('\n');
  document.getElementById('params-display').textContent = paramText;
}}

// ============================================================
// Save / Copy / Reset
// ============================================================
function buildParamsJSON() {{
  const p = getAllParams();
  const result = computeNorm(p.e0, p.pre1, p.pre2, p.norm1, p.norm2, p.nnorm);
  return JSON.stringify({{
    group_id: GROUP_ID,
    e0: p.e0,
    pre1: p.pre1,
    pre2: p.pre2,
    norm1: p.norm1,
    norm2: p.norm2,
    nnorm: p.nnorm,
    edge_step: result ? result.edge_step : null,
  }}, null, 2);
}}

function saveParams() {{
  const json = buildParamsJSON();
  const blob = new Blob([json], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `norm_params_${{GROUP_ID}}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
  showStatus('Parameters saved! Tell Claude: "apply normalization parameters from the downloaded file"');
}}

function copyParams() {{
  const json = buildParamsJSON();
  navigator.clipboard.writeText(json).then(() => {{
    showStatus('Copied to clipboard! Paste the JSON to Claude.');
  }}).catch(() => {{
    // Fallback: select the params display text
    const el = document.getElementById('params-display');
    const range = document.createRange();
    range.selectNodeContents(el);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    showStatus('Copy failed \u2014 text selected, use Ctrl+C.');
  }});
}}

function resetParams() {{
  params.forEach(name => {{
    document.getElementById('sl-' + name).value = DEFAULTS[name];
    document.getElementById('num-' + name).value = DEFAULTS[name];
  }});
  update();
  showStatus('Reset to default parameters.');
}}

function showStatus(msg) {{
  const el = document.getElementById('status-msg');
  el.textContent = msg;
  setTimeout(() => {{ el.textContent = ''; }}, 5000);
}}

// Initial render
update();
</script>
</body>
</html>"""
