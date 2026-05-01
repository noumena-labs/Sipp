#!/usr/bin/env node
/**
 * visualize_ops.mjs — Parse one or more test-backend-ops perf logs and generate
 * an interactive HTML comparison chart.
 *
 * Usage:
 *   node scripts/visualize_ops.mjs <file1> [file2] [-o output.html]
 *
 * If a single file contains multiple backends (e.g. CUDA and WebGPU in one run),
 * they will all be extracted automatically.  If you ran each backend separately
 * and have two log files, pass both.
 *
 * Examples:
 *   node scripts/visualize_ops.mjs conv2d_test_cuda conv2d_test -o conv2d_comparison.html
 *   node scripts/visualize_ops.mjs combined_results.log
 */

import { readFileSync, writeFileSync } from 'fs';
import { basename } from 'path';

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
let outputPath = 'ops_benchmark_comparison.html';
const inputFiles = [];

for (let i = 0; i < args.length; i++) {
  if (args[i] === '-o' || args[i] === '--output') {
    outputPath = args[++i];
  } else {
    inputFiles.push(args[i]);
  }
}

if (inputFiles.length === 0) {
  console.error('Usage: node visualize_ops.mjs <file1> [file2] [-o output.html]');
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Parser — extract backend → { label: gflops } from a log file
// ---------------------------------------------------------------------------
function parseLog(logPath) {
  const content = readFileSync(logPath, 'utf8');
  const lines = content.split(/\r?\n/);

  const backends = {};       // backendName → { label → gflops }
  let currentBackend = null;

  for (const rawLine of lines) {
    // Strip ANSI escape sequences everywhere
    const line = rawLine.replace(/\x1b\[[0-9;]*m/g, '');

    // Detect backend header: "Backend 1/2: CUDA0" or "Backend 1/2: WebGPU"
    const backendMatch = line.match(/Backend \d+\/\d+:\s+(.*)/);
    if (backendMatch) {
      let name = backendMatch[1].trim();
      // Normalise CUDA0/CUDA1 → CUDA
      if (/^CUDA\d*$/.test(name)) name = 'CUDA';
      currentBackend = name;
      if (!backends[currentBackend]) backends[currentBackend] = {};
      continue;
    }

    // Skip non-perf lines or if we haven't hit a backend header yet
    if (!currentBackend) continue;

    // Match any op line:
    //   OP_NAME(params):   N runs - T us/run - X MFLOP/run -  YYY.YY GFLOPS
    //   OP_NAME(params):   N runs - T us/run - X kB/run -  YYY.YY GB/s
    const m = line.match(
      /^\s*([A-Z0-9_]+\([^)]*\)):\s+\d+\s+runs\s+-\s+[\d.]+\s+us\/run\s+-\s+[\d.]+\s+[\w/]+\s+-\s+([\d.]+)\s+([A-Z/s]+)/
    );
    if (m) {
      const rawLabel = m[1];
      const val      = parseFloat(m[2]);
      const unit     = m[3];

      // Build a short, readable label by extracting only the most
      // distinguishing parameters.
      const shortLabel = shortenLabel(rawLabel);
      backends[currentBackend][shortLabel] = { val, unit };
    }
  }

  return backends;
}

/**
 * Turn a verbose op string like
 *   CONV_2D(ne_input=[19,19,256,16],ne_kernel=[4,4,256,4096],type_kernel=f32,stride0=1,...)
 * into something chart-friendly:
 *   CONV_2D in[19,19,256,16] k[4,4,256,4096] f32
 */
function shortenLabel(raw) {
  const opNameMatch = raw.match(/^([A-Z0-9_]+)\(/);
  const opName = opNameMatch ? opNameMatch[1] : raw;

  // Extract specific known keys
  const get = (key) => {
    const r = new RegExp(key + '=\\[?([^\\],]+(?:,[^\\],]+)*)]?');
    const m = raw.match(r);
    return m ? m[1] : null;
  };

  const parts = [opName];
  const input  = get('ne_input');
  const kernel = get('ne_kernel');
  const type   = get('type_kernel') || get('type');
  if (input)  parts.push(`in[${input}]`);
  if (kernel) parts.push(`k[${kernel}]`);
  if (type)   parts.push(type);

  // If none of the known keys matched, just strip ne_input= etc inline
  if (parts.length === 1) {
    return raw
      .replace(/ne_input=/g, 'in=')
      .replace(/ne_kernel=/g, 'k=')
      .replace(/type_kernel=/g, 't=')
      .replace(/,stride0=1,stride1=1/g, '')
      .replace(/,padding0=0,padding1=0/g, '')
      .replace(/,dilation0=1,dilation1=1/g, '')
      .replace(/,cwhn=0/g, '');
  }
  return parts.join(' ');
}

// ---------------------------------------------------------------------------
// Merge backends from all input files
// ---------------------------------------------------------------------------
const allBackends = {};  // backendName → { label → gflops }

for (const f of inputFiles) {
  const parsed = parseLog(f);
  for (const [beName, entries] of Object.entries(parsed)) {
    if (!allBackends[beName]) allBackends[beName] = {};
    Object.assign(allBackends[beName], entries);
  }
}

// Drop backends with no perf data (e.g. "CPU — Skipping")
for (const be of Object.keys(allBackends)) {
  if (Object.keys(allBackends[be]).length === 0) delete allBackends[be];
}

const backendNames = Object.keys(allBackends);
if (backendNames.length < 2) {
  console.error(`Found ${backendNames.length} backend(s): [${backendNames.join(', ')}]`);
  console.error('Need at least 2 backends to compare. Pass more log files or check the data.');
  process.exit(1);
}

// Collect the union of all labels, ordered by the first backend's insertion order
const labelSet = new Set();
for (const be of backendNames) {
  for (const l of Object.keys(allBackends[be])) labelSet.add(l);
}
const labels = Array.from(labelSet);

// Build per-backend data arrays
const datasets = backendNames.map(be => labels.map(l => allBackends[be][l]?.val ?? 0));

// Determine unit (assume same for all, pick first found)
let unit = 'GFLOPS';
for (const be of backendNames) {
  for (const l of labels) {
    if (allBackends[be][l]?.unit) {
      unit = allBackends[be][l].unit;
      break;
    }
  }
  if (unit !== 'GFLOPS') break;
}

// Compute speedup ratios (first backend / second backend)
const speedups = labels.map((_, i) => {
  const a = datasets[0][i];
  const b = datasets[1][i];
  return b > 0 ? (a / b).toFixed(1) : '—';
});

// ---------------------------------------------------------------------------
// Colour palette
// ---------------------------------------------------------------------------
const palette = [
  { bg: 'rgba(56, 189, 248, 0.85)',  border: 'rgb(56, 189, 248)'  },  // sky
  { bg: 'rgba(250, 204, 21, 0.85)',  border: 'rgb(250, 204, 21)'  },  // amber
  { bg: 'rgba(167, 139, 250, 0.85)', border: 'rgb(167, 139, 250)' },  // violet
  { bg: 'rgba(52, 211, 153, 0.85)',  border: 'rgb(52, 211, 153)'  },  // emerald
];

// ---------------------------------------------------------------------------
// Generate HTML
// ---------------------------------------------------------------------------
const chartDatasets = backendNames.map((name, i) => ({
  label: name,
  data: datasets[i],
  backgroundColor: palette[i % palette.length].bg,
  borderColor: palette[i % palette.length].border,
  borderWidth: 1,
}));

const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Backend Ops Comparison — ${backendNames.join(' vs ')}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      margin: 0; padding: 2rem;
      font-family: 'Inter', system-ui, sans-serif;
      background: #0c0f1a; color: #e2e8f0;
    }
    .card {
      max-width: 1400px; margin: 0 auto;
      background: linear-gradient(145deg, #161b2e 0%, #1a1f35 100%);
      border-radius: 16px; padding: 2rem 2.5rem;
      box-shadow: 0 8px 32px rgba(0,0,0,.45);
    }
    h1 { text-align: center; font-weight: 700; font-size: 1.6rem;
         background: linear-gradient(90deg, #38bdf8, #a78bfa);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         margin: 0 0 .25rem; }
    .subtitle { text-align: center; color: #94a3b8; font-size: .85rem; margin-bottom: 1.5rem; }
    .chart-wrap { position: relative; width: 100%; height: ${Math.max(500, labels.length * 38)}px; }

    /* Summary table */
    table { width: 100%; border-collapse: collapse; margin-top: 2rem; font-size: .82rem; }
    th, td { padding: .45rem .6rem; text-align: right; border-bottom: 1px solid #1e293b; }
    th { color: #94a3b8; font-weight: 600; position: sticky; top: 0; background: #161b2e; }
    td:first-child, th:first-child { text-align: left; }
    tr:hover td { background: rgba(56,189,248,.06); }
    .speedup { font-weight: 600; }
    .speedup.fast { color: #34d399; }
    .speedup.slow { color: #f87171; }
  </style>
</head>
<body>
<div class="card">
  <h1>Backend Performance — ${backendNames.join(' vs ')}</h1>
  <p class="subtitle">${unit} · higher is better · RTX 3070 Laptop GPU</p>

  <div class="chart-wrap"><canvas id="chart"></canvas></div>

  <table>
    <thead>
      <tr>
        <th>Configuration</th>
        ${backendNames.map(n => `<th>${n} (${unit})</th>`).join('\n        ')}
        <th>Speedup (${backendNames[0]}/${backendNames[1]})</th>
      </tr>
    </thead>
    <tbody>
      ${labels.map((l, i) => {
        const sp = speedups[i];
        const cls = sp === '—' ? '' : parseFloat(sp) >= 2 ? 'fast' : parseFloat(sp) < 1 ? 'slow' : '';
        return `<tr>
        <td>${l}</td>
        ${datasets.map(d => `<td>${d[i].toFixed(2)}</td>`).join('')}
        <td class="speedup ${cls}">${sp}×</td>
      </tr>`;
      }).join('\n      ')}
    </tbody>
  </table>
</div>

<script>
new Chart(document.getElementById('chart'), {
  type: 'bar',
  data: {
    labels: ${JSON.stringify(labels)},
    datasets: ${JSON.stringify(chartDatasets)}
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#e2e8f0', font: { family: 'Inter', size: 13 } } },
      tooltip: {
        callbacks: {
          label: ctx => ctx.dataset.label + ': ' + ctx.parsed.x.toFixed(2) + ' ${unit}'
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        grid: { color: 'rgba(255,255,255,.06)' },
        ticks: { color: '#94a3b8', font: { family: 'Inter' } },
        title: { display: true, text: '${unit} (higher is better)', color: '#cbd5e1',
                 font: { family: 'Inter', size: 13 } }
      },
      y: {
        grid: { display: false },
        ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
      }
    }
  }
});
</script>
</body>
</html>`;

writeFileSync(outputPath, html, 'utf8');
console.log(`✓ Wrote ${outputPath}  (${backendNames.length} backends, ${labels.length} test configs)`);
