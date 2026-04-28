import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, '..');
const logsDir = path.join(rootDir, 'logs');
const outFile = path.join(logsDir, 'dashboard.html');

console.log('Generating dashboard...');

const files = fs.readdirSync(logsDir).filter(f => f.endsWith('.json'));
const benchmarks = {};

for (const file of files) {
  const content = fs.readFileSync(path.join(logsDir, file), 'utf-8');
  try {
    benchmarks[file] = JSON.parse(content);
  } catch(e) {
    console.warn(`Failed to parse ${file}`);
  }
}

// NOTE: We use template literals for the HTML. 
// We must ensure that the benchmarks data is correctly injected.
const htmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cogent Benchmark Visualizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
       * { box-sizing: border-box; }
       body {
           margin: 0; font-family: 'Inter', sans-serif;
           background: radial-gradient(circle at top, #1e1e2f, #0c0c14);
           color: #fff; min-height: 100vh; padding: 2rem;
       }
       .header { text-align: center; margin-bottom: 2rem; }
       .header h1 { font-weight: 800; text-transform: uppercase; letter-spacing: 2px;
           background: -webkit-linear-gradient(45deg, #0cebeb, #20e3b2, #29ffc6);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;}
       .header p { margin: 0; color: #94a3b8; font-size: 1.1rem; }
       .select-container { display: flex; justify-content: center; margin-bottom: 2rem; }
       select {
           padding: 10px 20px; font-size: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2);
           background: rgba(255,255,255,0.05); color: #fff; outline: none; cursor: pointer;
           backdrop-filter: blur(10px); min-width: 300px; text-align: center;
       }
       select option { background: #1e1e2f; color: #fff; }
       .dashboard {
           display: grid;
           grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
           gap: 1.5rem; max-width: 1400px; margin: 0 auto;
       }
       .glass-panel {
           background: rgba(255, 255, 255, 0.03);
           border: 1px solid rgba(255,255,255,0.08);
           border-radius: 16px;
           padding: 1.5rem;
           backdrop-filter: blur(12px);
           box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
           transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
       }
       .glass-panel:hover { transform: translateY(-4px); background: rgba(255, 255, 255, 0.05); }
       .panel-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 1.5rem; color: #e2e8f0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem; }
       .chart-container { position: relative; height: 300px; width: 100%; display: flex; justify-content: center; }
       
       .kpi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; height: 100%; }
       .kpi-card { background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 12px; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px solid rgba(255,255,255,0.02);}
       .kpi-val { font-size: 2.2rem; font-weight: 800; display: flex; align-items: baseline; gap: 4px; }
       .kpi-val span { font-size: 1rem; color: #94a3b8; font-weight: 600;}
       .kpi-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; margin-top: 0.5rem; letter-spacing: 1px;}
       
       .env-details { font-size: 0.95rem; color: #cbd5e1; line-height: 1.8; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
       .env-details p { margin: 0; background: rgba(0,0,0,0.2); padding: 10px 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.02); }
       .env-details span { color: #60a5fa; font-weight: 600; display: block; font-size: 1.05rem; margin-top: 2px;}
       
       .c-primary { color: #38bdf8; }
       .c-success { color: #4ade80; }
       .c-warning { color: #fbbf24; }
       .c-danger { color: #f87171; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CogentEngine Benchmarks</h1>
        <p>Interactive performance metrics dashboard</p>
    </div>
    
    <div class="select-container">
        <select id="viewSelector" onchange="renderView(this.value)">
            <option value="COMPARE_ALL">📊 Compare All Runs by Date</option>
            <optgroup label="Individual Runs" id="runOptions"></optgroup>
        </select>
    </div>

    <div class="dashboard" id="dashboard">
        <!-- Panels injected here -->
    </div>

    <script>
        const benchmarks = ${JSON.stringify(benchmarks)};
        const colors = ['#38bdf8', '#4ade80', '#fbbf24', '#f87171', '#c084fc', '#f472b6'];
        
        Chart.defaults.color = "#94a3b8";
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.plugins.tooltip.backgroundColor = "rgba(15, 23, 42, 0.95)";
        Chart.defaults.plugins.tooltip.titleColor = "#fff";
        Chart.defaults.plugins.tooltip.padding = 12;
        Chart.defaults.plugins.tooltip.cornerRadius = 8;
        
        let chartInstances = [];

        function formatVal(val) {
           return typeof val === 'number' && Number.isFinite(val) ? val.toFixed(2) : 'n/a';
        }

        function createPanel(title, id, isChart = true) {
            const colStyle = isChart ? '' : 'style="grid-column: 1 / -1;"';
            return \`
            <div class="glass-panel" \${colStyle}>
                <div class="panel-title">\${title}</div>
                \${isChart ? \`<div class="chart-container"><canvas id="\${id}"></canvas></div>\` : \`<div id="\${id}"></div>\`}
            </div>
            \`;
        }

        function readMetricValue(container, key, property = 'meanMs') {
            if (container[key] === undefined || container[key] === null) {
                return null;
            }
            if (property && typeof container[key] === 'object' && container[key] !== null) {
                const nested = container[key][property];
                return typeof nested === 'number' && Number.isFinite(nested) ? nested : null;
            }
            return typeof container[key] === 'number' && Number.isFinite(container[key]) ? container[key] : null;
        }

        function getMetric(summary, metricKey, property = 'meanMs') {
            if (!summary) return null;
            const serving = summary.serving || {};
            const runtime = summary.runtime || {};

            const lookup = {
                ttftMs: [
                    [serving, 'appObservedTtftMs'],
                    [serving, 'ttftMs'],
                    [runtime, 'nativeTtftMs'],
                ],
                tpotMs: [
                    [serving, 'appObservedTpotMs'],
                    [serving, 'tpotMs'],
                ],
                itlMs: [
                    [serving, 'appObservedItlMs'],
                    [serving, 'itlMs'],
                    [runtime, 'nativeMeanItlMs'],
                ],
                nativeDecodeTps: [
                    [runtime, 'nativeDecodeTokensPerSecond'],
                    [runtime, 'decodeTokensPerSecond'],
                    [runtime, 'tokensPerSecond'],
                ],
                outputThroughputTps: [
                    [serving, 'outputTokenThroughputTps'],
                ],
                totalThroughputTps: [
                    [serving, 'totalTokenThroughputTps'],
                ],
                nativeTtftMs: [
                    [runtime, 'nativeTtftMs'],
                ],
                nativeMeanItlMs: [
                    [runtime, 'nativeMeanItlMs'],
                ],
            };

            const entries = lookup[metricKey] || [[serving, metricKey], [runtime, metricKey]];
            for (const [container, key] of entries) {
                const value = readMetricValue(container, key, property);
                if (value !== null) {
                    return value;
                }
            }

            return null;
        }

        // Generate options
        const runs = Object.keys(benchmarks).map(k => ({ file: k, data: benchmarks[k] }));
        runs.sort((a, b) => new Date(a.data.generatedAt) - new Date(b.data.generatedAt)); // chronological

        const runOptions = document.getElementById('runOptions');
        runs.slice().reverse().forEach(run => {
            const opt = document.createElement('option');
            opt.value = run.file;
            const d = new Date(run.data.generatedAt);
            opt.text = d.toLocaleString() + ' (' + run.file.split('-').pop().split('.')[0] + ')';
            runOptions.appendChild(opt);
        });

        function renderView(value) {
            chartInstances.forEach(c => c.destroy());
            chartInstances = [];
            
            const dashboard = document.getElementById('dashboard');
            dashboard.innerHTML = '';

            if (value === 'COMPARE_ALL') {
                renderComparisonDashboard();
            } else {
                renderSingleRunDashboard(value);
            }
        }

        function renderComparisonDashboard() {
            const dashboard = document.getElementById('dashboard');
            
            const dateLabels = runs.map(r => {
                const d = new Date(r.data.generatedAt);
                return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + '\\n' + d.toLocaleDateString();
            });

            const getTrend = (scenarioId, metricKey, mode = 'hotReuseContext', property = 'meanMs') => {
                return runs.map(r => {
                    if (!r.data.scenarios) return null;
                    const s = r.data.scenarios.find(sc => sc.definition.id === scenarioId);
                    if (!s || !s[mode]) return null;
                    return getMetric(s[mode].summary, metricKey, property);
                });
            };

            const allScenarioIds = new Set();
            runs.forEach(r => {
                if (r.data.scenarios) r.data.scenarios.forEach(s => allScenarioIds.add(s.definition.id));
            });
            const scenarioIds = Array.from(allScenarioIds);

            dashboard.innerHTML += createPanel('Native Decode TPS Trend (Hot Reuse)', 'chart-trend-native-decode');
            dashboard.innerHTML += createPanel('End-to-End Output TPS Trend (Hot Reuse)', 'chart-trend-output-through');
            dashboard.innerHTML += createPanel('TTFT Trend (Hot Reuse)', 'chart-trend-ttft');

            setTimeout(() => {
                chartInstances.push(new Chart(document.getElementById('chart-trend-native-decode'), {
                    type: 'line',
                    data: {
                        labels: dateLabels,
                        datasets: scenarioIds.map((id, i) => ({
                            label: \`\${id.toUpperCase()} Native Decode\`,
                            data: getTrend(id, 'nativeDecodeTps'),
                            borderColor: colors[i % colors.length],
                            backgroundColor: colors[i % colors.length],
                            borderWidth: 2, tension: 0.3, pointRadius: 4
                        }))
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { title: {display: true, text: 'Tokens / Sec'}, grid: {color:'rgba(255,255,255,0.05)'} } } }
                }));

                chartInstances.push(new Chart(document.getElementById('chart-trend-output-through'), {
                    type: 'line',
                    data: {
                        labels: dateLabels,
                        datasets: scenarioIds.map((id, i) => ({
                            label: \`\${id.toUpperCase()} Output TPS\`,
                            data: getTrend(id, 'outputThroughputTps'),
                            borderColor: colors[i % colors.length],
                            backgroundColor: colors[i % colors.length],
                            borderWidth: 2, tension: 0.3, pointRadius: 4
                        }))
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { title: {display: true, text: 'Tokens / Sec'}, grid: {color:'rgba(255,255,255,0.05)'} } } }
                }));

                const ttftDatasets = [];
                scenarioIds.forEach((id, i) => {
                    ttftDatasets.push({
                        label: \`\${id.toUpperCase()} App\`,
                        data: getTrend(id, 'ttftMs'),
                        backgroundColor: colors[i % colors.length],
                        borderRadius: 4
                    });
                    const nativeData = getTrend(id, 'nativeTtftMs');
                    if (nativeData.some(v => v > 0)) {
                        ttftDatasets.push({
                            label: \`\${id.toUpperCase()} Native\`,
                            data: nativeData,
                            backgroundColor: colors[i % colors.length],
                            borderWidth: 1,
                            borderColor: '#fff',
                            borderRadius: 4,
                            borderDash: [2, 2]
                        });
                    }
                });

                chartInstances.push(new Chart(document.getElementById('chart-trend-ttft'), {
                    type: 'bar',
                    data: { labels: dateLabels, datasets: ttftDatasets },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { title: {display: true, text: 'Milliseconds'}, grid: {color:'rgba(255,255,255,0.05)'} }, x: {grid: {display: false}} } }
                }));
            }, 0);
        }

        function renderSingleRunDashboard(filename) {
            const data = benchmarks[filename];
            if (!data) return;
            const dashboard = document.getElementById('dashboard');
            const runDate = new Date(data.generatedAt).toLocaleString();

            dashboard.innerHTML += \`
                <div class="glass-panel" style="grid-column: 1 / -1;">
                    <div class="panel-title">Summary & Environment - <span style="color:#38bdf8">\${runDate}</span></div>
                    <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 1.5rem;">
                        <div id="kpi-container"></div>
                        <div id="env-container"></div>
                    </div>
                </div>
            \`;
            
            let bestNativeDecodeTps = null;
            let bestOutputThroughputTps = null;
            let bestTPOT = null;
            if (data.scenarios) {
                const primary = data.scenarios.find(s => s.definition.id === "siso") || data.scenarios[0];
                if (primary && primary.hotReuseContext) {
                    bestNativeDecodeTps = getMetric(primary.hotReuseContext.summary, 'nativeDecodeTps');
                    bestOutputThroughputTps = getMetric(primary.hotReuseContext.summary, 'outputThroughputTps');
                    bestTPOT = getMetric(primary.hotReuseContext.summary, 'tpotMs');
                }
            }
            
            document.getElementById('kpi-container').innerHTML = \`
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-val c-success">\${formatVal(bestNativeDecodeTps)} <span>tk/s</span></div>
                        <div class="kpi-label">Native Decode TPS</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-val c-primary">\${formatVal(bestOutputThroughputTps)} <span>tk/s</span></div>
                        <div class="kpi-label">Output TPS (Hot Reuse)</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-val c-warning">\${formatVal(bestTPOT)} <span>ms</span></div>
                        <div class="kpi-label">TPOT (Hot Reuse)</div>
                    </div>
                </div>
            \`;
            
            const env = data.environment || {};
            const backend = data.backend || {};
            document.getElementById('env-container').innerHTML = \`
                <div class="env-details">
                    <p>Model<span>\${data.source ? data.source.label : (data.modelSource ? data.modelSource.label : 'N/A')}</span></p>
                    <p>Browser<span>\${env.browserLabel ? env.browserLabel.split(' ')[0] : 'N/A'}</span></p>
                    <p>GPU Adapter<span>\${env.adapterVendor || ''} \${env.adapterArchitecture || 'N/A'}</span></p>
                    <p>Backend<span>\${backend.inferredExecutionBackend || 'N/A'} (\${backend.runtimeBackendStatus || 'N/A'})</span></p>
                </div>
            \`;

            if (data.scenarios) {
                const scenariosLabels = data.scenarios.map(s => s.definition.label);
                const getModeSeries = (mode, metricKey) =>
                    data.scenarios.map(s => getMetric(s[mode] ? s[mode].summary : null, metricKey));

                dashboard.innerHTML += createPanel('Time to First Token (TTFT)', 'chart-ttft');
                dashboard.innerHTML += createPanel('Time per Output Token (TPOT)', 'chart-tpot');
                dashboard.innerHTML += createPanel('Native Decode TPS', 'chart-native-decode');
                dashboard.innerHTML += createPanel('End-to-End Output TPS', 'chart-output-throughput');
                dashboard.innerHTML += createPanel('Memory Usage Profile', 'chart-memory');

                setTimeout(() => {
                    chartInstances.push(new Chart(document.getElementById('chart-ttft'), {
                        type: 'bar',
                        data: {
                            labels: scenariosLabels,
                            datasets: [
                                { label: 'Cold Context', data: getModeSeries('coldPrompt', 'ttftMs'), backgroundColor: '#38bdf8', borderRadius: 4 },
                                { label: 'Hot Fresh', data: getModeSeries('hotFreshContext', 'ttftMs'), backgroundColor: '#fbbf24', borderRadius: 4 },
                                { label: 'Hot Reuse', data: getModeSeries('hotReuseContext', 'ttftMs'), backgroundColor: '#4ade80', borderRadius: 4 }
                            ]
                        },
                        options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, grid: {color:'rgba(255,255,255,0.05)'} } } }
                    }));

                    chartInstances.push(new Chart(document.getElementById('chart-tpot'), {
                        type: 'bar',
                        data: {
                            labels: scenariosLabels,
                            datasets: [
                                { label: 'Cold Context', data: getModeSeries('coldPrompt', 'tpotMs'), backgroundColor: '#38bdf8', borderRadius: 4 },
                                { label: 'Hot Fresh', data: getModeSeries('hotFreshContext', 'tpotMs'), backgroundColor: '#fbbf24', borderRadius: 4 },
                                { label: 'Hot Reuse', data: getModeSeries('hotReuseContext', 'tpotMs'), backgroundColor: '#4ade80', borderRadius: 4 }
                            ]
                        },
                        options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, grid: {color:'rgba(255,255,255,0.05)'} } } }
                    }));

                    chartInstances.push(new Chart(document.getElementById('chart-native-decode'), {
                        type: 'line',
                        data: {
                            labels: scenariosLabels,
                            datasets: [
                                { label: 'Cold', data: getModeSeries('coldPrompt', 'nativeDecodeTps'), borderColor: '#38bdf8', tension: 0.3 },
                                { label: 'Hot Fresh', data: getModeSeries('hotFreshContext', 'nativeDecodeTps'), borderColor: '#fbbf24', tension: 0.3 },
                                { label: 'Hot Reuse', data: getModeSeries('hotReuseContext', 'nativeDecodeTps'), borderColor: '#4ade80', tension: 0.3 }
                            ]
                        },
                        options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, grid: {color:'rgba(255,255,255,0.05)'} } } }
                    }));

                    chartInstances.push(new Chart(document.getElementById('chart-output-throughput'), {
                        type: 'line',
                        data: {
                            labels: scenariosLabels,
                            datasets: [
                                { label: 'Cold', data: getModeSeries('coldPrompt', 'outputThroughputTps'), borderColor: '#38bdf8', tension: 0.3 },
                                { label: 'Hot Fresh', data: getModeSeries('hotFreshContext', 'outputThroughputTps'), borderColor: '#fbbf24', tension: 0.3 },
                                { label: 'Hot Reuse', data: getModeSeries('hotReuseContext', 'outputThroughputTps'), borderColor: '#4ade80', tension: 0.3 }
                            ]
                        },
                        options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, grid: {color:'rgba(255,255,255,0.05)'} } } }
                    }));

                    if (data.memory && data.memory.snapshots) {
                        chartInstances.push(new Chart(document.getElementById('chart-memory'), {
                            type: 'line',
                            data: {
                                labels: data.memory.snapshots.map(s => s.label),
                                datasets: [{ label: 'JS Heap (MB)', data: data.memory.snapshots.map(s => s.usedJsHeapBytes / (1024*1024)), borderColor: '#f472b6', tension: 0.3 }]
                            },
                            options: { responsive: true, maintainAspectRatio: false, scales: { y: { grid: {color:'rgba(255,255,255,0.05)'} } } }
                        }));
                    }
                }, 0);
            }
        }

        // INIT
        if (runs.length > 0) {
            if (runs.length > 1) {
                renderView('COMPARE_ALL');
            } else {
                document.getElementById('viewSelector').value = runs[0].file;
                renderView(runs[0].file);
            }
        } else {
            document.getElementById('dashboard').innerHTML = '<div class="panel-title" style="grid-column: 1/-1; text-align:center;">No benchmark files found.</div>';
        }
    </script>
</body>
</html>`;

fs.writeFileSync(outFile, htmlTemplate);
console.log('✅ Visualizer generated at ' + outFile);
