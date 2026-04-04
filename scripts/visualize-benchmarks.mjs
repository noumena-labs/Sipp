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
        
        Chart.defaults.color = "#94a3b8";
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.plugins.tooltip.backgroundColor = "rgba(15, 23, 42, 0.95)";
        Chart.defaults.plugins.tooltip.titleColor = "#fff";
        Chart.defaults.plugins.tooltip.padding = 12;
        Chart.defaults.plugins.tooltip.cornerRadius = 8;
        
        let chartInstances = [];

        function formatVal(val) {
           return typeof val === 'number' ? val.toFixed(2) : val;
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

        // Generate options
        const runs = Object.keys(benchmarks).map(k => ({ file: k, data: benchmarks[k] }));
        runs.sort((a, b) => new Date(a.data.generatedAt) - new Date(b.data.generatedAt)); // chronological

        const runOptions = document.getElementById('runOptions');
        runs.reverse().forEach(run => {
            const opt = document.createElement('option');
            opt.value = run.file;
            const d = new Date(run.data.generatedAt);
            opt.text = d.toLocaleString() + ' (' + run.file.split('-').pop().split('.')[0] + ')';
            runOptions.appendChild(opt);
        });
        // Back to chronological for comparison plots
        runs.reverse(); 

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

            // Extract scenario specific metric natively across all runs
            const getTrend = (scenarioId, metricObj, metricProperty) => {
                return runs.map(r => {
                    if (!r.data.scenarios) return 0;
                    const s = r.data.scenarios.find(sc => sc.definition.id === scenarioId);
                    if (!s || !s.hotReuseContext || !s.hotReuseContext.summary) return 0;
                    if (metricProperty) return s.hotReuseContext.summary.serving[metricObj][metricProperty];
                    return s.hotReuseContext.summary.serving[metricObj];
                });
            };

            dashboard.innerHTML += createPanel('Tokens Throughput Trend (Hot Reuse)', 'chart-trend-through');
            dashboard.innerHTML += createPanel('Time to First Token Trend (Hot Reuse)', 'chart-trend-ttft');
            dashboard.innerHTML += createPanel('Time per Output Token Trend (Hot Reuse)', 'chart-trend-tpot');

            setTimeout(() => {
                // Throughput Trends
                chartInstances.push(new Chart(document.getElementById('chart-trend-through'), {
                    type: 'line',
                    data: {
                        labels: dateLabels,
                        datasets: [
                            { label: 'SISO Throughput', data: getTrend('siso', 'outputTokenThroughputTps'), borderColor: '#38bdf8', backgroundColor: '#38bdf8', borderWidth: 2, tension: 0.3, pointRadius: 5 },
                            { label: 'SILO Throughput', data: getTrend('silo', 'outputTokenThroughputTps'), borderColor: '#4ade80', backgroundColor: '#4ade80', borderWidth: 2, tension: 0.3, pointRadius: 5 }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { title: {display: true, text: 'Tokens / Sec'}, grid: {color:'rgba(255,255,255,0.05)'} } } }
                }));

                // TTFT Trends
                chartInstances.push(new Chart(document.getElementById('chart-trend-ttft'), {
                    type: 'bar',
                    data: {
                        labels: dateLabels,
                        datasets: [
                            { label: 'SISO TTFT', data: getTrend('siso', 'ttftMs', 'meanMs'), backgroundColor: '#fbbf24', borderRadius: 4 },
                            { label: 'SILO TTFT', data: getTrend('silo', 'ttftMs', 'meanMs'), backgroundColor: '#f87171', borderRadius: 4 }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { title: {display: true, text: 'Milliseconds'}, grid: {color:'rgba(255,255,255,0.05)'} }, x: {grid: {display: false}} } }
                }));

                // TPOT Trends
                chartInstances.push(new Chart(document.getElementById('chart-trend-tpot'), {
                    type: 'bar',
                    data: {
                        labels: dateLabels,
                        datasets: [
                            { label: 'SISO TPOT', data: getTrend('siso', 'tpotMs', 'meanMs'), backgroundColor: '#c084fc', borderRadius: 4 },
                            { label: 'SILO TPOT', data: getTrend('silo', 'tpotMs', 'meanMs'), backgroundColor: '#4ade80', borderRadius: 4 }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { title: {display: true, text: 'Milliseconds'}, grid: {color:'rgba(255,255,255,0.05)'} }, x: {grid: {display: false}} } }
                }));
            }, 0);
        }

        function renderSingleRunDashboard(filename) {
            const data = benchmarks[filename];
            if (!data) return;
            const dashboard = document.getElementById('dashboard');
            
            const runDate = new Date(data.generatedAt).toLocaleString();

            const createFullWidthMetricsPanel = () => {
                return \`
                <div class="glass-panel" style="grid-column: 1 / -1;">
                    <div class="panel-title">Summary & Environment - <span style="color:#38bdf8">\${runDate}</span></div>
                    <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 1.5rem;">
                        <div id="kpi-container"></div>
                        <div id="env-container"></div>
                    </div>
                </div>
                \`;
            }
            dashboard.innerHTML += createFullWidthMetricsPanel();
            
            let sisoTPOT = "N/A";
            let sisoThroughput = "N/A";
            if (data.scenarios) {
                const siso = data.scenarios.find(s => s.definition.id === "siso");
                if (siso && siso.hotReuseContext) {
                    sisoTPOT = siso.hotReuseContext.summary.serving.tpotMs.meanMs;
                    sisoThroughput = siso.hotReuseContext.summary.serving.outputTokenThroughputTps;
                }
            }
            
            document.getElementById('kpi-container').innerHTML = \`
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-val c-success">\${formatVal(sisoThroughput)} <span>tk/s</span></div>
                        <div class="kpi-label">Throughput (Hot)</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-val c-warning">\${formatVal(sisoTPOT)} <span>ms</span></div>
                        <div class="kpi-label">TPOT (Hot Reuse)</div>
                    </div>
                </div>
            \`;
            
            const env = data.environment || {};
            const backend = data.backend || {};
            document.getElementById('env-container').innerHTML = \`
                <div class="env-details">
                    <p>Model<span>\${data.modelSource ? data.modelSource.label : 'N/A'}</span></p>
                    <p>Browser<span>\${env.browserLabel ? env.browserLabel.split(' ')[0] : 'N/A'}</span></p>
                    <p>GPU Adapter<span>\${env.adapterVendor || ''} \${env.adapterArchitecture || 'N/A'}</span></p>
                    <p>Execution Mode<span>\${backend.inferredExecutionBackend || 'N/A'}</span></p>
                </div>
            \`;

            const scenariosLabels = data.scenarios ? data.scenarios.map(s => s.definition.label) : [];
            
            const getMetric = (scenarios, mode, metricObj, metricProperty) => {
                return scenarios.map(s => {
                    const m = s[mode];
                    if (m && m.summary && m.summary.serving[metricObj]) {
                        return m.summary.serving[metricObj][metricProperty];
                    }
                    return 0;
                });
            };

            const getThroughput = (scenarios, mode) => {
                return scenarios.map(s => {
                    const m = s[mode];
                    if (m && m.summary && m.summary.serving) return m.summary.serving.outputTokenThroughputTps;
                    return 0;
                });
            };

            if (data.scenarios) {
                dashboard.innerHTML += createPanel('Time to First Token (TTFT)', 'chart-ttft');
                dashboard.innerHTML += createPanel('Time per Output Token (TPOT)', 'chart-tpot');
                dashboard.innerHTML += createPanel('Generation Throughput', 'chart-through');
                dashboard.innerHTML += createPanel('Memory Usage Profile', 'chart-memory');

                setTimeout(() => {
                    chartInstances.push(new Chart(document.getElementById('chart-ttft'), {
                        type: 'bar',
                        data: {
                            labels: scenariosLabels,
                            datasets: [
                                { label: 'Cold Context', data: getMetric(data.scenarios, 'coldPrompt', 'ttftMs', 'meanMs'), backgroundColor: '#38bdf8', borderRadius: 4 },
                                { label: 'Hot Fresh Context', data: getMetric(data.scenarios, 'hotFreshContext', 'ttftMs', 'meanMs'), backgroundColor: '#fbbf24', borderRadius: 4 },
                                { label: 'Hot Reuse Context', data: getMetric(data.scenarios, 'hotReuseContext', 'ttftMs', 'meanMs'), backgroundColor: '#4ade80', borderRadius: 4 }
                            ]
                        },
                        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { beginAtZero: true, title: {display: true, text: 'Milliseconds'}, grid: {color:'rgba(255,255,255,0.05)'} }, x: {grid: {display: false}} } }
                    }));

                    chartInstances.push(new Chart(document.getElementById('chart-tpot'), {
                        type: 'bar',
                        data: {
                            labels: scenariosLabels,
                            datasets: [
                                { label: 'Cold Context', data: getMetric(data.scenarios, 'coldPrompt', 'tpotMs', 'meanMs'), backgroundColor: '#38bdf8', borderRadius: 4 },
                                { label: 'Hot Fresh Context', data: getMetric(data.scenarios, 'hotFreshContext', 'tpotMs', 'meanMs'), backgroundColor: '#fbbf24', borderRadius: 4 },
                                { label: 'Hot Reuse Context', data: getMetric(data.scenarios, 'hotReuseContext', 'tpotMs', 'meanMs'), backgroundColor: '#4ade80', borderRadius: 4 }
                            ]
                        },
                        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { beginAtZero: true, title: {display: true, text: 'Milliseconds'}, grid: {color:'rgba(255,255,255,0.05)'} }, x: {grid: {display: false}} } }
                    }));

                    chartInstances.push(new Chart(document.getElementById('chart-through'), {
                        type: 'line',
                        data: {
                            labels: scenariosLabels,
                            datasets: [
                                { label: 'Cold Context', data: getThroughput(data.scenarios, 'coldPrompt'), borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.2)', fill: true, tension: 0.3, pointRadius: 5 },
                                { label: 'Hot Fresh Context', data: getThroughput(data.scenarios, 'hotFreshContext'), borderColor: '#fbbf24', backgroundColor: 'rgba(251, 191, 36, 0.2)', fill: true, tension: 0.3, pointRadius: 5 },
                                { label: 'Hot Reuse Context', data: getThroughput(data.scenarios, 'hotReuseContext'), borderColor: '#4ade80', backgroundColor: 'rgba(74, 222, 128, 0.2)', fill: true, tension: 0.3, pointRadius: 5 }
                            ]
                        },
                        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { beginAtZero: true, title: {display: true, text: 'Tokens / Sec'}, grid: {color:'rgba(255,255,255,0.05)'} }, x: {grid: {display: false}} } }
                    }));

                    if (data.memory && data.memory.snapshots) {
                        const memLabels = data.memory.snapshots.map(s => s.label);
                        const jsHeapBytes = data.memory.snapshots.map(s => s.usedJsHeapBytes / (1024*1024));
                        const userAgentBytes = data.memory.snapshots.map(s => s.userAgentBytes ? (s.userAgentBytes / (1024*1024)) : null);

                        chartInstances.push(new Chart(document.getElementById('chart-memory'), {
                            type: 'line',
                            data: {
                                labels: memLabels,
                                datasets: [
                                    { label: 'JS Heap (MB)', data: jsHeapBytes, borderColor: '#f472b6', backgroundColor: '#f472b6', borderWidth: 2, tension: 0.3, pointRadius: 4 },
                                    { label: 'Agent Specific (MB)', data: userAgentBytes, borderColor: '#c084fc', backgroundColor: '#c084fc', borderWidth: 2, tension: 0.3, borderDash: [5, 5], pointRadius: 4 }
                                ]
                            },
                            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } }, scales: { y: { beginAtZero: false, title: {display: true, text: 'Megabytes'}, grid: {color:'rgba(255,255,255,0.05)'} }, x: {grid: {display: false}} } }
                        }));
                    }
                }, 0);
            }
        }

        // INIT
        if (runs.length > 0) {
            // Default to Compare All if multiple exist
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
</html>\n`;

fs.writeFileSync(outFile, htmlTemplate);
console.log('✅ Visualizer generated at ' + outFile);
