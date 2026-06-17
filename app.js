// ═══════════════════════════════════════════════════════════
//  ASTRA Mobility Intelligence — Dashboard Controller v2
//  Flipkart Gridlock Hackathon 2.0 | Prototype Phase
// ═══════════════════════════════════════════════════════════

'use strict';

// ── App State ──────────────────────────────────────────────
let data              = null;
let map               = null;
let markersGroup      = null;
let mapBounds         = null;
let currentTsIdx      = 9;   // Default: 02:15
let isPlaying         = false;
let playInterval      = null;

// Charts
let globalFlowChart   = null;
let roadTypeChart     = null;
let localInspectChart = null;

// Route Simulator
let isDrawingRoute    = false;
let routePoints       = [];
let routePolyline     = null;
let routeMarkers      = [];

// Traffic Mitigation States
let activeMitigations = {}; // geohash -> type ('none', 'signal', 'divert', 'patrol')
let currentInspectedGeohash = null;

function getAdjustedDemand(gh, tsIdx) {
    if (!data || !data.geohashes[gh]) return 0;
    const info = data.geohashes[gh];
    const val = (info.day49[tsIdx] !== undefined) ? info.day49[tsIdx] : 0;
    const mitigation = activeMitigations[gh] || 'none';
    if (mitigation === 'signal') return val * 0.75;
    if (mitigation === 'divert') return val * 0.85;
    if (mitigation === 'patrol') return val * 0.80;
    return val;
}


// ── Color Scale ────────────────────────────────────────────
const CONGESTION_LEVELS = [
    { max: 0.12, color: '#00b894', rgba: 'rgba(0,184,148,',   label: 'Light Traffic'      },
    { max: 0.28, color: '#00cec9', rgba: 'rgba(0,206,201,',   label: 'Normal Flow'        },
    { max: 0.50, color: '#e1b12c', rgba: 'rgba(225,177,44,',  label: 'Moderate Delay'     },
    { max: 0.75, color: '#e67e22', rgba: 'rgba(230,126,34,',  label: 'Heavy Congestion'   },
    { max: 1.00, color: '#d63031', rgba: 'rgba(214,48,49,',   label: 'Gridlock Status'    },
];

function getCongestionLevel(val) {
    for (const lvl of CONGESTION_LEVELS) {
        if (val <= lvl.max) return lvl;
    }
    return CONGESTION_LEVELS[CONGESTION_LEVELS.length - 1];
}

function getCongestionColor(val)  { return getCongestionLevel(val).color; }
function getCongestionRgba(val, a){ return getCongestionLevel(val).rgba + a + ')'; }
function getCongestionText(val)   { return getCongestionLevel(val).label; }

function getSeverityBadgeStyle(val) {
    if (val < 0.12) return { bg:'rgba(0,184,148,0.15)',   border:'rgba(0,184,148,0.4)',   color:'#00b894', label:'Light' };
    if (val < 0.28) return { bg:'rgba(0,206,201,0.15)',   border:'rgba(0,206,201,0.4)',   color:'#00cec9', label:'Normal' };
    if (val < 0.50) return { bg:'rgba(225,177,44,0.15)',  border:'rgba(225,177,44,0.4)',  color:'#e1b12c', label:'Moderate' };
    if (val < 0.75) return { bg:'rgba(230,126,34,0.15)',  border:'rgba(230,126,34,0.4)',  color:'#e67e22', label:'Heavy' };
    return              { bg:'rgba(214,48,49,0.15)',   border:'rgba(214,48,49,0.4)',   color:'#d63031', label:'Gridlock' };
}

// ══════════════════════════════════════════════════════════════
//  SPLASH SCREEN LOADER
// ══════════════════════════════════════════════════════════════
const SPLASH_STEPS = [
    { pct: 10, msg: 'Initializing ML inference engine...' },
    { pct: 25, msg: 'Loading geohash zone registry (1,190 zones)...' },
    { pct: 45, msg: 'Parsing spatiotemporal demand matrix...' },
    { pct: 65, msg: 'Calibrating XGBoost + LightGBM ensemble...' },
    { pct: 80, msg: 'Mapping Day 49 traffic forecast layer...' },
    { pct: 95, msg: 'Rendering dashboard and charts...' },
    { pct: 100, msg: 'System ready. Welcome to ASTRA.' },
];

function runSplashSequence(onDone) {
    const bar    = document.getElementById('splash-progress');
    const status = document.getElementById('splash-status');
    let step = 0;

    function nextStep() {
        if (step >= SPLASH_STEPS.length) {
            setTimeout(() => {
                const splash = document.getElementById('splash-screen');
                splash.classList.add('fade-out');
                document.getElementById('main-app').classList.remove('hidden');
                setTimeout(onDone, 100);
            }, 300);
            return;
        }
        const s = SPLASH_STEPS[step++];
        bar.style.width = s.pct + '%';
        status.textContent = s.msg;
        const delay = step === SPLASH_STEPS.length ? 500 : 200 + Math.random() * 250;
        setTimeout(nextStep, delay);
    }
    nextStep();
}

// ══════════════════════════════════════════════════════════════
//  LIVE CLOCK
// ══════════════════════════════════════════════════════════════
function startLiveClock() {
    const el = document.getElementById('h-live-time');
    function tick() {
        const now = new Date();
        const hh = String(now.getHours()).padStart(2,'0');
        const mm = String(now.getMinutes()).padStart(2,'0');
        const ss = String(now.getSeconds()).padStart(2,'0');
        if (el) el.textContent = `${hh}:${mm}:${ss}`;
    }
    tick();
    setInterval(tick, 1000);
}

// ══════════════════════════════════════════════════════════════
//  ANIMATED COUNTER (number roll-up)
// ══════════════════════════════════════════════════════════════
function animateCounter(el, from, to, duration = 700, decimals = 0) {
    if (!el) return;
    const start  = performance.now();
    const range  = to - from;
    function step(ts) {
        const progress = Math.min((ts - start) / duration, 1);
        // ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const val   = from + range * eased;
        el.textContent = decimals > 0 ? val.toFixed(decimals) : Math.round(val).toLocaleString();
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// ══════════════════════════════════════════════════════════════
//  MAP INITIALIZATION
// ══════════════════════════════════════════════════════════════
function initMap(centerLat, centerLon) {
    map = L.map('map', { zoomControl: true, attributionControl: false })
            .setView([centerLat, centerLon], 13);

    // Dark premium tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19, subdomains: 'abcd'
    }).addTo(map);

    markersGroup = L.layerGroup().addTo(map);
    map.on('click', handleMapClick);

    // Fit-to-all button
    document.getElementById('btn-fit-map').addEventListener('click', () => {
        if (mapBounds) map.fitBounds(mapBounds, { padding: [40, 40], animate: true });
    });
}

// ══════════════════════════════════════════════════════════════
//  RENDER GEOHASH MARKERS
// ══════════════════════════════════════════════════════════════
function renderGeohashMarkers() {
    markersGroup.clearLayers();
    if (!data) return;

    const geohashes = data.geohashes;
    const latLngs   = [];
    let   boundsArr = [];

    for (const gh in geohashes) {
        const info = geohashes[gh];
        const val  = getAdjustedDemand(gh, currentTsIdx);

        // Geohash-6 dimensions: Height ~0.005 lat, Width ~0.011 lon
        const dLat = 0.00225;
        const dLon = 0.0055;
        const rectBounds = [
            [info.lat - dLat, info.lon - dLon],
            [info.lat + dLat, info.lon + dLon]
        ];

        const rectangle = L.rectangle(rectBounds, {
            color:       'rgba(0,0,0,0.25)',
            fillColor:   getCongestionColor(val),
            fillOpacity: 0.45 + val * 0.35,
            weight:      1.0
        });

        // Rich popup
        const pct   = (val * 100).toFixed(1);
        const fillW = Math.round(val * 100);
        rectangle.bindPopup(`
            <div class="popup-title">
                <i class="fa-solid fa-location-dot"></i> Zone: ${gh}
            </div>
            <div class="popup-row">
                <span class="popup-lbl">Road Type</span>
                <span class="popup-val">${info.road_type}</span>
            </div>
            <div class="popup-row">
                <span class="popup-lbl">Lanes</span>
                <span class="popup-val">${info.lanes}</span>
            </div>
            <div class="popup-row">
                <span class="popup-lbl">Landmark</span>
                <span class="popup-val">${info.landmarks}</span>
            </div>
            <div class="popup-row">
                <span class="popup-lbl">Weather / Temp</span>
                <span class="popup-val">${info.weather} · ${info.temp}°C</span>
            </div>
            <div class="popup-row" style="margin-top:6px">
                <span class="popup-lbl">Congestion Index</span>
                <span class="popup-val" style="color:${getCongestionColor(val)};font-size:12px;font-weight:700">${val.toFixed(3)}</span>
            </div>
            <div class="popup-row">
                <span class="popup-lbl">Status</span>
                <span class="popup-val" style="color:${getCongestionColor(val)}">${getCongestionText(val)}</span>
            </div>
            <div class="popup-congestion-bar">
                <div class="popup-congestion-fill" style="width:${fillW}%;background:${getCongestionColor(val)}"></div>
            </div>
        `, { maxWidth: 240 });

        rectangle.on('click', () => inspectGeohash(gh));
        rectangle.addTo(markersGroup);
        boundsArr.push([info.lat, info.lon]);
    }

    if (boundsArr.length > 0 && !mapBounds) {
        mapBounds = L.latLngBounds(boundsArr);
    }
}

// ══════════════════════════════════════════════════════════════
//  ZONE INSPECTOR
// ══════════════════════════════════════════════════════════════
function inspectGeohash(gh) {
    if (!data || !data.geohashes[gh]) return;

    currentInspectedGeohash = gh;
    const info = data.geohashes[gh];
    const val  = getAdjustedDemand(gh, currentTsIdx);
    const sev  = getSeverityBadgeStyle(val);

    // Update active mitigation button state
    const currentMitigation = activeMitigations[gh] || 'none';
    document.querySelectorAll('.btn-mitigation').forEach(btn => {
        if (btn.getAttribute('data-mitigation') === currentMitigation) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Show inspector, hide placeholder
    document.getElementById('inspector-placeholder').classList.add('hidden');
    document.getElementById('inspector-content').classList.remove('hidden');

    // Basic info
    document.getElementById('inspect-geohash').textContent = gh;
    document.getElementById('inspect-coords').textContent  =
        `Lat: ${info.lat.toFixed(5)}  ·  Lon: ${info.lon.toFixed(5)}`;

    document.getElementById('inspect-lanes').textContent     = `${info.lanes} lanes`;
    document.getElementById('inspect-road-type').textContent = info.road_type;
    document.getElementById('inspect-temp').textContent      = `${info.temp}°C`;
    document.getElementById('inspect-weather').textContent   = info.weather;
    document.getElementById('inspect-landmark').textContent  = info.landmarks;

    // Severity badge
    const badge = document.getElementById('inspect-severity-badge');
    badge.textContent = sev.label;
    badge.style.background  = sev.bg;
    badge.style.border      = `1px solid ${sev.border}`;
    badge.style.color       = sev.color;

    // Demand gauge
    document.getElementById('inspect-demand-val').textContent = val.toFixed(3);
    document.getElementById('inspect-demand-val').style.color = sev.color;
    const fill = document.getElementById('inspect-gauge-fill');
    fill.style.width      = `${Math.min(val * 100, 100)}%`;
    fill.style.background = `linear-gradient(90deg, ${getCongestionColor(val < 0.3 ? 0 : val * 0.5)}, ${getCongestionColor(val)})`;

    // Mini chart
    renderLocalInspectChart(gh);
}

// ══════════════════════════════════════════════════════════════
//  TOP CONGESTED ZONES LEADERBOARD
// ══════════════════════════════════════════════════════════════
function updateLeaderboard() {
    if (!data) return;

    const zones = [];
    for (const gh in data.geohashes) {
        const val = getAdjustedDemand(gh, currentTsIdx);
        zones.push({ gh, val, info: data.geohashes[gh] });
    }
    zones.sort((a, b) => b.val - a.val);

    const top5 = zones.slice(0, 5);
    const list  = document.getElementById('leaderboard-list');
    list.innerHTML = '';

    const maxVal = top5[0]?.val || 1;
    const rankClass = ['gold', 'silver', 'bronze', '', ''];

    top5.forEach((z, i) => {
        const sev   = getSeverityBadgeStyle(z.val);
        const fillW = Math.round((z.val / maxVal) * 100);
        const item  = document.createElement('div');
        item.className = 'lb-item';
        item.innerHTML = `
            <span class="lb-rank ${rankClass[i]}">#${i + 1}</span>
            <div class="lb-info">
                <span class="lb-geohash">${z.gh}</span>
                <span class="lb-desc">${z.info.road_type} · ${z.info.landmarks.slice(0, 28)}${z.info.landmarks.length > 28 ? '…' : ''}</span>
            </div>
            <div class="lb-bar-col">
                <span class="lb-val" style="color:${sev.color}">${z.val.toFixed(3)}</span>
                <div class="lb-bar">
                    <div class="lb-bar-fill" style="width:${fillW}%;background:${getCongestionColor(z.val)}"></div>
                </div>
            </div>
        `;
        item.addEventListener('click', () => {
            map.panTo([z.info.lat, z.info.lon], { animate: true, duration: 0.6 });
            inspectGeohash(z.gh);
        });
        list.appendChild(item);
    });
}

// ══════════════════════════════════════════════════════════════
//  GLOBAL METRICS (Header KPIs + Left Cards)
// ══════════════════════════════════════════════════════════════
let prevAvgDemand = 0;
let prevCritical  = 0;

function updateGlobalMetrics() {
    if (!data) return;

    let sum = 0, count = 0, critical = 0;
    for (const gh in data.geohashes) {
        const v = getAdjustedDemand(gh, currentTsIdx);
        sum += v; count++;
        if (v > 0.50) critical++;
    }
    const avg = sum / count;

    // Animated counter updates
    const avgEl  = document.getElementById('m-avg-demand');
    const critEl = document.getElementById('m-critical-zones');

    if (avgEl) {
        // Re-render as decimal
        animateCounter(avgEl, prevAvgDemand * 1000, avg * 1000, 600, 0);
        setTimeout(() => {
            if (avgEl) avgEl.textContent = avg.toFixed(3);
        }, 650);
    }

    if (critEl) {
        animateCounter(critEl, prevCritical, critical, 500, 0);
    }

    prevAvgDemand = avg;
    prevCritical  = critical;

    // Critical pct
    const pct = ((critical / count) * 100).toFixed(1);
    const critPctEl = document.getElementById('m-critical-pct');
    if (critPctEl) critPctEl.textContent = `${pct}% of all zones`;

    // Avg status label
    const statusEl = document.getElementById('m-avg-status');
    if (statusEl) {
        const lvl = getCongestionLevel(avg);
        statusEl.textContent = lvl.label;
        statusEl.style.color = lvl.color;
    }

    // Header KPI bar
    const hAvg  = document.getElementById('h-avg-congestion');
    const hCrit = document.getElementById('h-critical-count');
    if (hAvg)  hAvg.textContent  = avg.toFixed(3);
    if (hCrit) hCrit.textContent = critical;
}

// ══════════════════════════════════════════════════════════════
//  TIME SLIDER TICK MARKS
// ══════════════════════════════════════════════════════════════
function buildSliderTicks() {
    if (!data) return;
    const container = document.getElementById('slider-ticks');
    if (!container) return;
    container.innerHTML = '';

    const total = data.day49_timestamps.length;
    for (let i = 0; i < total; i++) {
        const ts   = data.day49_timestamps[i];
        const mark = document.createElement('div');
        const isHour = ts.endsWith(':00');
        mark.className = `tick-mark${isHour ? ' hour-mark' : ''}`;
        container.appendChild(mark);
    }
}

// ══════════════════════════════════════════════════════════════
//  UPDATE TIMESTAMP (master update)
// ══════════════════════════════════════════════════════════════
function updateTimestamp(idx) {
    currentTsIdx = parseInt(idx);
    const ts     = data.day49_timestamps[currentTsIdx];
    const total  = data.day49_timestamps.length;

    // Slider UI
    document.getElementById('time-slider').value  = currentTsIdx;
    document.getElementById('display-time').textContent = ts;

    // Progress fill
    const pct = ((currentTsIdx + 1) / total) * 100;
    const fill = document.getElementById('tc-progress-fill');
    if (fill) fill.style.width = pct + '%';

    // Re-render map
    renderGeohashMarkers();

    // Update metrics, charts, alerts, leaderboard
    updateGlobalMetrics();
    updateLeaderboard();
    generateLiveAlerts();

    // Redraw vertical bar on global chart
    if (globalFlowChart) globalFlowChart.update();

    // Update route sim if active
    if (routePoints.length > 1) calculateRouteDelay();
}

// ══════════════════════════════════════════════════════════════
//  PLAY / PAUSE TIMELINE
// ══════════════════════════════════════════════════════════════
function togglePlay() {
    const btn      = document.getElementById('btn-play-pause');
    const speedSel = document.getElementById('playback-speed');

    if (isPlaying) {
        isPlaying = false;
        clearInterval(playInterval);
        btn.innerHTML = '<i class="fa-solid fa-play"></i>';
    } else {
        isPlaying = true;
        btn.innerHTML = '<i class="fa-solid fa-pause"></i>';
        const speed = parseInt(speedSel?.value || 800);
        playInterval = setInterval(() => {
            let next = currentTsIdx + 1;
            if (next >= data.day49_timestamps.length) next = 0;
            updateTimestamp(next);
        }, speed);

        // React to speed change during playback
        speedSel?.addEventListener('change', () => {
            if (isPlaying) { clearInterval(playInterval); togglePlay(); togglePlay(); }
        });
    }
}

// ══════════════════════════════════════════════════════════════
//  LIVE ALERTS
// ══════════════════════════════════════════════════════════════
function generateLiveAlerts() {
    const list = document.getElementById('alerts-list');
    const badge = document.getElementById('alert-count-badge');
    list.innerHTML = '';
    if (!data) return;

    const ts = data.day49_timestamps[currentTsIdx];
    const congested = [];

    for (const gh in data.geohashes) {
        const v = getAdjustedDemand(gh, currentTsIdx);
        if (v > 0.35) congested.push({ gh, val: v, info: data.geohashes[gh] });
    }
    congested.sort((a, b) => b.val - a.val);

    if (badge) badge.textContent = congested.length;

    if (congested.length === 0) {
        list.innerHTML = `
            <div class="alert-item normal">
                <i class="fa-solid fa-circle-check alert-icon"></i>
                <div class="alert-body">
                    <span class="alert-msg">All zones reporting normal traffic conditions.</span>
                    <span class="alert-meta">Forecast timestamp: ${ts} · Day 49</span>
                </div>
            </div>`;
        if (badge) badge.textContent = '0';
        return;
    }

    congested.slice(0, 5).forEach(item => {
        const isCrit = item.val > 0.75;
        const isWarn = item.val > 0.50;
        const cls    = isCrit ? 'critical' : isWarn ? 'warning' : 'info';
        const icon   = isCrit ? 'fa-triangle-exclamation' : isWarn ? 'fa-circle-exclamation' : 'fa-circle-dot';
        const delay  = Math.round(item.val * 22);

        const el = document.createElement('div');
        el.className = `alert-item ${cls}`;
        el.innerHTML = `
            <i class="fa-solid ${icon} alert-icon"></i>
            <div class="alert-body">
                <span class="alert-msg">
                    <strong>${item.gh}</strong> — ${getCongestionText(item.val)}
                    near <em>${item.info.landmarks}</em>
                </span>
                <span class="alert-meta">
                    Index: ${item.val.toFixed(3)} · +${delay} min delay · ${ts}
                </span>
            </div>`;
        el.style.cursor = 'pointer';
        el.addEventListener('click', () => {
            map.panTo([item.info.lat, item.info.lon], { animate: true, duration: 0.6 });
            inspectGeohash(item.gh);
        });
        list.appendChild(el);
    });
}

// ══════════════════════════════════════════════════════════════
//  ROUTE DELAY SIMULATOR
// ══════════════════════════════════════════════════════════════
function handleMapClick(e) {
    if (!isDrawingRoute || !data) return;

    const latlng = e.latlng;
    routePoints.push(latlng);

    const marker = L.circleMarker(latlng, {
        radius: 7, color: '#6c5ce7',
        fillColor: '#00f2fe', fillOpacity: 0.95, weight: 2.5
    }).addTo(map);
    routeMarkers.push(marker);

    if (routePoints.length > 1) {
        if (routePolyline) map.removeLayer(routePolyline);
        routePolyline = L.polyline(routePoints, {
            color: '#6c5ce7', weight: 4.5,
            opacity: 0.85, dashArray: '6, 12'
        }).addTo(map);
        calculateRouteDelay();
    }
}

function calculateRouteDelay() {
    if (routePoints.length < 2 || !data) return;

    let totalDist = 0;
    const traversed = new Set();

    for (let i = 0; i < routePoints.length; i++) {
        const pt = routePoints[i];
        let nearestGh = null, minDist = Infinity;

        for (const gh in data.geohashes) {
            const { lat, lon } = data.geohashes[gh];
            const d = (lat - pt.lat) ** 2 + (lon - pt.lng) ** 2;
            if (d < minDist) { minDist = d; nearestGh = gh; }
        }
        if (nearestGh) traversed.add(nearestGh);

        if (i > 0) {
            const prev = routePoints[i - 1];
            totalDist += Math.sqrt((pt.lat - prev.lat) ** 2 + (pt.lng - prev.lng) ** 2) * 111.3;
        }
    }

    const zones = [...traversed];
    const zoneCount = zones.length;

    let sumCong = 0;
    zones.forEach(gh => { sumCong += getAdjustedDemand(gh, currentTsIdx); });
    const avgCong = sumCong / (zoneCount || 1);

    const baseMin  = totalDist * 1.5;
    const extraMin = zoneCount * avgCong * 5.0;
    const totalMin = baseMin + extraMin;

    document.getElementById('route-stats').classList.remove('hidden');
    document.getElementById('route-dist').textContent       = `${totalDist.toFixed(2)} km`;
    document.getElementById('route-zones').textContent      = `${zoneCount} zones`;
    document.getElementById('route-congestion').textContent = `${avgCong.toFixed(2)} (${getCongestionText(avgCong)})`;
    document.getElementById('route-delay').textContent      = `+${Math.round(extraMin)} mins (Total: ~${Math.round(totalMin)} min)`;

    // Delay bar (capped at 30 min = 100%)
    const barPct = Math.min((extraMin / 30) * 100, 100);
    const bar    = document.getElementById('route-delay-bar');
    if (bar) bar.style.width = barPct + '%';

    // Recolor polyline
    if (routePolyline) {
        routePolyline.setStyle({ color: getCongestionColor(avgCong) });
    }
}

// ══════════════════════════════════════════════════════════════
//  CHARTS
// ══════════════════════════════════════════════════════════════

// Chart.js global defaults
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color       = '#8b98bb';

/* ── Global Spatiotemporal Flow Chart ── */
function renderGlobalFlowChart() {
    const ctx = document.getElementById('global-demand-chart').getContext('2d');

    const day48Sliced = data.global_day48.slice(0, data.day49_timestamps.length);
    const labels      = data.day49_timestamps;
    const maxVal      = Math.max(...data.global_day49, ...day48Sliced) * 1.15;

    // Gradient fills
    const gradCyan = ctx.createLinearGradient(0, 0, 0, 200);
    gradCyan.addColorStop(0, 'rgba(0,242,254,0.25)');
    gradCyan.addColorStop(1, 'rgba(0,242,254,0.00)');

    const gradIndigo = ctx.createLinearGradient(0, 0, 0, 200);
    gradIndigo.addColorStop(0, 'rgba(108,92,231,0.15)');
    gradIndigo.addColorStop(1, 'rgba(108,92,231,0.00)');

    globalFlowChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Day 48 Actual',
                    data: day48Sliced,
                    borderColor: 'rgba(108,92,231,0.55)',
                    borderWidth: 1.5,
                    borderDash: [5, 6],
                    pointRadius: 0,
                    fill: true,
                    backgroundColor: gradIndigo,
                    tension: 0.4,
                    order: 2,
                },
                {
                    label: 'Day 49 ML Forecast',
                    data: data.global_day49,
                    borderColor: '#00f2fe',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    backgroundColor: gradCyan,
                    tension: 0.4,
                    order: 1,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            animation: { duration: 600 },
            plugins: {
                legend: {
                    display: true,
                    align: 'end',
                    labels: {
                        color: '#8b98bb', font: { size: 9 },
                        boxWidth: 16, boxHeight: 2,
                        usePointStyle: true, pointStyle: 'line',
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10,14,26,0.95)',
                    borderColor: 'rgba(0,242,254,0.2)',
                    borderWidth: 1,
                    titleColor: '#00f2fe', bodyColor: '#8b98bb',
                    padding: 10, cornerRadius: 8,
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.03)', drawBorder: false },
                    ticks: { color: '#4a5475', font: { size: 8 }, maxTicksLimit: 8 }
                },
                y: {
                    max: maxVal,
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#4a5475', font: { size: 8 }, maxTicksLimit: 5 }
                }
            }
        },
        plugins: [{
            // Vertical "now" indicator
            afterDraw(chart) {
                const { ctx: c, scales: { x, y } } = chart;
                const px = x.getPixelForTick(currentTsIdx);
                if (!px) return;
                c.save();
                c.beginPath();
                c.moveTo(px, y.top);
                c.lineTo(px, y.bottom);
                c.lineWidth   = 1.5;
                c.strokeStyle = 'rgba(0,242,254,0.7)';
                c.setLineDash([3, 3]);
                c.stroke();

                // Label
                c.fillStyle    = 'rgba(0,242,254,0.85)';
                c.font         = '9px JetBrains Mono, monospace';
                c.textAlign    = 'center';
                c.fillText(data.day49_timestamps[currentTsIdx], px, y.top - 4);
                c.restore();
            }
        }]
    });
}

/* ── Road Type / Category Bar Chart ── */
function renderRoadTypeChart() {
    const ctx    = document.getElementById('road-type-chart').getContext('2d');
    const labels = Object.keys(data.road_type_stats);
    const vals   = Object.values(data.road_type_stats);

    const COLORS = ['#00f2fe','#4facfe','#6c5ce7','#e1b12c','#00b894'];
    const BGs    = ['rgba(0,242,254,0.18)','rgba(79,172,254,0.18)',
                    'rgba(108,92,231,0.18)','rgba(225,177,44,0.18)',
                    'rgba(0,184,148,0.18)'];

    roadTypeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Avg Predicted Demand',
                data: vals,
                backgroundColor: BGs,
                borderColor: COLORS,
                borderWidth: 1.5,
                borderRadius: 5,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10,14,26,0.95)',
                    borderColor: 'rgba(0,242,254,0.2)',
                    borderWidth: 1,
                    titleColor: '#00f2fe', bodyColor: '#8b98bb',
                    padding: 8, cornerRadius: 6,
                    callbacks: {
                        label: ctx => ` Index: ${ctx.parsed.y.toFixed(4)}`
                    }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#4a5475', font: { size: 8 } } },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#4a5475', font: { size: 8 }, maxTicksLimit: 4 }
                }
            }
        }
    });
}

/* ── Local Geohash Diurnal Chart ── */
function renderLocalInspectChart(gh) {
    const info = data.geohashes[gh];
    const ctx  = document.getElementById('geohash-demand-chart').getContext('2d');

    if (localInspectChart) localInspectChart.destroy();

    const gradDay49 = ctx.createLinearGradient(0, 0, 0, 130);
    gradDay49.addColorStop(0, 'rgba(0,242,254,0.20)');
    gradDay49.addColorStop(1, 'rgba(0,242,254,0.00)');

    localInspectChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.day49_timestamps,
            datasets: [
                {
                    label: 'Day 48',
                    data: info.day48.slice(0, data.day49_timestamps.length),
                    borderColor: 'rgba(255,255,255,0.20)',
                    borderWidth: 1.2, borderDash: [4, 5],
                    pointRadius: 0, fill: false, tension: 0.4,
                },
                {
                    label: 'Day 49 Pred',
                    data: info.day49,
                    borderColor: '#00f2fe',
                    borderWidth: 1.8,
                    pointRadius: 0,
                    fill: true,
                    backgroundColor: gradDay49,
                    tension: 0.4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 400 },
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#8b98bb', font: { size: 8 }, boxWidth: 12 }
                },
                tooltip: {
                    backgroundColor: 'rgba(10,14,26,0.95)',
                    borderColor: 'rgba(0,242,254,0.2)',
                    borderWidth: 1, padding: 6, cornerRadius: 6,
                    titleColor: '#00f2fe', bodyColor: '#8b98bb',
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#4a5475', font: { size: 7 }, maxTicksLimit: 5 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.03)', drawBorder: false },
                    ticks: { color: '#4a5475', font: { size: 7 }, maxTicksLimit: 4 }
                }
            }
        },
        plugins: [{
            // Vertical marker at current timestamp
            afterDraw(chart) {
                const { ctx: c, scales: { x, y } } = chart;
                const px = x.getPixelForTick(currentTsIdx);
                if (!px) return;
                c.save();
                c.beginPath();
                c.moveTo(px, y.top);
                c.lineTo(px, y.bottom);
                c.lineWidth   = 1;
                c.strokeStyle = 'rgba(0,242,254,0.5)';
                c.setLineDash([2, 2]);
                c.stroke();
                c.restore();
            }
        }]
    });
}

// ══════════════════════════════════════════════════════════════
//  DATA LOADING & BOOT
// ══════════════════════════════════════════════════════════════
async function loadData() {
    const response = await fetch('dashboard_data.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
    console.log('[ASTRA] Dashboard data loaded. Zones:', Object.keys(data.geohashes).length);
}

async function bootDashboard() {
    // Compute average center
    let latSum = 0, lonSum = 0, n = 0;
    for (const gh in data.geohashes) {
        latSum += data.geohashes[gh].lat;
        lonSum += data.geohashes[gh].lon;
        n++;
    }

    // Init map
    initMap(latSum / n, lonSum / n);

    // Render markers
    renderGeohashMarkers();

    // Fit map bounds
    if (mapBounds) map.fitBounds(mapBounds, { padding: [40, 40] });

    // Build slider ticks
    buildSliderTicks();

    // Render charts
    renderGlobalFlowChart();
    renderRoadTypeChart();

    // Populate panels
    updateGlobalMetrics();
    updateLeaderboard();
    generateLiveAlerts();

    // Initial progress bar
    const fill = document.getElementById('tc-progress-fill');
    if (fill) fill.style.width = ((currentTsIdx + 1) / data.day49_timestamps.length * 100) + '%';
    const display = document.getElementById('display-time');
    if (display) display.textContent = data.day49_timestamps[currentTsIdx];

    // Animate metric counters on first load
    animateCounter(document.getElementById('m-total-zones'), 0, n, 900, 0);

    // Start live clock
    startLiveClock();
}

// ══════════════════════════════════════════════════════════════
//  EVENT LISTENERS & INIT
// ══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {

    // Time slider
    document.getElementById('time-slider').addEventListener('input', e => {
        updateTimestamp(e.target.value);
    });

    // Play / Pause
    document.getElementById('btn-play-pause').addEventListener('click', togglePlay);

    // Route Draw button
    const btnDraw  = document.getElementById('btn-draw-route');
    const btnClear = document.getElementById('btn-clear-route');

    btnDraw.addEventListener('click', () => {
        if (!isDrawingRoute) {
            isDrawingRoute = true;
            routePoints = [];
            btnDraw.innerHTML = '<i class="fa-solid fa-square-check"></i> Finish Route';
            btnDraw.style.background = 'linear-gradient(135deg, #00b894, #00cec9)';
            btnClear.classList.remove('hidden');
        } else {
            isDrawingRoute = false;
            btnDraw.innerHTML = '<i class="fa-solid fa-pen-nib"></i> Draw Route';
            btnDraw.style.background = '';
        }
    });

    btnClear.addEventListener('click', () => {
        isDrawingRoute = false;
        routePoints    = [];
        if (routePolyline) { map.removeLayer(routePolyline); routePolyline = null; }
        routeMarkers.forEach(m => map.removeLayer(m));
        routeMarkers = [];

        btnDraw.innerHTML    = '<i class="fa-solid fa-pen-nib"></i> Draw Route';
        btnDraw.style.background = '';
        btnClear.classList.add('hidden');
        document.getElementById('route-stats').classList.add('hidden');
    });

    // Traffic mitigation button clicks
    document.querySelectorAll('.btn-mitigation').forEach(btn => {
        btn.addEventListener('click', e => {
            if (!currentInspectedGeohash) return;
            const targetBtn = e.currentTarget;
            const mitigationType = targetBtn.getAttribute('data-mitigation');
            
            // Set active mitigation for this geohash
            activeMitigations[currentInspectedGeohash] = mitigationType;
            
            // Update UI buttons active state
            document.querySelectorAll('.btn-mitigation').forEach(b => {
                b.classList.toggle('active', b === targetBtn);
            });
            
            // Force re-render of map and other stats
            updateTimestamp(currentTsIdx);
            inspectGeohash(currentInspectedGeohash);
        });
    });

    // Run splash then load data
    runSplashSequence(async () => {
        try {
            await loadData();
            await bootDashboard();
        } catch (err) {
            console.error('[ASTRA] Boot failed:', err);
            document.getElementById('splash-status').textContent = 'Error loading data. Check console.';
        }
    });
});
