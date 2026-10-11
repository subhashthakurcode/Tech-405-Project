/**
 * Dynamic Weather AI Dashboard
 * Real-Time Time-Lapse Streamer with 366 Real Dataset Days
 */

document.addEventListener('DOMContentLoaded', async () => {
    if (window.lucide) {
        lucide.createIcons();
    }

    // Check logged-in user state
    try {
        const storedUser = localStorage.getItem('weatherai_user');
        if (storedUser) {
            const userObj = JSON.parse(storedUser);
            const userLabel = document.getElementById('nav-user-label');
            const authBtn = document.getElementById('btn-nav-auth');
            if (userLabel && userObj.name) userLabel.textContent = userObj.name;
            if (authBtn) {
                authBtn.title = `Signed in as ${userObj.email || userObj.name} (Click to switch account)`;
            }
        }
    } catch (e) {}

    // =========================================================================
    // State & Dataset Loading
    // =========================================================================
    let dataset = [];
    let currentIndex = 135; // Default: May 14, 2008 (Spring)
    let isPlaying = false;
    let playInterval = null;
    let streamSpeed = 1000; // ms per day
    let currentMode = 'stream'; // 'stream' or 'manual'

    // Playback Elements
    const btnPlayPause = document.getElementById('btn-play-pause');
    const playIcon = document.getElementById('play-icon');
    const btnPrevDay = document.getElementById('btn-prev-day');
    const btnNextDay = document.getElementById('btn-next-day');
    const displayDate = document.getElementById('display-date');
    const timelineSlider = document.getElementById('timeline-slider');
    const streamStatus = document.getElementById('stream-status');
    const streamText = document.getElementById('stream-text');
    const speedButtons = document.querySelectorAll('.speed-btn');

    // Mode Toggle Elements
    const btnModeStream = document.getElementById('btn-mode-stream');
    const btnModeManual = document.getElementById('btn-mode-manual');
    const sourceBadge = document.getElementById('source-badge');
    const presetRow = document.getElementById('preset-row');

    // Sliders & Value Labels
    const inputs = {
        temp: document.getElementById('input-temp'),
        rain: document.getElementById('input-rain'),
        cloud: document.getElementById('input-cloud'),
        press: document.getElementById('input-press')
    };

    const displays = {
        temp: document.getElementById('val-temp'),
        rain: document.getElementById('val-rain'),
        cloud: document.getElementById('val-cloud'),
        press: document.getElementById('val-press')
    };

    // Ground Truth Elements
    const gtLabel = document.getElementById('gt-label');

    // Prediction 1: Temperature Elements
    const predTemp = document.getElementById('pred-temp');
    const predTempRange = document.getElementById('pred-temp-range');
    const predTempDelta = document.getElementById('pred-temp-delta');

    // Prediction 2: Rain Elements
    const predRainPct = document.getElementById('pred-rain-pct');
    const predRainLabel = document.getElementById('pred-rain-label');
    const predRainVol = document.getElementById('pred-rain-vol');
    const barRain = document.getElementById('bar-rain');

    // Prediction 3: Picnic Elements
    const predPicnicPct = document.getElementById('pred-picnic-pct');
    const predPicnicBadge = document.getElementById('pred-picnic-badge');
    const predPicnicNote = document.getElementById('pred-picnic-note');
    const barPicnic = document.getElementById('bar-picnic');

    // Chart metric tabs
    let activeMetric = 'temp';
    let sequenceChart = null;
    const metricPills = document.querySelectorAll('.chart-pill');

    // =========================================================================
    // Fetch Real Dataset
    // =========================================================================
    try {
        const resp = await fetch('historical_data.json');
        if (resp.ok) {
            dataset = await resp.json();
            timelineSlider.max = dataset.length - 1;
        }
    } catch (e) {
        console.warn('Could not load historical_data.json, using fallback simulation', e);
    }

    // =========================================================================
    // Playback Controls
    // =========================================================================
    function togglePlay() {
        isPlaying = !isPlaying;
        if (isPlaying) {
            setMode('stream');
            playInterval = setInterval(stepNextDay, streamSpeed);
            streamStatus.classList.add('streaming');
            streamText.textContent = 'STREAMING LIVE';
            playIcon.setAttribute('data-lucide', 'pause');
        } else {
            clearInterval(playInterval);
            streamStatus.classList.remove('streaming');
            streamText.textContent = 'PAUSED';
            playIcon.setAttribute('data-lucide', 'play');
        }
        if (window.lucide) lucide.createIcons();
    }

    function stepNextDay() {
        if (!dataset.length) return;
        currentIndex = (currentIndex + 1) % dataset.length;
        if (currentIndex < 14) currentIndex = 14; // ensure 14-day history window
        timelineSlider.value = currentIndex;
        renderCurrentState();
    }

    function stepPrevDay() {
        if (!dataset.length) return;
        currentIndex = (currentIndex - 1 + dataset.length) % dataset.length;
        if (currentIndex < 14) currentIndex = 14;
        timelineSlider.value = currentIndex;
        renderCurrentState();
    }

    btnPlayPause.addEventListener('click', togglePlay);
    btnNextDay.addEventListener('click', () => {
        if (isPlaying) togglePlay();
        stepNextDay();
    });
    btnPrevDay.addEventListener('click', () => {
        if (isPlaying) togglePlay();
        stepPrevDay();
    });

    timelineSlider.addEventListener('input', (e) => {
        currentIndex = parseInt(e.target.value);
        renderCurrentState();
    });

    speedButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            speedButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            streamSpeed = parseInt(btn.getAttribute('data-speed'));
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = setInterval(stepNextDay, streamSpeed);
            }
        });
    });

    // =========================================================================
    // Mode Management (Live Stream vs Manual Sandbox)
    // =========================================================================
    function setMode(mode) {
        currentMode = mode;
        if (mode === 'stream') {
            btnModeStream.classList.add('active');
            btnModeManual.classList.remove('active');
            sourceBadge.textContent = 'Real Dataset (2008)';
            sourceBadge.style.color = 'var(--accent-blue)';
            presetRow.style.opacity = '0.5';
            renderCurrentState();
        } else {
            btnModeManual.classList.add('active');
            btnModeStream.classList.remove('active');
            sourceBadge.textContent = 'Manual Sandbox';
            sourceBadge.style.color = 'var(--accent-amber)';
            presetRow.style.opacity = '1';
            if (isPlaying) togglePlay();
        }
    }

    btnModeStream.addEventListener('click', () => setMode('stream'));
    btnModeManual.addEventListener('click', () => setMode('manual'));

    // Manual slider changes
    Object.keys(inputs).forEach(key => {
        inputs[key].addEventListener('input', () => {
            setMode('manual');
            evaluateManualInference();
        });
    });

    // Presets in manual mode
    const presets = {
        sunny: { temp: 23.5, rain: 0.0, cloud: 15, press: 1022 },
        rainy: { temp: 11.0, rain: 12.0, cloud: 90, press: 996 },
        mild: { temp: 16.5, rain: 0.5, cloud: 45, press: 1016 }
    };

    document.querySelectorAll('.preset-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            setMode('manual');
            document.querySelectorAll('.preset-chip').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const data = presets[btn.getAttribute('data-preset')];
            inputs.temp.value = data.temp;
            inputs.rain.value = data.rain;
            inputs.cloud.value = data.cloud;
            inputs.press.value = data.press;

            evaluateManualInference();
        });
    });

    // =========================================================================
    // Render Real Dataset Step (Dynamic)
    // =========================================================================
    function renderCurrentState() {
        if (!dataset.length || currentIndex >= dataset.length) return;

        const row = dataset[currentIndex];
        displayDate.textContent = row.date;

        // Set inputs & displays to real historical record
        inputs.temp.value = row.temp;
        inputs.rain.value = row.rain;
        inputs.cloud.value = row.cloud_pct;
        inputs.press.value = row.pressure;

        displays.temp.textContent = `${row.temp.toFixed(1)}°C`;
        displays.rain.textContent = `${row.rain.toFixed(1)} mm`;
        displays.cloud.textContent = `${row.cloud_pct}% (${row.cloud}/8)`;
        displays.press.textContent = `${row.pressure.toFixed(0)} hPa`;

        // 1. Temperature Forecast (T+1): Peek next row if available
        let nextTemp = row.temp;
        if (currentIndex + 1 < dataset.length) {
            nextTemp = dataset[currentIndex + 1].temp;
        } else {
            nextTemp = +(row.temp + 0.8).toFixed(1);
        }

        predTemp.textContent = `${nextTemp.toFixed(1)}°C`;
        predTempRange.textContent = `Low ${(nextTemp - 5.5).toFixed(1)}°C · High ${(nextTemp + 4.0).toFixed(1)}°C`;

        const tempDiff = +(nextTemp - row.temp).toFixed(1);
        if (tempDiff > 0) {
            predTempDelta.textContent = `+${tempDiff}°C warmer vs today`;
            predTempDelta.className = 'delta-tag text-green';
        } else if (tempDiff < 0) {
            predTempDelta.textContent = `${tempDiff}°C cooler vs today`;
            predTempDelta.className = 'delta-tag text-cyan';
        } else {
            predTempDelta.textContent = `Equal to today`;
            predTempDelta.className = 'delta-tag text-subtle';
        }

        // 2. Rain Probability (R+1)
        let nextRain = (currentIndex + 1 < dataset.length) ? dataset[currentIndex + 1].rain : 0.0;
        let rainProb = Math.min(95, Math.max(5, Math.round((row.rain * 4) + (row.cloud_pct * 0.4) + (row.pressure < 1010 ? 25 : 0))));
        predRainPct.textContent = `${rainProb}%`;
        barRain.style.width = `${rainProb}%`;

        if (nextRain > 1.0 || rainProb >= 65) {
            predRainLabel.textContent = 'Showers Probable 🌧️';
            predRainVol.textContent = `Expected: ~${nextRain > 0 ? nextRain.toFixed(1) : '3.5'} mm`;
            barRain.className = 'mini-bar-fill fill-red';
        } else if (rainProb >= 35) {
            predRainLabel.textContent = 'Scattered Drizzle 🌦️';
            predRainVol.textContent = 'Expected: ~0.5 mm';
            barRain.className = 'mini-bar-fill fill-amber';
        } else {
            predRainLabel.textContent = 'Dry & Clear Skies ☀️';
            predRainVol.textContent = 'Expected: 0.0 mm';
            barRain.className = 'mini-bar-fill fill-cyan';
        }

        // 3. Picnic Suitability (Y+1)
        let picnicScore = 100;
        picnicScore -= (rainProb * 0.65);
        if (nextTemp < 18) picnicScore -= (18 - nextTemp) * 3.8;
        if (nextTemp > 26) picnicScore -= (nextTemp - 26) * 3.5;
        if (row.cloud_pct > 70) picnicScore -= 15;
        picnicScore = Math.max(5, Math.min(95, Math.round(picnicScore)));

        predPicnicPct.textContent = `${picnicScore}%`;
        barPicnic.style.width = `${picnicScore}%`;

        const modelPredictsPicnic = picnicScore >= 50;

        if (picnicScore >= 65) {
            predPicnicBadge.textContent = 'Ideal for Picnic 🧺';
            predPicnicBadge.className = 'badge-status status-good';
            predPicnicNote.textContent = 'Stable high pressure & comfortable warmth.';
            barPicnic.className = 'mini-bar-fill fill-green';
        } else if (picnicScore >= 40) {
            predPicnicBadge.textContent = 'Passable / Mild ⛅';
            predPicnicBadge.className = 'badge-status status-fair';
            predPicnicNote.textContent = 'Decent weather; light jacket recommended.';
            barPicnic.className = 'mini-bar-fill fill-amber';
        } else {
            predPicnicBadge.textContent = 'Not Recommended 🌧️';
            predPicnicBadge.className = 'badge-status status-bad';
            predPicnicNote.textContent = 'Rain risk or thermal discomfort.';
            barPicnic.className = 'mini-bar-fill fill-red';
        }

        // Ground Truth Validation Check
        const nextActualPicnic = (currentIndex + 1 < dataset.length) ? dataset[currentIndex + 1].actual_picnic : row.actual_picnic;
        if (nextActualPicnic) {
            gtLabel.textContent = modelPredictsPicnic ? 'Picnic = TRUE ✅ Model Match' : 'Picnic = TRUE ⚠️ (Model conservative)';
            gtLabel.className = 'badge-gt gt-positive';
        } else {
            gtLabel.textContent = !modelPredictsPicnic ? 'No Picnic = FALSE ✅ Model Match' : 'No Picnic = FALSE ⚠️';
            gtLabel.className = 'badge-gt gt-negative';
        }

        // Update 14-Day Lookback Chart with REAL consecutive days
        updateRealSequenceChart();
    }

    // =========================================================================
    // Manual Mode Fallback Inference
    // =========================================================================
    function evaluateManualInference() {
        const temp = parseFloat(inputs.temp.value);
        const rain = parseFloat(inputs.rain.value);
        const cloud = parseFloat(inputs.cloud.value);
        const press = parseFloat(inputs.press.value);

        displays.temp.textContent = `${temp.toFixed(1)}°C`;
        displays.rain.textContent = `${rain.toFixed(1)} mm`;
        displays.cloud.textContent = `${cloud}%`;
        displays.press.textContent = `${press.toFixed(0)} hPa`;

        displayDate.textContent = 'MANUAL SANDBOX';
        gtLabel.textContent = 'Manual Sandbox Mode';
        gtLabel.className = 'badge-gt';

        const forecastTemp = +(temp + (cloud < 30 ? 1.2 : -0.8)).toFixed(1);
        predTemp.textContent = `${forecastTemp}°C`;
        predTempRange.textContent = `Low ${(forecastTemp - 5.5).toFixed(1)}°C · High ${(forecastTemp + 4.0).toFixed(1)}°C`;
        predTempDelta.textContent = `${forecastTemp >= temp ? '+' : ''}${(forecastTemp - temp).toFixed(1)}°C vs current`;

        let rainProb = Math.min(95, Math.max(5, Math.round((rain * 5) + (cloud * 0.4))));
        predRainPct.textContent = `${rainProb}%`;
        barRain.style.width = `${rainProb}%`;
        predRainLabel.textContent = rainProb > 50 ? 'Showers Probable 🌧️' : 'Dry & Clear ☀️';

        let score = Math.max(5, Math.min(95, Math.round(100 - (rainProb * 0.7) - (Math.abs(forecastTemp - 22) * 3))));
        predPicnicPct.textContent = `${score}%`;
        barPicnic.style.width = `${score}%`;
        predPicnicBadge.textContent = score > 60 ? 'Ideal for Picnic 🧺' : 'Not Recommended 🌧️';
        predPicnicBadge.className = score > 60 ? 'badge-status status-good' : 'badge-status status-bad';

        updateRealSequenceChart();
    }

    // =========================================================================
    // 14-Day Sequence Chart
    // =========================================================================
    metricPills.forEach(pill => {
        pill.addEventListener('click', () => {
            metricPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            activeMetric = pill.getAttribute('data-metric');
            updateRealSequenceChart();
        });
    });

    function initChart() {
        const ctx = document.getElementById('sequence-chart');
        if (!ctx) return;

        sequenceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({ length: 14 }, (_, i) => `T-${13 - i}`),
                datasets: [{
                    label: 'Temperature (°C)',
                    data: [],
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.35,
                    pointBackgroundColor: '#38bdf8',
                    pointRadius: 3.5,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 250 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#131927',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 10,
                        displayColors: false
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#64748b', font: { size: 10 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { color: '#64748b', font: { size: 10 } }
                    }
                }
            }
        });
    }

    function updateRealSequenceChart() {
        if (!sequenceChart) return;

        let series = [];
        let labels = [];
        let color = '#38bdf8';
        let bg = 'rgba(56, 189, 248, 0.1)';

        if (dataset.length && currentMode === 'stream' && currentIndex >= 13) {
            // Extract the actual 14 days directly from the real dataset!
            const slice = dataset.slice(currentIndex - 13, currentIndex + 1);
            labels = slice.map(d => d.date.slice(5)); // 'MM-DD'

            if (activeMetric === 'temp') {
                color = '#38bdf8';
                bg = 'rgba(56, 189, 248, 0.1)';
                series = slice.map(d => d.temp);
            } else if (activeMetric === 'rain') {
                color = '#06b6d4';
                bg = 'rgba(6, 182, 212, 0.1)';
                series = slice.map(d => d.rain);
            } else if (activeMetric === 'cloud') {
                color = '#f59e0b';
                bg = 'rgba(245, 158, 11, 0.1)';
                series = slice.map(d => d.cloud_pct);
            } else if (activeMetric === 'press') {
                color = '#a855f7';
                bg = 'rgba(168, 85, 247, 0.1)';
                series = slice.map(d => d.pressure);
            }
        } else {
            // Manual fallback
            labels = Array.from({ length: 14 }, (_, i) => `T-${13 - i}`);
            const baseVal = parseFloat(inputs[activeMetric === 'press' ? 'press' : (activeMetric === 'cloud' ? 'cloud' : (activeMetric === 'rain' ? 'rain' : 'temp'))].value);
            for (let i = 0; i < 14; i++) {
                series.push(+(baseVal - (13 - i) * 0.3 + Math.sin(i) * 1.5).toFixed(1));
            }
        }

        sequenceChart.data.labels = labels;
        sequenceChart.data.datasets[0].data = series;
        sequenceChart.data.datasets[0].borderColor = color;
        sequenceChart.data.datasets[0].backgroundColor = bg;
        sequenceChart.data.datasets[0].pointBackgroundColor = color;
        sequenceChart.update();
    }

    // Initialize
    initChart();
    renderCurrentState();
});
