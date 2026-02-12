// Reachy Nova - AI Command Center
const POLL_INTERVAL = 500;

const els = {
    robotFace: document.getElementById('robot-face'),
    moodBadge: document.getElementById('mood-badge'),
    uptime: document.getElementById('uptime'),
    voiceDot: document.getElementById('voice-dot'),
    waveform: document.getElementById('waveform'),
    voiceLabel: document.getElementById('voice-label'),
    userText: document.querySelector('#user-text .text'),
    assistantText: document.querySelector('#assistant-text .text'),
    visionToggle: document.getElementById('vision-toggle'),
    visionSnap: document.getElementById('vision-snap'),
    eyeScanner: document.getElementById('eye-scanner'),
    visionDescription: document.getElementById('vision-description'),
    browserDot: document.getElementById('browser-dot'),
    browserInstruction: document.getElementById('browser-instruction'),
    browserGo: document.getElementById('browser-go'),
    browserScreenshot: document.getElementById('browser-screenshot'),
    browserResult: document.getElementById('browser-result'),
    trackingToggle: document.getElementById('tracking-toggle'),
    trackingMode: document.getElementById('tracking-mode'),
};

// --- State polling ---
async function pollState() {
    try {
        const resp = await fetch('/api/state');
        if (!resp.ok) return;
        const state = await resp.json();
        updateUI(state);
    } catch (e) {
        // Backend not ready yet
    }
}

function updateUI(state) {
    // Uptime
    const secs = Math.floor(state.uptime || 0);
    const mins = Math.floor(secs / 60);
    const hrs = Math.floor(mins / 60);
    els.uptime.textContent = hrs > 0
        ? `${hrs}:${String(mins % 60).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`
        : `${mins}:${String(secs % 60).padStart(2, '0')}`;

    // Mood
    els.moodBadge.textContent = state.mood || 'happy';
    els.moodBadge.className = 'mood-badge ' + (state.mood || 'happy');

    // Robot face
    els.robotFace.className = 'robot-face ' + (state.voice_state || '');
    if (state.mood === 'excited') els.robotFace.classList.add('excited');

    // Voice
    const voiceState = state.voice_state || 'idle';
    els.voiceLabel.textContent = voiceState;
    els.waveform.className = 'waveform';
    els.voiceDot.className = 'status-dot';

    if (voiceState === 'listening') {
        els.waveform.classList.add('active');
        els.voiceDot.classList.add('active');
    } else if (voiceState === 'speaking') {
        els.waveform.classList.add('active', 'speaking');
        els.voiceDot.classList.add('active');
    } else if (voiceState === 'thinking') {
        els.waveform.classList.add('thinking');
        els.voiceDot.classList.add('busy');
    }

    // Transcript
    if (state.last_user_text) {
        els.userText.textContent = state.last_user_text;
    }
    if (state.last_assistant_text) {
        els.assistantText.textContent = state.last_assistant_text;
    }

    // Vision
    els.eyeScanner.className = state.vision_analyzing ? 'eye-scanner analyzing' : 'eye-scanner';
    if (state.vision_description) {
        els.visionDescription.textContent = state.vision_description;
    }

    // Browser
    els.browserDot.className = 'status-dot';
    if (state.browser_state === 'busy') {
        els.browserDot.classList.add('busy');
        els.browserResult.textContent = 'Working: ' + (state.browser_task || '...');
    } else if (state.browser_state === 'error') {
        els.browserDot.classList.add('error');
    } else if (state.browser_result) {
        els.browserDot.classList.add('active');
    }

    if (state.browser_result && state.browser_state !== 'busy') {
        els.browserResult.textContent = state.browser_result;
    }

    if (state.browser_screenshot) {
        els.browserScreenshot.innerHTML = `<img src="data:image/png;base64,${state.browser_screenshot}" alt="Browser">`;
    }

    // Tracking
    if (els.trackingMode) {
        const mode = state.tracking_mode || 'idle';
        els.trackingMode.textContent = mode;
        els.trackingMode.className = 'tracking-mode-label ' + mode;
    }
    if (els.trackingToggle && state.tracking_enabled !== undefined) {
        els.trackingToggle.checked = state.tracking_enabled;
    }
}

// --- API helpers ---
async function postJSON(url, data = {}) {
    try {
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return await resp.json();
    } catch (e) {
        console.error('API error:', e);
    }
}

// --- Event handlers ---
els.visionToggle.addEventListener('change', (e) => {
    postJSON('/api/vision/toggle', { enabled: e.target.checked });
});

els.visionSnap.addEventListener('click', () => {
    postJSON('/api/vision/analyze');
    els.eyeScanner.classList.add('analyzing');
});

els.browserGo.addEventListener('click', () => {
    const instruction = els.browserInstruction.value.trim();
    if (!instruction) return;
    postJSON('/api/browser/task', { instruction });
    els.browserInstruction.value = '';
    els.browserResult.textContent = 'Queued: ' + instruction;
});

els.browserInstruction.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') els.browserGo.click();
});

// Tracking toggle
els.trackingToggle?.addEventListener('change', (e) => {
    postJSON('/api/tracking/toggle', { enabled: e.target.checked });
});

// Antenna mode buttons
document.querySelectorAll('.antenna-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.antenna-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        postJSON('/api/antenna/mode', { mode: btn.dataset.mode });
    });
});

// Mood buttons
document.querySelectorAll('.mood-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        postJSON('/api/mood', { mood: btn.dataset.mood });
    });
});

// Quick commands
document.getElementById('cmd-look')?.addEventListener('click', () => {
    postJSON('/api/vision/analyze');
});

document.getElementById('cmd-weather')?.addEventListener('click', () => {
    postJSON('/api/browser/task', {
        instruction: 'Search for current weather and read the temperature',
        url: 'https://www.google.com',
    });
});

document.getElementById('cmd-news')?.addEventListener('click', () => {
    postJSON('/api/browser/task', {
        instruction: 'Go to Google News and read the top 3 headlines',
        url: 'https://news.google.com',
    });
});

document.getElementById('cmd-joke')?.addEventListener('click', () => {
    postJSON('/api/browser/task', {
        instruction: 'Search for a random joke and read it aloud',
        url: 'https://www.google.com',
    });
});

// Start polling
setInterval(pollState, POLL_INTERVAL);
pollState();
