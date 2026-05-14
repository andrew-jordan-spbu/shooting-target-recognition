const canvas     = document.getElementById('calib-canvas');
const ctx        = canvas.getContext('2d');
const zonesList  = document.getElementById('zones-list');
const hint       = document.getElementById('calib-hint');
const statusEl   = document.getElementById('detect-status');

let zones    = [];
let bgImage  = null;
let editMode = false;
let dragIdx  = -1;
let dragMode = '';   // 'move' | 'resize'

// ── Initialise ──────────────────────────────────────────────────────────────
async function init() {
  statusEl.textContent = 'Collecting calibration frames…';

  // Poll until calibration frame is available (pipeline needs CALIBRATION_FRAMES)
  let attempts = 0;
  while (attempts < 60) {
    const [frameResp, statusResp] = await Promise.all([
      fetch('/calibrate/frame'),
      fetch('/status'),
    ]);
    const statusData = await statusResp.json().catch(() => ({}));
    if (statusData.error) {
      statusEl.textContent = '✗ Pipeline error: ' + statusData.error;
      statusEl.style.color = '#ef4444';
      return;
    }
    if (!statusData.running && attempts > 2) {
      statusEl.textContent = '✗ Pipeline stopped unexpectedly. Check camera/stream settings and restart.';
      statusEl.style.color = '#ef4444';
      return;
    }
    if (frameResp.status === 200) {
      const blob = await frameResp.blob();
      const url  = URL.createObjectURL(blob);
      bgImage = await loadImage(url);
      canvas.width  = bgImage.naturalWidth  || 640;
      canvas.height = bgImage.naturalHeight || 480;
      break;
    }
    await sleep(500);
    attempts++;
  }

  statusEl.textContent = 'Running auto-detection…';
  let zonesFound = false;
  for (let i = 0; i < 30; i++) {   // poll up to 15 s
    const zonesResp = await fetch('/calibrate/zones');
    const data = await zonesResp.json().catch(() => ({}));
    if (data && data.zones && data.zones.length) {
      zones = data.zones;
      statusEl.textContent = data.auto_detected
        ? '✓ Target auto-detected. Confirm or edit zones below.'
        : '⚠ Auto-detect failed — using fallback centred zones. Please edit manually.';
      zonesFound = true;
      break;
    }
    await sleep(500);
  }

  if (!zonesFound) {
    statusEl.textContent = '⚠ Auto-detection timed out. Please edit zones manually.';
  }

  renderZoneList();
  render();
}

function loadImage(src) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Render ───────────────────────────────────────────────────────────────────
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (bgImage) ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);

  zones.forEach((z, i) => {
    const isSelected = editMode && i === dragIdx;
    const color = z.score >= 7 ? '#00e676' : '#ffab40';

    ctx.beginPath();
    ctx.arc(z.cx, z.cy, z.radius, 0, Math.PI * 2);
    ctx.strokeStyle = isSelected ? '#ffffff' : color;
    ctx.lineWidth   = isSelected ? 2.5 : 1.5;
    ctx.stroke();

    // Score label near inner edge
    ctx.fillStyle = color;
    ctx.font = 'bold 11px monospace';
    ctx.fillText(String(z.score), z.cx - z.radius + 5, z.cy - z.radius + 14);
  });

  // Draw centre crosshair if in edit mode
  if (editMode && zones.length) {
    const z = zones[0];
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(z.cx - 10, z.cy); ctx.lineTo(z.cx + 10, z.cy);
    ctx.moveTo(z.cx, z.cy - 10); ctx.lineTo(z.cx, z.cy + 10);
    ctx.stroke();
  }
}

// ── Zone list sidebar ─────────────────────────────────────────────────────────
function renderZoneList() {
  zonesList.innerHTML = zones.map((z, i) =>
    `<div class="zone-row">
      <span>Zone ${11 - z.score} &nbsp; <small style="color:#4b5563">r=${z.radius}px</small></span>
      <input type="number" min="0" max="10" value="${z.score}"
        ${editMode ? '' : 'disabled'}
        onchange="updateZoneScore(${i}, this.value)">
      <span>pts</span>
    </div>`
  ).join('');
}

function updateZoneScore(idx, val) {
  zones[idx].score = parseInt(val) || 0;
}

// ── Edit / confirm ────────────────────────────────────────────────────────────
function enableEdit() {
  editMode = true;
  hint.classList.remove('hidden');
  renderZoneList();
  render();
}

async function confirmZones() {
  const payload = { zones, auto_detected: true };
  const resp = await fetch('/calibrate/zones', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (resp.ok) {
    window.location.href = '/session';
  }
}

// ── Drag interactions ─────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {
  if (!editMode) return;
  const { x, y } = getCanvasPos(e);
  dragIdx = -1;
  dragMode = '';

  // Prefer innermost zone for edge-resize, then centre-move
  for (let i = 0; i < zones.length; i++) {
    const z = zones[i];
    const d = Math.hypot(x - z.cx, y - z.cy);
    if (Math.abs(d - z.radius) <= 14) {
      dragIdx = i;
      dragMode = 'resize';
      break;
    }
  }
  if (dragIdx === -1) {
    // Check near any zone centre → move all
    for (let i = 0; i < zones.length; i++) {
      if (Math.hypot(x - zones[i].cx, y - zones[i].cy) <= 16) {
        dragIdx = i;
        dragMode = 'move';
        break;
      }
    }
  }
  render();
});

canvas.addEventListener('mousemove', e => {
  if (dragIdx === -1 || !editMode) return;
  const { x, y } = getCanvasPos(e);
  const z = zones[dragIdx];

  if (dragMode === 'resize') {
    const newR = Math.max(4, Math.round(Math.hypot(x - z.cx, y - z.cy)));
    // Scale all zones proportionally relative to this one's change
    const ratio = newR / z.radius;
    zones = zones.map((zz, i) => ({
      ...zz,
      radius: i === dragIdx ? newR : Math.max(4, Math.round(zz.radius * ratio)),
    }));
  } else {
    // Move all zones together
    const dx = x - z.cx;
    const dy = y - z.cy;
    zones = zones.map(zz => ({ ...zz, cx: zz.cx + dx, cy: zz.cy + dy }));
  }

  renderZoneList();
  render();
});

canvas.addEventListener('mouseup',    () => { dragIdx = -1; dragMode = ''; });
canvas.addEventListener('mouseleave', () => { dragIdx = -1; dragMode = ''; });

function getCanvasPos(e) {
  const rect   = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY,
  };
}

// ── Boot ──────────────────────────────────────────────────────────────────────
init();
