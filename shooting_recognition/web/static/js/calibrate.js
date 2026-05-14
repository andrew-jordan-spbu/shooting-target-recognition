const canvas    = document.getElementById('calib-canvas');
const ctx       = canvas.getContext('2d');
const zonesList = document.getElementById('zones-list');
const hint      = document.getElementById('calib-hint');
const statusEl  = document.getElementById('detect-status');
const semiBanner = document.getElementById('semi-banner');
const semiTitle  = document.getElementById('semi-title');
const zoneDesc   = document.getElementById('zone-desc');

// ── State ─────────────────────────────────────────────────────────────────
let zones    = [];
let bgImage  = null;
let editMode = false;
let dragIdx  = -1;
let dragMode = '';

// Semi-auto state
let mode = 'auto';   // 'auto' | 'semi'
let semiScore = 10;  // next zone to collect (10 → 1)
let targetCx  = null;
let targetCy  = null;
let previewR  = 0;   // live preview radius while hovering in semi mode

// ── Mode switching ────────────────────────────────────────────────────────

function switchMode(val) {
  mode = val;
  if (mode === 'semi') {
    semiBanner.classList.remove('hidden');
    zoneDesc.style.display = 'none';
    hint.classList.add('hidden');
    editMode = false;
    resetSemiAuto();
  } else {
    semiBanner.classList.add('hidden');
    zoneDesc.style.display = '';
    semiScore = 10;
  }
  render();
}

function resetSemiAuto() {
  semiScore = 10;
  zones     = [];
  targetCx  = null;   // must re-click centre on each reset
  targetCy  = null;
  previewR  = 0;
  updateSemiStatus();
  renderZoneList();
  render();
}

function updateSemiStatus() {
  if (targetCx === null) {
    semiTitle.textContent =
      'Step 1 — Click the TARGET CENTRE (bullseye dot)';
  } else if (semiScore === 10) {
    semiTitle.textContent =
      'Step 2 — Move mouse to the Zone 10 boundary, then click';
  } else if (semiScore > 0) {
    semiTitle.textContent =
      `Move mouse to Zone ${semiScore} boundary, then click`;
  } else {
    semiTitle.textContent = '✓ All zones set — confirm or edit.';
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function effectiveR(z) {
  return z.rx !== undefined
    ? Math.round(Math.sqrt(z.rx * z.ry))
    : z.radius;
}

function drawZoneRing(z, selected) {
  const color = z.score >= 7 ? '#00e676' : '#ffab40';
  ctx.strokeStyle = selected ? '#ffffff' : color;
  ctx.lineWidth   = selected ? 2.5 : 1.5;
  ctx.fillStyle   = color;
  ctx.font        = 'bold 11px monospace';

  ctx.beginPath();
  if (z.rx !== undefined && z.ry !== undefined) {
    ctx.save();
    ctx.translate(z.cx, z.cy);
    ctx.rotate((z.angle || 0) * Math.PI / 180);
    ctx.ellipse(0, 0, z.rx, z.ry, 0, 0, Math.PI * 2);
    ctx.restore();
    ctx.stroke();
    ctx.fillText(String(z.score), z.cx - z.rx + 5, z.cy - z.ry + 14);
  } else {
    ctx.arc(z.cx, z.cy, z.radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillText(String(z.score),
      z.cx - z.radius + 5, z.cy - z.radius + 14);
  }
}

function scaleZone(z, ratio) {
  if (z.rx !== undefined && z.ry !== undefined) {
    return {
      ...z,
      rx: Math.max(2, Math.round(z.rx * ratio)),
      ry: Math.max(2, Math.round(z.ry * ratio)),
    };
  }
  return { ...z, radius: Math.max(4, Math.round(z.radius * ratio)) };
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function loadImage(src) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });
}

// ── Initialise ────────────────────────────────────────────────────────────

async function init() {
  statusEl.textContent = 'Collecting calibration frames…';

  // Wait for first frame
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
      statusEl.textContent =
        '✗ Pipeline stopped. Check camera/stream settings and restart.';
      statusEl.style.color = '#ef4444';
      return;
    }
    if (frameResp.status === 200) {
      const blob = await frameResp.blob();
      bgImage = await loadImage(URL.createObjectURL(blob));
      canvas.width  = bgImage.naturalWidth  || 640;
      canvas.height = bgImage.naturalHeight || 480;
      break;
    }
    await sleep(500);
    attempts++;
  }

  // Poll for auto-detected zones (used for centre in both modes)
  if (mode === 'semi') {
    // In semi-auto the centre comes from the user's first click (zone 10).
    // No auto-detection needed — skip the polling loop.
    statusEl.textContent = 'Ready. Click inside Zone 10 (bullseye) to begin.';
  } else {
    // Auto mode: poll for detected zones
    statusEl.textContent = 'Running auto-detection…';
    let zonesFound = false;
    for (let i = 0; i < 30; i++) {
      const resp = await fetch('/calibrate/zones');
      const data = await resp.json().catch(() => ({}));
      if (data && data.zones && data.zones.length) {
        zones = data.zones;
        targetCx = data.zones[0].cx;
        targetCy = data.zones[0].cy;
        statusEl.textContent = data.auto_detected
          ? '✓ Target auto-detected. Confirm or edit zones below.'
          : '⚠ Auto-detect failed — fallback zones. Try semi-auto mode.';
        zonesFound = true;
        break;
      }
      await sleep(500);
    }
    if (!zonesFound) {
      statusEl.textContent =
        '⚠ Auto-detection timed out. Edit zones manually or switch to semi-auto.';
    }
  }

  if (mode === 'semi') {
    resetSemiAuto();
  }
  renderZoneList();
  render();
}

// ── Render ────────────────────────────────────────────────────────────────

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (bgImage) ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);

  zones.forEach((z, i) => drawZoneRing(z, editMode && i === dragIdx));

  // Semi-auto: crosshair at confirmed centre
  if (mode === 'semi' && targetCx !== null) {
    ctx.save();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(targetCx - 14, targetCy); ctx.lineTo(targetCx + 14, targetCy);
    ctx.moveTo(targetCx, targetCy - 14); ctx.lineTo(targetCx, targetCy + 14);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(targetCx, targetCy, 4, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // Semi-auto: live preview circle (all zones, once centre is known)
  if (mode === 'semi' && semiScore > 0 && previewR > 0 && targetCx !== null) {
    const color = semiScore >= 7 ? '#00e676' : '#ffab40';
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.65;
    ctx.beginPath();
    ctx.arc(targetCx, targetCy, previewR, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = color;
    ctx.font = 'bold 13px monospace';
    ctx.fillText(
      String(semiScore),
      targetCx - 6,
      targetCy - previewR - 6,
    );
  }

  // Centre crosshair
  const showCross = (editMode && zones.length) ||
                    (mode === 'semi' && targetCx !== null);
  if (showCross) {
    const cx = (zones.length ? zones[0].cx : targetCx) ?? canvas.width / 2;
    const cy = (zones.length ? zones[0].cy : targetCy) ?? canvas.height / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy);
    ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12);
    ctx.stroke();
  }
}

// ── Zone list sidebar ─────────────────────────────────────────────────────

function renderZoneList() {
  const sorted = [...zones].sort((a, b) => b.score - a.score); // 10 → 1
  zonesList.innerHTML = sorted.map((z, i) => {
    const origIdx = zones.indexOf(z);
    const sizeLabel = z.rx !== undefined
      ? `${z.rx}×${z.ry}px`
      : `r=${z.radius}px`;
    return `<div class="zone-row">
      <span>Zone ${11 - z.score} &nbsp;
        <small style="color:#4b5563">${sizeLabel}</small></span>
      <input type="number" min="0" max="10" value="${z.score}"
        ${editMode ? '' : 'disabled'}
        onchange="updateZoneScore(${origIdx}, this.value)">
      <span>pts</span>
    </div>`;
  }).join('');
}

function updateZoneScore(idx, val) {
  zones[idx].score = parseInt(val) || 0;
}

// ── Edit / confirm ────────────────────────────────────────────────────────

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
  if (resp.ok) window.location.href = '/session';
}

// ── Semi-auto: canvas click places next zone ──────────────────────────────

canvas.addEventListener('click', e => {
  if (mode !== 'semi') return;
  if (dragIdx !== -1) return;   // ignore if mid-drag in edit mode

  const { x, y } = getCanvasPos(e);

  // ── Step 1: set target centre ─────────────────────────────────────────
  if (targetCx === null) {
    targetCx = Math.round(x);
    targetCy = Math.round(y);
    previewR = 0;
    updateSemiStatus();
    render();
    return;
  }

  // ── Step 2+: all zones done? ──────────────────────────────────────────
  if (semiScore <= 0) return;

  // Place zone boundary at raw distance from known centre.
  // No server call, no snapping — the user sees the live preview circle
  // and clicks exactly on the ring line they want.
  const radius = Math.round(Math.hypot(x - targetCx, y - targetCy));
  zones = zones.filter(z => z.score !== semiScore);
  zones.push({ score: semiScore, cx: targetCx, cy: targetCy, radius });
  zones.sort((a, b) => b.score - a.score);   // innermost first

  semiScore--;
  previewR = 0;
  updateSemiStatus();
  renderZoneList();
  render();
});

// ── Semi-auto preview: live circle that follows the mouse ─────────────────


// ── Edit mode: drag interactions ──────────────────────────────────────────

canvas.addEventListener('mousedown', e => {
  if (!editMode) return;
  const { x, y } = getCanvasPos(e);
  dragIdx = -1; dragMode = '';

  for (let i = 0; i < zones.length; i++) {
    const d = Math.hypot(x - zones[i].cx, y - zones[i].cy);
    if (Math.abs(d - effectiveR(zones[i])) <= 14) {
      dragIdx = i; dragMode = 'resize'; break;
    }
  }
  if (dragIdx === -1) {
    for (let i = 0; i < zones.length; i++) {
      if (Math.hypot(x - zones[i].cx, y - zones[i].cy) <= 16) {
        dragIdx = i; dragMode = 'move'; break;
      }
    }
  }
  render();
});

canvas.addEventListener('mousemove', e => {
  const { x, y } = getCanvasPos(e);

  // Semi-auto mode: show preview circle whenever centre is known
  if (mode === 'semi' && semiScore > 0 && targetCx !== null) {
    previewR = Math.round(Math.hypot(x - targetCx, y - targetCy));
    render();
    return;
  }

  // Edit mode drag
  if (dragIdx === -1 || !editMode) return;
  const z = zones[dragIdx];

  if (dragMode === 'resize') {
    const newR  = Math.max(4, Math.round(Math.hypot(x - z.cx, y - z.cy)));
    const ratio = newR / effectiveR(z);
    zones = zones.map(zz => scaleZone(zz, ratio));
  } else {
    const dx = x - z.cx, dy = y - z.cy;
    zones = zones.map(zz => ({ ...zz, cx: zz.cx + dx, cy: zz.cy + dy }));
    // Keep targetCx/Cy in sync
    if (zones.length) { targetCx = zones[0].cx; targetCy = zones[0].cy; }
  }
  renderZoneList();
  render();
});

canvas.addEventListener('mouseup',    () => { dragIdx = -1; dragMode = ''; });
canvas.addEventListener('mouseleave', () => { dragIdx = -1; dragMode = ''; });

// Double-click: move all zone centres (and targetCx/Cy) to clicked point
canvas.addEventListener('dblclick', e => {
  const { x, y } = getCanvasPos(e);
  targetCx = Math.round(x);
  targetCy = Math.round(y);
  if (zones.length) {
    zones = zones.map(z => ({ ...z, cx: targetCx, cy: targetCy }));
  }
  renderZoneList();
  render();
});

function getCanvasPos(e) {
  const rect   = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY,
  };
}

// ── Boot ──────────────────────────────────────────────────────────────────
init();
