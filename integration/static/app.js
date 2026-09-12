/**
 * app.js — LocateAnything Web Dashboard & Vision Grounding Interactive Studio
 */

// Color Palette for bounding boxes
const LABEL_COLORS = [
  '#06b6d4', // Cyan
  '#10b981', // Emerald
  '#f59e0b', // Amber
  '#ec4899', // Pink
  '#8b5cf6', // Purple
  '#3b82f6', // Blue
  '#f43f5e', // Rose
  '#14b8a6', // Teal
];

// App State
const state = {
  currentBlob: null,
  currentUrl: null,
  naturalWidth: 0,
  naturalHeight: 0,
  detections: [],
  labelColorMap: {},
  colorCounter: 0,
  hoveredIdx: null,
  servingNode: 'None',
  isBusy: false,
};

// DOM Elements
const DOM = {
  // Navigation & Badges
  clusterStatusBadge: document.getElementById('clusterStatusBadge'),
  clusterPulseDot: document.getElementById('clusterPulseDot'),
  clusterStatusText: document.getElementById('clusterStatusText'),
  
  // Inputs
  fileInput: document.getElementById('fileInput'),
  imageDropzone: document.getElementById('imageDropzone'),
  dropzoneContent: document.getElementById('dropzoneContent'),
  dropzonePreview: document.getElementById('dropzonePreview'),
  previewThumbnail: document.getElementById('previewThumbnail'),
  clearImageBtn: document.getElementById('clearImageBtn'),
  promptInput: document.getElementById('promptInput'),
  presetChips: document.getElementById('presetChips'),
  runDetectBtn: document.getElementById('runDetectBtn'),
  btnSpinner: document.getElementById('btnSpinner'),
  sampleCardsContainer: document.getElementById('sampleCardsContainer'),

  // Workspace / Canvas
  canvasViewport: document.getElementById('canvasViewport'),
  canvasWrapper: document.getElementById('canvasWrapper'),
  sourceImage: document.getElementById('sourceImage'),
  overlayCanvas: document.getElementById('overlayCanvas'),
  canvasTooltip: document.getElementById('canvasTooltip'),
  emptyState: document.getElementById('emptyState'),
  loadingOverlay: document.getElementById('loadingOverlay'),
  imageDimBadge: document.getElementById('imageDimBadge'),
  resetZoomBtn: document.getElementById('resetZoomBtn'),
  downloadAnnotatedBtn: document.getElementById('downloadAnnotatedBtn'),

  // Results & Telemetry
  metricLatency: document.getElementById('metricLatency'),
  metricCount: document.getElementById('metricCount'),
  metricNode: document.getElementById('metricNode'),
  tabDetectionsBtn: document.getElementById('tabDetectionsBtn'),
  tabJsonBtn: document.getElementById('tabJsonBtn'),
  tabDetections: document.getElementById('tabDetections'),
  tabJson: document.getElementById('tabJson'),
  detectionsList: document.getElementById('detectionsList'),
  detTabCount: document.getElementById('detTabCount'),
  jsonViewer: document.getElementById('jsonViewer'),
  copyJsonBtn: document.getElementById('copyJsonBtn'),

  // Cluster Monitoring
  clusterNodeCountBadge: document.getElementById('clusterNodeCountBadge'),
  clusterNodesList: document.getElementById('clusterNodesList'),
  footerTelemetry: document.getElementById('footerTelemetry'),
};

// ==============================================================================
// 1. INITIALIZATION & SAMPLES LOADING
// ==============================================================================
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  startClusterMonitoring();

  // Preload first sample (Bus)
  loadSampleImage('/samples/bus_in.png', 'bus');
});

function setupEventListeners() {
  // Sample card buttons
  DOM.sampleCardsContainer.addEventListener('click', (e) => {
    const btn = e.target.closest('.sample-btn');
    if (!btn) return;
    
    document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const imgUrl = btn.dataset.img;
    const prompt = btn.dataset.prompt;
    loadSampleImage(imgUrl, prompt);
  });

  // Prompt chip clicks
  DOM.presetChips.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;

    if (chip.dataset.mode === 'od') {
      const modeOD = document.getElementById('modeOD');
      if (modeOD) modeOD.checked = true;
      DOM.promptInput.value = '<OD>';
      return;
    }

    if (chip.dataset.mode === 'ocr') {
      const modeOCR = document.getElementById('modeOCR');
      if (modeOCR) modeOCR.checked = true;
      DOM.promptInput.value = '<OCR>';
      return;
    }

    const tag = chip.dataset.tag;
    if (!tag) return;

    // If current mode is OD or OCR, switch back to grounding
    const modeGrounding = document.getElementById('modeGrounding');
    if (modeGrounding && (document.getElementById('modeOD')?.checked || document.getElementById('modeOCR')?.checked)) {
      modeGrounding.checked = true;
    }

    const current = DOM.promptInput.value.trim();
    if (!current || current === '<OD>' || current === '<OCR>') {
      DOM.promptInput.value = tag;
    } else if (!current.includes(tag)) {
      DOM.promptInput.value = `${current}</c>${tag}`;
    }
  });

  // Task mode radio change listener
  document.querySelectorAll('input[name="decodeMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      if (e.target.value === 'od') {
        DOM.promptInput.value = '<OD>';
      } else if (e.target.value === 'ocr') {
        DOM.promptInput.value = '<OCR>';
      } else if (e.target.value === 'grounding' && (DOM.promptInput.value.trim() === '<OD>' || DOM.promptInput.value.trim() === '<OCR>')) {
        DOM.promptInput.value = 'bus';
      }
    });
  });

  // File Input Change
  DOM.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelected(file);
  });

  // Drag & Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    DOM.imageDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.imageDropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    DOM.imageDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.imageDropzone.classList.remove('drag-over');
    });
  });

  DOM.imageDropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFileSelected(file);
    }
  });

  // Clear Image
  DOM.clearImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearLoadedImage();
  });

  // Run Locate Objects
  DOM.runDetectBtn.addEventListener('click', runDetection);

  // Tabs
  DOM.tabDetectionsBtn.addEventListener('click', () => switchTab('detections'));
  DOM.tabJsonBtn.addEventListener('click', () => switchTab('json'));

  // Copy JSON
  DOM.copyJsonBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(DOM.jsonViewer.textContent);
    DOM.copyJsonBtn.textContent = 'Copied!';
    setTimeout(() => { DOM.copyJsonBtn.textContent = 'Copy JSON'; }, 2000);
  });

  // Download Annotated
  DOM.downloadAnnotatedBtn.addEventListener('click', downloadAnnotatedImage);

  // Reset Zoom
  DOM.resetZoomBtn.addEventListener('click', () => {
    DOM.sourceImage.style.transform = 'none';
  });

  // Canvas Mouse Move for Hover Detection
  DOM.overlayCanvas.addEventListener('mousemove', handleCanvasMouseMove);
  DOM.overlayCanvas.addEventListener('mouseleave', handleCanvasMouseLeave);

  // Window resize re-aligns canvas
  window.addEventListener('resize', syncCanvasDimensions);
}

// ==============================================================================
// 2. IMAGE HANDLING
// ==============================================================================
async function loadSampleImage(url, defaultPrompt) {
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const blob = await resp.blob();
    const file = new File([blob], url.split('/').pop(), { type: blob.type || 'image/png' });
    
    if (defaultPrompt) {
      DOM.promptInput.value = defaultPrompt;
    }
    
    handleFileSelected(file);
  } catch (err) {
    console.warn('Could not load sample image:', err);
  }
}

function handleFileSelected(file) {
  state.currentBlob = file;
  if (state.currentUrl) {
    URL.revokeObjectURL(state.currentUrl);
  }
  state.currentUrl = URL.createObjectURL(file);

  // Update Dropzone Thumbnail
  DOM.previewThumbnail.src = state.currentUrl;
  DOM.dropzoneContent.classList.add('hidden');
  DOM.dropzonePreview.classList.remove('hidden');

  // Load into main workspace image
  DOM.sourceImage.onload = () => {
    state.naturalWidth = DOM.sourceImage.naturalWidth;
    state.naturalHeight = DOM.sourceImage.naturalHeight;
    DOM.imageDimBadge.textContent = `${state.naturalWidth} × ${state.naturalHeight} px`;

    DOM.emptyState.classList.add('hidden');
    DOM.canvasWrapper.classList.remove('hidden');
    syncCanvasDimensions();
    clearCanvas();
  };
  DOM.sourceImage.src = state.currentUrl;

  // Reset detections
  state.detections = [];
  renderDetectionsList([]);
  DOM.detTabCount.textContent = '0';
  DOM.metricCount.textContent = '0';
}

function clearLoadedImage() {
  state.currentBlob = null;
  if (state.currentUrl) {
    URL.revokeObjectURL(state.currentUrl);
    state.currentUrl = null;
  }
  DOM.dropzoneContent.classList.remove('hidden');
  DOM.dropzonePreview.classList.add('hidden');
  DOM.fileInput.value = '';
  DOM.canvasWrapper.classList.add('hidden');
  DOM.emptyState.classList.remove('hidden');
  clearCanvas();
}

function syncCanvasDimensions() {
  if (!DOM.sourceImage || !DOM.sourceImage.clientWidth) return;
  const rect = DOM.sourceImage.getBoundingClientRect();
  const width = Math.round(rect.width);
  const height = Math.round(rect.height);

  DOM.overlayCanvas.width = width;
  DOM.overlayCanvas.height = height;
  DOM.overlayCanvas.style.width = `${width}px`;
  DOM.overlayCanvas.style.height = `${height}px`;

  if (state.detections.length > 0) {
    renderCanvasBoxes();
  }
}

// ==============================================================================
// 3. DETECTION WORKFLOW & API CALL
// ==============================================================================
async function runDetection() {
  if (!state.currentBlob) {
    alert('Please upload an image or choose a quick sample first.');
    return;
  }
  const selectedMode = document.querySelector('input[name="decodeMode"]:checked')?.value || 'grounding';
  let prompt = DOM.promptInput.value.trim();
  if (selectedMode === 'ocr') {
    if (!prompt) prompt = '<OCR>';
  } else if (selectedMode === 'od') {
    if (!prompt) prompt = '<OD>';
  }

  if (!prompt) {
    alert('Please enter at least one target description in the prompt.');
    DOM.promptInput.focus();
    return;
  }

  // Set loading state
  setBusy(true);

  const isOCR = (selectedMode === 'ocr');
  const endpoint = isOCR ? '/v1/ocr' : '/v1/locate';
  const formData = new FormData();
  formData.append('image', state.currentBlob);
  if (!isOCR) {
    formData.append('prompt', prompt);
    formData.append('mode', selectedMode);
  }

  const t0 = performance.now();
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errJson.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    const elapsedMs = Math.round(performance.now() - t0);
    const servedBy = response.headers.get('X-Cluster-Served-By') || 'Cluster Worker';

    // Store state
    if (isOCR) {
      state.detections = (data.regions || []).map(r => ({
        label: r.text,
        box: r.box
      }));
    } else {
      state.detections = data.detections || [];
    }
    state.servingNode = servedBy;

    // Update telemetry metrics
    DOM.metricLatency.textContent = `${Math.round(data.latency_ms || elapsedMs)} ms`;
    DOM.metricCount.textContent = state.detections.length;
    DOM.metricNode.textContent = servedBy;

    // Assign colors to labels
    assignLabelColors(state.detections);

    // Render Canvas & Lists
    syncCanvasDimensions();
    renderCanvasBoxes();
    renderDetectionsList(state.detections);

    // Update JSON viewer
    DOM.jsonViewer.textContent = JSON.stringify(data, null, 2);
    DOM.detTabCount.textContent = state.detections.length;

  } catch (err) {
    console.error('Detection error:', err);
    alert(`Detection failed: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  state.isBusy = busy;
  if (busy) {
    DOM.runDetectBtn.disabled = true;
    DOM.btnSpinner.classList.remove('hidden');
    DOM.loadingOverlay.classList.remove('hidden');
  } else {
    DOM.runDetectBtn.disabled = false;
    DOM.btnSpinner.classList.add('hidden');
    DOM.loadingOverlay.classList.add('hidden');
  }
}

function assignLabelColors(detections) {
  for (const det of detections) {
    if (!state.labelColorMap[det.label]) {
      state.labelColorMap[det.label] = LABEL_COLORS[state.colorCounter % LABEL_COLORS.length];
      state.colorCounter++;
    }
  }
}

// ==============================================================================
// 4. HTML5 CANVAS RENDERING & INTERACTIVE HOVER
// ==============================================================================
function clearCanvas() {
  const ctx = DOM.overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, DOM.overlayCanvas.width, DOM.overlayCanvas.height);
  DOM.canvasTooltip.classList.add('hidden');
}

function renderCanvasBoxes() {
  const canvas = DOM.overlayCanvas;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!state.naturalWidth || !state.naturalHeight || !state.detections.length) return;

  const scaleX = canvas.width / state.naturalWidth;
  const scaleY = canvas.height / state.naturalHeight;

  state.detections.forEach((det, idx) => {
    const isHovered = (state.hoveredIdx === idx);
    const color = state.labelColorMap[det.label] || '#06b6d4';
    const [x1, y1, x2, y2] = det.box;

    const rx = Math.round(x1 * scaleX);
    const ry = Math.round(y1 * scaleY);
    const rw = Math.round((x2 - x1) * scaleX);
    const rh = Math.round((y2 - y1) * scaleY);

    // Box stroke
    ctx.save();
    ctx.lineWidth = isHovered ? 3.5 : 2;
    ctx.strokeStyle = color;
    if (isHovered) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.fillStyle = `${color}25`; // 15% opacity fill
      ctx.fillRect(rx, ry, rw, rh);
    }
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.restore();

    // Box label badge
    const labelText = `${det.label}`;
    ctx.save();
    ctx.font = '600 11px Inter, sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    const badgeHeight = 18;
    const badgeWidth = textWidth + 12;

    const badgeX = rx;
    const badgeY = Math.max(0, ry - badgeHeight);

    // Badge background
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(badgeX, badgeY, badgeWidth, badgeHeight, [3, 3, 0, 0])
                  : ctx.fillRect(badgeX, badgeY, badgeWidth, badgeHeight);
    ctx.fill();

    // Badge text
    ctx.fillStyle = '#ffffff';
    ctx.fillText(labelText, badgeX + 6, badgeY + 13);
    ctx.restore();
  });
}

function handleCanvasMouseMove(e) {
  if (!state.detections.length || !state.naturalWidth) return;

  const canvas = DOM.overlayCanvas;
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const scaleX = canvas.width / state.naturalWidth;
  const scaleY = canvas.height / state.naturalHeight;

  let foundIdx = null;
  // Check in reverse order so top-most box gets selected first
  for (let i = state.detections.length - 1; i >= 0; i--) {
    const [x1, y1, x2, y2] = state.detections[i].box;
    const rx = x1 * scaleX;
    const ry = y1 * scaleY;
    const rw = (x2 - x1) * scaleX;
    const rh = (y2 - y1) * scaleY;

    if (mouseX >= rx && mouseX <= rx + rw && mouseY >= ry && mouseY <= ry + rh) {
      foundIdx = i;
      break;
    }
  }

  if (foundIdx !== state.hoveredIdx) {
    state.hoveredIdx = foundIdx;
    renderCanvasBoxes();
    highlightListItem(foundIdx);

    if (foundIdx !== null) {
      const det = state.detections[foundIdx];
      DOM.canvasTooltip.textContent = `${det.label} [${det.box.map(Math.round).join(', ')}]`;
      DOM.canvasTooltip.style.left = `${mouseX}px`;
      DOM.canvasTooltip.style.top = `${mouseY}px`;
      DOM.canvasTooltip.classList.remove('hidden');
    } else {
      DOM.canvasTooltip.classList.add('hidden');
    }
  }
}

function handleCanvasMouseLeave() {
  if (state.hoveredIdx !== null) {
    state.hoveredIdx = null;
    renderCanvasBoxes();
    highlightListItem(null);
    DOM.canvasTooltip.classList.add('hidden');
  }
}

// ==============================================================================
// 5. RESULTS LIST & SIDEBAR
// ==============================================================================
function renderDetectionsList(detections) {
  DOM.detectionsList.innerHTML = '';

  if (!detections || detections.length === 0) {
    DOM.detectionsList.innerHTML = `
      <div class="empty-results-msg">
        <span>No detections found. Try adjusting prompt keywords.</span>
      </div>`;
    return;
  }

  detections.forEach((det, idx) => {
    const color = state.labelColorMap[det.label] || '#06b6d4';
    const item = document.createElement('div');
    item.className = 'detection-item';
    item.id = `detItem_${idx}`;
    item.style.borderLeftColor = color;

    item.innerHTML = `
      <span class="detection-label" style="color: ${color};">${det.label}</span>
      <span class="detection-coords">${det.box.map(Math.round).join(', ')}</span>
    `;

    item.addEventListener('mouseenter', () => {
      state.hoveredIdx = idx;
      renderCanvasBoxes();
      item.classList.add('highlighted');
    });

    item.addEventListener('mouseleave', () => {
      state.hoveredIdx = null;
      renderCanvasBoxes();
      item.classList.remove('highlighted');
    });

    DOM.detectionsList.appendChild(item);
  });
}

function highlightListItem(idx) {
  document.querySelectorAll('.detection-item').forEach((el, i) => {
    if (i === idx) {
      el.classList.add('highlighted');
      el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    } else {
      el.classList.remove('highlighted');
    }
  });
}

function switchTab(tab) {
  if (tab === 'detections') {
    DOM.tabDetectionsBtn.classList.add('active');
    DOM.tabJsonBtn.classList.remove('active');
    DOM.tabDetections.classList.remove('hidden');
    DOM.tabJson.classList.add('hidden');
  } else {
    DOM.tabJsonBtn.classList.add('active');
    DOM.tabDetectionsBtn.classList.remove('active');
    DOM.tabJson.classList.remove('hidden');
    DOM.tabDetections.classList.add('hidden');
  }
}

// Download Annotated Canvas
function downloadAnnotatedImage() {
  if (!DOM.sourceImage || !DOM.sourceImage.naturalWidth) return;

  const offscreen = document.createElement('canvas');
  offscreen.width = state.naturalWidth;
  offscreen.height = state.naturalHeight;
  const ctx = offscreen.getContext('2d');

  // Draw source
  ctx.drawImage(DOM.sourceImage, 0, 0);

  // Draw detections in native resolution
  state.detections.forEach(det => {
    const color = state.labelColorMap[det.label] || '#06b6d4';
    const [x1, y1, x2, y2] = det.box;
    const w = x2 - x1;
    const h = y2 - y1;

    ctx.lineWidth = Math.max(3, Math.round(state.naturalWidth * 0.003));
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, w, h);

    ctx.font = `bold ${Math.max(16, Math.round(state.naturalWidth * 0.018))}px Inter, sans-serif`;
    const tw = ctx.measureText(det.label).width;
    const bh = Math.max(24, Math.round(state.naturalWidth * 0.025));
    ctx.fillStyle = color;
    ctx.fillRect(x1, Math.max(0, y1 - bh), tw + 14, bh);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(det.label, x1 + 7, Math.max(bh - 6, y1 - 6));
  });

  const link = document.createElement('a');
  link.download = `annotated_${Date.now()}.png`;
  link.href = offscreen.toDataURL('image/png');
  link.click();
}

// ==============================================================================
// 6. CLUSTER TELEMETRY & STATUS MONITORING
// ==============================================================================
function startClusterMonitoring() {
  fetchClusterTelemetry();
  setInterval(fetchClusterTelemetry, 4000);
}

async function fetchClusterTelemetry() {
  try {
    const resp = await fetch('/cluster/status');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    const nodes = data.nodes || [];
    const onlineNodes = nodes.filter(n => n.ready);

    // Update Badge
    if (onlineNodes.length > 0) {
      DOM.clusterPulseDot.className = 'pulse-dot';
      DOM.clusterStatusText.textContent = `${onlineNodes.length} Node(s) Online`;
    } else {
      DOM.clusterPulseDot.className = 'pulse-dot warning';
      DOM.clusterStatusText.textContent = '0 Ready Nodes';
    }

    // Update Nodes Summary
    DOM.clusterNodeCountBadge.textContent = `${nodes.length} Node(s)`;
    DOM.clusterNodesList.innerHTML = nodes.map(node => `
      <div class="node-item">
        <div class="node-item-header">
          <span class="node-indicator ${node.ready ? 'online' : 'offline'}"></span>
          <strong>${node.name}</strong>
          <span class="node-type-badge ${node.type}">${node.type}</span>
        </div>
        <div class="node-item-stats">
          <span>In-flight: ${node.in_flight_requests}</span>
          <span>Served: ${node.total_served}</span>
          <span>Avg: ${Math.round(node.avg_latency_ms)}ms</span>
        </div>
      </div>
    `).join('');

    // Footer summary
    DOM.footerTelemetry.textContent = `Cluster: ${onlineNodes.length}/${nodes.length} nodes ready | Total Served: ${data.total_served_requests} requests`;

  } catch (err) {
    DOM.clusterPulseDot.className = 'pulse-dot warning';
    DOM.clusterStatusText.textContent = 'Cluster Disconnected';
  }
}
