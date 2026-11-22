/**
 * Smart Crime Detection Dashboard — Frontend Logic
 * =================================================
 * - WebSocket connection to /ws/stream
 * - Renders annotated JPEG frames onto canvas
 * - Updates threat meter, algorithm indicators, incident feed
 * - Controls: start/stop, source selection, file upload
 */

/* ── Constants ─────────────────────────────────────────────────────── */
const API_BASE = "";
const WS_URL = `ws://${location.host}/ws/stream`;

const THREAT_COLORS = ["#4a7a3a", "#8a7a22", "#a05a10", "#8a1a0a", "#6a0808"];
const THREAT_LABELS = ["Clear", "Suspicious", "Weapon Detected", "Armed Threat", "Active Threat"];
const THREAT_CLASSES = ["level-none", "level-low", "level-medium", "level-high", "level-critical"];

/* ── State ──────────────────────────────────────────────────────────── */
let ws = null;
let isRunning = false;
let incidentCount = 0;
let currentThreatScore = 0;
let uploadedFilename = null;
let animFrame = null;
let statusInterval = null;

/* ── DOM Refs ───────────────────────────────────────────────────────── */
const canvas = document.getElementById("video-canvas");
const ctx = canvas.getContext("2d");
const waitingOverlay = document.getElementById("waiting-overlay");
const alertFlash = document.getElementById("alert-flash");
const statusBadge = document.getElementById("status-badge");
const fpsDisplay = document.getElementById("fps-display");
const frameDisplay = document.getElementById("frame-display");
const incidentCountEl = document.getElementById("incident-count");
const incidentList = document.getElementById("incident-list");
const arcFill = document.getElementById("threat-arc-fill");
const arcScoreText = document.getElementById("arc-score-text");
const threatLevelLabel = document.getElementById("threat-level-label");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnDemoGreen = document.getElementById("btn-demo-green");
const btnDownload = document.getElementById("btn-download");
const sourceSelect = document.getElementById("source-select");
const rtspInput = document.getElementById("rtsp-input");
const fileInput = document.getElementById("file-input");
const alg1Count = document.getElementById("alg1-count");
const alg2Count = document.getElementById("alg2-count");
const alg3Count = document.getElementById("alg3-count");
const alg4Count = document.getElementById("alg4-count");

/* ── Source Select Handler ──────────────────────────────────────────── */
sourceSelect.addEventListener("change", () => {
  rtspInput.classList.add("hidden");
  fileInput.classList.add("hidden");

  if (sourceSelect.value === "rtsp") {
    rtspInput.classList.remove("hidden");
  } else if (sourceSelect.value === "uploaded") {
    fileInput.classList.remove("hidden");
  }
});

/* ── File Upload ─────────────────────────────────────────────────────── */
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);
    btnStart.textContent = "[ Uploading... ]";
  btnStart.disabled = true;

  try {
    const resp = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
    const data = await resp.json();
    uploadedFilename = data.filename;
    btnStart.textContent = `▶ START (${file.name.slice(0, 20)})`;
    btnStart.disabled = false;
    showToast(`✅ Uploaded: ${file.name}`);
  } catch (err) {
    showToast(`❌ Upload failed: ${err}`, true);
    btnStart.textContent = "▶ START";
    btnStart.disabled = false;
  }
});

/* ── Start Stream ─────────────────────────────────────────────────────── */
btnStart.addEventListener("click", async () => {
  let source = "0";
  if (sourceSelect.value === "rtsp") {
    source = rtspInput.value.trim();
    if (!source) { showToast("Enter RTSP URL", true); return; }
  } else if (sourceSelect.value === "uploaded") {
    if (!uploadedFilename) { showToast("Upload a file first", true); return; }
    source = uploadedFilename;
  }

  try {
    const resp = await fetch(`${API_BASE}/start?source=${encodeURIComponent(source)}`, { method: "POST" });
    const data = await resp.json();
    if (resp.ok) {
      isRunning = true;
      startWebSocket();
      setStatus("online");
      btnStart.disabled = true;
      btnStop.disabled = false;
      btnDownload.disabled = false;
      setAlgorithmDots("active");
      showToast(`▶ Stream started: ${data.source}`);
    }
  } catch (err) {
    showToast(`❌ Start failed: ${err}`, true);
  }
});

/* ── Stop Stream ──────────────────────────────────────────────────────── */
btnStop.addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/stop`, { method: "POST" });
  } catch (_) {}

  stopWebSocket();
  isRunning = false;
  setStatus("offline");
  btnStart.disabled = false;
  btnStop.disabled = true;
  setAlgorithmDots("idle");
  waitingOverlay.classList.remove("hidden");
  alertFlash.classList.add("hidden");
  showToast("⏹ Stream stopped.");
});

/* ── Demo Safe Action (Green Frame Toggle) ────────────────────────── */
if (btnDemoGreen) {
  btnDemoGreen.addEventListener("click", async () => {
    try {
      const resp = await fetch(`${API_BASE}/demo_action`, { method: "POST" });
      const data = await resp.json();
      if (data.status === "ok") {
        if (data.demo_green_action) {
          btnDemoGreen.classList.add("active");
          btnDemoGreen.textContent = "> GREEN FRAME [ACTIVE]";
          showToast("🟢 Safe Action recognized: Green frame active!");
        } else {
          btnDemoGreen.classList.remove("active");
          btnDemoGreen.textContent = "> DEMO SAFE ACTION [G]";
          showToast("⚪ Green frame deactivated.");
        }
      }
    } catch (err) {
      showToast(`Error toggling demo action: ${err}`, true);
    }
  });
}

// Shortcut: press 'g' or 'G' to toggle demo action
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.key === "g" || e.key === "G") {
    if (btnDemoGreen) btnDemoGreen.click();
  }
});

/* ── WebSocket ────────────────────────────────────────────────────────── */
function startWebSocket() {
  if (ws) stopWebSocket();

  ws = new WebSocket(WS_URL);
  ws.binaryType = "blob";

  ws.onopen = () => console.log("[WS] Connected");
  ws.onerror = (e) => console.error("[WS] Error:", e);
  ws.onclose = () => {
    console.log("[WS] Closed");
    if (isRunning) setTimeout(startWebSocket, 1000); // reconnect
  };

  ws.onmessage = (event) => {
    if (typeof event.data !== "string") return;
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      console.warn("[WS] Non-JSON payload received:", e);
      return;
    }

    if (msg.type === "frame") {
      waitingOverlay.classList.add("hidden");
      fpsDisplay.textContent = msg.fps.toFixed(1);
      frameDisplay.textContent = msg.frame_id;
      drawBase64Frame(msg.data);
    } else if (msg.type === "alerts") {
      msg.alerts.forEach(handleAlert);
    } else if (msg.type === "waiting") {
      waitingOverlay.classList.remove("hidden");
    }
  };
}

function stopWebSocket() {
  if (ws) {
    ws.onclose = null;
    ws.close();
    ws = null;
  }
}

/* ── Frame Rendering ─────────────────────────────────────────────────── */
const _img = new Image();
_img.onload = () => {
  // Resize canvas to match image if needed
  if (canvas.width !== _img.naturalWidth) {
    canvas.width = _img.naturalWidth;
    canvas.height = _img.naturalHeight;
  }
  ctx.drawImage(_img, 0, 0, canvas.width, canvas.height);
};

function drawBase64Frame(b64) {
  _img.src = "data:image/jpeg;base64," + b64;
}

/* ── Alert Handling ──────────────────────────────────────────────────── */
function handleAlert(alert) {
  incidentCount++;
  incidentCountEl.textContent = incidentCount;
  updateThreatMeter(alert.threat_score, alert.threat_level, alert.threat_label);
  addIncidentCard(alert);

  // Flash warning on critical
  if (alert.threat_level >= 3) {
    setStatus("alert");
    alertFlash.classList.remove("hidden");
    alg4Count.textContent = incidentCount;
    setTimeout(() => {
      alertFlash.classList.add("hidden");
      if (isRunning) setStatus("online");
    }, 2000);
  }
}

/* ── Threat Meter ────────────────────────────────────────────────────── */
function updateThreatMeter(score, level, label) {
  const ARC_LEN = 251;
  const offset = ARC_LEN - (ARC_LEN * score / 100);
  const color = THREAT_COLORS[level] || THREAT_COLORS[0];

  arcFill.style.strokeDashoffset = offset;
  arcFill.setAttribute("stroke", color);
  arcScoreText.textContent = Math.round(score);
  arcScoreText.setAttribute("fill", color);

  threatLevelLabel.textContent = label;
  threatLevelLabel.className = "threat-level-label " + THREAT_CLASSES[level];

  // Gradually decay meter to 0 after 5 seconds of no alerts
  clearTimeout(window._meterDecay);
  window._meterDecay = setTimeout(() => {
    updateThreatMeter(0, 0, "CLEAR");
  }, 5000);
}

/* ── Incident Feed Card ──────────────────────────────────────────────── */
function addIncidentCard(alert) {
  // Remove placeholder
  const placeholder = incidentList.querySelector(".no-incidents");
  if (placeholder) placeholder.remove();

  const card = document.createElement("div");
  card.className = `incident-card level-${alert.threat_level}`;
  const color = THREAT_COLORS[alert.threat_level] || THREAT_COLORS[0];
  const tagsHtml = alert.action_tags.slice(0, 5)
    .map(t => `<span class="tag">${t}</span>`).join("");

  card.innerHTML = `
    <div class="inc-header">
      <span class="inc-time">${alert.timestamp_str}</span>
      <span class="inc-label" style="color:${color}">${alert.threat_label}</span>
      <span class="inc-score" style="color:${color}">${alert.threat_score.toFixed(0)}%</span>
    </div>
    <div class="inc-weapon">
      Person #${alert.person_track_id}
      ${alert.weapon_class ? `• <b>${alert.weapon_class.toUpperCase()}</b>` : ''}
      ${alert.target_track_id != null ? `• → Target #${alert.target_track_id}` : ''}
    </div>
    <div class="inc-tags">${tagsHtml}</div>
  `;

  // Prepend (newest first)
  incidentList.insertBefore(card, incidentList.firstChild);

  // Keep only last 50 cards
  while (incidentList.children.length > 50) {
    incidentList.removeChild(incidentList.lastChild);
  }
}

/* ── Status Polling (backup for algorithm counts) ──────────────────── */
async function pollStatus() {
  try {
    const resp = await fetch(`${API_BASE}/status`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (isRunning) {
      alg1Count.textContent = data.frame_count;
      alg2Count.textContent = data.frame_count;
      alg3Count.textContent = data.frame_count;
      alg4Count.textContent = data.incident_count;
    }
  } catch (_) {}
}

/* ── Helper Functions ─────────────────────────────────────────────────── */
function setStatus(state) {
  statusBadge.className = `status-badge ${state}`;
  statusBadge.textContent = state.toUpperCase();
}

function setAlgorithmDots(state) {
  ["alg1", "alg2", "alg3", "alg4"].forEach(id => {
    const dot = document.querySelector(`#${id} .algo-dot`);
    dot.className = `algo-dot dot-${state}`;
  });
}

function showToast(msg, isError = false) {
  console.log(`[Toast] ${msg}`);
  // Simple console log for now; could be a proper toast
}

function clearIncidents() {
  incidentList.innerHTML = '<div class="no-incidents">No incidents detected yet.</div>';
  incidentCount = 0;
  incidentCountEl.textContent = "0";
  updateThreatMeter(0, 0, "CLEAR");
}

/* ── Export Incidents ────────────────────────────────────────────────── */
btnDownload.addEventListener("click", async () => {
  try {
    const resp = await fetch(`${API_BASE}/incidents`);
    const data = await resp.json();
    const blob = new Blob([JSON.stringify(data.incidents, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `crime_incidents_${Date.now()}.json`;
    a.click();
  } catch (err) {
    console.error("Export failed:", err);
  }
});

/* ── Init ────────────────────────────────────────────────────────────── */
// Poll server status every 2s
statusInterval = setInterval(pollStatus, 2000);
// Set initial threat meter
updateThreatMeter(0, 0, "CLEAR");

/* ══════════════════════════════════════════════════════════════════════ */
/* TAB SWITCHING                                                          */
/* ══════════════════════════════════════════════════════════════════════ */
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const targetId = btn.getAttribute("data-tab");

    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

    btn.classList.add("active");
    const targetEl = document.getElementById(targetId);
    if (targetEl) targetEl.classList.add("active");

    if (targetId === "tab-watchlist") {
      loadWatchlist();
      loadRecentSightings();
    } else if (targetId === "tab-vehicles") {
      loadRecentVehicles();
    } else if (targetId === "tab-regulars") {
      loadStreetRegulars();
    }
  });
});

window.addEventListener("keydown", (e) => {
  if (e.key === "F1") { e.preventDefault(); document.querySelector('[data-tab="tab-live"]')?.click(); }
  if (e.key === "F2") { e.preventDefault(); document.querySelector('[data-tab="tab-watchlist"]')?.click(); }
  if (e.key === "F3") { e.preventDefault(); document.querySelector('[data-tab="tab-vehicles"]')?.click(); }
  if (e.key === "F4") { e.preventDefault(); document.querySelector('[data-tab="tab-regulars"]')?.click(); }
});

/* ══════════════════════════════════════════════════════════════════════ */
/* TAB 2: PERSON WATCHLIST (RE-ID)                                        */
/* ══════════════════════════════════════════════════════════════════════ */
const formEnroll = document.getElementById("form-enroll");
const targetPhotoInput = document.getElementById("target-photo");
const targetPreviewWrap = document.getElementById("target-preview-wrap");
const targetPreviewImg = document.getElementById("target-preview");
const watchlistGallery = document.getElementById("watchlist-gallery");
const sightingsList = document.getElementById("sightings-list");

// Photo preview on file select
if (targetPhotoInput) {
  targetPhotoInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        targetPreviewImg.src = ev.target.result;
        targetPreviewWrap.classList.remove("hidden");
      };
      reader.readAsDataURL(file);
    }
  });
}

// Enroll Target Submit
if (formEnroll) {
  formEnroll.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = document.getElementById("target-name").value.trim();
    const notes = document.getElementById("target-notes").value.trim();
    const file = targetPhotoInput.files[0];
    if (!name || !file) return;

    const btn = document.getElementById("btn-enroll");
    btn.disabled = true;
    btn.textContent = "> ENROLLING...";

    const formData = new FormData();
    formData.append("name", name);
    formData.append("notes", notes);
    formData.append("file", file);

    try {
      const resp = await fetch("/api/watchlist/enroll", {
        method: "POST",
        body: formData,
      });
      const data = await resp.json();
      if (resp.ok) {
        formEnroll.reset();
        targetPreviewWrap.classList.add("hidden");
        loadWatchlist();
        alert(`Target '${name}' enrolled successfully into visual memory.`);
      } else {
        alert(`Enrollment failed: ${data.detail || "Server error"}`);
      }
    } catch (err) {
      alert(`Network error: ${err}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "> ENROLL INTO MEMORY";
    }
  });
}

async function loadWatchlist() {
  if (!watchlistGallery) return;
  try {
    const resp = await fetch("/api/watchlist/list");
    const data = await resp.json();
    const persons = data.watchlist || [];

    if (persons.length === 0) {
      watchlistGallery.innerHTML = '<div class="no-incidents">No target persons enrolled. Use form on left to add targets.</div>';
      return;
    }

    watchlistGallery.innerHTML = persons.map(p => `
      <div class="target-card">
        <img src="${p.photo_url || '/ui/placeholder.png'}" class="target-thumb" alt="${p.name}" onerror="this.style.display='none'" />
        <div class="target-name">${p.name.toUpperCase()}</div>
        <div class="target-notes">${p.notes || "No additional case notes."}</div>
        <div class="target-footer">
          <span style="font-size:0.55rem; color:var(--text-dim);">ID #${p.id}</span>
          <button class="btn-delete" onclick="deleteWatchlistTarget(${p.id})">REMOVE</button>
        </div>
      </div>
    `).join("");
  } catch (err) {
    console.error("Error loading watchlist:", err);
  }
}

window.deleteWatchlistTarget = async function(id) {
  if (!confirm(`Delete target person #${id} from memory?`)) return;
  try {
    const resp = await fetch(`/api/watchlist/${id}`, { method: "DELETE" });
    if (resp.ok) loadWatchlist();
  } catch (err) {
    console.error("Delete failed:", err);
  }
};

async function loadRecentSightings() {
  if (!sightingsList) return;
  try {
    const resp = await fetch("/api/sightings/recent");
    const data = await resp.json();
    const sightings = data.sightings || [];

    if (sightings.length === 0) {
      sightingsList.innerHTML = '<div class="no-incidents">No live target sightings recorded yet.</div>';
      return;
    }

    sightingsList.innerHTML = sightings.map(s => `
      <div class="incident-card level-4">
        <div class="inc-header">
          <span class="inc-time">${s.timestamp_str}</span>
          <span class="inc-label" style="color:var(--threat-high);">TARGET IDENTIFIED</span>
          <span class="inc-score" style="color:var(--amber-bright);">${Math.round(s.confidence * 100)}% MATCH</span>
        </div>
        <div style="font-size:0.75rem; color:var(--amber-bright); font-weight:700;">
          ${s.person_name.toUpperCase()}
        </div>
        <div style="font-size:0.6rem; color:var(--tan-dim);">
          Source: ${s.camera_source} | Coordinates: (${s.x}, ${s.y}) | Frame: ${s.frame_id}
        </div>
        ${s.snapshot_url ? `<img src="${s.snapshot_url}" style="width:100%; height:90px; object-fit:cover; margin-top:4px; border:1px solid var(--border-color);" />` : ''}
      </div>
    `).join("");
  } catch (err) {
    console.error("Error loading sightings:", err);
  }
}

/* ══════════════════════════════════════════════════════════════════════ */
/* TAB 3: VEHICLE & LICENSE PLATE REGISTRY (ANPR)                         */
/* ══════════════════════════════════════════════════════════════════════ */
const plateSearchInput = document.getElementById("plate-search-input");
const btnSearchPlate = document.getElementById("btn-search-plate");
const btnRefreshVehicles = document.getElementById("btn-refresh-vehicles");
const vehicleCardsContainer = document.getElementById("vehicle-cards-container");
const vehicleCountBadge = document.getElementById("vehicle-count-badge");
const vehicleResultsTitle = document.getElementById("vehicle-results-title");

if (btnSearchPlate) {
  btnSearchPlate.addEventListener("click", () => searchPlates());
}
if (plateSearchInput) {
  plateSearchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") searchPlates();
  });
}
if (btnRefreshVehicles) {
  btnRefreshVehicles.addEventListener("click", () => loadRecentVehicles());
}

async function searchPlates() {
  const query = plateSearchInput.value.trim();
  if (!query) {
    loadRecentVehicles();
    return;
  }

  vehicleResultsTitle.textContent = `// Search Results for: "${query.toUpperCase()}"`;
  vehicleCardsContainer.innerHTML = '<div class="no-incidents">Searching vehicle database...</div>';

  try {
    const resp = await fetch(`/api/vehicles/search?plate=${encodeURIComponent(query)}`);
    const data = await resp.json();
    renderVehicleCards(data.vehicles || []);
  } catch (err) {
    vehicleCardsContainer.innerHTML = `<div class="no-incidents">Search failed: ${err}</div>`;
  }
}

async function loadRecentVehicles() {
  if (!vehicleCardsContainer) return;
  vehicleResultsTitle.textContent = '// Registered Vehicle Passes';
  vehicleCardsContainer.innerHTML = '<div class="no-incidents">Loading vehicle records...</div>';

  try {
    const resp = await fetch("/api/vehicles/recent");
    const data = await resp.json();
    renderVehicleCards(data.vehicles || []);
  } catch (err) {
    vehicleCardsContainer.innerHTML = `<div class="no-incidents">Load failed: ${err}</div>`;
  }
}

function renderVehicleCards(vehicles) {
  if (!vehicleCardsContainer) return;

  vehicleCountBadge.textContent = `${vehicles.length} VEHICLES`;

  if (vehicles.length === 0) {
    vehicleCardsContainer.innerHTML = '<div class="no-incidents">No matching vehicle records found.</div>';
    return;
  }

  vehicleCardsContainer.innerHTML = vehicles.map(v => {
    const trajStr = (v.trajectory && v.trajectory.length > 0)
      ? v.trajectory.map(p => `(${p[0]},${p[1]})`).join(" → ")
      : "No coordinate path";

    return `
      <div class="vehicle-card">
        <div class="veh-header">
          <span class="veh-plate">${v.plate_number}</span>
          <span class="veh-type">${v.vehicle_type}</span>
        </div>
        ${v.snapshot_url ? `<img src="${v.snapshot_url}" class="veh-thumb" alt="${v.plate_number}" onerror="this.style.display='none'" />` : ''}
        <div class="veh-meta">
          <div><b>First Seen:</b> ${v.first_seen_str}</div>
          <div><b>Last Seen:</b> ${v.last_seen_str}</div>
          <div><b>Passes Count:</b> ${v.sightings_count} time(s)</div>
          <div><b>Vector:</b> ${Math.round(v.direction_deg)}° @ ${v.avg_speed.toFixed(1)} px/s</div>
        </div>
        <div class="ctrl-label" style="margin-top:2px;">Movement Trail:</div>
        <div class="veh-traj">${trajStr}</div>
      </div>
    `;
  }).join("");
}

/* ══════════════════════════════════════════════════════════════════════ */
/* TAB 4: STREET REGULARS & JOBS (CROWDED STREET AI)                     */
/* ══════════════════════════════════════════════════════════════════════ */
const regularsCardsContainer = document.getElementById("regulars-cards-container");
const regularsCountBadge = document.getElementById("regulars-count-badge");
const regTotalCount = document.getElementById("reg-total-count");
const regVendorCount = document.getElementById("reg-vendor-count");
const regShopCount = document.getElementById("reg-shop-count");
const regTransitCount = document.getElementById("reg-transit-count");
const regLoiterCount = document.getElementById("reg-loiter-count");

async function loadStreetRegulars() {
  if (!regularsCardsContainer) return;
  regularsCardsContainer.innerHTML = '<div class="no-incidents">Loading street regular profiles...</div>';

  try {
    const resp = await fetch("/api/regulars/list");
    const data = await resp.json();
    const regulars = data.regulars || [];
    renderRegularCards(regulars);
  } catch (err) {
    regularsCardsContainer.innerHTML = `<div class="no-incidents">Failed loading regular profiles: ${err}</div>`;
  }
}

function renderRegularCards(regulars) {
  if (!regularsCardsContainer) return;

  // Update metrics
  if (regTotalCount) regTotalCount.textContent = regulars.length;
  if (regularsCountBadge) regularsCountBadge.textContent = `${regulars.length} REGULARS DETECTED`;

  let vendors = 0, shops = 0, transit = 0, loiter = 0;
  regulars.forEach(r => {
    const role = (r.detected_role || "").toUpperCase();
    if (role === "STREET_VENDOR") vendors++;
    else if (role === "SHOPKEEPER") shops++;
    else if (role === "COMMUTER" || role === "DELIVERY_WORKER") transit++;
    else if (role === "SUSPICIOUS_LOITERER") loiter++;
  });

  if (regVendorCount) regVendorCount.textContent = vendors;
  if (regShopCount) regShopCount.textContent = shops;
  if (regTransitCount) regTransitCount.textContent = transit;
  if (regLoiterCount) regLoiterCount.textContent = loiter;

  if (regulars.length === 0) {
    regularsCardsContainer.innerHTML = '<div class="no-incidents">No street regular profiles discovered yet. Run surveillance feed with street video to discover vendors, workers, and regulars.</div>';
    return;
  }

  regularsCardsContainer.innerHTML = regulars.map(r => {
    const role = (r.detected_role || "UNKNOWN").toLowerCase();
    let badgeClass = "badge-vendor";
    if (role === "shopkeeper") badgeClass = "badge-shop";
    else if (role === "commuter" || role === "delivery_worker") badgeClass = "badge-transit";
    else if (role === "suspicious_loiterer") badgeClass = "badge-loiter";

    const displayName = r.custom_name ? r.custom_name : r.alias;
    const hoursStr = (r.active_hours && r.active_hours.length > 0)
      ? r.active_hours.map(h => `${h}:00`).join(", ")
      : "Throughout Day";

    return `
      <div class="regular-card role-${role}">
        <div class="reg-card-top">
          ${r.snapshot_url 
            ? `<img src="${r.snapshot_url}" class="reg-thumb" alt="${r.alias}" onerror="this.style.display='none'" />` 
            : `<div class="reg-thumb" style="display:flex;align-items:center;justify-content:center;color:var(--text-dim);font-size:0.6rem;">NO PHOTO</div>`
          }
          <div class="reg-info">
            <div class="reg-alias">// ID: ${r.alias}</div>
            <div class="reg-name" title="${displayName}">${displayName}</div>
            <div class="reg-role-badge ${badgeClass}">${r.detected_role} (${Math.round(r.role_confidence * 100)}%)</div>
            
            <div class="reg-stats">
              <div><b>Days Seen:</b> ${r.days_seen_count} day(s)</div>
              <div><b>Appearances:</b> ${r.total_appearances} session(s)</div>
              <div><b>Total Dwell:</b> ${Math.round(r.total_dwell_seconds)} sec</div>
              <div><b>Active Hours:</b> ${hoursStr}</div>
              <div><b>Anchor Pos:</b> (${r.anchor_x}, ${r.anchor_y})</div>
              <div><b>First Seen:</b> ${r.first_seen_str}</div>
              <div><b>Last Seen:</b> ${r.last_seen_str}</div>
            </div>

            <div class="reg-actions">
              <button class="btn-reg-action" onclick="renameRegularPrompt(${r.id}, '${escapeQuote(displayName)}')">&gt; RENAME</button>
              <button class="btn-reg-action btn-reg-del" onclick="deleteRegularConfirm(${r.id})">&gt; REMOVE</button>
            </div>
          </div>
        </div>
      </div>
    `;
  }).join("");
}

function escapeQuote(str) {
  return (str || "").replace(/'/g, "\\'");
}

async function renameRegularPrompt(id, currentName) {
  const newName = prompt("Enter custom name/job title for this street regular (e.g. Ramesh Chaiwala, Fruit Cart Uncle):", currentName);
  if (newName === null || !newName.trim()) return;
  
  const formData = new FormData();
  formData.append("custom_name", newName.trim());

  try {
    const resp = await fetch(`/api/regulars/${id}/rename`, {
      method: "POST",
      body: formData,
    });
    if (resp.ok) {
      showToast(`Renamed regular to '${newName}'`);
      loadStreetRegulars();
    } else {
      alert("Failed to rename regular");
    }
  } catch (err) {
    alert(`Error: ${err}`);
  }
}

async function deleteRegularConfirm(id) {
  if (!confirm("Are you sure you want to remove this street regular profile?")) return;
  try {
    const resp = await fetch(`/api/regulars/${id}`, { method: "DELETE" });
    if (resp.ok) {
      showToast("Regular profile removed");
      loadStreetRegulars();
    } else {
      alert("Failed to delete regular profile");
    }
  } catch (err) {
    alert(`Error: ${err}`);
  }
}


