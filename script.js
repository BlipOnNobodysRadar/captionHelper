const chatLog = document.getElementById('chatLog');
const sendBtn = document.getElementById('sendBtn');
const clipInput = document.getElementById('clipInput');

const imageModeToggle = document.getElementById('imageMode');

function refreshAccept(){
  if (imageModeToggle && imageModeToggle.checked) {
    clipInput.setAttribute('accept', 'image/*');
  } else {
    clipInput.setAttribute('accept', 'video/*');
  }
}
if (imageModeToggle) {
  imageModeToggle.addEventListener('change', refreshAccept);
  refreshAccept();
}

function addMsg(who, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function postChatCaption(file) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('system_prompt', document.getElementById('systemPrompt').value);
  fd.append('num_frames', document.getElementById('numFrames').value);
  fd.append('sampling_type', document.getElementById('samplingType').value);
  fd.append('model', document.getElementById('modelName').value);
  fd.append('prefill', document.getElementById('prefill').value);
  fd.append('use_existing_caption', document.getElementById('useExistingCaption').checked);
  fd.append('existing_caption', document.getElementById('existingCaption').value);
  fd.append('image_mode', imageModeToggle && imageModeToggle.checked);

  const res = await fetch('/api/chat-caption', { method:'POST', body: fd });
  return await res.json();
}

sendBtn.addEventListener('click', async () => {
  const file = clipInput.files?.[0];
  if (!file) { addMsg('assistant', 'Attach a file first.'); return; }
  addMsg('user', `Attached: ${file.name}`);
  addMsg('assistant', 'Thinking... preparing inputs and querying LM Studio');

  const out = await postChatCaption(file);
  if (out.error) {
    addMsg('assistant', 'Error: ' + out.error);
  } else {
    const framesInfo = (typeof out.frames_used === 'number') ? ` [inputs: ${out.frames_used}] ` : ' ';
    addMsg('assistant', framesInfo + out.caption);
  }
});

// ---- Batch with progress & cancel ----
const startBtn = document.getElementById('startBatch');
const cancelBtn = document.getElementById('cancelBatch');
const batchLog = document.getElementById('batchLog');
const batchStatus = document.getElementById('batchStatus');
const batchProgress = document.getElementById('batchProgress');
const batchStats = document.getElementById('batchStats');
const activeFiles = document.getElementById('activeFiles');

let currentJobId = null;
let progressTimer = null;
let lastResultCount = 0;

function logBatch(line){
  batchLog.textContent += line + "\n";
  batchLog.scrollTop = batchLog.scrollHeight;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return '—';
  seconds = Math.max(0, Number(seconds));
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const rem = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${rem}s`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours}h ${mins}m`;
}

function formatRate(itemsPerMinute) {
  const n = Number(itemsPerMinute);
  if (!Number.isFinite(n) || n <= 0) return '—';
  return `${n.toFixed(n < 10 ? 2 : 1)}/min`;
}

function renderActiveFiles(active) {
  if (!activeFiles) return;
  if (!Array.isArray(active) || active.length === 0) {
    activeFiles.textContent = '';
    activeFiles.style.display = 'none';
    return;
  }

  activeFiles.style.display = 'block';
  activeFiles.textContent = active.map(item => {
    const file = typeof item === 'string' ? item : item.file;
    const elapsed = typeof item === 'string' ? null : item.elapsed_sec;
    return elapsed === null || elapsed === undefined ? `• ${file}` : `• ${file} (${formatDuration(elapsed)})`;
  }).join('\n');
}

function getBatchBody() {
  return {
    target_folder: document.getElementById('targetFolder').value,
    system_prompt: document.getElementById('systemPrompt').value,
    model: document.getElementById('modelName').value,
    prefill: document.getElementById('prefill').value,
    num_frames: Number(document.getElementById('numFrames').value),
    max_concurrent: Number(document.getElementById('maxConcurrent').value),
    abort_after_server_errors: Number(document.getElementById('abortAfterServerErrors').value),
    sampling_type: document.getElementById('samplingType').value,
    overwrite: document.getElementById('overwrite').checked,
    prepend_existing: document.getElementById('prependExisting').checked,
    use_existing_caption: document.getElementById('useExistingCaption').checked,
    image_mode: imageModeToggle && imageModeToggle.checked
  };
}

async function pollProgress() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/api/batch-progress?job_id=${encodeURIComponent(currentJobId)}`);
    const out = await res.json();
    if (out.error) {
      batchStatus.textContent = 'Error: ' + out.error;
      stopProgress();
      return;
    }
    const total = out.total || 0;
    const completed = out.completed || 0;
    const percent = total ? Math.floor((completed/total) * 100) : 0;
    batchProgress.value = percent;

    const parallel = out.max_concurrent || 1;
    const activeCount = out.active_count || (Array.isArray(out.active) ? out.active.length : 0);
    batchStatus.textContent = `${out.status} — ${completed}/${total} — parallel: ${parallel} — active: ${activeCount}`;
    if (out.abort_reason) {
      batchStatus.textContent += ` — ${out.abort_reason}`;
    }

    if (batchStats) {
      const elapsed = formatDuration(out.elapsed_sec);
      const eta = formatDuration(out.eta_sec);
      const rate = formatRate(out.throughput_per_min);
      const avg = formatDuration(out.avg_item_sec);
      batchStats.textContent = `elapsed: ${elapsed}  |  ETA: ${eta}  |  speed: ${rate}  |  avg item: ${avg}`;
    }
    renderActiveFiles(out.active);

    // Append only new lines
    if (Array.isArray(out.results) && out.results.length > lastResultCount) {
      for (let i = lastResultCount; i < out.results.length; i++) {
        const r = out.results[i];
        const took = r.duration_sec !== undefined ? ` [${formatDuration(r.duration_sec)}]` : '';
        if (r.ok) logBatch(`✓ ${r.file} -> ${r.out}${took}`);
        else if (r.skipped) logBatch(`↷ ${r.file} (skipped: ${r.reason})${took}`);
        else {
          const code = r.status_code ? ` [HTTP ${r.status_code}]` : '';
          logBatch(`✗ ${r.file}${code}: ${r.error}${took}`);
        }
      }
      lastResultCount = out.results.length;
    }

    if (out.status === 'done' || out.status === 'cancelled' || out.status === 'failed') {
      if (document.getElementById('notifyDone').checked) {
        if (out.status === 'done') alert('Batch complete.');
        else if (out.status === 'failed') alert(out.abort_reason || 'Batch failed.');
        else alert('Batch cancelled.');
      }
      stopProgress();
    }
  } catch (e) {
    batchStatus.textContent = 'Error: ' + e;
    stopProgress();
  }
}

function stopProgress() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
  currentJobId = null;
  startBtn.disabled = false;
  cancelBtn.disabled = true;
  startBtn.textContent = 'Start Batch Process';
}

startBtn.addEventListener('click', async () => {
  if (currentJobId) return; // already running
  batchLog.textContent = '';
  batchStatus.textContent = 'Starting...';
  if (batchStats) batchStats.textContent = '';
  if (activeFiles) { activeFiles.textContent = ''; activeFiles.style.display = 'none'; }
  batchProgress.value = 0;
  lastResultCount = 0;

  try {
    const body = getBatchBody();
    const res = await fetch('/api/batch-start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const out = await res.json();
    if (out.error) {
      batchStatus.textContent = 'Error: ' + out.error;
      return;
    }
    currentJobId = out.job_id;
    startBtn.disabled = true;
    startBtn.textContent = 'Running...';
    cancelBtn.disabled = false;
    batchStatus.textContent = 'queued';
    progressTimer = setInterval(pollProgress, 1000);
  } catch (e) {
    batchStatus.textContent = 'Error: ' + e;
  }
});

cancelBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  cancelBtn.disabled = true;
  try {
    await fetch('/api/batch-cancel', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({job_id: currentJobId})
    });
    // Let pollProgress pick up the 'cancelled' status
  } catch (e) {
    batchStatus.textContent = 'Error: ' + e;
    cancelBtn.disabled = false;
  }
});
