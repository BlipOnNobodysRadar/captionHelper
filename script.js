const chatLog = document.getElementById('chatLog');
const sendBtn = document.getElementById('sendBtn');
const clipInput = document.getElementById('clipInput');

const imageModeToggle = document.getElementById('imageMode');
const captionPresetSelect = document.getElementById('captionPreset');
const savePresetBtn = document.getElementById('savePreset');
const deletePresetBtn = document.getElementById('deletePreset');
const presetStatus = document.getElementById('presetStatus');
let captionPresets = [];

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

function getSelectedPreset() {
  if (!captionPresetSelect) return null;
  return captionPresets.find(p => p.id === captionPresetSelect.value) || null;
}

function setPresetStatus(message, isError = false) {
  if (!presetStatus) return;
  presetStatus.textContent = message || '';
  presetStatus.classList.toggle('error', Boolean(isError));
}

function setFieldValue(id, value) {
  const el = document.getElementById(id);
  if (!el || value === undefined || value === null) return;
  if (el.type === 'checkbox') {
    el.checked = Boolean(value);
  } else {
    el.value = value;
  }
}

function applyPreset(preset) {
  if (!preset) return;
  const systemPrompt = document.getElementById('systemPrompt');
  const userTemplate = document.getElementById('userTemplate');
  const prefill = document.getElementById('prefill');
  const maxOutputTokens = document.getElementById('maxOutputTokens');
  const presetDescription = document.getElementById('presetDescription');

  if (systemPrompt) systemPrompt.value = preset.system_prompt || '';
  if (userTemplate) userTemplate.value = preset.user_template || '';
  if (prefill) prefill.value = preset.prefill || '';
  if (maxOutputTokens && Number.isFinite(Number(preset.max_output_tokens))) {
    maxOutputTokens.value = preset.max_output_tokens;
  }

  const savedSettings = preset.saved_settings || {};
  setFieldValue('modelName', savedSettings.model ?? preset.model);
  setFieldValue('targetFolder', savedSettings.target_folder ?? preset.target_folder);
  setFieldValue('numFrames', savedSettings.num_frames ?? preset.num_frames);
  setFieldValue('samplingType', savedSettings.sampling_type ?? preset.sampling_type);
  setFieldValue('maxImageSide', savedSettings.max_image_side ?? preset.max_image_side);
  setFieldValue('maxConcurrent', savedSettings.max_concurrent ?? preset.max_concurrent);
  setFieldValue('abortAfterServerErrors', savedSettings.abort_after_server_errors ?? preset.abort_after_server_errors);
  setFieldValue('overwrite', savedSettings.overwrite ?? preset.overwrite);
  setFieldValue('prependExisting', savedSettings.prepend_existing ?? preset.prepend_existing);
  setFieldValue('filenameAffixText', savedSettings.filename_affix_text ?? preset.filename_affix_text);
  setFieldValue('filenameAffixPosition', savedSettings.filename_affix_position ?? preset.filename_affix_position);
  setFieldValue('outputToSubdir', savedSettings.output_to_subdir ?? preset.output_to_subdir);
  setFieldValue('outputSubdirName', savedSettings.output_subdir_name ?? preset.output_subdir_name);
  setFieldValue('useExistingCaption', savedSettings.use_existing_caption ?? preset.use_existing_caption);
  setFieldValue('existingCaption', savedSettings.existing_caption ?? preset.existing_caption);
  setFieldValue('sourceTags', savedSettings.source_tags ?? preset.source_tags);
  setFieldValue('characterTags', savedSettings.character_tags ?? preset.character_tags);
  setFieldValue('copyrightTags', savedSettings.copyright_tags ?? preset.copyright_tags);
  setFieldValue('artistTags', savedSettings.artist_tags ?? preset.artist_tags);
  setFieldValue('generalTags', savedSettings.general_tags ?? preset.general_tags);
  setFieldValue('ratingTags', savedSettings.rating_tags ?? preset.rating_tags);
  setFieldValue('qualityTags', savedSettings.quality_tags ?? preset.quality_tags);
  if (presetDescription) {
    const source = preset.source === 'user' ? 'User preset' : 'Built-in preset';
    presetDescription.textContent = `${source}: ${preset.description || 'Customize the system prompt and user message template below.'}`;
  }
  if (deletePresetBtn) {
    deletePresetBtn.disabled = preset.source !== 'user' || preset.readonly === true;
  }
  setPresetStatus('');
  if (imageModeToggle && preset.media) {
    imageModeToggle.checked = preset.media === 'image';
    refreshAccept();
  }
}

function populatePresets(presets) {
  captionPresets = Array.isArray(presets) ? presets : [];
  if (!captionPresetSelect) return;
  captionPresetSelect.innerHTML = '';
  for (const preset of captionPresets) {
    const option = document.createElement('option');
    option.value = preset.id;
    option.textContent = preset.name || preset.id;
    captionPresetSelect.appendChild(option);
  }
  const initial = captionPresets.find(p => p.id === 'video_basic') || captionPresets[0];
  if (initial) {
    captionPresetSelect.value = initial.id;
    applyPreset(initial);
  }
}

if (captionPresetSelect) {
  captionPresetSelect.addEventListener('change', () => applyPreset(getSelectedPreset()));
}

function collectPresetPayload(name, id = '') {
  const batchSettings = getBatchBody();
  return {
    id,
    name,
    description: `User-saved preset${name ? `: ${name}` : ''}`,
    media: imageModeToggle && imageModeToggle.checked ? 'image' : 'video',
    system_prompt: document.getElementById('systemPrompt').value,
    user_template: document.getElementById('userTemplate').value,
    prefill: document.getElementById('prefill').value,
    max_output_tokens: Number(document.getElementById('maxOutputTokens').value),
    saved_settings: {
      ...batchSettings,
      existing_caption: document.getElementById('existingCaption').value
    }
  };
}

async function saveCurrentPreset() {
  const selected = getSelectedPreset();
  const defaultName = selected && selected.source === 'user' ? selected.name : 'My caption preset';
  const name = window.prompt('Name for this user preset:', defaultName);
  if (!name || !name.trim()) return;
  const updateExisting = selected && selected.source === 'user';
  const payload = collectPresetPayload(name.trim(), updateExisting ? selected.id : '');

  try {
    setPresetStatus('Saving preset…');
    const res = await fetch('/api/user-presets', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const out = await res.json();
    if (!res.ok || out.error) throw new Error(out.error || `HTTP ${res.status}`);
    populatePresets(out.caption_presets || []);
    if (out.preset && captionPresetSelect) {
      captionPresetSelect.value = out.preset.id;
      applyPreset(getSelectedPreset());
    }
    setPresetStatus('Saved user preset.');
  } catch (e) {
    setPresetStatus(`Save failed: ${e.message || e}`, true);
  }
}

async function deleteCurrentPreset() {
  const selected = getSelectedPreset();
  if (!selected || selected.source !== 'user') return;
  if (!window.confirm(`Delete user preset "${selected.name}"?`)) return;
  try {
    setPresetStatus('Deleting preset…');
    const res = await fetch(`/api/user-presets/${encodeURIComponent(selected.id)}`, { method: 'DELETE' });
    const out = await res.json();
    if (!res.ok || out.error) throw new Error(out.error || `HTTP ${res.status}`);
    populatePresets(out.caption_presets || []);
    setPresetStatus('Deleted user preset.');
  } catch (e) {
    setPresetStatus(`Delete failed: ${e.message || e}`, true);
  }
}

if (savePresetBtn) {
  savePresetBtn.addEventListener('click', saveCurrentPreset);
}
if (deletePresetBtn) {
  deletePresetBtn.addEventListener('click', deleteCurrentPreset);
}

function appendMetadataFields(target) {
  target.append('user_template', document.getElementById('userTemplate').value);
  target.append('source_tags', document.getElementById('sourceTags').value);
  target.append('character_tags', document.getElementById('characterTags').value);
  target.append('copyright_tags', document.getElementById('copyrightTags').value);
  target.append('artist_tags', document.getElementById('artistTags').value);
  target.append('general_tags', document.getElementById('generalTags').value);
  target.append('rating_tags', document.getElementById('ratingTags').value);
  target.append('quality_tags', document.getElementById('qualityTags').value);
}

function readMetadataFields() {
  return {
    user_template: document.getElementById('userTemplate').value,
    source_tags: document.getElementById('sourceTags').value,
    character_tags: document.getElementById('characterTags').value,
    copyright_tags: document.getElementById('copyrightTags').value,
    artist_tags: document.getElementById('artistTags').value,
    general_tags: document.getElementById('generalTags').value,
    rating_tags: document.getElementById('ratingTags').value,
    quality_tags: document.getElementById('qualityTags').value
  };
}

async function loadBackendConfig() {
  try {
    const res = await fetch('/api/config');
    const cfg = await res.json();
    const backendInfo = document.getElementById('backendInfo');
    if (backendInfo) {
      backendInfo.textContent = `Backend: ${cfg.backend_display_name || cfg.backend} at ${cfg.api_base_url}`;
    }

    const modelName = document.getElementById('modelName');
    if (modelName && cfg.default_model && !modelName.value) {
      modelName.value = cfg.default_model;
      modelName.placeholder = cfg.default_model;
    }

    const maxImageSide = document.getElementById('maxImageSide');
    if (maxImageSide && Number.isFinite(Number(cfg.max_image_side))) {
      maxImageSide.value = cfg.max_image_side;
    }

    const maxOutputTokens = document.getElementById('maxOutputTokens');
    if (maxOutputTokens && Number.isFinite(Number(cfg.max_output_tokens))) {
      maxOutputTokens.value = cfg.max_output_tokens;
    }

    populatePresets(cfg.caption_presets);

    const maxConcurrent = document.getElementById('maxConcurrent');
    if (maxConcurrent && Number.isFinite(Number(cfg.default_batch_concurrency))) {
      maxConcurrent.value = cfg.default_batch_concurrency;
      if (Number.isFinite(Number(cfg.max_batch_concurrency))) {
        maxConcurrent.max = cfg.max_batch_concurrency;
      }
    }

    const abortAfter = document.getElementById('abortAfterServerErrors');
    if (abortAfter && Number.isFinite(Number(cfg.abort_after_server_errors))) {
      abortAfter.value = cfg.abort_after_server_errors;
    }
  } catch (e) {
    const backendInfo = document.getElementById('backendInfo');
    if (backendInfo) backendInfo.textContent = 'Backend: unable to load config';
  }
}
loadBackendConfig();

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
  fd.append('max_image_side', document.getElementById('maxImageSide').value);
  fd.append('max_output_tokens', document.getElementById('maxOutputTokens').value);
  fd.append('use_existing_caption', document.getElementById('useExistingCaption').checked);
  fd.append('existing_caption', document.getElementById('existingCaption').value);
  appendMetadataFields(fd);
  fd.append('image_mode', imageModeToggle && imageModeToggle.checked);

  const res = await fetch('/api/chat-caption', { method:'POST', body: fd });
  return await res.json();
}

sendBtn.addEventListener('click', async () => {
  const file = clipInput.files?.[0];
  if (!file) { addMsg('assistant', 'Attach a file first.'); return; }
  addMsg('user', `Attached: ${file.name}`);
  addMsg('assistant', 'Thinking... preparing inputs and querying the local vision backend');

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
const resumeBtn = document.getElementById('resumeBatch');
const batchLog = document.getElementById('batchLog');
const batchStatus = document.getElementById('batchStatus');
const batchProgress = document.getElementById('batchProgress');
const batchStats = document.getElementById('batchStats');
const activeFiles = document.getElementById('activeFiles');

let currentJobId = null;
let resumableJobId = null;
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
    max_image_side: Number(document.getElementById('maxImageSide').value),
    max_output_tokens: Number(document.getElementById('maxOutputTokens').value),
    max_concurrent: Number(document.getElementById('maxConcurrent').value),
    abort_after_server_errors: Number(document.getElementById('abortAfterServerErrors').value),
    sampling_type: document.getElementById('samplingType').value,
    overwrite: document.getElementById('overwrite').checked,
    prepend_existing: document.getElementById('prependExisting').checked,
    filename_affix_text: document.getElementById('filenameAffixText').value,
    filename_affix_position: document.getElementById('filenameAffixPosition').value,
    output_to_subdir: document.getElementById('outputToSubdir').checked,
    output_subdir_name: document.getElementById('outputSubdirName').value,
    use_existing_caption: document.getElementById('useExistingCaption').checked,
    image_mode: imageModeToggle && imageModeToggle.checked,
    ...readMetadataFields()
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
        if (r.ok) {
          const mediaOut = r.media_out ? ` + ${r.media_out}` : '';
          logBatch(`✓ ${r.file} -> ${r.out}${mediaOut}${took}`);
        }
        else if (r.skipped) logBatch(`↷ ${r.file} (skipped: ${r.reason})${took}`);
        else {
          const code = r.status_code ? ` [HTTP ${r.status_code}]` : '';
          logBatch(`✗ ${r.file}${code}: ${r.error}${took}`);
        }
      }
      lastResultCount = out.results.length;
    }

    if (out.status === 'done' || out.status === 'cancelled' || out.status === 'failed') {
      resumableJobId = out.status === 'failed' ? currentJobId : null;
      if (resumeBtn) {
        resumeBtn.disabled = out.status !== 'failed';
      }
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
  if (resumeBtn) resumeBtn.disabled = !resumableJobId;
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
  resumableJobId = null;
  if (resumeBtn) resumeBtn.disabled = true;

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

if (resumeBtn) {
  resumeBtn.addEventListener('click', async () => {
    if (currentJobId || !resumableJobId) return;
    const resumeFrom = resumableJobId;
    batchLog.textContent += `\nResuming failed batch ${resumeFrom}; successful/skipped items from the previous run will be reused.\n`;
    batchStatus.textContent = 'Resuming...';
    if (batchStats) batchStats.textContent = '';
    if (activeFiles) { activeFiles.textContent = ''; activeFiles.style.display = 'none'; }
    batchProgress.value = 0;
    lastResultCount = 0;
    resumeBtn.disabled = true;

    try {
      const res = await fetch('/api/batch-resume', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({job_id: resumeFrom})
      });
      const out = await res.json();
      if (out.error) {
        batchStatus.textContent = 'Error: ' + out.error;
        resumableJobId = resumeFrom;
        resumeBtn.disabled = false;
        return;
      }
      currentJobId = out.job_id;
      resumableJobId = null;
      startBtn.disabled = true;
      startBtn.textContent = 'Running...';
      cancelBtn.disabled = false;
      batchStatus.textContent = `resumed — retrying ${out.total || 0} item(s)`;
      progressTimer = setInterval(pollProgress, 1000);
      pollProgress();
    } catch (e) {
      batchStatus.textContent = 'Error: ' + e;
      resumableJobId = resumeFrom;
      resumeBtn.disabled = false;
    }
  });
}

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
