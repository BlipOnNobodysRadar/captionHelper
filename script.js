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
  return {
    id,
    name,
    description: `User-saved preset${name ? `: ${name}` : ''}`,
    media: imageModeToggle && imageModeToggle.checked ? 'image' : 'video',
    system_prompt: document.getElementById('systemPrompt').value,
    user_template: document.getElementById('userTemplate').value,
    prefill: document.getElementById('prefill').value,
    max_output_tokens: Number(document.getElementById('maxOutputTokens').value)
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
    if (cfg.region_preprocess) {
      const regionDetector = document.getElementById('regionDetector');
      const regionSegmenter = document.getElementById('regionSegmenter');
      const regionOcr = document.getElementById('regionOcr');
      const regionMaxRegions = document.getElementById('regionMaxRegions');
      const regionOcrThreshold = document.getElementById('regionOcrThreshold');
      const regionDetectorBoxThreshold = document.getElementById('regionDetectorBoxThreshold');
      const regionDetectorTextThreshold = document.getElementById('regionDetectorTextThreshold');
      const regionModelRoot = document.getElementById('regionModelRoot');
      const regionAutoDownload = document.getElementById('regionAutoDownload');
      const regionLoadModels = document.getElementById('regionLoadModels');
      const regionDetectorModelPath = document.getElementById('regionDetectorModelPath');
      const regionSegmenterModelPath = document.getElementById('regionSegmenterModelPath');
      const regionOcrModelPath = document.getElementById('regionOcrModelPath');
      if (regionDetector) regionDetector.value = cfg.region_preprocess.detector || regionDetector.value;
      if (regionSegmenter) regionSegmenter.value = cfg.region_preprocess.segmenter || regionSegmenter.value;
      if (regionOcr) regionOcr.value = cfg.region_preprocess.ocr || regionOcr.value;
      if (regionMaxRegions) regionMaxRegions.value = cfg.region_preprocess.max_regions ?? regionMaxRegions.value;
      if (regionOcrThreshold) regionOcrThreshold.value = cfg.region_preprocess.ocr_threshold ?? regionOcrThreshold.value;
      if (regionDetectorBoxThreshold) regionDetectorBoxThreshold.value = cfg.region_preprocess.detector_box_threshold ?? regionDetectorBoxThreshold.value;
      if (regionDetectorTextThreshold) regionDetectorTextThreshold.value = cfg.region_preprocess.detector_text_threshold ?? regionDetectorTextThreshold.value;
      if (regionModelRoot) regionModelRoot.value = cfg.region_preprocess.model_root || '';
      if (regionAutoDownload) regionAutoDownload.checked = Boolean(cfg.region_preprocess.auto_download);
      if (regionLoadModels) regionLoadModels.checked = Boolean(cfg.region_preprocess.load_models);
      if (regionDetectorModelPath) regionDetectorModelPath.value = cfg.region_preprocess.detector_model_path || '';
      if (regionSegmenterModelPath) regionSegmenterModelPath.value = cfg.region_preprocess.segmenter_model_path || '';
      if (regionOcrModelPath) regionOcrModelPath.value = cfg.region_preprocess.ocr_model_path || '';
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

function formatRegionPreprocessNotice(summary) {
  if (!summary || !summary.enabled) return '';
  const warnings = Array.isArray(summary.warnings) ? summary.warnings.filter(Boolean) : [];
  const count = Number(summary.candidate_count || 0);
  const prefix = summary.skipped
    ? '⚠️ Region preprocessing was skipped or produced no usable candidates.'
    : `ℹ️ Region preprocessing produced ${count} candidate${count === 1 ? '' : 's'}.`;
  if (!warnings.length) return prefix;
  return `${prefix}\n${warnings.map(w => `• ${w}`).join('\n')}`;
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
  fd.append('enable_region_preprocess', document.getElementById('enableRegionPreprocess').checked);
  fd.append('validate_ideogram_json', document.getElementById('validateIdeogramJson').checked);
  fd.append('region_detector', document.getElementById('regionDetector').value);
  fd.append('region_segmenter', document.getElementById('regionSegmenter').value);
  fd.append('region_ocr', document.getElementById('regionOcr').value);
  fd.append('region_max_regions', document.getElementById('regionMaxRegions').value);
  fd.append('region_ocr_threshold', document.getElementById('regionOcrThreshold').value);
  fd.append('region_detector_box_threshold', document.getElementById('regionDetectorBoxThreshold').value);
  fd.append('region_detector_text_threshold', document.getElementById('regionDetectorTextThreshold').value);
  fd.append('region_model_root', document.getElementById('regionModelRoot').value);
  fd.append('region_auto_download', document.getElementById('regionAutoDownload').checked);
  fd.append('region_load_models', document.getElementById('regionLoadModels').checked);
  fd.append('region_detector_model_path', document.getElementById('regionDetectorModelPath').value);
  fd.append('region_segmenter_model_path', document.getElementById('regionSegmenterModelPath').value);
  fd.append('region_ocr_model_path', document.getElementById('regionOcrModelPath').value);
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
  const regionEnabled = document.getElementById('enableRegionPreprocess')?.checked;
  const autoDownload = document.getElementById('regionAutoDownload')?.checked;
  const downloadHint = regionEnabled && autoDownload ? ' If selected preprocessing models are missing, they will be downloaded before captioning.' : '';
  addMsg('assistant', 'Thinking... preparing inputs and querying the local vision backend' + downloadHint);

  const out = await postChatCaption(file);
  if (out.error) {
    addMsg('assistant', 'Error: ' + out.error);
  } else {
    const framesInfo = (typeof out.frames_used === 'number') ? ` [inputs: ${out.frames_used}] ` : ' ';
    const preprocessNotice = formatRegionPreprocessNotice(out.region_preprocess_summary);
    addMsg('assistant', (preprocessNotice ? preprocessNotice + '\n\n' : '') + framesInfo + out.caption);
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
    const preprocess = typeof item === 'string' ? null : item.preprocess;
    const progress = preprocess && preprocess.message
      ? ` — ${preprocess.message}${Number.isFinite(Number(preprocess.percent)) ? ` (${Math.round(Number(preprocess.percent))}%)` : ''}`
      : '';
    return elapsed === null || elapsed === undefined ? `• ${file}${progress}` : `• ${file} (${formatDuration(elapsed)})${progress}`;
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
    enable_region_preprocess: document.getElementById('enableRegionPreprocess').checked,
    validate_ideogram_json: document.getElementById('validateIdeogramJson').checked,
    region_detector: document.getElementById('regionDetector').value,
    region_segmenter: document.getElementById('regionSegmenter').value,
    region_ocr: document.getElementById('regionOcr').value,
    region_max_regions: Number(document.getElementById('regionMaxRegions').value),
    region_ocr_threshold: Number(document.getElementById('regionOcrThreshold').value),
    region_detector_box_threshold: Number(document.getElementById('regionDetectorBoxThreshold').value),
    region_detector_text_threshold: Number(document.getElementById('regionDetectorTextThreshold').value),
    region_model_root: document.getElementById('regionModelRoot').value,
    region_auto_download: document.getElementById('regionAutoDownload').checked,
    region_load_models: document.getElementById('regionLoadModels').checked,
    region_detector_model_path: document.getElementById('regionDetectorModelPath').value,
    region_segmenter_model_path: document.getElementById('regionSegmenterModelPath').value,
    region_ocr_model_path: document.getElementById('regionOcrModelPath').value,
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
          const notice = formatRegionPreprocessNotice(r.region_preprocess_summary);
          if (notice) logBatch(notice);
        }
        else if (r.skipped) logBatch(`↷ ${r.file} (skipped: ${r.reason})${took}`);
        else {
          const code = r.status_code ? ` [HTTP ${r.status_code}]` : '';
          logBatch(`✗ ${r.file}${code}: ${r.error}${took}`);
          const notice = formatRegionPreprocessNotice(r.region_preprocess_summary);
          if (notice) logBatch(notice);
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
