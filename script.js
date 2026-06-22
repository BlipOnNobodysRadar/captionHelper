const chatLog = document.getElementById('chatLog');
const sendBtn = document.getElementById('sendBtn');
const clipInput = document.getElementById('clipInput');
const fileDrop = document.getElementById('fileDrop');
const attachmentPreview = document.getElementById('attachmentPreview');
const chatMessage = document.getElementById('chatMessage');
const newChatBtn = document.getElementById('newChatBtn');
const clearChatBtn = document.getElementById('clearChatBtn');
let currentAttachmentUrl = '';
let chatHistory = [];

const imageModeToggle = document.getElementById('imageMode');
const captionPresetSelect = document.getElementById('captionPreset');
const savePresetBtn = document.getElementById('savePreset');
const deletePresetBtn = document.getElementById('deletePreset');
const presetStatus = document.getElementById('presetStatus');
let captionPresets = [];

function refreshAccept(){
  clipInput.setAttribute('accept', 'image/*,video/*');
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
  const modelNameField = document.getElementById('modelName');
  if (modelNameField && !modelNameField.value) modelNameField.value = modelNameField.placeholder || 'gemma4-12b-vision';
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

async function collectPresetPayload(name, id = '') {
  const batchSettings = await getBatchBody();
  delete batchSettings.few_shot_examples;
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
  const payload = await collectPresetPayload(name.trim(), updateExisting ? selected.id : '');

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


function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener('load', () => resolve(reader.result));
    reader.addEventListener('error', () => reject(reader.error || new Error('Unable to read file')));
    reader.readAsDataURL(file);
  });
}

async function collectFewShotExamples() {
  const examples = [];
  for (let i = 1; i <= 3; i++) {
    const file = document.getElementById(`fewShotImage${i}`)?.files?.[0];
    const caption = document.getElementById(`fewShotCaption${i}`)?.value?.trim() || '';
    if (!file && !caption) continue;
    if (!file || !caption) {
      throw new Error(`Few-shot example ${i} needs both an image and a matching caption.`);
    }
    if (!selectedFileLooksLikeImage(file)) {
      throw new Error(`Few-shot example ${i} must be an image file.`);
    }
    examples.push({
      name: file.name,
      caption,
      image_data_url: await fileToDataUrl(file)
    });
  }
  return examples;
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
      const regionDevice = document.getElementById('regionDevice');
      const regionModelRoot = document.getElementById('regionModelRoot');
      const regionAutoDownload = document.getElementById('regionAutoDownload');
      const regionLoadModels = document.getElementById('regionLoadModels');
      const regionDetectorModelPath = document.getElementById('regionDetectorModelPath');
      const regionSegmenterModelPath = document.getElementById('regionSegmenterModelPath');
      const regionOcrModelPath = document.getElementById('regionOcrModelPath');
      if (regionDetector) regionDetector.value = cfg.region_preprocess.detector || regionDetector.value;
      if (regionSegmenter) regionSegmenter.value = 'none';
      if (regionOcr) regionOcr.value = 'none';
      if (regionMaxRegions) regionMaxRegions.value = cfg.region_preprocess.max_regions ?? regionMaxRegions.value;
      if (regionOcrThreshold) regionOcrThreshold.value = cfg.region_preprocess.ocr_threshold ?? regionOcrThreshold.value;
      if (regionDetectorBoxThreshold) regionDetectorBoxThreshold.value = cfg.region_preprocess.detector_box_threshold ?? regionDetectorBoxThreshold.value;
      if (regionDetectorTextThreshold) regionDetectorTextThreshold.value = cfg.region_preprocess.detector_text_threshold ?? regionDetectorTextThreshold.value;
      if (regionDevice) regionDevice.value = cfg.region_preprocess.device || regionDevice.value;
      if (regionModelRoot) regionModelRoot.value = cfg.region_preprocess.model_root || '';
      if (regionAutoDownload) regionAutoDownload.checked = Boolean(cfg.region_preprocess.auto_download);
      if (regionLoadModels) regionLoadModels.checked = Boolean(cfg.region_preprocess.load_models);
      if (regionDetectorModelPath) regionDetectorModelPath.value = cfg.region_preprocess.detector_model_path || '';
      if (regionSegmenterModelPath) regionSegmenterModelPath.value = cfg.region_preprocess.segmenter_model_path || '';
      if (regionOcrModelPath) regionOcrModelPath.value = cfg.region_preprocess.ocr_model_path || '';
    }
    const llamaUnload = document.getElementById('llamaCppUnloadDuringPreprocess');
    if (llamaUnload && cfg.llama_cpp_model_management) {
      llamaUnload.checked = Boolean(cfg.llama_cpp_model_management.unload_during_preprocess);
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

function formatModelManagementNotice(modelManagement) {
  if (!modelManagement) return '';
  const unload = modelManagement.unload_before_preprocess;
  const reload = modelManagement.reload_after_preprocess;
  const lines = [];
  if (unload && !unload.ok) {
    lines.push(`⚠️ llama.cpp model unload did not happen: ${unload.reason || unload.error || 'unknown error'}`);
  }
  if (reload && !reload.ok) {
    lines.push(`⚠️ llama.cpp model reload did not complete: ${reload.reason || reload.error || 'unknown error'}`);
  }
  return lines.join('\n');
}

function selectedFileLooksLikeImage(file) {
  return Boolean(file && (file.type || '').startsWith('image/')) || /\.(png|jpe?g|webp|bmp|gif)$/i.test(file?.name || '');
}

function selectedFileLooksLikeVideo(file) {
  return Boolean(file && (file.type || '').startsWith('video/')) || /\.(mp4|mov|avi|webm|mkv|m4v)$/i.test(file?.name || '');
}

function setAttachmentFile(file) {
  if (!file) return;
  if (imageModeToggle) {
    if (selectedFileLooksLikeImage(file)) imageModeToggle.checked = true;
    if (selectedFileLooksLikeVideo(file)) imageModeToggle.checked = false;
    refreshAccept();
  }
  renderAttachmentPreview(file);
}

function attachFileToInput(file) {
  if (!file || !clipInput) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  clipInput.files = transfer.files;
  setAttachmentFile(file);
}

function firstUsableFile(fileList) {
  return Array.from(fileList || []).find(file => selectedFileLooksLikeImage(file) || selectedFileLooksLikeVideo(file)) || null;
}

function autoSizeChatInput() {
  if (!chatMessage) return;
  chatMessage.style.height = 'auto';
  const maxHeight = Math.round(window.innerHeight * 0.32);
  chatMessage.style.height = `${Math.min(chatMessage.scrollHeight, maxHeight)}px`;
}

function renderAttachmentPreview(file) {
  if (!attachmentPreview) return;
  if (currentAttachmentUrl) {
    URL.revokeObjectURL(currentAttachmentUrl);
    currentAttachmentUrl = '';
  }
  attachmentPreview.hidden = !file;
  attachmentPreview.innerHTML = '';
  if (!file) return;

  const media = document.createElement(selectedFileLooksLikeImage(file) ? 'img' : 'video');
  media.className = 'attachment-media';
  media.alt = file.name;
  if (media.tagName === 'VIDEO') {
    currentAttachmentUrl = URL.createObjectURL(file);
    media.src = currentAttachmentUrl;
    media.muted = true;
    media.controls = true;
    media.playsInline = true;
  } else {
    const reader = new FileReader();
    reader.addEventListener('load', () => { media.src = reader.result; });
    reader.readAsDataURL(file);
  }

  const meta = document.createElement('div');
  meta.className = 'attachment-meta';
  meta.innerHTML = `<strong>${file.name}</strong><small>${selectedFileLooksLikeImage(file) ? 'Image ready for chat' : 'Video ready for frame sampling'}</small>`;
  attachmentPreview.append(media, meta);
}

if (clipInput) {
  clipInput.addEventListener('change', () => setAttachmentFile(clipInput.files?.[0]));
}

if (fileDrop) {
  ['dragenter', 'dragover'].forEach(eventName => {
    fileDrop.addEventListener(eventName, (event) => {
      event.preventDefault();
      fileDrop.classList.add('drag-over');
    });
  });
  ['dragleave', 'drop'].forEach(eventName => {
    fileDrop.addEventListener(eventName, (event) => {
      event.preventDefault();
      fileDrop.classList.remove('drag-over');
    });
  });
  fileDrop.addEventListener('drop', (event) => {
    const file = firstUsableFile(event.dataTransfer?.files);
    if (!file) return;
    attachFileToInput(file);
  });
}

if (chatMessage) {
  chatMessage.addEventListener('input', autoSizeChatInput);
  chatMessage.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!sendBtn.disabled) sendBtn.click();
    }
  });
  ['dragenter', 'dragover'].forEach(eventName => {
    chatMessage.addEventListener(eventName, (event) => {
      if (!Array.from(event.dataTransfer?.types || []).includes('Files')) return;
      event.preventDefault();
      chatMessage.classList.add('drag-over');
    });
  });
  ['dragleave', 'drop'].forEach(eventName => {
    chatMessage.addEventListener(eventName, (event) => {
      chatMessage.classList.remove('drag-over');
      if (eventName === 'drop') event.preventDefault();
    });
  });
  chatMessage.addEventListener('drop', (event) => {
    const file = firstUsableFile(event.dataTransfer?.files);
    if (file) attachFileToInput(file);
  });
  chatMessage.addEventListener('paste', (event) => {
    const file = firstUsableFile(event.clipboardData?.files);
    if (file) attachFileToInput(file);
  });
  autoSizeChatInput();
}

async function buildChatFormData(file, messageOverride = null, historyOverride = null) {
  const fd = new FormData();
  if (file) fd.append('file', file);
  fd.append('chat_message', messageOverride !== null ? messageOverride : (chatMessage ? chatMessage.value : ''));
  fd.append('chat_history', JSON.stringify(historyOverride || chatHistory.slice(-12)));
  fd.append('few_shot_examples', JSON.stringify(await collectFewShotExamples()));
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
  fd.append('region_device', document.getElementById('regionDevice').value);
  fd.append('region_model_root', document.getElementById('regionModelRoot').value);
  fd.append('region_auto_download', document.getElementById('regionAutoDownload').checked);
  fd.append('region_load_models', document.getElementById('regionLoadModels').checked);
  fd.append('region_detector_model_path', document.getElementById('regionDetectorModelPath').value);
  fd.append('region_segmenter_model_path', document.getElementById('regionSegmenterModelPath').value);
  fd.append('region_ocr_model_path', document.getElementById('regionOcrModelPath').value);
  fd.append('llama_cpp_unload_during_preprocess', document.getElementById('llamaCppUnloadDuringPreprocess').checked);
  fd.append('use_existing_caption', document.getElementById('useExistingCaption').checked);
  fd.append('existing_caption', document.getElementById('existingCaption').value);
  appendMetadataFields(fd);
  fd.append('image_mode', imageModeToggle && imageModeToggle.checked);
  return fd;
}

function addMsg(who, text, extraClass = '') {
  const div = document.createElement('div');
  div.className = ['msg', who, extraClass].filter(Boolean).join(' ');
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

function clearAttachment() {
  if (currentAttachmentUrl) {
    URL.revokeObjectURL(currentAttachmentUrl);
    currentAttachmentUrl = '';
  }
  if (clipInput) clipInput.value = '';
  if (attachmentPreview) {
    attachmentPreview.hidden = true;
    attachmentPreview.innerHTML = '';
  }
}

function clearChat({ clearDraft = false, clearFile = false } = {}) {
  chatHistory = [];
  chatLog.innerHTML = '';
  if (clearDraft && chatMessage) {
    chatMessage.value = '';
    autoSizeChatInput();
  }
  if (clearFile) clearAttachment();
}

if (clearChatBtn) {
  clearChatBtn.addEventListener('click', () => clearChat());
}
if (newChatBtn) {
  newChatBtn.addEventListener('click', () => clearChat({ clearDraft: true, clearFile: true }));
}

function createStreamingAssistantMsg(turn = null) {
  const div = document.createElement('div');
  div.className = 'msg assistant streaming';

  const thinkingButton = document.createElement('button');
  thinkingButton.type = 'button';
  thinkingButton.className = 'thinking-toggle thinking';
  thinkingButton.textContent = 'Thinking';
  thinkingButton.setAttribute('aria-expanded', 'false');

  const thinkingPanel = document.createElement('pre');
  thinkingPanel.className = 'thinking-panel';
  thinkingPanel.hidden = true;
  thinkingPanel.textContent = 'No thinking tokens received yet.';

  const content = document.createElement('div');
  content.className = 'assistant-content';
  content.textContent = 'Waiting for response';

  const retryButton = document.createElement('button');
  retryButton.type = 'button';
  retryButton.className = 'retry-turn';
  retryButton.textContent = 'Retry turn';
  retryButton.hidden = true;
  retryButton.addEventListener('click', () => {
    if (turn) retryTurn(turn);
  });

  thinkingButton.addEventListener('click', () => {
    thinkingPanel.hidden = !thinkingPanel.hidden;
    thinkingButton.setAttribute('aria-expanded', String(!thinkingPanel.hidden));
  });

  div.append(thinkingButton, thinkingPanel, content, retryButton);
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return { div, thinkingButton, thinkingPanel, content, retryButton };
}

function parseSseEvents(buffer, onEvent) {
  const blocks = buffer.split('\n\n');
  const remainder = blocks.pop() || '';
  for (const block of blocks) {
    let event = 'message';
    const dataLines = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) continue;
    try {
      onEvent(event, JSON.parse(dataLines.join('\n')));
    } catch (e) {
      console.warn('Unable to parse stream event', e);
    }
  }
  return remainder;
}

async function streamChatCaption(turn) {
  const res = await fetch('/api/chat-caption-stream', { method: 'POST', body: await buildChatFormData(turn.file, turn.message, turn.historyBefore) });
  if (!res.ok || !res.body) {
    let message = `HTTP ${res.status}`;
    try {
      const out = await res.json();
      message = out.error || message;
    } catch (_) {}
    throw new Error(message);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let assistantText = '';
  let thinkingText = '';
  let notices = '';
  const assistantMsg = createStreamingAssistantMsg(turn);

  function renderAssistant() {
    assistantMsg.thinkingButton.classList.toggle('thinking', !assistantText);
    assistantMsg.thinkingButton.textContent = thinkingText ? 'Thinking tokens' : 'Thinking';
    assistantMsg.thinkingPanel.textContent = thinkingText || 'No thinking tokens received yet.';
    assistantMsg.content.textContent = (notices ? notices + '\n\n' : '') + (assistantText || 'Waiting for response');
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSseEvents(buffer, (event, payload) => {
      if (event === 'meta') {
        const framesInfo = (typeof payload.frames_used === 'number' && payload.frames_used > 0) ? ` [inputs: ${payload.frames_used}] ` : '';
        const preprocessNotice = formatRegionPreprocessNotice(payload.region_preprocess_summary);
        const modelNotice = formatModelManagementNotice(payload.model_management);
        notices = [preprocessNotice, modelNotice, framesInfo].filter(Boolean).join('\n');
        renderAssistant();
      } else if (event === 'thinking') {
        thinkingText += payload.token || '';
        renderAssistant();
      } else if (event === 'token') {
        assistantText += payload.token || '';
        renderAssistant();
      } else if (event === 'error') {
        assistantMsg.thinkingButton.classList.remove('thinking');
        assistantMsg.div.classList.add('error');
        assistantMsg.content.textContent = 'Error: ' + (payload.error || 'Unknown error');
        assistantMsg.retryButton.hidden = false;
      } else if (event === 'done' && payload.caption) {
        assistantText = payload.caption;
        renderAssistant();
        assistantMsg.retryButton.hidden = false;
      }
    });
  }
  assistantMsg.thinkingButton.classList.remove('thinking');
  assistantMsg.retryButton.hidden = false;
  return assistantText.trim();
}

async function retryTurn(turn) {
  if (sendBtn.disabled) return;
  sendBtn.disabled = true;
  try {
    const caption = await streamChatCaption(turn);
    if (caption && !turn.recorded) {
      chatHistory.push({ role: 'user', content: turn.userText });
      chatHistory.push({ role: 'assistant', content: caption });
      turn.recorded = true;
    }
  } catch (e) {
    addMsg('assistant', 'Error: ' + (e.message || e), 'error');
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', async () => {
  const file = clipInput.files?.[0];
  const message = chatMessage ? chatMessage.value.trim() : '';
  if (!file && !message) { addMsg('assistant', 'Attach a file or type a message first.'); return; }

  const userText = [file ? `Attached: ${file.name}` : '', message].filter(Boolean).join('\n\n');
  const turn = { file, message, userText, historyBefore: chatHistory.slice(-12), recorded: false };
  addMsg('user', userText);
  sendBtn.disabled = true;
  try {
    const caption = await streamChatCaption(turn);
    if (caption) {
      chatHistory.push({ role: 'user', content: userText });
      chatHistory.push({ role: 'assistant', content: caption });
      turn.recorded = true;
    }
    if (chatMessage) {
      chatMessage.value = '';
      autoSizeChatInput();
    }
  } catch (e) {
    addMsg('assistant', 'Error: ' + (e.message || e), 'error');
  } finally {
    if (file) clearAttachment();
    sendBtn.disabled = false;
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

async function getBatchBody() {
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
    region_device: document.getElementById('regionDevice').value,
    region_model_root: document.getElementById('regionModelRoot').value,
    region_auto_download: document.getElementById('regionAutoDownload').checked,
    region_load_models: document.getElementById('regionLoadModels').checked,
    region_detector_model_path: document.getElementById('regionDetectorModelPath').value,
    region_segmenter_model_path: document.getElementById('regionSegmenterModelPath').value,
    region_ocr_model_path: document.getElementById('regionOcrModelPath').value,
    llama_cpp_unload_during_preprocess: document.getElementById('llamaCppUnloadDuringPreprocess').checked,
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
    few_shot_examples: await collectFewShotExamples(),
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
    if (out.status === 'preprocessing') {
      const pc = Number(out.preprocess_completed || 0);
      const pt = Number(out.preprocess_total || total || 0);
      batchStatus.textContent = `preprocessing — ${pc}/${pt} — active: ${activeCount}`;
    } else {
      batchStatus.textContent = `${out.status} — ${completed}/${total} — parallel: ${parallel} — active: ${activeCount}`;
    }
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
          const modelNotice = formatModelManagementNotice(r.model_management);
          if (modelNotice) logBatch(modelNotice);
        }
        else if (r.skipped) logBatch(`↷ ${r.file} (skipped: ${r.reason})${took}`);
        else {
          const code = r.status_code ? ` [HTTP ${r.status_code}]` : '';
          logBatch(`✗ ${r.file}${code}: ${r.error}${took}`);
          const notice = formatRegionPreprocessNotice(r.region_preprocess_summary);
          if (notice) logBatch(notice);
          const modelNotice = formatModelManagementNotice(r.model_management);
          if (modelNotice) logBatch(modelNotice);
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
    const body = await getBatchBody();
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
