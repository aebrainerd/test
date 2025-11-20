const DEFAULT_SETTINGS = {
  apiBase: 'https://api.openai.com',
  apiKey: '',
  model: 'gpt-3.5-turbo-instruct',
  top_k: 5
};

document.addEventListener('DOMContentLoaded', restoreOptions);

document.getElementById('save').addEventListener('click', saveOptions);

async function saveOptions() {
  const apiBase = document.getElementById('apiBase').value.trim() || DEFAULT_SETTINGS.apiBase;
  const apiKey = document.getElementById('apiKey').value.trim();
  const model = document.getElementById('model').value.trim() || DEFAULT_SETTINGS.model;
  const top_k = parseInt(document.getElementById('top_k').value, 10) || DEFAULT_SETTINGS.top_k;

  await chrome.storage.local.set({ apiBase, apiKey, model, top_k });
  showStatus('Saved!');
}

async function restoreOptions() {
  const stored = await chrome.storage.local.get(['apiBase', 'apiKey', 'model', 'top_k']);
  document.getElementById('apiBase').value = stored.apiBase || DEFAULT_SETTINGS.apiBase;
  document.getElementById('apiKey').value = stored.apiKey || DEFAULT_SETTINGS.apiKey;
  document.getElementById('model').value = stored.model || DEFAULT_SETTINGS.model;
  document.getElementById('top_k').value = typeof stored.top_k === 'number' ? stored.top_k : DEFAULT_SETTINGS.top_k;
}

function showStatus(msg) {
  const status = document.getElementById('status');
  status.textContent = msg;
  setTimeout(() => {
    status.textContent = '';
  }, 1500);
}
