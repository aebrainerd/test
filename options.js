const DEFAULT_SETTINGS = {
  model: 'Qwen/Qwen1.5-4B',
  top_k: 5
};

document.addEventListener('DOMContentLoaded', restoreOptions);

document.getElementById('save').addEventListener('click', saveOptions);

async function saveOptions() {
  const model = document.getElementById('model').value.trim() || DEFAULT_SETTINGS.model;
  const top_k = parseInt(document.getElementById('top_k').value, 10) || DEFAULT_SETTINGS.top_k;

  await chrome.storage.local.set({ model, top_k });
  showStatus('Saved!');
}

async function restoreOptions() {
  const stored = await chrome.storage.local.get(['model', 'top_k']);
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
