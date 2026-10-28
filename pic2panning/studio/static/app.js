import { createEditor, initialComposition } from './editor.js';
import { createActions } from './actions.js';

const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) =>
  String(value ?? '').replace(
    /[&<>"']/g,
    (character) =>
      ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      })[character],
  );
const clone = (value) => structuredClone(value);
const state = {
  token: '',
  workspace: '',
  assets: [],
  projects: [],
  jobs: [],
  pending: new Map(),
  doc: null,
  path: 'projects/untitled.json',
  savedRevision: 0,
  dirty: false,
  selected: null,
  selectedNode: 'main',
  library: 'media',
  inspector: 'transform',
  session: 0,
  time: 0,
  playing: false,
  undo: [],
  redo: [],
  previewBusy: false,
  previewAgain: false,
};
let editor;
let actions;
let previewTimer;
let playback;

function notify(message, error = false) {
  $('#message').textContent = message;
  $('#message').title = message;
  $('#message').classList.toggle('error-text', error);
}

async function api(path, options = {}) {
  const method = options.method || 'GET';
  const headers = {};
  if (method !== 'GET') headers['X-Studio-Token'] = state.token;
  let body = options.body;
  if (options.data !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(options.data);
  }
  const response = await fetch(path, { method, headers, body });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.message || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function composition() {
  return (
    state.doc.nodes.find(
      (node) => node.id === 'main' && node.job.kind === 'compose',
    )?.job || state.doc.nodes.find((node) => node.job.kind === 'compose')?.job
  );
}
function mainNode() {
  return state.doc.nodes.find((node) => node.job === composition());
}
function selectedLayer() {
  return composition()?.layers.find((layer) => layer.id === state.selected);
}
function fileURL(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}
function relative(path) {
  return path.startsWith(state.workspace + '/')
    ? path.slice(state.workspace.length + 1)
    : path;
}
function slug(value) {
  return (
    value
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, '-')
      .replace(/^-|-$/g, '') || 'untitled'
  );
}

function render() {
  if (!state.doc) return;
  $('#project-name').value = state.doc.name;
  $('#save-state').textContent = state.dirty
    ? 'Unsaved'
    : state.savedRevision
      ? `Saved · r${state.savedRevision}`
      : 'Unsaved';
  $('#undo').disabled = !state.undo.length;
  $('#redo').disabled = !state.redo.length;
  editor.render();
  actions.renderJobs();
}

function change(callback, preview = true) {
  const before = JSON.stringify(state.doc);
  callback();
  if (before === JSON.stringify(state.doc)) return;
  state.undo.push(before);
  if (state.undo.length > 60) state.undo.shift();
  state.redo = [];
  state.dirty = true;
  render();
  if (preview) schedulePreview();
}

function restore(direction) {
  const source = direction === 'undo' ? state.undo : state.redo;
  const target = direction === 'undo' ? state.redo : state.undo;
  if (!source.length) return;
  target.push(JSON.stringify(state.doc));
  state.doc = JSON.parse(source.pop());
  state.dirty = true;
  if (!selectedLayer()) state.selected = composition().layers[0]?.id || null;
  state.time = Math.min(
    state.time,
    Math.max(0, composition().duration - 1 / composition().fps),
  );
  render();
  schedulePreview();
}

function schedulePreview(delay = 300) {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(requestPreview, delay);
}

async function requestPreview() {
  const current = composition();
  if (!current?.layers.length) return;
  if (state.previewBusy) {
    state.previewAgain = true;
    return;
  }
  state.previewBusy = true;
  const frame = Math.min(
    Math.ceil(current.duration * current.fps) - 1,
    Math.floor(state.time * current.fps),
  );
  $('#preview-state').textContent = `Rendering frame ${frame}`;
  try {
    const result = await api('/api/preview', {
      method: 'POST',
      data: { composition: clone(current), frame },
    });
    $('#preview').src = fileURL(result.path);
    $('#preview-state').textContent =
      `FRAME ${String(result.frame).padStart(4, '0')} · ${current.size.width} × ${current.size.height}`;
  } catch (error) {
    if (error.status === 429) state.previewAgain = true;
    else {
      $('#preview-state').textContent = 'Preview unavailable';
      notify(error.message, true);
    }
  } finally {
    state.previewBusy = false;
    if (state.previewAgain) {
      state.previewAgain = false;
      schedulePreview(100);
    }
  }
}

function setTime(seconds, refresh = true) {
  const current = composition();
  state.time = Math.max(
    0,
    Math.min(
      seconds,
      (Math.ceil(current.duration * current.fps) - 1) / current.fps,
    ),
  );
  editor.renderTimeline();
  if (refresh) editor.renderInspector();
  schedulePreview(state.playing ? 0 : 180);
}

function togglePlay() {
  state.playing = !state.playing;
  $('#play').textContent = state.playing ? 'Ⅱ' : '▶';
  clearInterval(playback);
  if (!state.playing) {
    render();
    return;
  }
  let last = performance.now();
  playback = setInterval(() => {
    const now = performance.now();
    const duration = composition().duration;
    const next = state.time + (now - last) / 1000;
    last = now;
    setTime(next >= duration ? 0 : next, false);
  }, 180);
}

async function refreshAssets() {
  const result = await api('/api/assets');
  state.assets = result.assets;
  editor.renderLibrary();
}

async function upload(file, batch = null) {
  notify(`Importing ${file.name}`);
  const asset = await api(
    `/api/upload?name=${encodeURIComponent(file.name)}${batch ? '&batch=' + batch : ''}`,
    { method: 'POST', body: file },
  );
  state.assets = [
    ...state.assets.filter((item) => item.path !== asset.path),
    asset,
  ];
  editor.renderLibrary();
  return asset;
}

async function importFiles(files) {
  try {
    const batch = crypto.randomUUID().replaceAll('-', '');
    const manifests = [];
    const companions = new Set();
    for (const file of files.filter((item) =>
      item.name.endsWith('.depth.json'),
    )) {
      const document = JSON.parse(await file.text());
      manifests.push(file.name);
      for (const field of ['depth', 'preview', 'normals', 'mask']) {
        if (typeof document[field]?.path !== 'string')
          throw new Error(
            'The depth metadata is missing a companion file path.',
          );
        companions.add(document[field].path.split('/').pop());
      }
    }
    const imported = [];
    for (const file of files) {
      const asset = await upload(file, batch);
      imported.push(asset);
    }
    for (const asset of imported) {
      if (companions.has(asset.name)) continue;
      if (asset.kind === 'image' || asset.kind === 'video')
        editor.addMedia(asset);
      else if (asset.kind === 'audio') editor.addAudio(asset);
    }
    for (const name of manifests) {
      const metadata = imported.find((asset) => asset.name === name);
      const verified = await api('/api/depth', {
        method: 'POST',
        data: {
          metadata: metadata.path,
          source:
            selectedLayer()?.kind === 'image' ? selectedLayer().source : null,
        },
      });
      let layer = composition().layers.find(
        (item) =>
          item.kind === 'image' && relative(item.source) === verified.source,
      );
      if (!layer) {
        const asset = state.assets.find(
          (item) => item.path === verified.source,
        );
        if (!asset)
          throw new Error(
            'Depth is valid. Add its source image from Media before attaching it.',
          );
        editor.addMedia(asset);
        layer = selectedLayer();
      }
      change(() => {
        layer.depth = {
          metadata: verified.metadata,
          focus_metres: verified.focus_metres,
        };
        state.selected = layer.id;
        state.inspector = 'depth';
      });
    }
    notify(`Imported ${files.length} file${files.length === 1 ? '' : 's'}`);
  } catch (error) {
    notify(
      error.message +
        (files.some((file) => file.name.endsWith('.depth.json'))
          ? '. Import depth metadata together with its TIFF, preview, normals, and mask companions.'
          : ''),
      true,
    );
  }
}

function newProject() {
  if (state.playing) togglePlay();
  const job = initialComposition();
  state.session++;
  state.doc = {
    name: 'Untitled',
    revision: 0,
    nodes: [{ id: 'main', job, depends_on: [], enabled: true }],
  };
  state.selected = job.layers[0].id;
  state.selectedNode = 'main';
  state.path = 'projects/untitled.json';
  state.savedRevision = 0;
  state.undo = [];
  state.redo = [];
  state.time = 0;
  state.dirty = false;
  render();
  const first = state.assets.find(
    (asset) =>
      asset.kind === 'image' &&
      !/(\.tiff?$|\.(preview|normals|mask)\.png$)/i.test(asset.name),
  );
  if (first) editor.addMedia(first, true);
  else schedulePreview();
}

async function saveProject() {
  try {
    if (!state.savedRevision)
      state.path = `projects/${slug(state.doc.name)}.json`;
    const snapshot = clone(state.doc);
    const session = state.session;
    const saved = await api('/api/project', {
      method: 'POST',
      data: {
        path: state.path,
        project: snapshot,
        expected_revision: state.savedRevision,
      },
    });
    state.projects = [...new Set([...state.projects, saved.path])];
    if (state.session !== session) {
      notify(`Saved ${saved.path}`);
      return;
    }
    state.savedRevision = saved.project.revision;
    state.doc.revision = saved.project.revision;
    state.dirty =
      JSON.stringify({ ...state.doc, revision: snapshot.revision }) !==
      JSON.stringify(snapshot);
    render();
    notify(`Saved ${saved.path} · revision ${state.savedRevision}`);
  } catch (error) {
    notify(error.message, true);
  }
}

async function openProject(path) {
  try {
    const saved = await api(`/api/project?path=${encodeURIComponent(path)}`);
    if (!saved.project.nodes.some((node) => node.job.kind === 'compose'))
      throw new Error(
        'This graph has no composition. Open its job JSON through the CLI.',
      );
    state.session++;
    state.doc = saved.project;
    state.path = saved.path;
    state.savedRevision = saved.project.revision;
    state.selectedNode = mainNode().id;
    state.selected = composition().layers[0]?.id || null;
    state.time = 0;
    state.undo = [];
    state.redo = [];
    state.dirty = false;
    $('#open-dialog').close();
    render();
    schedulePreview();
    notify(`Opened ${saved.path}`);
  } catch (error) {
    notify(error.message, true);
  }
}

async function submit(job, context = {}) {
  const record = await api('/api/jobs', {
    method: 'POST',
    data: { job: clone(job) },
  });
  state.jobs.unshift(record);
  if (context.nodeId) state.pending.set(record.id, context);
  actions.renderJobs();
  notify(`Queued ${context.label || job.kind}`);
  return record;
}

async function pollJobs() {
  try {
    const result = await api('/api/jobs');
    state.jobs = result.jobs;
    for (const record of state.jobs) {
      const context = state.pending.get(record.id);
      if (
        !context ||
        !['completed', 'failed', 'cancelled'].includes(record.status)
      )
        continue;
      state.pending.delete(record.id);
      if (record.status === 'completed')
        await actions.completed(record, context);
      else
        notify(
          record.error || `${context.label} ${record.status}`,
          record.status === 'failed',
        );
    }
    actions.renderJobs();
  } catch (error) {
    notify(error.message, true);
  }
}

const app = {
  state,
  $,
  escapeHTML,
  clone,
  api,
  notify,
  change,
  render,
  composition,
  mainNode,
  selectedLayer,
  fileURL,
  relative,
  slug,
  setTime,
  schedulePreview,
  refreshAssets,
  upload,
  submit,
  openProject,
};
editor = createEditor(app);
actions = createActions(app, editor);
editor.bind();
actions.bind();

$('#import-media').addEventListener('click', () => $('#file-input').click());
$('#file-input').addEventListener('change', (event) =>
  importFiles([...event.target.files]),
);
$('#project-name').addEventListener('change', (event) =>
  change(() => {
    state.doc.name = event.target.value.trim() || 'Untitled';
  }, false),
);
$('#save-project').addEventListener('click', saveProject);
$('#new-project').addEventListener('click', newProject);
$('#open-project').addEventListener('click', () => {
  $('#project-list').innerHTML = state.projects
    .map(
      (path) =>
        `<option value="${escapeHTML(path)}">${escapeHTML(path)}</option>`,
    )
    .join('');
  $('#project-path').value = state.path;
  $('#open-dialog').showModal();
});
$('#project-list').addEventListener('change', (event) => {
  $('#project-path').value = event.target.value;
});
$('#confirm-open').addEventListener('click', () =>
  openProject($('#project-path').value),
);
$('#undo').addEventListener('click', () => restore('undo'));
$('#redo').addEventListener('click', () => restore('redo'));
$('#play').addEventListener('click', togglePlay);
$('#time-range').addEventListener('input', (event) =>
  setTime(Number(event.target.value) / composition().fps),
);
$('#guides').addEventListener('change', (event) => {
  $('#safe-guides').hidden = !event.target.checked;
});
document
  .querySelectorAll('[data-close]')
  .forEach((button) =>
    button.addEventListener('click', () =>
      $('#' + button.dataset.close).close(),
    ),
  );
document.querySelectorAll('[data-library]').forEach((button) =>
  button.addEventListener('click', () => {
    state.library = button.dataset.library;
    render();
  }),
);
document.querySelectorAll('[data-inspector]').forEach((button) =>
  button.addEventListener('click', () => {
    state.inspector = button.dataset.inspector;
    render();
  }),
);
document.querySelectorAll('[data-aspect]').forEach((button) =>
  button.addEventListener('click', async () => {
    try {
      const [width, height] = button.dataset.aspect.split(',').map(Number);
      const result = await api('/api/reframe', {
        method: 'POST',
        data: { composition: clone(composition()), size: { width, height } },
      });
      change(() => {
        mainNode().job = result.job;
      });
    } catch (error) {
      notify(error.message, true);
    }
  }),
);
document.addEventListener('keydown', (event) => {
  const editing =
    event.target.matches('input,textarea,select') ||
    event.target.isContentEditable;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
    event.preventDefault();
    saveProject();
  }
  if (editing || $('dialog[open]')) return;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
    event.preventDefault();
    restore(event.shiftKey ? 'redo' : 'undo');
  }
  if (event.code === 'Space') {
    event.preventDefault();
    togglePlay();
  }
});
document.addEventListener('dragover', (event) => {
  event.preventDefault();
  document.body.classList.add('drop-active');
});
document.addEventListener('dragleave', (event) => {
  if (!event.relatedTarget) document.body.classList.remove('drop-active');
});
document.addEventListener('drop', (event) => {
  event.preventDefault();
  document.body.classList.remove('drop-active');
  if (event.dataTransfer.files.length)
    importFiles([...event.dataTransfer.files]);
});

try {
  const bootstrap = await api('/api/bootstrap');
  Object.assign(state, {
    token: bootstrap.token,
    workspace: bootstrap.workspace,
    assets: bootstrap.assets,
    projects: bootstrap.projects,
    jobs: bootstrap.jobs,
  });
  $('#workspace-label').textContent = state.workspace;
  $('#workspace-label').title = state.workspace;
  newProject();
  notify('Local workspace. Ready.');
  setInterval(pollJobs, 1200);
} catch (error) {
  notify(error.message, true);
  $('#preview-state').textContent = 'Studio could not open workspace';
}
