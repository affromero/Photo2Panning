export function createActions(app, editor) {
  const {
    state,
    $,
    escapeHTML: esc,
    clone,
    composition,
    selectedLayer,
    change,
    notify,
    api,
    fileURL,
  } = app;
  state.ai = {
    prompt: '',
    maskPrompt: '',
    seed: 7,
    steps: 16,
    quantization: 4,
    context: 64,
    maxEdge: 768,
    feather: 0,
    padding: 0,
    videoPrompt: '',
    videoFrames: 65,
    videoSteps: 20,
    lastFrame: '',
    references: [],
  };
  let maskLayer;
  let maskSource;
  let painting = false;
  let previousPoint = null;
  let erase = false;
  const unique = (prefix) => `${prefix}-${crypto.randomUUID().slice(0, 8)}`;
  const filename = (prefix, suffix) => `exports/${unique(prefix)}.${suffix}`;
  const aiField = (label, key, value, attributes = '') =>
    `<label>${esc(label)}<input data-ai="${key}" type="number" value="${value}" ${attributes}></label>`;

  function aiPanel() {
    const layer = selectedLayer();
    const image = layer?.kind === 'image';
    const asset = image ? editor.assetFor(layer.source) : null;
    const width = asset?.width || 1024;
    const height = asset?.height || 688;
    if (state.ai.roiLayer !== layer?.id) {
      state.ai.roiLayer = layer?.id;
      state.ai.roi = {
        left: Math.floor(width * 0.3),
        top: Math.floor(height * 0.3),
        right: Math.ceil(width * 0.7),
        bottom: Math.ceil(height * 0.7),
      };
    }
    const images = state.assets.filter(
      (item) => item.kind === 'image' && !/\.tiff?$/i.test(item.name),
    );
    return `<h3>Local AI jobs</h3><p>Jobs run only when you press a run button. Scrubbing uses existing media and cached depth.</p><label>Image edit prompt<textarea rows="4" data-ai="prompt" placeholder="Describe the complete image you want.">${esc(state.ai.prompt)}</textarea></label><div class="field-grid">${aiField('Seed', 'seed', state.ai.seed, 'min="0" step="1"')}${aiField('Steps', 'steps', state.ai.steps, 'min="1" step="1"')}<label>Weight bits<select data-ai="quantization">${[4, 6, 8].map((bits) => `<option ${bits === state.ai.quantization ? 'selected' : ''}>${bits}</option>`).join('')}</select></label></div><h3>FLUX 3 references</h3><p>${image ? 'The selected image is the first reference.' : 'Choose references or generate from text.'} Up to ten images total.</p><div class="references-list">${
      images
        .filter((item) => !image || item.path !== app.relative(layer.source))
        .map(
          (item) =>
            `<label class="check"><input type="checkbox" data-reference="${esc(item.path)}" ${state.ai.references.includes(item.path) ? 'checked' : ''}>${esc(item.name)}</label>`,
        )
        .join('') ||
      '<span class="tiny">Import images to use references.</span>'
    }</div><div class="ai-action"><button data-action="generate-image">Run FLUX 3 edit <span>↗</span></button></div><h3>Context-crop repair</h3><p>Repair an exact region at source resolution. A painted mask limits the selected pixels.</p><div class="field-grid">${Object.entries(
      state.ai.roi,
    )
      .map(([key, value]) =>
        aiField(
          key[0].toUpperCase() + key.slice(1),
          'roi.' + key,
          value,
          'min="0" step="1"',
        ),
      )
      .join(
        '',
      )}${aiField('Context / px', 'context', state.ai.context, 'min="0" max="4096" step="1"')}${aiField('Model edge / px', 'maxEdge', state.ai.maxEdge, 'min="16" max="4096" step="16"')}${aiField('Feather / px', 'feather', state.ai.feather, 'min="0" max="128" step="1"')}</div><div class="button-row"><button data-action="paint-mask" ${image ? '' : 'disabled'}>Paint region</button><button data-action="repair-image" ${image ? '' : 'disabled'}>Run repair ↗</button></div><h3>Text selection</h3><label>Objects to select<input data-ai="maskPrompt" value="${esc(state.ai.maskPrompt)}" placeholder="person . vehicle"></label>${aiField('Expand mask / px', 'padding', state.ai.padding, 'min="0" max="128" step="1"')}<button class="wide" data-action="select-mask" ${image ? '' : 'disabled'}>Run text selection ↗</button><h3>Image-guided video</h3><label>Last guide image<select data-ai="lastFrame"><option value="">Choose an image</option>${images.map((item) => `<option value="${esc(item.path)}" ${state.ai.lastFrame === item.path ? 'selected' : ''}>${esc(item.name)}</option>`).join('')}</select></label><label>Camera movement<textarea rows="3" data-ai="videoPrompt" placeholder="Describe one continuous camera move.">${esc(state.ai.videoPrompt)}</textarea></label><div class="field-grid">${aiField('Frames (8k + 1)', 'videoFrames', state.ai.videoFrames, 'min="9" step="8"')}${aiField('Steps', 'videoSteps', state.ai.videoSteps, 'min="1" step="1"')}</div><p>Uses the selected photo as the first guide. Output is 576 × 1024 at 24 fps.</p><button class="wide" data-action="generate-video" ${image ? '' : 'disabled'}>Run guided video ↗</button>`;
  }
  app.aiPanel = aiPanel;

  function sourcesOf(value, key = '') {
    if (Array.isArray(value))
      return value.flatMap((item) => sourcesOf(item, key));
    if (value && typeof value === 'object')
      return Object.entries(value).flatMap(([name, item]) =>
        sourcesOf(item, name),
      );
    return typeof value === 'string' &&
      [
        'source',
        'mask',
        'metadata',
        'first_frame',
        'last_frame',
        'reference',
        'references',
      ].includes(key)
      ? [app.relative(value)]
      : [];
  }

  function dependencies(job) {
    const sources = sourcesOf(job);
    return state.doc.nodes
      .filter((node) => {
        const output = app.relative(node.job.output || '');
        const artifacts = [output];
        if (node.job.kind === 'depth')
          artifacts.push(output.replace(/\.tiff?$/i, '.depth.json'));
        return artifacts.some((path) => path && sources.includes(path));
      })
      .map((node) => node.id);
  }

  async function runNode(node, context = {}) {
    const request = {
      kind: 'project',
      project: clone(state.doc),
      workspace: state.workspace,
      output: `reports/${unique('run')}.json`,
      targets: [node.id],
      cache: 'reuse',
      overwrite: true,
    };
    return app.submit(request, {
      nodeId: node.id,
      label: node.job.kind,
      ...context,
      output: node.job.output,
      jobJSON: JSON.stringify(node.job),
    });
  }

  async function runOperation(job, action, context = {}) {
    const layer = selectedLayer();
    const node = {
      id: unique(job.kind.replace(/[^a-z]/g, '')),
      job,
      depends_on: dependencies(job),
      enabled: true,
    };
    change(() => {
      state.doc.nodes.push(node);
    }, false);
    return runNode(node, {
      action,
      layerId: layer?.id,
      source: layer?.source,
      mask: layer?.mask,
      ...context,
    });
  }

  function requireImage() {
    const layer = selectedLayer();
    if (layer?.kind !== 'image')
      throw new Error('Select an image layer first.');
    return layer;
  }

  async function runAI(action) {
    const config = state.ai;
    const prompt = config.prompt.trim();
    const settings = {
      seed: config.seed,
      steps: config.steps,
      quantization: config.quantization,
    };
    if (action === 'generate-image') {
      if (!prompt) throw new Error('Write an image edit prompt first.');
      const layer = selectedLayer();
      const source = layer?.kind === 'image' ? layer.source : null;
      const references = config.references.filter(
        (path) => path !== app.relative(source || ''),
      );
      if (references.length + Number(Boolean(source)) > 10)
        throw new Error(
          'Choose at most ten reference images including the selected image.',
        );
      const asset = source ? editor.assetFor(source) : null;
      const ratio =
        asset?.width && asset?.height ? asset.width / asset.height : 1;
      const width =
        ratio >= 1 ? 512 : Math.max(16, Math.round((512 * ratio) / 16) * 16);
      const height =
        ratio >= 1 ? Math.max(16, Math.round(512 / ratio / 16) * 16) : 512;
      await runOperation(
        {
          kind: 'generate',
          source,
          references,
          output: filename('flux3', 'png'),
          prompt,
          model: 'flux3',
          size: { width, height },
          ...settings,
        },
        'image',
        { source },
      );
    } else if (action === 'repair-image') {
      const layer = requireImage();
      if (!prompt) throw new Error('Describe the desired repair first.');
      await runOperation(
        {
          kind: 'repair',
          source: layer.source,
          output: filename('repair', 'png'),
          region: { op: 'crop', ...config.roi },
          mask: layer.mask || null,
          prompt,
          context: config.context,
          max_edge: config.maxEdge,
          feather: config.feather,
          ...settings,
        },
        'image',
      );
    } else if (action === 'select-mask') {
      const layer = requireImage();
      if (!config.maskPrompt.trim())
        throw new Error('Name the objects to select first.');
      await runOperation(
        {
          kind: 'mask',
          source: layer.source,
          output: filename('selection', 'png'),
          prompt: config.maskPrompt.trim(),
          padding: config.padding,
          merge: true,
        },
        'mask',
      );
    } else if (action === 'extract-depth') {
      const layer = requireImage();
      await runOperation(
        {
          kind: 'depth',
          source: layer.source,
          output: filename('depth', 'tiff'),
        },
        'depth',
      );
    } else if (action === 'generate-video') {
      const layer = requireImage();
      if (!config.lastFrame || !config.videoPrompt.trim())
        throw new Error(
          'Choose the last guide image and describe the camera movement.',
        );
      await runOperation(
        {
          kind: 'generate-video',
          first_frame: layer.source,
          last_frame: config.lastFrame,
          output: filename('guided-video', 'mp4'),
          prompt: config.videoPrompt.trim(),
          size: { width: 576, height: 1024 },
          frames: config.videoFrames,
          fps: 24,
          seed: config.seed,
          steps: config.videoSteps,
        },
        'video',
      );
    }
  }

  async function completed(record, context) {
    await app.refreshAssets();
    const node = state.doc.nodes.find((item) => item.id === context.nodeId);
    if (!node) {
      notify(`${context.label} completed. Outputs are in Jobs.`);
      return;
    }
    if (context.jobJSON && JSON.stringify(node.job) !== context.jobJSON) {
      notify(
        'Job completed from an earlier graph revision. Outputs are in Media and Jobs.',
      );
      return;
    }
    const layer = composition().layers.find(
      (item) => item.id === context.layerId,
    );
    const output = app.relative(context.output || node.job.output || '');
    const canApply =
      layer &&
      ['image', 'video'].includes(layer.kind) &&
      typeof context.source === 'string' &&
      layer.source === context.source;
    if (context.action === 'beats') {
      const analysis = await fetch(fileURL(output)).then((response) =>
        response.json(),
      );
      change(() => {
        composition().beats = [
          ...new Set(
            analysis.markers
              .map((marker) => marker + (context.start || 0))
              .filter((marker) => marker < composition().duration),
          ),
        ].sort((left, right) => left - right);
        editor.linkSource(output);
      });
    } else if (context.action === 'depth' && canApply) {
      const metadata = output.replace(/\.tiff?$/i, '.depth.json');
      const values = await fetch(fileURL(metadata)).then((response) =>
        response.json(),
      );
      change(() => {
        layer.depth = {
          metadata,
          focus_metres:
            (values.preview_near_metres + values.preview_far_metres) / 2,
        };
        if (!app.mainNode().depends_on.includes(node.id))
          app.mainNode().depends_on.push(node.id);
      });
    } else if (
      context.action === 'mask' &&
      canApply &&
      layer.mask === context.mask
    ) {
      change(() => {
        layer.mask = output;
        if (!app.mainNode().depends_on.includes(node.id))
          app.mainNode().depends_on.push(node.id);
      });
    } else if (['image', 'video'].includes(context.action)) {
      const asset = state.assets.find((item) => item.path === output);
      if (canApply && asset) {
        change(() => {
          layer.source = output;
          delete layer.depth;
          for (const other of composition().layers)
            if (other.occlusion?.layer_id === layer.id) delete other.occlusion;
          delete layer.mask;
          delete layer.cinemagraph;
          layer.kind = context.action === 'video' ? 'video' : 'image';
          if (layer.kind === 'video') {
            layer.source_start = 0;
            layer.speed = 1;
            layer.duration = Math.min(
              layer.duration,
              asset.duration || layer.duration,
            );
            layer.keyframes = layer.keyframes.filter(
              (frame) => frame.at <= layer.duration,
            );
          } else {
            delete layer.speed;
            delete layer.source_start;
            delete layer.source_times;
          }
          if (!app.mainNode().depends_on.includes(node.id))
            app.mainNode().depends_on.push(node.id);
        });
      } else if (asset) editor.addMedia(asset);
    }
    app.render();
    app.schedulePreview();
    notify(
      `${context.label} completed. ${record.result?.nodes?.filter((node) => node.cache_hit).length || 0} cached nodes reused.`,
    );
  }

  function renderJobs() {
    const active = state.jobs.filter((job) =>
      ['queued', 'running'].includes(job.status),
    ).length;
    $('#job-count').textContent = active || state.jobs.length;
    $('#jobs').innerHTML =
      state.jobs
        .map(
          (job) =>
            `<section class="job-item"><div class="job-heading"><strong>${esc(state.pending.get(job.id)?.label || job.kind)}</strong><span class="badge">${esc(job.status)}</span>${['queued', 'running'].includes(job.status) ? `<button data-cancel-job="${esc(job.id)}">Cancel</button>` : ''}</div>${job.started ? `<p class="tiny">${((job.finished || Date.now() / 1000) - job.started).toFixed(1)} seconds</p>` : ''}${job.error ? `<p class="error-text">${esc(job.error)}</p>` : ''}${job.outputs.map((path) => `<a href="${fileURL(path)}" target="_blank" rel="noopener">${esc(path)} ↗</a>`).join('')}${job.log ? `<details><summary>Worker log</summary><pre>${esc(job.log)}</pre></details>` : ''}</section>`,
        )
        .join('') || '<p>No jobs submitted in this session.</p>';
  }

  async function openMask() {
    const layer = selectedLayer();
    if (!layer || !['image', 'video'].includes(layer.kind))
      throw new Error('Select an image or video layer.');
    maskLayer = layer.id;
    maskSource = layer.source;
    const asset = editor.assetFor(layer.source);
    let source = fileURL(layer.source);
    if (layer.kind === 'video') {
      const video = document.createElement('video');
      video.preload = 'auto';
      video.src = source;
      await new Promise((resolve, reject) => {
        video.addEventListener('loadeddata', resolve, { once: true });
        video.addEventListener(
          'error',
          () => reject(new Error('Could not read the video frame.')),
          { once: true },
        );
      });
      const frame = document.createElement('canvas');
      frame.width = video.videoWidth;
      frame.height = video.videoHeight;
      frame.getContext('2d').drawImage(video, 0, 0);
      source = frame.toDataURL('image/png');
      video.removeAttribute('src');
      video.load();
    }
    const reference = $('#mask-reference');
    reference.src = source;
    await reference.decode();
    const canvas = $('#mask-canvas');
    canvas.width = asset?.width || reference.naturalWidth;
    canvas.height = asset?.height || reference.naturalHeight;
    const context = canvas.getContext('2d');
    context.fillStyle = 'black';
    context.fillRect(0, 0, canvas.width, canvas.height);
    const existing = layer.cinemagraph?.mask || layer.mask;
    if (existing) {
      const image = new Image();
      image.src = fileURL(existing);
      await image.decode();
      if (
        image.naturalWidth !== canvas.width ||
        image.naturalHeight !== canvas.height
      )
        throw new Error(
          'Mask dimensions differ from the source. Clear the mask before painting.',
        );
      context.drawImage(image, 0, 0);
    }
    $('#mask-area').style.aspectRatio = String(canvas.width / canvas.height);
    const maxWidth = Math.min(1030, window.innerWidth * 0.87);
    const maxHeight = window.innerHeight * 0.63;
    $('#mask-area').style.width =
      `${Math.min(maxWidth, (maxHeight * canvas.width) / canvas.height)}px`;
    $('#mask-area').style.height =
      `${Math.min(maxHeight, (maxWidth * canvas.height) / canvas.width)}px`;
    $('#mask-dialog').showModal();
  }

  function paint(event) {
    if (!painting) return;
    const canvas = $('#mask-canvas');
    const rect = canvas.getBoundingClientRect();
    const point = {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
    const context = canvas.getContext('2d');
    context.strokeStyle = erase ? 'black' : 'white';
    context.fillStyle = context.strokeStyle;
    context.lineWidth = Number($('#brush-size').value);
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.beginPath();
    if (previousPoint) {
      context.moveTo(previousPoint.x, previousPoint.y);
      context.lineTo(point.x, point.y);
      context.stroke();
    } else {
      context.arc(point.x, point.y, context.lineWidth / 2, 0, Math.PI * 2);
      context.fill();
    }
    previousPoint = point;
  }

  async function saveMask() {
    const canvas = $('#mask-canvas');
    const data = canvas
      .getContext('2d')
      .getImageData(0, 0, canvas.width, canvas.height).data;
    let left = canvas.width,
      top = canvas.height,
      right = 0,
      bottom = 0;
    for (let y = 0; y < canvas.height; y++)
      for (let x = 0; x < canvas.width; x++)
        if (data[(y * canvas.width + x) * 4] > 0) {
          left = Math.min(left, x);
          right = Math.max(right, x + 1);
          top = Math.min(top, y);
          bottom = Math.max(bottom, y + 1);
        }
    if (right <= left || bottom <= top)
      throw new Error('Paint a selected region before saving the mask.');
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, 'image/png'),
    );
    const asset = await app.upload(
      new File([blob], unique('painted-mask') + '.png', { type: 'image/png' }),
    );
    const layer = composition().layers.find((item) => item.id === maskLayer);
    if (!layer)
      throw new Error(
        'The selected layer was removed. The mask is available in Media.',
      );
    if (layer.source !== maskSource)
      throw new Error(
        'The layer source changed while painting. The mask is available in Media.',
      );
    change(() => {
      if (layer.cinemagraph) layer.cinemagraph.mask = asset.path;
      else layer.mask = asset.path;
      state.ai.roi = { left, top, right, bottom };
      state.ai.roiLayer = layer.id;
    });
    $('#mask-dialog').close();
    notify('Selection mask saved.');
  }

  async function exportJob(mode) {
    const current = clone(composition());
    current.output = $('#export-path').value.trim();
    current.overwrite = $('#overwrite').checked;
    if (mode === 'render-full') {
      change(() => {
        composition().output = current.output;
        composition().overwrite = current.overwrite;
      }, false);
      await app.submit(current, {
        nodeId: app.mainNode().id,
        label: 'Movie export',
        action: 'export',
      });
    } else if (mode === 'render-draft') {
      const factor = Math.min(
        1,
        720 / Math.max(current.size.width, current.size.height),
      );
      const size = {
        width: Math.max(2, Math.round((current.size.width * factor) / 2) * 2),
        height: Math.max(
          2,
          Math.round((current.size.height * factor) / 2) * 2,
        ),
      };
      const reframed = await api('/api/reframe', {
        method: 'POST',
        data: { composition: current, size },
      });
      const job = {
        ...reframed.job,
        output: filename('draft', 'mp4'),
        fps: Math.min(12, current.fps),
      };
      await runOperation(job, 'export', { label: '720p draft' });
    } else if (mode === 'render-social') {
      const moments = $('#carousel-times')
        .value.split(',')
        .map((value) => value.trim())
        .filter(Boolean)
        .map(Number);
      if (
        moments.some(
          (value) =>
            !Number.isFinite(value) || value < 0 || value >= current.duration,
        )
      )
        throw new Error('Carousel times must be within the movie duration.');
      const job = {
        kind: 'social-pack',
        composition: current,
        output_directory: `exports/${unique('social')}`,
        cover_at: state.time,
        carousel_at: moments,
        preview_guides: $('#pack-guides').checked,
        overwrite: current.overwrite,
      };
      await runOperation(job, 'export', { label: 'Social pack' });
    } else {
      change(() => {
        composition().output = current.output;
        composition().overwrite = current.overwrite;
      }, false);
      const request = {
        kind: 'project',
        project: clone(state.doc),
        workspace: state.workspace,
        output: `reports/${unique('graph')}.json`,
        targets: [app.mainNode().id],
        cache: 'reuse',
        overwrite: current.overwrite,
      };
      await app.submit(request, {
        nodeId: app.mainNode().id,
        label: 'Project graph',
        action: 'export',
      });
    }
    $('#export-dialog').close();
    $('#jobs-dialog').showModal();
  }

  function bind() {
    $('#export').addEventListener('click', () => {
      $('#export-path').value = composition().output;
      $('#overwrite').checked = composition().overwrite || false;
      $('#carousel-times').value = [
        0,
        composition().duration / 3,
        (composition().duration * 2) / 3,
      ]
        .map((value) => value.toFixed(2))
        .join(', ');
      $('#export-dialog').showModal();
    });
    for (const name of [
      'render-full',
      'render-draft',
      'render-social',
      'run-project',
    ])
      $('#' + name).addEventListener('click', () =>
        exportJob(name).catch((error) => notify(error.message, true)),
      );
    $('#show-jobs').addEventListener('click', () =>
      $('#jobs-dialog').showModal(),
    );
    $('#jobs').addEventListener('click', async (event) => {
      const button = event.target.closest('[data-cancel-job]');
      if (button) {
        try {
          await api(
            `/api/cancel?id=${encodeURIComponent(button.dataset.cancelJob)}`,
            { method: 'POST' },
          );
          notify('Job cancelled.');
        } catch (error) {
          notify(error.message, true);
        }
      }
    });
    $('#inspector').addEventListener('input', (event) => {
      const input = event.target;
      if (input.dataset.ai) {
        const value =
          input.type === 'number' || input.dataset.ai === 'quantization'
            ? Number(input.value)
            : input.value;
        if (input.dataset.ai.startsWith('roi.'))
          state.ai.roi[input.dataset.ai.slice(4)] = value;
        else state.ai[input.dataset.ai] = value;
      }
      if (input.dataset.reference)
        state.ai.references = input.checked
          ? [...new Set([...state.ai.references, input.dataset.reference])]
          : state.ai.references.filter(
              (path) => path !== input.dataset.reference,
            );
    });
    document.addEventListener('click', async (event) => {
      const button = event.target.closest('[data-action]');
      if (!button) return;
      const action = button.dataset.action;
      try {
        if (
          [
            'generate-image',
            'repair-image',
            'select-mask',
            'extract-depth',
            'generate-video',
          ].includes(action)
        )
          await runAI(action);
        if (action === 'paint-mask') await openMask();
        if (action === 'run-node' || action === 'run-graph') {
          const node =
            state.doc.nodes.find((item) => item.id === state.selectedNode) ||
            app.mainNode();
          await runNode(node, { label: node.id, action: 'export' });
        }
        if (action === 'beats') {
          const track = composition().audio[Number(button.dataset.trackIndex)];
          await runOperation(
            {
              kind: 'beats',
              source: track.source,
              output: filename('beats', 'json'),
              start: track.source_start,
              duration: track.duration,
            },
            'beats',
            { start: track.start },
          );
        }
      } catch (error) {
        notify(error.message, true);
      }
    });
    const canvas = $('#mask-canvas');
    canvas.addEventListener('pointerdown', (event) => {
      painting = true;
      previousPoint = null;
      canvas.setPointerCapture(event.pointerId);
      paint(event);
    });
    canvas.addEventListener('pointermove', paint);
    canvas.addEventListener('pointerup', () => {
      painting = false;
      previousPoint = null;
    });
    canvas.addEventListener('pointercancel', () => {
      painting = false;
      previousPoint = null;
    });
    $('#brush-erase').addEventListener('click', () => {
      erase = !erase;
      $('#brush-erase').setAttribute('aria-pressed', String(erase));
    });
    $('#mask-clear').addEventListener('click', () => {
      const context = canvas.getContext('2d');
      context.fillStyle = 'black';
      context.fillRect(0, 0, canvas.width, canvas.height);
    });
    $('#mask-save').addEventListener('click', () =>
      saveMask().catch((error) => notify(error.message, true)),
    );
  }

  return { bind, renderJobs, completed };
}
