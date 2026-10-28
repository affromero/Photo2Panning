export function initialComposition() {
  return {
    kind: 'compose',
    canvas: { width: 1920, height: 1080 },
    duration: 5,
    camera: [{ at: 0, x: 960, y: 540, zoom: 1, ease: 'linear' }],
    layers: [
      {
        id: 'starter',
        kind: 'text',
        text: 'UNTITLED',
        font_size: 90,
        color: '#eeeede',
        duration: 5,
        start: 0,
        size: { width: 900, height: 170 },
        space: 'world',
        keyframes: [
          {
            at: 0,
            x: 960,
            y: 540,
            scale: 1,
            rotation: 0,
            opacity: 1,
            ease: 'linear',
          },
        ],
      },
    ],
    audio: [],
    beats: [],
    background: '#11140f',
    output: 'exports/untitled.mp4',
    size: { width: 1080, height: 1920 },
    fps: 30,
    overwrite: false,
  };
}

export function createEditor(app) {
  const {
    state,
    $,
    escapeHTML: esc,
    change,
    composition,
    selectedLayer,
    fileURL,
    notify,
  } = app;
  const id = (prefix) => `${prefix}-${crypto.randomUUID().slice(0, 8)}`;
  const get = (object, path) =>
    path.split('.').reduce((value, key) => value?.[key], object);
  const set = (object, path, value) => {
    const keys = path.split('.');
    const last = keys.pop();
    let target = object;
    for (const key of keys) target = target[key] ||= {};
    target[last] = value;
  };
  const field = (label, scope, path, value, extra = '') =>
    `<label>${esc(label)}<input data-scope="${scope}" data-field="${path}" type="number" value="${Number.isFinite(value) ? value : 0}" ${extra.includes('step=') ? '' : 'step="any"'} ${extra}></label>`;
  const text = (label, scope, path, value, extra = '') =>
    `<label>${esc(label)}<input data-scope="${scope}" data-field="${path}" value="${esc(value)}" ${extra}></label>`;
  const select = (label, scope, path, value, options) =>
    `<label>${esc(label)}<select data-scope="${scope}" data-field="${path}">${options.map((option) => `<option value="${esc(option[0])}" ${option[0] === value ? 'selected' : ''}>${esc(option[1])}</option>`).join('')}</select></label>`;
  const check = (label, scope, path, value) =>
    `<label class="check"><input type="checkbox" data-scope="${scope}" data-field="${path}" ${value ? 'checked' : ''}>${esc(label)}</label>`;
  const easings = [
    ['linear', 'Linear'],
    ['ease-in', 'Ease in'],
    ['ease-out', 'Ease out'],
    ['ease-in-out', 'Ease in & out'],
    ['hold', 'Hold'],
  ];
  const assetFor = (source) =>
    state.assets.find((asset) => asset.path === app.relative(source));

  function poseAt(frames, at) {
    let previous = frames[0];
    for (const frame of frames) {
      if (frame.at > at) {
        let progress = Math.max(
          0,
          (at - previous.at) / (frame.at - previous.at),
        );
        if (previous.ease === 'hold') progress = 0;
        else if (previous.ease === 'ease-in') progress *= progress;
        else if (previous.ease === 'ease-out')
          progress = 1 - (1 - progress) ** 2;
        else if (previous.ease === 'ease-in-out')
          progress = progress * progress * (3 - 2 * progress);
        const pose = { ...previous, at };
        for (const key of ['x', 'y', 'scale', 'rotation', 'opacity', 'zoom'])
          if (key in frame)
            pose[key] =
              previous[key] + (frame[key] - previous[key]) * progress;
        return pose;
      }
      previous = frame;
    }
    return { ...previous, at };
  }

  function snapTime(seconds) {
    const markers = composition().beats || [];
    if (!$('#snap-beats').checked || !markers.length) return seconds;
    const nearest = markers.reduce((best, marker) =>
      Math.abs(marker - seconds) < Math.abs(best - seconds) ? marker : best,
    );
    return Math.abs(nearest - seconds) <= 0.15 ? nearest : seconds;
  }

  function ensureFrame(layer = selectedLayer()) {
    const current = composition();
    const global = snapTime(state.time);
    const at = layer
      ? Math.max(0, Math.min(layer.duration, global - layer.start))
      : global;
    const frames = layer ? layer.keyframes : current.camera;
    let frame = frames.find((item) => Math.abs(item.at - at) < 1e-5);
    if (!frame) {
      frame = poseAt(frames, at);
      frames.push(frame);
      frames.sort((left, right) => left.at - right.at);
    }
    return frame;
  }

  function linkSource(source, node = app.mainNode()) {
    const producer = state.doc.nodes.find(
      (item) =>
        item.id !== node.id &&
        (app.relative(item.job.output || '') === app.relative(source) ||
          (item.job.kind === 'depth' &&
            app.relative(source) ===
              app.relative(
                item.job.output.replace(/\.tiff?$/i, '.depth.json'),
              ))),
    );
    if (producer && !node.depends_on.includes(producer.id))
      node.depends_on.push(producer.id);
  }

  function addMedia(asset, replace = false) {
    if (!['image', 'video'].includes(asset.kind)) return;
    change(() => {
      const current = composition();
      const first =
        replace ||
        (current.layers.length === 1 && current.layers[0].id === 'starter');
      if (first) {
        const height = Math.max(
          2,
          Math.round((1920 * (asset.height || 1080)) / (asset.width || 1920)),
        );
        current.canvas = { width: 1920, height };
        current.camera = [
          { at: 0, x: 960, y: height / 2, zoom: 1, ease: 'linear' },
        ];
        current.layers = [];
        if (asset.kind === 'video' && asset.duration)
          current.duration = Math.min(asset.duration, 30);
      }
      const start = first
        ? 0
        : Math.min(state.time, current.duration - 1 / current.fps);
      const height = current.canvas.height;
      const width = Math.max(
        1,
        Math.round((height * (asset.width || 1920)) / (asset.height || 1080)),
      );
      const right = first
        ? 0
        : Math.max(
            0,
            ...current.layers
              .filter((item) => item.space !== 'viewport')
              .map(
                (item) =>
                  item.keyframes[0].x +
                  (item.size.width * (item.keyframes[0].scale || 1)) / 2,
              ),
          );
      const layer = {
        id: id(asset.kind),
        kind: asset.kind,
        source: asset.path,
        fit: 'contain',
        size: { width, height },
        space: 'world',
        start,
        duration: Math.min(
          current.duration - start,
          asset.kind === 'video'
            ? asset.duration || current.duration
            : current.duration,
        ),
        keyframes: [
          {
            at: 0,
            x: right + width / 2,
            y: current.canvas.height / 2,
            scale: 1,
            rotation: 0,
            opacity: 1,
            ease: 'linear',
          },
        ],
      };
      if (asset.kind === 'video')
        Object.assign(layer, { source_start: 0, speed: 1 });
      current.layers.push(layer);
      current.canvas.width = Math.max(
        current.canvas.width,
        Math.ceil(right + width),
      );
      state.selected = layer.id;
      state.selectedNode = app.mainNode().id;
      state.inspector = 'transform';
      state.library = 'layers';
      linkSource(asset.path);
    });
  }

  function addAudio(asset) {
    change(() => {
      const duration = Math.min(
        asset.duration || composition().duration,
        composition().duration,
      );
      composition().audio ||= [];
      composition().audio.push({
        source: asset.path,
        role: 'music',
        start: 0,
        source_start: 0,
        duration,
        loop: false,
        gain_db: -6,
        fade_in: Math.min(0.2, duration / 4),
        fade_out: Math.min(0.2, duration / 4),
      });
      state.library = 'sound';
      linkSource(asset.path);
    });
  }

  function addText() {
    change(() => {
      const current = composition();
      const layer = {
        id: id('text'),
        kind: 'text',
        text: 'New text',
        font_size: 64,
        color: 'white',
        size: { width: 900, height: 180 },
        start: 0,
        duration: current.duration,
        space: 'world',
        keyframes: [
          {
            at: 0,
            x: current.canvas.width / 2,
            y: current.canvas.height / 2,
            scale: 1,
            rotation: 0,
            opacity: 1,
            ease: 'linear',
          },
        ],
      };
      current.layers.push(layer);
      state.selected = layer.id;
      state.inspector = 'transform';
      state.library = 'layers';
    });
  }

  function renderLibrary() {
    if (!state.doc) return;
    document
      .querySelectorAll('[data-library]')
      .forEach((button) =>
        button.classList.toggle(
          'active',
          button.dataset.library === state.library,
        ),
      );
    const current = composition();
    let html = '';
    if (state.library === 'media') {
      html = `<div class="tiny">${state.assets.length} WORKSPACE ASSETS<br>Click media to add a layer.</div>`;
      html += state.assets
        .map((asset) => {
          const thumb =
            asset.kind === 'image' && !/\.tiff?$/i.test(asset.name)
              ? `<img class="asset-thumb" src="${fileURL(asset.path)}" alt="" loading="lazy">`
              : `<span class="asset-thumb">${esc(asset.kind.slice(0, 3).toUpperCase())}</span>`;
          return `<div class="asset-row" role="button" tabindex="0" data-asset="${esc(asset.path)}">${thumb}<span class="asset-info"><strong>${esc(asset.name)}</strong><small>${asset.width ? `${asset.width} × ${asset.height}` : asset.duration ? `${asset.duration.toFixed(2)} SEC` : asset.kind.toUpperCase()} · ${(asset.bytes / 1024 ** 2).toFixed(1)} MB</small></span></div>`;
        })
        .join('');
      if (!state.assets.length)
        html +=
          '<div class="empty-state">Import photos, footage, and sound, or drop files into the workspace.</div>';
    } else if (state.library === 'layers') {
      html = `<div class="button-row"><button data-action="camera">Camera</button><button data-action="add-text">+ Text</button></div>`;
      html += [...current.layers]
        .reverse()
        .map(
          (layer) =>
            `<div class="layer-row ${layer.id === state.selected ? 'selected' : ''}" data-layer="${esc(layer.id)}"><span class="layer-type">${layer.kind.toUpperCase().slice(0, 3)}</span><strong>${esc(layer.kind === 'text' ? layer.text : assetFor(layer.source)?.name || layer.id)}</strong><button data-order="1" data-layer-id="${esc(layer.id)}" aria-label="Move layer forward">↑</button><button data-order="-1" data-layer-id="${esc(layer.id)}" aria-label="Move layer backward">↓</button></div>`,
        )
        .join('');
    } else if (state.library === 'sound') {
      html = '<h3>Soundtracks</h3>';
      html += (current.audio || [])
        .map(
          (track, index) =>
            `<div class="sound-row" data-track="${index}"><div class="sound-heading"><strong>${esc(assetFor(track.source)?.name || track.source.split('/').pop())}</strong><button data-remove-track="${index}" aria-label="Remove track">×</button></div>${select(
              'Role',
              'track',
              'role',
              track.role,
              [
                ['music', 'Music'],
                ['voice', 'Voice'],
                ['ambience', 'Ambience'],
              ],
            )}<div class="field-grid">${field('Start / sec', 'track', 'start', track.start, 'min="0"')}${field('Length / sec', 'track', 'duration', track.duration, 'min="0.01"')}${field('Source offset', 'track', 'source_start', track.source_start, 'min="0"')}${field('Gain / dB', 'track', 'gain_db', track.gain_db, 'min="-120" max="36"')}${field('Fade in', 'track', 'fade_in', track.fade_in, 'min="0"')}${field('Fade out', 'track', 'fade_out', track.fade_out, 'min="0"')}</div>${check('Loop source', 'track', 'loop', track.loop)}${track.role !== 'voice' ? `<div class="button-row">${check('Duck under voice', 'track', 'ducking', Boolean(track.ducking))}</div>` : ''}<div class="button-row"><button data-action="beats" data-track-index="${index}">Detect beats ↗</button></div></div>`,
        )
        .join('');
      html +=
        '<h3>Available audio</h3>' +
        state.assets
          .filter((asset) => asset.kind === 'audio')
          .map(
            (asset) =>
              `<button class="wide" data-add-audio="${esc(asset.path)}">+ ${esc(asset.name)}</button>`,
          )
          .join('');
      if (!(current.audio || []).length)
        html +=
          '<p>Import a soundtrack, choose its role, and set its timing. Video sound is selected here explicitly.</p>';
    } else {
      html =
        '<div class="tiny">SAVED OPERATION GRAPH<br>Dependencies use node identifiers.</div>';
      html += state.doc.nodes
        .map(
          (node) =>
            `<div class="node-row"><div class="node-heading"><input type="checkbox" data-node-enabled="${esc(node.id)}" ${node.enabled ? 'checked' : ''} aria-label="Enable ${esc(node.id)}"><button data-node="${esc(node.id)}">${esc(node.id)}</button><span class="badge">${esc(node.job.kind)}</span></div><small>Depends on</small><input data-node-deps="${esc(node.id)}" value="${esc(node.depends_on.join(', '))}" aria-label="Dependencies for ${esc(node.id)}"></div>`,
        )
        .join('');
      html +=
        '<div class="button-row"><button data-action="run-graph">Run selected graph ↗</button></div>';
    }
    $('#library').innerHTML = html;
  }

  function transformPanel(layer) {
    const current = composition();
    if (!layer) {
      const pose = poseAt(current.camera, state.time);
      return `<h3>Camera</h3><p>Position uses world pixels. Zoom changes the camera viewport.</p><div class="field-grid">${field('Center X', 'camera', 'x', pose.x)}${field('Center Y', 'camera', 'y', pose.y)}${field('Zoom', 'camera', 'zoom', pose.zoom, 'min="0.001"')}</div>${select('Following segment', 'camera', 'ease', pose.ease, easings)}<div class="button-row"><button data-action="keyframe">◇ Add camera keyframe</button></div><h3>Composition</h3><div class="field-grid">${field('Duration / sec', 'composition', 'duration', current.duration, 'min="0.1"')}${field('Frames / sec', 'composition', 'fps', current.fps, 'min="1" max="240" step="1"')}${field('Canvas width', 'composition', 'canvas.width', current.canvas.width, 'min="1" step="1"')}${field('Canvas height', 'composition', 'canvas.height', current.canvas.height, 'min="1" step="1"')}${field('Output width', 'composition', 'size.width', current.size.width, 'min="2" step="2"')}${field('Output height', 'composition', 'size.height', current.size.height, 'min="2" step="2"')}</div>${text('Background color', 'composition', 'background', current.background)}`;
    }
    const local = Math.max(
      0,
      Math.min(layer.duration, state.time - layer.start),
    );
    const pose = poseAt(layer.keyframes, local);
    let html = `<h3>${esc(layer.kind)} layer</h3><div class="tiny" title="${esc(layer.source || layer.id)}">${esc(layer.source ? assetFor(layer.source)?.name || layer.source.split('/').pop() : layer.id)}</div><div class="button-row"><button data-action="frame-layer">Camera to layer</button></div>`;
    if (layer.kind === 'text')
      html += `<label>Text<textarea rows="3" data-scope="layer" data-field="text">${esc(layer.text)}</textarea></label><div class="field-grid">${field('Font size', 'layer', 'font_size', layer.font_size || 64, 'min="1" step="1"')}${text('Color', 'layer', 'color', layer.color || 'white')}</div>${select('Font', 'layer', 'font', layer.font || '', [['', 'Bundled font'], ...state.assets.filter((asset) => asset.kind === 'font').map((asset) => [asset.path, asset.name])])}<div class="field-grid">${field('Stroke width', 'layer', 'stroke_width', layer.stroke_width || 0, 'min="0" step="1"')}${text('Stroke color', 'layer', 'stroke_color', layer.stroke_color || 'black')}</div>`;
    html += `<h3>Transform at ${local.toFixed(2)} s</h3><div class="field-grid">${field('Center X', 'pose', 'x', pose.x)}${field('Center Y', 'pose', 'y', pose.y)}${field('Scale', 'pose', 'scale', pose.scale, 'min="0.001"')}${field('Rotation / °', 'pose', 'rotation', pose.rotation)}${field('Opacity', 'pose', 'opacity', pose.opacity, 'min="0" max="1"')}</div>${select('Following segment', 'pose', 'ease', pose.ease || 'linear', easings)}<div class="button-row"><button data-action="keyframe">◇ Add keyframe</button><button data-action="remove-keyframe">Remove keyframe</button></div><h3>Layer timing &amp; frame</h3><div class="field-grid">${field('Start / sec', 'layer', 'start', layer.start || 0, 'min="0"')}${field('Length / sec', 'layer', 'duration', layer.duration, 'min="0.01"')}${field('Width', 'layer', 'size.width', layer.size.width, 'min="1" step="1"')}${field('Height', 'layer', 'size.height', layer.size.height, 'min="1" step="1"')}</div>${select(
      'Coordinate space',
      'layer',
      'space',
      layer.space || 'world',
      [
        ['world', 'World canvas'],
        ['viewport', 'Output pixels'],
      ],
    )}`;
    if (layer.kind !== 'text')
      html += select('Source fit', 'layer', 'fit', layer.fit || 'contain', [
        ['contain', 'Contain'],
        ['cover', 'Cover'],
      ]);
    if (layer.kind === 'video')
      html += `<h3>Video source</h3><div class="field-grid">${field('Source offset', 'layer', 'source_start', layer.source_start || 0, 'min="0"')}${field('Speed', 'layer', 'speed', layer.speed || 1, 'min="0.001"')}</div>`;
    if (layer.kind !== 'text') {
      const mask = layer.cinemagraph?.mask || layer.mask;
      html += `<h3>Selection &amp; motion</h3>${mask ? `<div class="mask-chip" title="${esc(mask)}">${esc(mask.split('/').pop())}</div>` : '<p>Paint a selection or use AI text selection.</p>'}<div class="button-row"><button data-action="paint-mask">Paint mask</button><button data-action="clear-mask" ${mask ? '' : 'disabled'}>Clear</button></div>`;
      if (layer.kind === 'video')
        html += `${check('Freeze outside the motion mask', 'layer', 'cinemagraph', Boolean(layer.cinemagraph))}${layer.cinemagraph ? field('Frozen source time', 'cinemagraph', 'reference_time', layer.cinemagraph.reference_time || 0, 'min="0"') : ''}`;
      else
        html += `${select('Clean background plate', 'plate', 'source', '', [['', 'Choose an imported clean image'], ...state.assets.filter((asset) => asset.kind === 'image' && asset.path !== app.relative(layer.source)).map((asset) => [asset.path, asset.name])])}<div class="button-row"><button data-action="add-plate">Add plate beneath layer</button></div>`;
    }
    const occluders = current.layers.filter(
      (item) => item.id !== layer.id && item.depth,
    );
    if (occluders.length)
      html += `<h3>Depth occlusion</h3>${select('Place behind', 'layer', 'occlusion.layer_id', layer.occlusion?.layer_id || '', [['', 'No occlusion'], ...occluders.map((item) => [item.id, item.id])])}${layer.occlusion ? `<div class="field-grid">${field('Plane / metres', 'occlusion', 'plane_metres', layer.occlusion.plane_metres || 3, 'min="0.001"')}${field('Edge / metres', 'occlusion', 'feather_metres', layer.occlusion.feather_metres || 0, 'min="0"')}</div>` : ''}`;
    return (
      html +
      '<div class="section-line"></div><button class="wide" data-action="delete-layer">Delete layer</button>'
    );
  }

  function depthPanel(layer) {
    if (!layer || layer.kind !== 'image')
      return '<h3>Depth</h3><p>Select an image layer to extract depth or edit a cached depth asset.</p>';
    const style = layer.depth;
    let html =
      '<h3>Metric depth</h3><button class="wide" data-action="extract-depth">Extract depth ↗</button><p>Extract once, then preview focus, fog, lighting, and small parallax moves.</p>';
    html += `<label>Cached depth metadata<input id="depth-metadata" value="${esc(style?.metadata || '')}" placeholder="exports/photo.depth.json"></label><button class="wide" data-action="attach-depth">Use cached depth</button>`;
    if (!style) return html;
    html += `<h3>Focus &amp; atmosphere</h3><div class="field-grid">${field('Focus / metres', 'depth', 'focus_metres', style.focus_metres || 3, 'min="0.001"')}${field('End focus / metres', 'depth', 'focus_end_metres', style.focus_end_metres || style.focus_metres || 3, 'min="0.001"')}${field('Blur / pixels', 'depth', 'blur_radius', style.blur_radius || 0, 'min="0" max="64"')}${field('Fog density', 'depth', 'fog_density', style.fog_density || 0, 'min="0"')}</div>${text('Fog color', 'depth', 'fog_color', style.fog_color || 'white')}<h3>Lighting</h3><div class="field-grid">${field('Ambient', 'depth', 'ambient', style.ambient ?? 1, 'min="0" max="1"')}${field('Relight strength', 'depth', 'relight_strength', style.relight_strength || 0, 'min="0" max="1"')}</div><div class="field-grid three">${field('Light X', 'light', '0', (style.light_direction || [0, 0, -1])[0])}${field('Light Y', 'light', '1', (style.light_direction || [0, 0, -1])[1])}${field('Light Z', 'light', '2', (style.light_direction || [0, 0, -1])[2])}</div><h3>Small parallax move</h3><div class="field-grid">${field('Horizontal / px', 'depth', 'parallax_x', style.parallax_x || 0, 'min="-32" max="32"')}${field('Vertical / px', 'depth', 'parallax_y', style.parallax_y || 0, 'min="-32" max="32"')}</div><p>Exposed image edges repeat their nearest pixels.</p><button class="wide" data-action="remove-depth">Remove depth effects</button>`;
    return html;
  }

  function renderInspector() {
    if (!state.doc) return;
    document
      .querySelectorAll('[data-inspector]')
      .forEach((button) =>
        button.classList.toggle(
          'active',
          button.dataset.inspector === state.inspector,
        ),
      );
    const layer = selectedLayer();
    let html;
    if (state.inspector === 'transform') html = transformPanel(layer);
    else if (state.inspector === 'depth') html = depthPanel(layer);
    else if (state.inspector === 'ai') html = app.aiPanel?.() || '';
    else {
      const node =
        state.doc.nodes.find((item) => item.id === state.selectedNode) ||
        app.mainNode();
      html = `<h3>Validated job JSON</h3><div class="tiny">${esc(node.id)} · ${esc(node.job.kind)}</div><p>The same request is accepted by Python and the CLI.</p><textarea id="job-json" class="json-area" spellcheck="false" aria-label="Validated job JSON">${esc(JSON.stringify(node.job, null, 2))}</textarea><div class="button-row"><button data-action="apply-json">Validate &amp; apply</button><button data-action="run-node">Run job ↗</button></div>`;
    }
    $('#inspector').innerHTML = html;
  }

  function renderTimeline() {
    if (!state.doc) return;
    const current = composition();
    $('#time-range').max = Math.max(
      0,
      Math.ceil(current.duration * current.fps) - 1,
    );
    $('#time-range').value = Math.floor(state.time * current.fps);
    $('#timecode').textContent =
      `${String(Math.floor(state.time / 60)).padStart(2, '0')}:${(state.time % 60).toFixed(2).padStart(5, '0')}`;
    $('#timeline-ruler').innerHTML = Array.from(
      { length: 6 },
      (_, index) =>
        `<span class="ruler-tick" data-position="${index * 20}">${((current.duration * index) / 5).toFixed(1)}s</span>`,
    ).join('');
    const rows = [
      {
        id: 'camera',
        kind: 'camera',
        start: 0,
        duration: current.duration,
        keyframes: current.camera,
      },
      ...current.layers,
    ];
    $('#timeline-layers').innerHTML = rows
      .map(
        (layer) =>
          `<div class="timeline-row ${layer.id === (state.selected || 'camera') ? 'selected' : ''}" data-timeline-layer="${esc(layer.id)}"><span>${esc(layer.kind === 'text' ? layer.text : layer.id)}</span><div class="timeline-track"><div class="clip-band" data-start="${((layer.start || 0) / current.duration) * 100}" data-length="${(layer.duration / current.duration) * 100}"></div>${layer.keyframes.map((frame) => `<button class="keyframe-dot" title="${frame.at.toFixed(2)} s" data-time="${(layer.start || 0) + frame.at}" data-position="${(((layer.start || 0) + frame.at) / current.duration) * 100}" aria-label="Keyframe at ${frame.at.toFixed(2)} seconds"></button>`).join('')}${(current.beats || []).map((beat) => `<span class="beat-line" data-position="${(beat / current.duration) * 100}"></span>`).join('')}<span class="playhead" data-position="${(state.time / current.duration) * 100}"></span></div></div>`,
      )
      .join('');
    document.querySelectorAll('[data-position]').forEach((element) => {
      element.style.left = `${element.dataset.position}%`;
    });
    document.querySelectorAll('[data-start]').forEach((element) => {
      element.style.left = `${element.dataset.start}%`;
      element.style.width = `${element.dataset.length}%`;
    });
  }

  function render() {
    const current = composition();
    $('#focus-layer').disabled = !selectedLayer();
    $('#view-label').textContent =
      `${current.size.width} × ${current.size.height} · ${current.fps} FPS`;
    const ratio = current.size.width / current.size.height;
    $('#preview-frame').style.aspectRatio = String(ratio);
    const stage = $('#stage');
    const height = Math.max(20, stage.clientHeight - 46);
    const width = Math.max(20, stage.clientWidth - 70);
    $('#preview-frame').style.width = `${Math.min(width, height * ratio)}px`;
    $('#preview-frame').style.height = `${Math.min(height, width / ratio)}px`;
    document
      .querySelectorAll('[data-aspect]')
      .forEach((button) =>
        button.classList.toggle(
          'active',
          button.dataset.aspect ===
            `${current.size.width},${current.size.height}`,
        ),
      );
    renderLibrary();
    renderInspector();
    renderTimeline();
  }

  function updateField(input) {
    const scope = input.dataset.scope;
    const path = input.dataset.field;
    if (!scope || scope === 'plate') return;
    if (!input.checkValidity()) {
      input.reportValidity();
      notify(input.validationMessage, true);
      return;
    }
    const value =
      input.type === 'checkbox'
        ? input.checked
        : input.type === 'number'
          ? Number(input.value)
          : input.value;
    if (input.type === 'number' && !Number.isFinite(value)) return;
    const activeLayer = selectedLayer();
    if (scope === 'layer' && path === 'source_start') {
      const source = assetFor(activeLayer.source);
      if (source?.duration && value >= source.duration) {
        notify('The source offset must be before the end of the video.', true);
        renderInspector();
        return;
      }
    }
    if (scope === 'track') {
      const track =
        composition().audio[
          Number(input.closest('[data-track]').dataset.track)
        ];
      if (path === 'start' && value >= composition().duration) {
        notify('Audio must begin before the end of the composition.', true);
        renderLibrary();
        return;
      }
      const fades =
        path === 'fade_in'
          ? value + track.fade_out
          : path === 'fade_out'
            ? value + track.fade_in
            : path === 'duration'
              ? track.fade_in + track.fade_out
              : 0;
      if (fades > (path === 'duration' ? value : track.duration)) {
        notify('Fade times must fit within the audio track duration.', true);
        renderLibrary();
        return;
      }
    }
    if (scope === 'composition' && path === 'duration') {
      const current = composition();
      const required = Math.max(
        ...current.layers.map((layer) => layer.start + layer.duration),
        ...current.camera.map((frame) => frame.at),
        ...(current.audio || []).map((track) => track.start + track.duration),
      );
      if (value < required - 1e-9) {
        notify(
          `Shorten layers, keyframes, and audio to ${value} seconds before reducing the composition.`,
          true,
        );
        renderInspector();
        return;
      }
    }
    change(() => {
      const layer = selectedLayer();
      let target;
      if (scope === 'composition') target = composition();
      else if (scope === 'layer') target = layer;
      else if (scope === 'pose') target = ensureFrame(layer);
      else if (scope === 'camera') target = ensureFrame(null);
      else if (scope === 'depth') target = layer.depth;
      else if (scope === 'light') {
        layer.depth.light_direction ||= [0, 0, -1];
        target = layer.depth.light_direction;
      } else if (scope === 'occlusion') target = layer.occlusion;
      else if (scope === 'cinemagraph') target = layer.cinemagraph;
      else if (scope === 'track')
        target =
          composition().audio[
            Number(input.closest('[data-track]').dataset.track)
          ];
      if (!target) return;
      if (scope === 'layer' && path === 'cinemagraph') {
        if (value) {
          if (!layer.mask) {
            notify('Paint or select a mask first.', true);
            return;
          }
          layer.cinemagraph = {
            mask: layer.mask,
            reference_time: layer.source_start || 0,
          };
          delete layer.mask;
        } else {
          layer.mask = layer.cinemagraph.mask;
          delete layer.cinemagraph;
        }
      } else if (scope === 'layer' && path === 'occlusion.layer_id') {
        if (value)
          layer.occlusion = {
            layer_id: value,
            plane_metres: 3,
            feather_metres: 0,
          };
        else delete layer.occlusion;
      } else if (scope === 'track' && path === 'ducking') {
        if (value)
          target.ducking = {
            threshold_db: -24,
            ratio: 6,
            attack_ms: 20,
            release_ms: 250,
          };
        else delete target.ducking;
      } else {
        set(target, path, value);
        if (path === 'font' && !value) delete target.font;
      }
      if (scope === 'track' && path === 'role' && value === 'voice')
        delete target.ducking;
      if (scope === 'track') {
        target.duration = Math.max(
          0.01,
          Math.min(target.duration, composition().duration - target.start),
        );
        target.fade_in = Math.min(target.fade_in, target.duration / 2);
        target.fade_out = Math.min(target.fade_out, target.duration / 2);
      }
      if (
        scope === 'layer' &&
        ['start', 'duration', 'source_start', 'speed'].includes(path)
      ) {
        if (layer.kind === 'video') {
          if (['source_start', 'speed'].includes(path))
            delete layer.source_times;
          const asset = assetFor(layer.source);
          if (asset?.duration)
            layer.duration = Math.min(
              layer.duration,
              (asset.duration - (layer.source_start || 0)) /
                (layer.speed || 1),
            );
        }
        composition().duration = Math.max(
          composition().duration,
          layer.start + layer.duration,
        );
        layer.keyframes = layer.keyframes.filter(
          (frame) => frame.at <= layer.duration,
        );
        if (layer.source_times)
          layer.source_times = layer.source_times.filter(
            (frame) => frame.at <= layer.duration,
          );
      }
    });
  }

  function bind() {
    $('#inspector').addEventListener('change', (event) =>
      updateField(event.target),
    );
    $('#library').addEventListener('change', (event) => {
      const input = event.target;
      if (input.dataset.scope) updateField(input);
      else if (input.dataset.nodeDeps)
        change(() => {
          state.doc.nodes.find(
            (node) => node.id === input.dataset.nodeDeps,
          ).depends_on = input.value
            .split(',')
            .map((value) => value.trim())
            .filter(Boolean);
        }, false);
      else if (input.dataset.nodeEnabled)
        change(() => {
          state.doc.nodes.find(
            (node) => node.id === input.dataset.nodeEnabled,
          ).enabled = input.checked;
        }, false);
    });
    $('#library').addEventListener('click', (event) => {
      const assetRow = event.target.closest('[data-asset]');
      if (assetRow) {
        const asset = state.assets.find(
          (item) => item.path === assetRow.dataset.asset,
        );
        if (asset.kind === 'audio') addAudio(asset);
        else if (asset.kind === 'data') app.openProject(asset.path);
        else addMedia(asset);
        return;
      }
      const audio = event.target.closest('[data-add-audio]');
      if (audio) {
        addAudio(
          state.assets.find((item) => item.path === audio.dataset.addAudio),
        );
        return;
      }
      const order = event.target.closest('[data-order]');
      if (order) {
        change(() => {
          const layers = composition().layers;
          const index = layers.findIndex(
            (layer) => layer.id === order.dataset.layerId,
          );
          const other = Math.max(
            0,
            Math.min(layers.length - 1, index + Number(order.dataset.order)),
          );
          [layers[index], layers[other]] = [layers[other], layers[index]];
        });
        return;
      }
      const row = event.target.closest('[data-layer]');
      if (row) {
        state.selected = row.dataset.layer;
        state.selectedNode = app.mainNode().id;
        state.inspector = 'transform';
        app.render();
        return;
      }
      const node = event.target.closest('[data-node]');
      if (node) {
        state.selectedNode = node.dataset.node;
        state.inspector = 'json';
        app.render();
        return;
      }
      const remove = event.target.closest('[data-remove-track]');
      if (remove)
        change(() =>
          composition().audio.splice(Number(remove.dataset.removeTrack), 1),
        );
    });
    $('#timeline-layers').addEventListener('click', (event) => {
      const row = event.target.closest('[data-timeline-layer]');
      if (!row) return;
      state.selected =
        row.dataset.timelineLayer === 'camera'
          ? null
          : row.dataset.timelineLayer;
      state.inspector = 'transform';
      const frame = event.target.closest('[data-time]');
      if (frame) app.setTime(Number(frame.dataset.time));
      else {
        const track = event.target.closest('.timeline-track');
        if (track) {
          const rect = track.getBoundingClientRect();
          app.setTime(
            ((event.clientX - rect.left) / rect.width) *
              composition().duration,
          );
        }
      }
      app.render();
    });
    $('#add-keyframe').addEventListener('click', () =>
      change(() => ensureFrame()),
    );
    $('#add-text').addEventListener('click', addText);
    $('#camera-mode').addEventListener('click', () => {
      state.selected = null;
      state.inspector = 'transform';
      app.render();
    });
    document.addEventListener('click', async (event) => {
      const button = event.target.closest('[data-action]');
      if (!button) return;
      const action = button.dataset.action;
      if (action === 'camera') {
        state.selected = null;
        state.inspector = 'transform';
        app.render();
      }
      if (action === 'add-text') addText();
      if (action === 'keyframe') change(() => ensureFrame());
      if (action === 'frame-layer')
        change(() => {
          const layer = selectedLayer();
          const pose = poseAt(
            layer.keyframes,
            Math.max(0, state.time - layer.start),
          );
          Object.assign(ensureFrame(null), { x: pose.x, y: pose.y, zoom: 1 });
        });
      if (action === 'remove-keyframe')
        change(() => {
          const layer = selectedLayer();
          const local = state.time - (layer?.start || 0);
          const frames = layer?.keyframes || composition().camera;
          const index = frames.findIndex(
            (frame) => Math.abs(frame.at - local) < 1 / composition().fps,
          );
          if (index > 0) frames.splice(index, 1);
        });
      if (action === 'delete-layer') {
        if (composition().layers.length === 1) {
          notify('A composition needs at least one layer.', true);
          return;
        }
        change(() => {
          composition().layers = composition().layers.filter(
            (layer) => layer.id !== state.selected,
          );
          for (const layer of composition().layers)
            if (layer.occlusion?.layer_id === state.selected)
              delete layer.occlusion;
          state.selected = composition().layers[0].id;
        });
      }
      if (action === 'clear-mask')
        change(() => {
          const layer = selectedLayer();
          delete layer.mask;
          delete layer.cinemagraph;
        });
      if (action === 'remove-depth')
        change(() => {
          delete selectedLayer().depth;
          for (const layer of composition().layers)
            if (layer.occlusion?.layer_id === state.selected)
              delete layer.occlusion;
        });
      if (action === 'attach-depth') {
        const metadata = $('#depth-metadata').value.trim();
        const layer = selectedLayer();
        if (metadata)
          try {
            const verified = await app.api('/api/depth', {
              method: 'POST',
              data: { metadata, source: layer.source },
            });
            change(() => {
              layer.depth = {
                metadata: verified.metadata,
                focus_metres: verified.focus_metres,
              };
              linkSource(metadata);
            });
          } catch (error) {
            notify(error.message, true);
          }
      }
      if (action === 'add-plate') {
        const source = $('[data-scope="plate"]').value;
        if (source)
          change(() => {
            const layer = selectedLayer();
            const plate = { ...app.clone(layer), id: id('plate'), source };
            delete plate.mask;
            delete plate.depth;
            delete plate.occlusion;
            composition().layers.splice(
              composition().layers.indexOf(layer),
              0,
              plate,
            );
            linkSource(source);
          });
      }
      if (action === 'apply-json') {
        try {
          const value = JSON.parse($('#job-json').value);
          const selectedNode =
            state.doc.nodes.find((item) => item.id === state.selectedNode) ||
            app.mainNode();
          if (selectedNode === app.mainNode() && value.kind !== 'compose')
            throw new Error(
              'Keep the main job as a composition. Add other operations through the AI controls.',
            );
          const validated = await app.api('/api/validate', {
            method: 'POST',
            data: { job: value },
          });
          change(() => {
            selectedNode.job = validated.job;
          });
          notify('Job JSON validated and applied.');
        } catch (error) {
          notify(error.message, true);
        }
      }
    });
    window.addEventListener('resize', render);
  }

  return {
    render,
    renderLibrary,
    renderInspector,
    renderTimeline,
    bind,
    addMedia,
    addAudio,
    ensureFrame,
    linkSource,
    assetFor,
  };
}
