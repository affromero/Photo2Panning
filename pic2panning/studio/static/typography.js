export function createTypography(app, helpers) {
  const { state, composition, selectedLayer, clone, api, change, notify } =
    app;
  const { field, select, check, assetFor, ensureFrame } = helpers;
  const esc = app.escapeHTML;
  let pending = Promise.resolve(true);
  let acceptedGeneration = state.editGeneration;
  let panelSession = state.session;
  const openCues = new Map();
  const cueCounts = new Map();

  function expandedCues(layer, count) {
    if (panelSession !== state.session) {
      openCues.clear();
      cueCounts.clear();
      panelSession = state.session;
    }
    const existing = app.$('#inspector .typography-panel');
    if (existing)
      openCues.set(
        existing.dataset.typographyLayer,
        new Set(
          [...existing.querySelectorAll('.text-cue[open]')].map((element) =>
            Number(element.dataset.cueIndex),
          ),
        ),
      );
    const selected = openCues.get(layer.id) || new Set(count ? [0] : []);
    if (count > (cueCounts.get(layer.id) ?? count)) selected.add(count - 1);
    cueCounts.set(layer.id, count);
    openCues.set(layer.id, selected);
    const identifiers = new Set(composition().layers.map((item) => item.id));
    for (const identifier of openCues.keys())
      if (!identifiers.has(identifier)) {
        openCues.delete(identifier);
        cueCounts.delete(identifier);
      }
    return selected;
  }

  function panel(layer) {
    if (layer.kind !== 'text') return '';
    const cues = layer.text_cues || [];
    const expanded = expandedCues(layer, cues.length);
    let html = `<div class="typography-panel" data-typography-layer="${esc(layer.id)}"><h3>Sentence morphs</h3><p>The initial text holds until a cue starts. Each transition finishes before the next cue.</p>`;
    html += cues
      .map(
        (cue, index) =>
          `<details class="text-cue" data-cue-index="${index}" ${expanded.has(index) ? 'open' : ''}><summary>Sentence ${index + 2} · ${cue.at.toFixed(2)} s</summary><label>Target sentence<textarea rows="3" data-typography="cue" data-type-index="${index}" data-type-cue-at="${cue.at}" data-type-field="text">${esc(cue.text)}</textarea></label><div class="field-grid">${control('Start / sec', 'cue', 'at', cue.at, 'min="0.001"', index)}${control('Transition / sec', 'cue', 'transition', cue.transition ?? 0.8, 'min="0.01" max="10"', index)}${control('Stagger / fraction', 'cue', 'stagger', cue.stagger ?? 0.25, 'min="0" max="0.8"', index)}${control('Blur / px', 'cue', 'blur', cue.blur ?? 1.5, 'min="0" max="32"', index)}${control('Curved travel / em', 'cue', 'travel', cue.travel ?? 0.6, 'min="0" max="4"', index)}</div><button data-typography-action="remove-cue" data-type-index="${index}" data-type-cue-at="${cue.at}">Remove sentence ${index + 2}</button></details>`,
      )
      .join('');
    html += `<button class="wide" data-typography-action="add-cue" ${cues.length >= 64 ? 'disabled' : ''}>+ Morph to another sentence</button><h3>Letter entrance</h3>${check('Drop letters one by one', 'typography', 'letter_drop', Boolean(layer.letter_drop)).replace('data-scope="typography"', 'data-typography="drop-toggle"')}`;
    if (layer.letter_drop) {
      const drop = layer.letter_drop;
      html += `<div class="field-grid">${control('Drop duration / sec', 'drop', 'duration', drop.duration, 'min="0.01"')}${control('Stagger / fraction', 'drop', 'stagger', drop.stagger, 'min="0" max="0.99"')}${control('Fall height / em', 'drop', 'height', drop.height, 'min="0.01" max="20"')}${control('Bounce / fraction', 'drop', 'bounce', drop.bounce, 'min="0" max="0.5"')}</div><div class="field-grid three">${control('Initial X / °', 'drop', 'rotation.0', drop.rotation?.[0] ?? 65)}${control('Initial Y / °', 'drop', 'rotation.1', drop.rotation?.[1] ?? -25)}${control('Initial Z / °', 'drop', 'rotation.2', drop.rotation?.[2] ?? -20)}</div><p>Letters land in the initial sentence before its first morph begins.</p>`;
    }
    const current = composition();
    const index = current.layers.indexOf(layer);
    const hosts = current.layers
      .slice(0, index)
      .filter((item) => item.kind === 'image' && item.depth);
    html += `<h3>Depth surface</h3>${select('Attach to image surface', 'typography', 'surface-host', layer.surface?.layer_id || '', [['', 'Free text plane'], ...hosts.map((host) => [host.id, assetFor(host.source)?.name || host.id])]).replace('data-scope="typography"', 'data-typography="surface-host"')}`;
    if (!hosts.length)
      html +=
        '<p>Add a depth-enabled image beneath this text to attach it to a table or another flat surface.</p>';
    if (layer.surface) {
      const surface = layer.surface;
      const host = current.layers.find((item) => item.id === surface.layer_id);
      const asset = host && assetFor(host.source);
      const region = surface.region;
      html += `<p>Patch and text centers use the original image pixels. Choose a flat part of the table; foreground objects hide the text automatically.</p><div class="field-grid">${control('Patch left / px', 'surface', 'region.0', region[0], `min="0" ${asset ? `max="${asset.width}"` : ''}`)}${control('Patch top / px', 'surface', 'region.1', region[1], `min="0" ${asset ? `max="${asset.height}"` : ''}`)}${control('Patch right / px', 'surface', 'region.2', region[2], `min="1" ${asset ? `max="${asset.width}"` : ''}`)}${control('Patch bottom / px', 'surface', 'region.3', region[3], `min="1" ${asset ? `max="${asset.height}"` : ''}`)}</div><div class="field-grid three">${control('Surface tilt / °', 'surface', 'tilt', surface.tilt ?? 0, 'min="-80" max="80"')}${control('Surface yaw / °', 'surface', 'yaw', surface.yaw ?? 0, 'min="-80" max="80"')}${control('Surface roll / °', 'surface', 'roll', surface.roll ?? 0)}</div><div class="field-grid">${control('Right offset / m', 'surface', 'offset_metres.0', surface.offset_metres?.[0] ?? 0)}${control('Down offset / m', 'surface', 'offset_metres.1', surface.offset_metres?.[1] ?? 0)}${control('Lift / m', 'surface', 'lift_metres', surface.lift_metres ?? 0, 'min="0"')}${control('Depth tolerance / m', 'surface', 'depth_tolerance_metres', surface.depth_tolerance_metres ?? 0.02, 'min="0"')}${control('Depth edge / m', 'surface', 'feather_metres', surface.feather_metres ?? 0.01, 'min="0"')}${control('Fit error / fraction', 'surface', 'max_relative_error', surface.max_relative_error ?? 0.02, 'min="0.0001" max="0.25"')}</div>`;
      html += `<h3>Solid letters</h3>${check('Render beveled 3D letters', 'typography', 'solid', Boolean(layer.solid)).replace('data-scope="typography"', 'data-typography="solid-toggle"')}<p>Physical thickness, lighting and cast shadows. Requires Blender with Metal. Bitmap strokes, shadow offsets, masks and cue blur must be zero.</p>`;
      if (layer.solid) html += solidPanel(layer.solid);
    }
    return html + '</div>';
  }

  function solidPanel(solid) {
    let html = `<div class="field-grid">${control('Thickness / em', 'solid', 'thickness', solid.thickness ?? 0.2, 'min="0.001" max="2"')}${control('Bevel / em', 'solid', 'bevel', solid.bevel ?? 0.02, 'min="0" max="0.25"')}${control('Roughness', 'solid', 'roughness', solid.roughness ?? 0.36, 'min="0" max="1"')}${control('Metallic', 'solid', 'metallic', solid.metallic ?? 0, 'min="0" max="1"')}${control('Render samples', 'solid', 'samples', solid.samples ?? 64, 'min="1" max="4096" step="1"')}${control('Ambient light', 'solid', 'ambient', solid.ambient ?? 0.25, 'min="0" max="10"')}</div>`;
    for (const name of ['key', 'fill']) {
      const light = solid[name];
      if (!light) continue;
      html += `<details><summary>${name === 'key' ? 'Key' : 'Fill'} light</summary><div class="field-grid">${control('Power / W', 'solid', `${name}.energy`, light.energy, 'min="0.001"')}${control('Emitter diameter / m', 'solid', `${name}.size`, light.size, 'min="0.001"')}</div><div class="field-grid three">${control('Right / m', 'solid', `${name}.position.0`, light.position[0])}${control('Down / m', 'solid', `${name}.position.1`, light.position[1])}${control('Height / m', 'solid', `${name}.position.2`, light.position[2], 'min="0.001"')}</div><label>Light color<input data-typography="solid" data-type-field="${name}.color" value="${esc(light.color)}"></label></details>`;
    }
    return html;
  }

  function control(label, group, path, value, constraints = '', index = null) {
    return `<label>${esc(label)}<input type="number" step="any" data-typography="${group}" data-type-field="${path}" ${index === null ? '' : `data-type-index="${index}" data-type-cue-at="${selectedLayer()?.text_cues?.[index]?.at}"`} value="${Number.isFinite(value) ? value : 0}" ${constraints}></label>`;
  }

  function posePanel(layer, pose) {
    return `<h3>3D orientation</h3><div class="field-grid three">${field('Rotation X / °', 'pose', 'rotation_x', pose.rotation_x ?? 0)}${field('Rotation Y / °', 'pose', 'rotation_y', pose.rotation_y ?? 0)}${field('Perspective / px', 'layer', 'perspective', layer.perspective ?? 1200, 'min="1"')}</div>`;
  }

  function set(object, path, value) {
    const parts = path.split('.');
    const key = parts.pop();
    let current = object;
    for (const part of parts) current = current[part] ||= {};
    current[key] = value;
  }

  function dirtyInspectorInput(input) {
    const active = globalThis.document?.activeElement;
    return active !== input &&
      active?.matches('#inspector input, #inspector textarea') &&
      active.value !== active.defaultValue
      ? active
      : null;
  }

  function mutate(callback, message = 'Typography updated.', input = null) {
    const session = state.session;
    const intentGeneration = state.editGeneration;
    pending = pending.then(async () => {
      if (session !== state.session) return false;
      if (
        intentGeneration !== state.editGeneration &&
        acceptedGeneration !== state.editGeneration
      )
        return false;
      const before = JSON.stringify(composition());
      const generation = state.editGeneration;
      const candidate = clone(composition());
      const overwrite = candidate.overwrite ?? false;
      try {
        callback(candidate);
        const validated = await api('/api/validate', {
          method: 'POST',
          data: { job: { ...candidate, overwrite: true } },
        });
        if (session !== state.session) return false;
        if (
          generation !== state.editGeneration ||
          before !== JSON.stringify(composition())
        ) {
          notify(
            'The composition changed during validation. Repeat your edit.',
            true,
          );
          app.render({
            preserveInspector: Boolean(dirtyInspectorInput(input)),
          });
          return false;
        }
        const active = dirtyInspectorInput(input);
        const preserveInspector = Boolean(active);
        let retimed = null;
        if (preserveInspector && active.dataset.typeCueAt !== undefined) {
          const oldAt = Number(active.dataset.typeCueAt);
          const oldLayer = selectedLayer();
          const nextLayer = validated.job.layers.find(
            (layer) => layer.id === oldLayer?.id,
          );
          if (!nextLayer?.text_cues?.some((cue) => cue.at === oldAt)) {
            if (nextLayer?.text_cues?.length !== oldLayer?.text_cues?.length) {
              notify('Finish this sentence edit before removing it.', true);
              return false;
            }
            const index = oldLayer.text_cues.findIndex(
              (cue) => cue.at === oldAt,
            );
            if (index >= 0) retimed = [oldAt, nextLayer.text_cues[index].at];
          }
        }
        validated.job.overwrite = overwrite;
        const changed = change(
          () => {
            app.mainNode().job = validated.job;
          },
          true,
          preserveInspector,
        );
        if (!changed) app.render({ preserveInspector });
        if (retimed)
          for (const element of app
            .$('#inspector')
            .querySelectorAll('[data-type-cue-at]'))
            if (Number(element.dataset.typeCueAt) === retimed[0])
              element.dataset.typeCueAt = retimed[1];
        acceptedGeneration = state.editGeneration;
        notify(message);
        return true;
      } catch (error) {
        if (session !== state.session) return false;
        notify(error.message, true);
        app.render({ preserveInspector: Boolean(dirtyInspectorInput(input)) });
        return false;
      }
    });
    return pending;
  }

  function mutateLayer(callback, message, input = null) {
    const identifier = selectedLayer()?.id;
    return mutate(
      (candidate) => {
        const layer = candidate.layers.find((item) => item.id === identifier);
        if (!layer)
          throw new Error('The selected layer is no longer available.');
        callback(layer, candidate);
      },
      message,
      input,
    );
  }

  function update(input) {
    if (!input.dataset.typography) return false;
    if (!input.checkValidity()) {
      input.reportValidity();
      notify(input.validationMessage, true);
      return true;
    }
    const group = input.dataset.typography;
    const path = input.dataset.typeField;
    const index = Number(input.dataset.typeIndex);
    const cueAt =
      group === 'cue'
        ? Number(
            input.dataset.typeCueAt ?? selectedLayer()?.text_cues?.[index]?.at,
          )
        : null;
    const value =
      input.type === 'checkbox'
        ? input.checked
        : input.type === 'number'
          ? Number(input.value)
          : input.value;
    if (input.type === 'number' && !Number.isFinite(value)) return true;
    mutateLayer(
      (layer, candidate) => {
        if (group === 'drop-toggle') {
          if (value) {
            const earliest = layer.text_cues?.[0]?.at ?? layer.duration;
            layer.letter_drop = {
              duration: Math.min(2, earliest, layer.duration),
              stagger: 0.6,
              height: 6,
              rotation: [65, -25, -20],
              bounce: 0.15,
            };
          } else delete layer.letter_drop;
        } else if (group === 'solid-toggle') {
          if (value) layer.solid = {};
          else delete layer.solid;
        } else if (group === 'surface-host') {
          if (!value) {
            if (layer.solid)
              throw new Error('Turn off solid letters before detaching the surface.');
            delete layer.surface;
            return;
          }
          if (layer.occlusion)
            throw new Error(
              'Remove the existing depth occlusion before attaching a surface.',
            );
          const host = candidate.layers.find((item) => item.id === value);
          const asset = host && assetFor(host.source);
          if (!asset?.width || !asset?.height)
            throw new Error(
              'Import the original host image before attaching its surface.',
            );
          const region = [
            Math.round(asset.width * 0.15),
            Math.round(asset.height * 0.5),
            Math.round(asset.width * 0.85),
            Math.round(asset.height * 0.85),
          ];
          layer.surface = { layer_id: value, region };
          layer.space = 'world';
          for (const pose of layer.keyframes) {
            pose.x = (region[0] + region[2]) / 2;
            pose.y = (region[1] + region[3]) / 2;
          }
        } else if (group === 'cue') {
          const cue = layer.text_cues?.find((item) => item.at === cueAt);
          if (!cue)
            throw new Error(
              'The sentence changed before this edit could be applied. Repeat your edit.',
            );
          set(cue, path, value);
        } else if (group === 'drop') set(layer.letter_drop, path, value);
        else if (group === 'surface') set(layer.surface, path, value);
        else if (group === 'solid') set(layer.solid, path, value);
      },
      undefined,
      input,
    );
    return true;
  }

  function standardField(scope, path, value, input = null) {
    const layer = selectedLayer();
    if (!layer) return false;
    const linked =
      layer.kind === 'text' ||
      composition().layers.some((item) => item.surface?.layer_id === layer.id);
    if (!linked || !['layer', 'pose'].includes(scope)) return false;
    const seconds = state.time;
    mutateLayer(
      (candidateLayer, candidate) => {
        const target =
          scope === 'pose'
            ? ensureFrame(candidateLayer, candidate, seconds)
            : candidateLayer;
        if (scope === 'layer' && path === 'occlusion.layer_id') {
          if (value)
            target.occlusion = {
              layer_id: value,
              plane_metres: 3,
              feather_metres: 0,
            };
          else delete target.occlusion;
        } else set(target, path, value);
        if (path === 'font' && !value) delete target.font;
      },
      undefined,
      input,
    );
    return true;
  }

  function action(button) {
    const action = button.dataset.typographyAction;
    if (!action) return false;
    const index = Number(button.dataset.typeIndex);
    const cueAt = Number(
      button.dataset.typeCueAt ?? selectedLayer()?.text_cues?.[index]?.at,
    );
    const seconds = state.time;
    mutateLayer((layer) => {
      layer.text_cues ||= [];
      if (action === 'remove-cue') {
        const current = layer.text_cues.findIndex((cue) => cue.at === cueAt);
        if (current < 0)
          throw new Error(
            'The sentence changed before it could be removed. Repeat your edit.',
          );
        layer.text_cues.splice(current, 1);
      } else if (action === 'add-cue') {
        const last = layer.text_cues.at(-1);
        const end = last
          ? last.at + last.transition
          : layer.letter_drop?.duration || 0;
        const at = Math.max(
          end + 0.35,
          Math.min(layer.duration - 0.8, seconds - layer.start),
        );
        if (at + 0.8 > layer.duration + 1e-9)
          throw new Error(
            'Increase the layer duration before adding another sentence.',
          );
        layer.text_cues.push({
          at,
          text: 'Next sentence',
          transition: 0.8,
          stagger: 0.25,
          blur: layer.solid ? 0 : 1.5,
          travel: 0.6,
        });
      }
    });
    return true;
  }

  function guardHost(identifier, action) {
    const attached = composition().layers.filter(
      (layer) => layer.surface?.layer_id === identifier,
    );
    if (!attached.length) return true;
    notify(
      `Detach ${attached.map((layer) => layer.id).join(', ')} before ${action}.`,
      true,
    );
    return false;
  }

  function reorder(identifier, direction) {
    return mutate((candidate) => {
      const index = candidate.layers.findIndex(
        (layer) => layer.id === identifier,
      );
      const next = Math.max(
        0,
        Math.min(candidate.layers.length - 1, index + direction),
      );
      [candidate.layers[index], candidate.layers[next]] = [
        candidate.layers[next],
        candidate.layers[index],
      ];
    }, 'Layer order updated.');
  }

  function cameraLayer(layer) {
    if (!layer?.surface) return layer;
    return (
      composition().layers.find(
        (item) => item.id === layer.surface.layer_id,
      ) || layer
    );
  }

  async function flushEdits() {
    let task;
    let result;
    do {
      task = pending;
      result = await task;
    } while (task !== pending);
    pending = Promise.resolve(true);
    return result !== false;
  }

  return {
    panel,
    posePanel,
    update,
    action,
    standardField,
    guardHost,
    reorder,
    cameraLayer,
    flushEdits,
  };
}
