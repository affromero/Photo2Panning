import assert from 'node:assert/strict';
import { setImmediate as nextTurn } from 'node:timers/promises';
import test from 'node:test';

import { createEditor } from '../../pic2panning/studio/static/editor.js';
import { createTypography } from '../../pic2panning/studio/static/typography.js';

function deferred() {
  let resolve;
  const promise = new Promise((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}

function studio() {
  const state = {
    session: 1,
    editGeneration: 0,
    selected: 'caption',
    time: 0,
    assets: [],
    doc: {
      nodes: [{
        id: 'main',
        job: {
          kind: 'compose',
          duration: 8,
          output: 'exports/movie.mp4',
          overwrite: false,
          layers: [{
            id: 'caption',
            kind: 'text',
            text: 'ONE',
            font_size: 32,
            start: 0,
            duration: 8,
            keyframes: [{ at: 0, x: 10, y: 10 }],
            text_cues: [
              { at: 1, text: 'TWO', transition: 0.5 },
              { at: 3, text: 'THREE', transition: 0.5 },
              { at: 5, text: 'FOUR', transition: 0.5 },
            ],
          }],
        },
      }],
    },
  };
  const started = deferred();
  const response = deferred();
  const errors = [];
  const history = [];
  let holdResponse = true;
  const app = {
    state,
    $: () => ({ checked: false }),
    escapeHTML: String,
    clone: structuredClone,
    composition: () => state.doc.nodes[0].job,
    mainNode: () => state.doc.nodes[0],
    selectedLayer: () => app.composition().layers[0],
    relative: (value) => value,
    render: () => {},
    notify: (message, error) => {
      if (error) errors.push(message);
    },
    change: (callback) => {
      const before = structuredClone(state.doc);
      callback();
      if (JSON.stringify(before) === JSON.stringify(state.doc)) return;
      history.push(before);
      state.editGeneration++;
    },
    api: async (path, options) => {
      assert.equal(path, '/api/validate');
      if (holdResponse) {
        holdResponse = false;
        started.resolve();
        await response.promise;
      }
      return { job: structuredClone(options.data.job) };
    },
  };
  const editor = createEditor(app);
  const typography = createTypography(app, {
    ensureFrame: editor.ensureFrame,
  });
  return {
    app,
    state,
    typography,
    errors,
    started: started.promise,
    release: response.resolve,
    undo: () => {
      state.doc = history.pop();
      state.editGeneration++;
    },
  };
}

function cueInput(index, text) {
  return {
    dataset: {
      typography: 'cue',
      typeIndex: String(index),
      typeField: 'text',
    },
    type: 'textarea',
    value: text,
    checkValidity: () => true,
  };
}

test('queued pose edits keep their original timeline time after scrubbing', async () => {
  const scene = studio();
  scene.typography.standardField('layer', 'font_size', 40);
  await scene.started;
  scene.typography.standardField('pose', 'x', 99);
  scene.state.time = 4;
  scene.release();
  await nextTurn();

  assert.deepEqual(scene.app.selectedLayer().keyframes, [
    { at: 0, x: 99, y: 10 },
  ]);
  assert.equal(scene.app.composition().overwrite, false);
  assert.deepEqual(scene.errors, []);
});

test('removing an earlier cue preserves the identity of a queued sentence edit', async () => {
  const scene = studio();
  scene.typography.action({
    dataset: { typographyAction: 'remove-cue', typeIndex: '0' },
  });
  await scene.started;
  scene.typography.update(cueInput(1, 'EDITED THREE'));
  scene.release();
  await nextTurn();

  assert.deepEqual(
    scene.app.selectedLayer().text_cues.map(({ text }) => text),
    ['EDITED THREE', 'FOUR'],
  );
  assert.deepEqual(scene.errors, []);
});

test('undoing another edit does not let an older validation overwrite history', async () => {
  const scene = studio();
  const original = structuredClone(scene.app.composition());
  scene.typography.standardField('layer', 'font_size', 40);
  await scene.started;
  scene.app.change(() => {
    scene.app.selectedLayer().text = 'ANOTHER EDIT';
  });
  scene.undo();
  scene.release();
  await nextTurn();

  assert.deepEqual(scene.app.composition(), original);
  assert.ok(scene.errors.some((message) => /changed/i.test(message)));
});

test('flushing edits waits for validation and preserves the chosen overwrite option', async () => {
  const scene = studio();
  scene.app.composition().overwrite = true;
  scene.typography.standardField('layer', 'font_size', 40);
  await scene.started;
  let completed = false;
  const flushed = scene.typography.flushEdits().then((accepted) => {
    completed = true;
    return accepted;
  });
  await nextTurn();
  assert.equal(completed, false);
  scene.release();

  assert.equal(await flushed, true);
  assert.equal(scene.app.selectedLayer().font_size, 40);
  assert.equal(scene.app.composition().overwrite, true);
  assert.equal(scene.app.composition().output, 'exports/movie.mp4');
  assert.deepEqual(scene.errors, []);
});

test('new solid sentences use geometry without bitmap blur', async () => {
  const scene = studio();
  const layer = scene.app.selectedLayer();
  layer.solid = { thickness: 0.2 };
  layer.duration = 10;
  scene.typography.action({
    dataset: { typographyAction: 'add-cue' },
  });
  await scene.started;
  scene.release();
  assert.equal(await scene.typography.flushEdits(), true);

  const cue = scene.app.selectedLayer().text_cues.at(-1);
  assert.equal(cue.blur, 0);
  assert.ok(cue.at >= 5.5);
  assert.equal(scene.app.selectedLayer().solid.thickness, 0.2);
});

test('detaching a surface requires turning off its solid letters first', async () => {
  const scene = studio();
  scene.app.selectedLayer().solid = { thickness: 0.2 };
  scene.app.selectedLayer().surface = {
    layer_id: 'table',
    region: [10, 10, 90, 90],
  };
  scene.typography.update({
    dataset: { typography: 'surface-host' },
    type: 'select-one',
    value: '',
    checkValidity: () => true,
  });
  assert.equal(await scene.typography.flushEdits(), false);
  assert.equal(scene.app.selectedLayer().surface.layer_id, 'table');
  assert.equal(scene.app.selectedLayer().solid.thickness, 0.2);
  assert.ok(scene.errors.some((message) => /solid.*detach/i.test(message)));
});
