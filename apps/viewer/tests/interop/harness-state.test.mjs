import test from 'node:test';
import assert from 'node:assert/strict';
import { createHarness, localModelUrl } from './common.js';

function mockDom(context) {
  const handlers = new Map();
  const buttons = [{ disabled: false }, { disabled: false }];
  const elements = {
    '#status': { dataset: {}, textContent: '' }, '#details': { textContent: '' },
    form: { querySelectorAll: () => buttons, addEventListener: (event, fn) => handlers.set(event, fn) },
    '#file': { files: [] }, '#url': { value: '' },
    '#fixture': { addEventListener: (event, fn) => handlers.set(`fixture-${event}`, fn) },
  };
  context.mock.method(console, 'error', () => {});
  const old = { document: globalThis.document, window: globalThis.window, location: globalThis.location };
  globalThis.document = { querySelector: (selector) => elements[selector] };
  globalThis.window = {};
  globalThis.location = { href: 'http://127.0.0.1:4873/tests/interop/mkkellogg.html', search: '' };
  context.after(() => Object.assign(globalThis, old));
  return { elements, handlers, buttons };
}

test('failed model switch clears prior count, SH, bounds and all extension metadata', async (context) => {
  const { elements, buttons } = mockDom(context);
  let cleaned = 0;
  const harness = createHarness('test upstream', '1', async (source, update) => {
    if (source === 'bad') throw new Error('decode failed');
    update({ state: 'rendered', loadedCount: 9, loadedShDegree: 3, renderFrames: 3,
      center: [1, 2, 3], radius: 4, customPreviousField: 'old' });
    return () => { cleaned++; };
  });
  await harness.run('fixture', 'fixture');
  await harness.run('bad', 'broken PLY');
  assert.equal(cleaned, 1);
  assert.equal(harness.result.state, 'error');
  for (const key of ['loadedCount', 'loadedShDegree', 'center', 'radius']) assert.equal(harness.result[key], null);
  assert.equal(harness.result.renderFrames, 0);
  assert.equal('customPreviousField' in harness.result, false);
  assert.equal(elements['#status'].dataset.loadedCount, '');
  assert.ok(buttons.every((button) => !button.disabled));
});

test('late callbacks from a disposed viewer cannot overwrite the next model status', async (context) => {
  mockDom(context);
  const callbacks = [];
  let cleaned = 0;
  const harness = createHarness('test upstream', '1', async (source, update) => {
    callbacks.push(update);
    update({ state: 'rendered', loadedCount: source === 'fixture' ? 9 : 204649, loadedShDegree: 2, renderFrames: 3 });
    return () => { cleaned++; };
  });
  await harness.run('fixture', 'fixture');
  await harness.run('person', 'person');
  callbacks[0]({ state: 'rendered', loadedCount: 9, loadedShDegree: 1 });
  assert.equal(harness.result.loadedCount, 204649);
  assert.equal(harness.result.loadedShDegree, 2);
  await harness.run('fixture', 'fixture again');
  callbacks[1]({ state: 'error', error: 'late previous viewer failure' });
  assert.equal(harness.result.loadedCount, 9);
  assert.equal(harness.result.state, 'rendered');
  assert.equal(cleaned, 2);
});

test('model URL input accepts localhost and rejects external or credentialed sources', () => {
  const base = 'http://127.0.0.1:4873/tests/interop/index.html';
  assert.equal(localModelUrl('/model.ply', base), 'http://127.0.0.1:4873/model.ply');
  assert.equal(localModelUrl('http://localhost:4874/person.ply', base), 'http://localhost:4874/person.ply');
  assert.throws(() => localModelUrl('https://example.com/model.ply', base), /Only local/);
  assert.throws(() => localModelUrl('http://user:secret@localhost/model.ply', base), /Credentials/);
});
