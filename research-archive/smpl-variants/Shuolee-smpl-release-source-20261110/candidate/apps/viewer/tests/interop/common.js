/** Harness plumbing only. Gaussian decoding is performed by the named upstream library. */
export function localModelUrl(value, base = location.href) {
  const url = new URL(value, base);
  if (!['http:', 'https:'].includes(url.protocol) ||
      !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) {
    throw new Error('Only local HTTP(S) model URLs are accepted. Use the file picker for a local PLY.');
  }
  if (url.username || url.password) throw new Error('Credentials in a model URL are not accepted.');
  return url.href;
}

export function createHarness(name, version, load) {
  const status = document.querySelector('#status');
  const details = document.querySelector('#details');
  const form = document.querySelector('form');
  const fileInput = document.querySelector('#file');
  const urlInput = document.querySelector('#url');
  const fixture = new URL('./anisotropic-sh3.ply', import.meta.url).href;
  const freshResult = (state = 'idle', source = null) => ({
    library: name, version, state, loadedCount: null, loadedShDegree: null,
    rendererShLimit: null, renderFrames: 0, source, error: null, qualityAccepted: false,
    center: null, radius: null, cameraPosition: null, cameraLookAt: null, gaussianProperties: null,
  });
  const result = freshResult();
  window.__interop = result;
  function update(values) {
    Object.assign(result, values);
    status.dataset.state = result.state;
    status.dataset.loadedCount = result.loadedCount === null ? '' : String(result.loadedCount);
    status.dataset.renderFrames = String(result.renderFrames);
    status.textContent = `${result.state} | Gaussian count: ${result.loadedCount ?? '—'} | Render frames: ${result.renderFrames}`;
    details.textContent = JSON.stringify(result, null, 2);
  }
  function reset(values) {
    for (const key of Object.keys(result)) delete result[key];
    update({ ...freshResult(), ...values });
  }
  update({});
  let cleanup;
  let runId = 0;
  async function run(source, displayName) {
    if (result.state === 'loading') return;
    const currentRun = ++runId;
    const updateCurrentRun = (values) => { if (currentRun === runId) update(values); };
    form.querySelectorAll('button').forEach((button) => { button.disabled = true; });
    reset({ state: 'loading', source: displayName });
    try {
      if (cleanup) { const previous = cleanup; cleanup = undefined; await previous(); }
      cleanup = await load(source, updateCurrentRun);
    } catch (error) {
      console.error(error);
      reset({ state: 'error', source: displayName, error: String(error?.stack || error) });
    } finally {
      form.querySelectorAll('button').forEach((button) => { button.disabled = false; });
    }
  }
  function invalidSource(error) {
    runId++;
    reset({ state: 'error', error: String(error) });
  }
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    try {
      const file = fileInput.files[0];
      if (file) {
        if (!file.name.toLowerCase().endsWith('.ply')) throw new Error('Select a standard 3DGS .ply file.');
        const blobUrl = URL.createObjectURL(file);
        run(blobUrl, file.name).finally(() => URL.revokeObjectURL(blobUrl));
      } else {
        const source = localModelUrl(urlInput.value);
        run(source, source);
      }
    } catch (error) { invalidSource(error); }
  });
  document.querySelector('#fixture').addEventListener('click', () => run(fixture, 'synthetic anisotropic SH3 fixture'));
  const query = new URLSearchParams(location.search);
  if (query.get('fixture') === '1') run(fixture, 'synthetic anisotropic SH3 fixture');
  else if (query.has('model')) {
    try { const source = localModelUrl(query.get('model')); run(source, source); }
    catch (error) { invalidSource(error); }
  }
  return { run, result };
}
