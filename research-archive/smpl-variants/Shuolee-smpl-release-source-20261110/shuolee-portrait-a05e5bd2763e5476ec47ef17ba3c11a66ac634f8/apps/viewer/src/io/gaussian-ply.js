/**
 * Graphdeco 3D Gaussian PLY inspection and lossless format export.
 *
 * Values remain training parameters: opacity is a logit, scales are log
 * standard deviations, rotation is Hamilton wxyz, and SH is channel-major.
 * This module never reads a renderer's packed/quantized representation.
 */

const MAX_HEADER_BYTES = 1024 * 1024;
const MAX_VERTICES = 10_000_000;
const MAX_PROPERTIES = 512;
const decoder = new TextDecoder('utf-8', { fatal: true });
const encoder = new TextEncoder();
const numberToken = /^[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)$/i;
const float32Types = new Set(['float', 'float32']);
const types = {
  char: [1, 'getInt8'], int8: [1, 'getInt8'],
  uchar: [1, 'getUint8'], uint8: [1, 'getUint8'],
  short: [2, 'getInt16'], int16: [2, 'getInt16'],
  ushort: [2, 'getUint16'], uint16: [2, 'getUint16'],
  int: [4, 'getInt32'], int32: [4, 'getInt32'],
  uint: [4, 'getUint32'], uint32: [4, 'getUint32'],
  float: [4, 'getFloat32'], float32: [4, 'getFloat32'],
  double: [8, 'getFloat64'], float64: [8, 'getFloat64'],
};
const mandatory = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
  'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
const normals = ['nx', 'ny', 'nz'];
const gaussianSignal = /^(?:opacity$|f_dc(?:_|$)|f_rest(?:_|$)|scale_\d|rot_\d)/;
const supportedRestCounts = new Map([[0, 0], [9, 1], [24, 2], [45, 3]]);

const semantics = Object.freeze({
  opacity: 'logit',
  scale: 'log-standard-deviation',
  rotation: 'Hamilton-wxyz-raw-nonzero',
  sh: 'Graphdeco-real-SH-channel-major',
  worldTransform: 'none',
});

function fail(message, code = 'INVALID_GAUSSIAN_PLY') {
  const error = new Error(message);
  error.code = code;
  throw error;
}

function bytesOf(buffer) {
  if (buffer instanceof ArrayBuffer) return new Uint8Array(buffer);
  if (ArrayBuffer.isView(buffer)) return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  fail('PLY input must be an ArrayBuffer or typed array.');
}

function unsignedCount(token, label) {
  if (!/^\d+$/.test(token ?? '')) fail(`PLY: invalid ${label}.`);
  const count = Number(token);
  if (!Number.isSafeInteger(count) || count > MAX_VERTICES) fail(`PLY: ${label} exceeds ${MAX_VERTICES}.`);
  return count;
}

function headerOf(bytes) {
  if (!bytes.byteLength) fail('PLY file is empty.');
  let start = 0, end = -1;
  const lines = [];
  for (let i = 0; i < Math.min(bytes.length, MAX_HEADER_BYTES); i++) {
    if (bytes[i] !== 10) continue;
    let line;
    try { line = decoder.decode(bytes.subarray(start, bytes[i - 1] === 13 ? i - 1 : i)); }
    catch { fail('PLY header is not valid UTF-8.'); }
    start = i + 1;
    if (line === 'end_header') { end = start; break; }
    lines.push(line);
  }
  if (end < 0) fail('PLY: missing end_header newline or header exceeds 1 MiB.');
  if (lines.shift() !== 'ply') fail('PLY: missing ply magic.');
  let format = null, current = null;
  const elements = [];
  for (const line of lines) {
    const words = line.trim().split(/\s+/);
    if (!line.trim() || words[0] === 'comment' || words[0] === 'obj_info') continue;
    if (words[0] === 'format') {
      if (format || elements.length || words.length !== 3 || words[2] !== '1.0'
        || !['ascii', 'binary_little_endian', 'binary_big_endian'].includes(words[1])) fail('PLY: invalid format declaration.');
      format = words[1];
    } else if (words[0] === 'element') {
      if (!format || words.length !== 3 || elements.length >= 64 || elements.some(e => e.name === words[1])) fail('PLY: invalid or duplicate element.');
      current = { name: words[1], count: unsignedCount(words[2], 'element count'), properties: [] };
      elements.push(current);
    } else if (words[0] === 'property') {
      if (!current || current.properties.length >= MAX_PROPERTIES) fail('PLY: property outside element or too many properties.');
      const list = words[1] === 'list';
      if (words.length !== (list ? 5 : 3)) fail('PLY: malformed property.');
      const type = list ? words[3] : words[1], name = list ? words[4] : words[2];
      if (!types[type] || (list && (!types[words[2]] || /^(?:float|double)/.test(words[2])))) fail('PLY: unsupported property type.');
      if (!/^[A-Za-z_][\w.-]*$/.test(name) || current.properties.some(p => p.name === name)) fail('PLY: invalid or duplicate property name.');
      current.properties.push({ name, type, list });
    } else fail(`PLY: unknown header directive ${words[0]}.`);
  }
  const vertex = elements.find(e => e.name === 'vertex');
  if (!format || !vertex || !vertex.count || !vertex.properties.length) fail('PLY: requires a format and nonempty vertex element.');
  return { format, elements, vertex, headerBytes: end };
}

function standardProperties(restCount) {
  return ['x', 'y', 'z', ...normals, 'f_dc_0', 'f_dc_1', 'f_dc_2',
    ...Array.from({ length: restCount }, (_, i) => `f_rest_${i}`),
    'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
}

function schemaOf(header) {
  const props = header.vertex.properties;
  const names = props.map(p => p.name);
  if (!names.some(name => gaussianSignal.test(name))) return { kind: 'points', shDegree: null };
  if (header.elements.length !== 1 || props.some(p => p.list)) fail('Gaussian PLY requires one scalar vertex element.');
  const missing = mandatory.filter(name => !names.includes(name));
  if (missing.length) fail(`Incomplete Gaussian PLY schema; missing ${missing.join(', ')}.`, 'INCOMPLETE_GAUSSIAN_SCHEMA');
  const rest = names.filter(name => name.startsWith('f_rest'));
  const shDegree = supportedRestCounts.get(rest.length);
  if (shDegree === undefined || rest.some(name => !/^f_rest_(?:0|[1-9]\d*)$/.test(name))
    || Array.from({ length: rest.length }, (_, i) => `f_rest_${i}`).some(name => !names.includes(name))) {
    fail('Gaussian SH rest fields must be continuous f_rest_0..N with count 0, 9, 24 or 45.');
  }
  const allowed = new Set(standardProperties(rest.length));
  if (names.some(name => gaussianSignal.test(name) && !allowed.has(name))) fail('Gaussian PLY contains an unsupported Gaussian parameter field.');
  const requiredProps = props.filter(p => allowed.has(p.name));
  if (requiredProps.some(p => !float32Types.has(p.type))) fail('Gaussian parameters must be declared float/float32; implicit precision conversion is unsupported.', 'UNSUPPORTED_GAUSSIAN_ENCODING');
  const normalCount = normals.filter(name => names.includes(name)).length;
  if (normalCount !== 0 && normalCount !== 3) fail('Gaussian PLY normals must include nx, ny and nz together.');
  const expected = standardProperties(rest.length);
  const canonical = header.format === 'binary_little_endian'
    && props.every(p => p.type === 'float') && names.length === expected.length
    && names.every((name, i) => name === expected[i]);
  return { kind: 'gaussian', shDegree, restCount: rest.length, canonical, hasNormals: normalCount === 3,
    extraProperties: names.filter(name => !allowed.has(name)) };
}

// One reusable row avoids allocating an object per Gaussian during validation.
function forEachVertex(bytes, header, callback) {
  const { properties, count } = header.vertex;
  const stride = properties.reduce((n, p) => n + types[p.type][0], 0);
  const row = new Float64Array(properties.length);
  if (header.format === 'ascii') {
    let text;
    try { text = decoder.decode(bytes.subarray(header.headerBytes)); }
    catch { fail('PLY ASCII body is not valid UTF-8.'); }
    let at = 0;
    const token = () => {
      while (at < text.length && /\s/.test(text[at])) at++;
      const start = at;
      while (at < text.length && !/\s/.test(text[at])) at++;
      return text.slice(start, at);
    };
    for (let i = 0; i < count; i++) {
      for (let j = 0; j < properties.length; j++) {
        const value = token();
        if (!value || !numberToken.test(value)) fail(`PLY ASCII: missing or invalid value at vertex ${i}, ${properties[j].name}.`);
        const numeric = Number(value);
        row[j] = float32Types.has(properties[j].type) ? Math.fround(numeric) : numeric;
      }
      callback(row, i);
    }
    if (token()) fail('PLY ASCII: unexpected trailing values.');
  } else {
    if (header.headerBytes + count * stride !== bytes.byteLength) fail('Gaussian PLY body length does not match its vertex declaration.');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const littleEndian = header.format === 'binary_little_endian';
    let offset = header.headerBytes;
    for (let i = 0; i < count; i++) {
      for (let j = 0; j < properties.length; j++) {
        const [size, getter] = types[properties[j].type];
        row[j] = view[getter](offset, littleEndian);
        offset += size;
      }
      callback(row, i);
    }
  }
}

function validatedInfo(bytes) {
  const header = headerOf(bytes), schema = schemaOf(header);
  if (schema.kind === 'gaussian') {
    const props = header.vertex.properties;
    const rotations = ['rot_0', 'rot_1', 'rot_2', 'rot_3'].map(name => props.findIndex(p => p.name === name));
    forEachVertex(bytes, header, (row, i) => {
      for (let j = 0; j < row.length; j++) if (!Number.isFinite(row[j])) fail(`Gaussian PLY vertex ${i}: ${props[j].name} must be finite.`);
      if (rotations.every(j => row[j] === 0)) fail(`Gaussian PLY vertex ${i}: quaternion must be nonzero.`);
    });
  }
  const info = { ...schema, count: header.vertex.count, format: header.format,
    properties: header.vertex.properties.map(p => p.name),
    propertyTypes: header.vertex.properties.map(p => p.type),
    headerBytes: header.headerBytes, byteLength: bytes.byteLength,
    bodyValidated: schema.kind === 'gaussian',
    ...(schema.kind === 'gaussian' ? { semantics: { ...semantics } } : {}) };
  return { header, info };
}

/**
 * Inspect a PLY. Gaussian bodies are fully validated. A points result is only
 * header classification; use the ordinary point-cloud parser to validate it.
 * Accepts ArrayBuffer, Uint8Array, DataView and Node Buffer without mutation.
 */
export function inspectGaussianPly(buffer) {
  return validatedInfo(bytesOf(buffer)).info;
}

/**
 * Return {bytes: Uint8Array, metadata}. Canonical Graphdeco input stays byte
 * identical. Other supported float32 PLYs are explicitly canonicalized: only
 * encoding/field order changes and absent normals become zero. No point, SH,
 * opacity, scale or raw quaternion value is transformed or normalized.
 */
export function exportStandardGaussianPly(buffer) {
  const source = bytesOf(buffer), { header, info } = validatedInfo(source);
  if (info.kind !== 'gaussian') fail('Use the ordinary point-cloud exporter for this PLY.', 'NOT_GAUSSIAN_PLY');
  if (info.extraProperties.length) fail(`Cannot export standard 3DGS without discarding extra fields: ${info.extraProperties.join(', ')}.`, 'UNSUPPORTED_GAUSSIAN_FIELDS');
  let bytes;
  if (info.canonical) bytes = source.slice();
  else {
    const names = standardProperties(info.restCount);
    const text = 'ply\nformat binary_little_endian 1.0\n'
      + 'comment Graphdeco 3DGS parameters; original world coordinates\n'
      + `element vertex ${info.count}\n`
      + names.map(name => `property float ${name}\n`).join('') + 'end_header\n';
    const encoded = encoder.encode(text);
    bytes = new Uint8Array(encoded.length + info.count * names.length * 4);
    bytes.set(encoded);
    const view = new DataView(bytes.buffer);
    const indices = names.map(name => info.properties.indexOf(name));
    forEachVertex(source, header, (row, i) => {
      let offset = encoded.length + i * names.length * 4;
      for (const index of indices) { view.setFloat32(offset, index < 0 ? 0 : row[index], true); offset += 4; }
    });
  }
  return { bytes, metadata: {
    format: 'graphdeco-3dgs-ply', count: info.count, shDegree: info.shDegree,
    mode: info.canonical ? 'byte-preserved' : 'canonical-format-conversion',
    sourceFormat: info.format, preservedBytes: info.canonical,
    addedProperties: info.hasNormals ? [] : normals.slice(),
    reorderedProperties: !info.canonical && info.properties.join(',') !== standardProperties(info.restCount).join(','),
    numericTransforms: [], discardedProperties: [], semantics: { ...semantics },
    sourceByteLength: source.byteLength, byteLength: bytes.byteLength,
  } };
}
