/**
 * Point cloud import, pure ESM; no rendering or model mutation.
 * Format handling reviewed against Three.js r180 PLYLoader / PCDLoader:
 * https://github.com/mrdoob/three.js/tree/r180/examples/jsm/loaders
 * This implementation adds strict counts, bounded lists and explicit failures.
 * RGB is normalized sRGB, NOT converted to linear light here. The renderer
 * should convert it once. Coordinates are not reoriented/recentered/rescaled;
 * returned positions have the precision of the required Float32Array.
 */

const MAX_POINTS = 10_000_000;
const MAX_HEADER_BYTES = 1024 * 1024;
const MAX_PROPERTIES = 512;
const MAX_COMPONENTS = 65536;
const FLOAT32_MAX = 3.4028234663852886e38;
const decoder = new TextDecoder('utf-8', { fatal: true });
const FLOAT_TOKEN = /^[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)$/i;
const INTEGER_TOKEN = /^[+-]?\d+$/;
const GS_FIELDS = ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2'];

function invalid(message, code = 'INVALID_POINT_CLOUD') {
  const error = new Error(message);
  error.code = code;
  throw error;
}

function bytesOf(buffer) {
  if (!(buffer instanceof ArrayBuffer)) invalid('Point cloud input must be an ArrayBuffer.');
  if (buffer.byteLength === 0) invalid('Point cloud file is empty.');
  return new Uint8Array(buffer);
}

function decode(bytes, label) {
  try { return decoder.decode(bytes); }
  catch { invalid(`${label}: invalid UTF-8 or binary data in a text section.`); }
}

function countValue(token, label, allowZero = false, max = MAX_POINTS) {
  if (!/^\d+$/.test(token ?? '')) invalid(`${label}: expected an unsigned integer.`);
  const value = Number(token);
  if (!Number.isSafeInteger(value) || value < (allowZero ? 0 : 1) || value > max) {
    invalid(`${label}: count must be ${allowZero ? '0' : '1'}–${max}; no point subsampling is performed.`);
  }
  return value;
}

function makeType(name, bytes, getter, signed = null) {
  const floating = signed === null;
  const min = floating ? -Infinity : signed ? -(2 ** (8 * bytes - 1)) : 0;
  const max = floating ? Infinity : signed ? 2 ** (8 * bytes - 1) - 1 : 2 ** (8 * bytes) - 1;
  return { name, bytes, getter, floating, signed, min, max };
}

const TYPES = {
  int8: makeType('int8', 1, 'getInt8', true), uint8: makeType('uint8', 1, 'getUint8', false),
  int16: makeType('int16', 2, 'getInt16', true), uint16: makeType('uint16', 2, 'getUint16', false),
  int32: makeType('int32', 4, 'getInt32', true), uint32: makeType('uint32', 4, 'getUint32', false),
  float32: makeType('float32', 4, 'getFloat32'), float64: makeType('float64', 8, 'getFloat64'),
  int64: makeType('int64', 8, 'getBigInt64', true), uint64: makeType('uint64', 8, 'getBigUint64', false),
};
const PLY_TYPES = { char: 'int8', uchar: 'uint8', short: 'int16', ushort: 'uint16', int: 'int32', uint: 'uint32', float: 'float32', double: 'float64' };

function plyType(name) {
  const canonical = PLY_TYPES[name] ?? name;
  const type = TYPES[canonical];
  if (!type || canonical === 'int64' || canonical === 'uint64') invalid(`PLY: unsupported property type "${name}".`, 'UNSUPPORTED_ENCODING');
  return type;
}

function numeric(token, type, label) {
  if (!type.floating) {
    if (!INTEGER_TOKEN.test(token ?? '')) invalid(`${label}: invalid integer "${token}".`);
    if (type.bytes === 8) {
      const value = BigInt(token);
      const low = type.signed ? -(1n << 63n) : 0n;
      const high = type.signed ? (1n << 63n) - 1n : (1n << 64n) - 1n;
      if (value < low || value > high) invalid(`${label}: integer exceeds ${type.name}.`);
      return Number(value);
    }
    const value = Number(token);
    if (!Number.isInteger(value) || value < type.min || value > type.max) invalid(`${label}: integer exceeds ${type.name}.`);
    return value;
  }
  if (!FLOAT_TOKEN.test(token ?? '')) invalid(`${label}: invalid floating-point value "${token}".`);
  if (/^[+-]?nan$/i.test(token)) return NaN;
  if (/^[+-]?inf(?:inity)?$/i.test(token)) return token.startsWith('-') ? -Infinity : Infinity;
  const value = Number(token);
  if (!Number.isFinite(value) || (type.bytes === 4 && Math.abs(value) > FLOAT32_MAX)) invalid(`${label}: numeric overflow.`);
  return value;
}

// Header offsets are byte offsets. CRLF consumes exactly two bytes, never an
// arbitrary whitespace run that could eat the first binary payload byte.
function headerLines(bytes, stop) {
  const lines = [];
  let at = bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf ? 3 : 0;
  while (at < bytes.length && at < MAX_HEADER_BYTES) {
    const start = at;
    while (at < bytes.length && bytes[at] !== 10 && bytes[at] !== 13) {
      if (at >= MAX_HEADER_BYTES) invalid('Point cloud header exceeds 1 MiB.');
      at++;
    }
    const line = decode(bytes.subarray(start, at), 'Header').trim();
    if (at < bytes.length) {
      const newline = bytes[at++];
      if (newline === 13 && bytes[at] === 10) at++;
    }
    lines.push(line);
    if (stop(line)) return { lines, offset: at };
  }
  invalid('Missing or overlong point cloud header terminator.');
}

function requireXYZ(properties, label) {
  const indices = ['x', 'y', 'z'].map(name => properties.findIndex(p => p.name === name));
  if (indices.some(i => i < 0) || indices.some(i => properties[i].list || properties[i].count !== 1)) {
    invalid(`${label}: x, y and z must all be scalar properties.`);
  }
  return indices;
}

function plyHeader(bytes) {
  const { lines, offset } = headerLines(bytes, line => line === 'end_header');
  if (lines[0] !== 'ply') invalid('PLY: missing ply magic.');
  let format = null, current = null;
  const elements = [];
  for (const line of lines.slice(1, -1)) {
    if (!line) continue;
    const words = line.split(/\s+/);
    if (words[0] === 'comment' || words[0] === 'obj_info') continue;
    if (words[0] === 'format') {
      if (format || elements.length || words.length !== 3 || words[2] !== '1.0' || !['ascii', 'binary_little_endian', 'binary_big_endian'].includes(words[1])) invalid('PLY: invalid or unsupported format declaration.');
      format = words[1];
    } else if (words[0] === 'element') {
      if (!format || words.length !== 3 || !/^[\w.-]+$/.test(words[1]) || elements.some(e => e.name === words[1]) || elements.length >= 64) invalid('PLY: invalid, duplicate or excessive element declaration.');
      current = { name: words[1], count: countValue(words[2], `PLY element ${words[1]}`, true), properties: [] };
      elements.push(current);
    } else if (words[0] === 'property') {
      if (!current || current.properties.length >= MAX_PROPERTIES) invalid('PLY: property outside an element or too many properties.');
      let property;
      if (words[1] === 'list') {
        if (words.length !== 5) invalid('PLY: malformed list property.');
        const countType = plyType(words[2]);
        if (countType.floating) invalid('PLY: list count must have an integer type.');
        property = { name: words[4], list: true, countType, type: plyType(words[3]) };
      } else {
        if (words.length !== 3) invalid('PLY: malformed scalar property.');
        property = { name: words[2], list: false, count: 1, type: plyType(words[1]) };
      }
      if (!/^[\w.-]+$/.test(property.name) || current.properties.some(p => p.name === property.name)) invalid('PLY: invalid or duplicate property name.');
      current.properties.push(property);
    } else invalid(`PLY: unknown header directive "${words[0]}".`);
  }
  if (!format) invalid('PLY: format declaration is missing.');
  if (elements.some(e => e.count > 0 && !e.properties.length)) invalid('PLY: nonempty element without properties.');
  const vertex = elements.find(e => e.name === 'vertex');
  if (!vertex || vertex.count === 0) invalid('PLY: no vertices declared.');
  requireXYZ(vertex.properties, 'PLY');
  const kind = GS_FIELDS.every(name => vertex.properties.some(p => p.name === name && !p.list)) ? 'gaussian' : 'points';
  return { format, offset, elements, vertex, kind };
}

/** Header-only classification. Body validation happens in parsePointCloud. */
export function inspectPly(arrayBuffer) {
  const header = plyHeader(bytesOf(arrayBuffer));
  return { kind: header.kind, count: header.vertex.count, format: header.format, properties: header.vertex.properties.map(p => p.name) };
}

class TextLines {
  constructor(text, comments = false) { this.text = text; this.at = 0; this.line = 0; this.comments = comments; }
  next() {
    while (this.at < this.text.length) {
      const start = this.at;
      while (this.at < this.text.length && this.text[this.at] !== '\n' && this.text[this.at] !== '\r') this.at++;
      let line = this.text.slice(start, this.at);
      if (this.at < this.text.length) {
        const newline = this.text[this.at++];
        if (newline === '\r' && this.text[this.at] === '\n') this.at++;
      }
      this.line++;
      if (this.comments) line = line.split('#', 1)[0];
      if (line.includes('\0')) invalid(`Text line ${this.line}: unexpected NUL byte.`);
      line = line.trim();
      if (line) return line;
    }
    return null;
  }
}

class AsciiRow {
  constructor(line, label) { this.values = line.split(/\s+/); this.at = 0; this.label = label; }
  read(type, packed = false) {
    if (this.at >= this.values.length) invalid(`${this.label}: missing property value.`);
    const value = numeric(this.values[this.at++], type, this.label);
    return packed ? packedBits(value, type, this.label) : value;
  }
  skip(type, count) { for (let n = 0; n < count; n++) this.read(type); }
  finish() { if (this.at !== this.values.length) invalid(`${this.label}: extra property values.`); }
}

class BinaryCursor {
  constructor(buffer, offset, littleEndian, label) { this.view = new DataView(buffer); this.at = offset; this.little = littleEndian; this.label = label; }
  ensure(size) { if (!Number.isSafeInteger(size) || size < 0 || this.at + size > this.view.byteLength) invalid(`${this.label}: truncated binary data.`); }
  read(type, packed = false) {
    this.ensure(type.bytes);
    const value = packed ? this.view.getUint32(this.at, this.little) : this.view[type.getter](this.at, this.little);
    this.at += type.bytes;
    return typeof value === 'bigint' ? Number(value) : value;
  }
  skip(type, count) { const size = type.bytes * count; this.ensure(size); this.at += size; }
  finish() { if (this.at !== this.view.byteLength) invalid(`${this.label}: unexpected trailing binary bytes.`); }
}

const packedView = new DataView(new ArrayBuffer(4));
function packedBits(value, type, label) {
  if (type.floating) {
    if (!Number.isFinite(value)) invalid(`${label}: non-finite ASCII packed RGB cannot preserve its bit payload; use binary or an unsigned integer RGB field.`);
    packedView.setFloat32(0, value, true);
    return packedView.getUint32(0, true);
  }
  return value >>> 0;
}

function colorLayout(properties, label) {
  const aliases = [['red', 'green', 'blue'], ['r', 'g', 'b'], ['diffuse_red', 'diffuse_green', 'diffuse_blue']];
  for (const names of aliases) {
    const indices = names.map(name => properties.findIndex(p => p.name === name));
    if (indices.every(i => i >= 0)) {
      if (indices.some(i => properties[i].list || properties[i].count !== 1)) invalid(`${label}: RGB channels must be scalar.`);
      const types = indices.map(i => properties[i].type);
      if (new Set(types.map(t => t.name)).size !== 1) invalid(`${label}: mixed RGB channel types are ambiguous.`);
      const type = types[0];
      if (type.bytes === 8 && !type.floating) invalid(`${label}: 64-bit integer RGB is unsupported.`, 'UNSUPPORTED_ENCODING');
      // Standard RGB is 8 bit. An explicitly unsigned16/32 channel uses its
      // declared full range. Floating channels use a per-file 0–1/0–255 scale.
      const divisor = type.floating ? null : (!type.signed && type.bytes > 1 ? type.max : 255);
      return { kind: 'channels', indices, divisor, packedIndex: -1 };
    }
  }
  const packedIndex = properties.findIndex(p => p.name === 'rgb' || p.name === 'rgba');
  if (packedIndex >= 0) {
    const p = properties[packedIndex];
    if (p.list || p.count !== 1 || p.type.bytes !== 4) invalid(`${label}: packed rgb/rgba must be a scalar32-bit float or integer.`);
    return { kind: 'packed', indices: [], divisor: 255, packedIndex };
  }
  if (aliases.flat().some(name => properties.some(p => p.name === name))) invalid(`${label}: incomplete RGB channels.`);
  return null;
}

class PointOutput {
  constructor(sourceCount, color, format) {
    this.sourceCount = sourceCount; this.color = color; this.format = format; this.count = 0;
    try {
      this.positions = new Float32Array(sourceCount * 3);
      this.colors = color ? new Float32Array(sourceCount * 3) : null;
    } catch { invalid('Insufficient memory for the complete point cloud; no points were silently truncated.'); }
    this.min = [Infinity, Infinity, Infinity]; this.max = [-Infinity, -Infinity, -Infinity]; this.maxColor = 0;
  }
  add(xyz, rgb) {
    if (rgb) {
      const upper = this.color.divisor ?? 255;
      if (rgb.some(c => !Number.isFinite(c) || c < 0 || c > upper)) invalid(`${this.format}: RGB must be finite and within its channel range.`);
      // Infer floating RGB units from the entire file, even if a point has an
      // explicitly non-finite position and is later reported as skipped.
      this.maxColor = Math.max(this.maxColor, ...rgb);
    }
    if (xyz.some(v => !Number.isFinite(v))) return; // Valid NaN/Inf samples are explicitly reported as skipped.
    const row = this.count * 3;
    for (let j = 0; j < 3; j++) {
      const value = Math.fround(xyz[j]);
      if (!Number.isFinite(value) || (value === 0 && xyz[j] !== 0)) invalid(`${this.format}: a coordinate is not representable in Float32; coordinates are not rescaled.`);
      this.positions[row + j] = value;
      this.min[j] = Math.min(this.min[j], value); this.max[j] = Math.max(this.max[j], value);
      if (rgb) this.colors[row + j] = rgb[j];
    }
    this.count++;
  }
  finish() {
    if (this.count === 0) invalid(`${this.format}: no finite points remain; the file is not loaded as an empty cloud.`);
    const length = this.count * 3;
    const colors = this.colors ? this.colors.subarray(0, length) : null;
    if (colors) {
      const divisor = this.color.divisor ?? (this.maxColor <= 1 ? 1 : 255);
      for (let i = 0; i < colors.length; i++) colors[i] /= divisor;
    }
    return { positions: this.positions.subarray(0, length), colors, count: this.count,
      sourceCount: this.sourceCount, skipped: this.sourceCount - this.count, format: this.format,
      colorEncoding: 'srgb', bounds: { min: this.min, max: this.max } };
  }
}

function colorsFromRow(row, color) {
  if (!color) return null;
  if (color.kind === 'channels') return color.indices.map(i => row[i]);
  const bits = row[color.packedIndex];
  return [(bits >>> 16) & 255, (bits >>> 8) & 255, bits & 255];
}

function parsePly(buffer, bytes) {
  const header = plyHeader(bytes);
  if (header.kind === 'gaussian') invalid('PLY contains a complete Gaussian schema; use the Gaussian renderer instead of the point-cloud loader.', 'GAUSSIAN_PLY');
  const xyz = requireXYZ(header.vertex.properties, 'PLY');
  const color = colorLayout(header.vertex.properties, 'PLY');
  const format = `ply/${header.format}`;
  let lines = null, cursor = null;
  if (header.format === 'ascii') {
    const text = decode(bytes.subarray(header.offset), 'PLY ASCII');
    const counter = new TextLines(text); let rows = 0;
    while (counter.next() !== null) rows++;
    if (rows !== header.elements.reduce((n, e) => n + e.count, 0)) invalid('PLY ASCII: element/row count mismatch.');
    lines = new TextLines(text);
  } else {
    const minimum = header.elements.reduce((n, e) => n + e.count * e.properties.reduce((m, p) => m + (p.list ? p.countType.bytes : p.type.bytes), 0), 0);
    if (!Number.isSafeInteger(minimum) || bytes.length - header.offset < minimum) invalid('PLY: truncated binary elements.');
    cursor = new BinaryCursor(buffer, header.offset, header.format === 'binary_little_endian', 'PLY');
  }
  const output = new PointOutput(header.vertex.count, color, format);
  const row = new Array(header.vertex.properties.length);
  for (const element of header.elements) {
    const vertex = element === header.vertex;
    for (let n = 0; n < element.count; n++) {
      const reader = lines ? new AsciiRow(lines.next(), `PLY ${element.name} ${n}`) : cursor;
      for (let j = 0; j < element.properties.length; j++) {
        const p = element.properties[j];
        if (p.list) {
          const count = reader.read(p.countType);
          if (!Number.isSafeInteger(count) || count < 0 || count > MAX_POINTS) invalid('PLY: invalid or excessive list length.');
          reader.skip(p.type, count);
        } else {
          const value = reader.read(p.type, vertex && j === color?.packedIndex);
          if (vertex) row[j] = value;
        }
      }
      if (lines) reader.finish();
      if (vertex) output.add(xyz.map(i => row[i]), colorsFromRow(row, color));
    }
  }
  if (cursor) cursor.finish();
  return output.finish();
}

function pcdHeader(bytes) {
  const { lines, offset } = headerLines(bytes, line => /^DATA\s+/i.test(line));
  const values = Object.create(null);
  const supported = new Set(['VERSION', 'FIELDS', 'SIZE', 'TYPE', 'COUNT', 'WIDTH', 'HEIGHT', 'VIEWPOINT', 'POINTS', 'DATA']);
  for (let line of lines) {
    line = line.split('#', 1)[0].trim();
    if (!line) continue;
    const [keyRaw, ...words] = line.split(/\s+/); const key = keyRaw.toUpperCase();
    if (!supported.has(key) || values[key]) invalid(`PCD: unknown or duplicate header field "${key}".`);
    values[key] = words;
  }
  if (values.VERSION && (values.VERSION.length !== 1 || !['.7', '0.7', '0.6', '.6'].includes(values.VERSION[0]))) invalid('PCD: unsupported version.');
  if (!values.FIELDS?.length || values.FIELDS.length > MAX_PROPERTIES || !values.SIZE || !values.TYPE || values.DATA?.length !== 1) invalid('PCD: FIELDS, SIZE, TYPE and DATA are required.');
  const format = values.DATA[0].toLowerCase();
  if (format === 'binary_compressed') invalid('PCD binary_compressed is not supported by this importer. Export uncompressed binary or ASCII PCD.', 'UNSUPPORTED_ENCODING');
  if (!['ascii', 'binary'].includes(format)) invalid('PCD: unsupported DATA encoding.', 'UNSUPPORTED_ENCODING');
  if ([values.SIZE, values.TYPE, ...(values.COUNT ? [values.COUNT] : [])].some(v => v.length !== values.FIELDS.length)) invalid('PCD: FIELDS/SIZE/TYPE/COUNT lengths differ.');
  const scalar = name => {
    if (values[name]?.length !== 1) invalid(`PCD: ${name} must have exactly one value.`);
    return countValue(values[name][0], `PCD ${name}`);
  };
  let dimensions = null;
  if (values.WIDTH || values.HEIGHT) {
    dimensions = scalar('WIDTH') * scalar('HEIGHT');
    if (!Number.isSafeInteger(dimensions) || dimensions > MAX_POINTS) invalid('PCD: WIDTH × HEIGHT exceeds 10000000 points.');
  }
  const count = values.POINTS ? scalar('POINTS') : dimensions;
  if (!count || (dimensions !== null && dimensions !== count)) invalid('PCD: POINTS and WIDTH × HEIGHT do not match or are absent.');
  if (values.VIEWPOINT) {
    if (values.VIEWPOINT.length !== 7 || values.VIEWPOINT.some(t => !Number.isFinite(numeric(t, TYPES.float64, 'PCD VIEWPOINT')))) invalid('PCD: VIEWPOINT must have seven finite values.');
    // VIEWPOINT records sensor pose. It does not transform the stored points.
  }
  let stride = 0, components = 0;
  const properties = values.FIELDS.map((name, i) => {
    if (!/^[\w.-]+$/.test(name) || (name !== '_' && values.FIELDS.indexOf(name) !== i)) invalid('PCD: invalid or duplicate FIELDS name.');
    const size = countValue(values.SIZE[i], 'PCD SIZE', false, 8);
    const family = values.TYPE[i].toUpperCase();
    const type = TYPES[`${family === 'F' ? 'float' : family === 'I' ? 'int' : family === 'U' ? 'uint' : 'invalid'}${size * 8}`];
    if (!type) invalid(`PCD: unsupported TYPE/SIZE ${family}/${size}.`, 'UNSUPPORTED_ENCODING');
    const n = values.COUNT ? countValue(values.COUNT[i], 'PCD COUNT', false, MAX_COMPONENTS) : 1;
    const p = { name, type, count: n, list: false, offset: stride };
    stride += type.bytes * n; components += n;
    if (components > MAX_COMPONENTS) invalid('PCD: excessive per-point component count.');
    return p;
  });
  requireXYZ(properties, 'PCD');
  return { offset, format, count, properties, stride, components };
}

function parsePcd(buffer, bytes) {
  const h = pcdHeader(bytes); const xyz = requireXYZ(h.properties, 'PCD'); const color = colorLayout(h.properties, 'PCD');
  let lines = null, cursor = null;
  if (h.format === 'binary') {
    if (bytes.length - h.offset !== h.count * h.stride) invalid('PCD binary: byte length does not match POINTS × point stride.');
    cursor = new BinaryCursor(buffer, h.offset, true, 'PCD'); // PCD native/PCL convention is little-endian.
  } else {
    const text = decode(bytes.subarray(h.offset), 'PCD ASCII'); const counter = new TextLines(text, true); let count = 0;
    while (counter.next() !== null) count++;
    if (count !== h.count) invalid('PCD ASCII: POINTS/row count mismatch.');
    lines = new TextLines(text, true);
  }
  const output = new PointOutput(h.count, color, `pcd/${h.format}`); const row = new Array(h.properties.length);
  for (let n = 0; n < h.count; n++) {
    const reader = lines ? new AsciiRow(lines.next(), `PCD point ${n}`) : cursor;
    for (let j = 0; j < h.properties.length; j++) {
      const p = h.properties[j]; row[j] = reader.read(p.type, j === color?.packedIndex);
      if (p.count > 1) reader.skip(p.type, p.count - 1);
    }
    if (lines) reader.finish();
    output.add(xyz.map(i => row[i]), colorsFromRow(row, color));
  }
  if (cursor) cursor.finish();
  return output.finish();
}

function parseXyz(bytes, format) {
  const text = decode(bytes, format.toUpperCase()).replace(/^\uFEFF/, '');
  const lines = new TextLines(text, true); let line = lines.next();
  if (line === null) invalid(`${format}: no point records.`);
  let declared = null;
  if (/^\d+$/.test(line)) { declared = countValue(line, `${format} header`); line = lines.next(); }
  let sourceCount = 0, columns = null;
  while (line !== null) {
    const tokens = line.split(/\s+/);
    if (![3, 6].includes(tokens.length)) invalid(`${format} line ${lines.line}: expected XYZ or XYZ RGB (3 or6 columns).`);
    if (columns !== null && columns !== tokens.length) invalid(`${format}: inconsistent point columns.`);
    columns = tokens.length;
    // Syntax is checked before allocating full buffers, including skipped rows.
    tokens.forEach(t => numeric(t, TYPES.float64, `${format} line ${lines.line}`));
    if (++sourceCount > MAX_POINTS) invalid(`${format}: exceeds 10000000 points; no subsampling is performed.`);
    line = lines.next();
  }
  if (sourceCount === 0 || (declared !== null && declared !== sourceCount)) invalid(`${format}: declared point count differs from the number of rows.`);
  const color = columns === 6 ? { kind: 'channels', divisor: null } : null;
  const output = new PointOutput(sourceCount, color, format);
  const reader = new TextLines(text, true); if (declared !== null) reader.next();
  while ((line = reader.next()) !== null) {
    const values = line.split(/\s+/).map(t => numeric(t, TYPES.float64, `${format} line ${reader.line}`));
    output.add(values.slice(0, 3), color ? values.slice(3) : null);
  }
  return output.finish();
}

/**
 * Load all valid points; skipped records are only explicit NaN/Inf XYZ rows.
 * Errors are thrown for unsupported encoding, malformed/truncated records,
 * invalid color ranges, empty clouds and more than10M declared/source points.
 */
export function parsePointCloud(arrayBuffer, fileName = '') {
  const bytes = bytesOf(arrayBuffer);
  const extension = String(fileName).split(/[?#]/, 1)[0].toLowerCase().match(/\.([^.\\/]+)$/)?.[1];
  if (extension === 'ply') return parsePly(arrayBuffer, bytes);
  if (extension === 'pcd') return parsePcd(arrayBuffer, bytes);
  if (extension === 'xyz' || extension === 'pts') return parseXyz(bytes, extension);
  invalid('Unsupported point-cloud extension. Use .ply, .pcd, .xyz or .pts.', 'UNSUPPORTED_FORMAT');
}
