"""Create the independent viewport patch from the verified upstream source."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path('D:/Projects/msv2-paired')
REPO = Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/msv2-clean-runtime-20261105/vendor/mini-splatting2')
SOURCE = REPO / 'submodules/diff-gaussian-rasterization_ms'
original = {}
changed = {}


def read(name):
    text = (SOURCE / name).read_text(encoding='utf-8')
    original[name] = text
    return text


def replace(text, old, new, count=1):
    actual = text.count(old)
    if actual != count:
        raise ValueError((old[:100], count, actual))
    return text.replace(old, new)


def section(text, start, end, callback):
    first = text.index(start)
    last = text.index(end, first) if end else len(text)
    return text[:first] + callback(text[first:last]) + text[last:]


# Python ABI: new fields are appended to Settings; only render_imp uses them.
name = 'diff_gaussian_rasterization_ms/__init__.py'
s = read(name)
def pyforward(t):
    return replace(t, '            raster_settings.image_width,\n',
        '            raster_settings.image_width,\n'
        '            raster_settings.full_image_height,\n'
        '            raster_settings.full_image_width,\n'
        '            raster_settings.viewport_x,\n'
        '            raster_settings.viewport_y,\n')
s = section(s, '    def forward(\n', '    def render_simp(', pyforward)
def pyback(t):
    return replace(t, '                grad_out_color, \n',
        '                raster_settings.full_image_height,\n'
        '                raster_settings.full_image_width,\n'
        '                raster_settings.viewport_x,\n'
        '                raster_settings.viewport_y,\n'
        '                grad_out_color, \n')
s = section(s, '    def backward(ctx,', 'class GaussianRasterizationSettings', pyback)
s = replace(s, '    debug : bool\n', '    debug : bool\n'
    '    full_image_height: int = 0\n    full_image_width: int = 0\n'
    '    viewport_x: int = 0\n    viewport_y: int = 0\n')
changed[name] = s

# Public C++ entry points and internal Rasterizer forward/backward signatures.
for name in ['rasterize_points.h', 'rasterize_points.cu']:
    s = read(name)
    def cppforward(t):
        t = replace(t, '    const int image_width,\n',
            '    const int image_width,\n'
            '    const int full_image_height,\n    const int full_image_width,\n'
            '    const int viewport_x,\n    const int viewport_y,\n')
        if name.endswith('.cu'):
            t = replace(t, '  const int W = image_width;\n',
                '  const int W = image_width;\n'
                '  const int full_W = full_image_width > 0 ? full_image_width : W;\n'
                '  const int full_H = full_image_height > 0 ? full_image_height : H;\n'
                '  TORCH_CHECK(W > 0 && H > 0 && full_W > 0 && full_H > 0, "Invalid viewport dimensions");\n'
                '  TORCH_CHECK(viewport_x >= 0 && viewport_y >= 0 && viewport_x < full_W && viewport_y < full_H, "Viewport must intersect the full image");\n'
                '  TORCH_CHECK(viewport_x % BLOCK_X == 0 && viewport_y % BLOCK_Y == 0, "Viewport origin must align with native tiles");\n'
                '  TORCH_CHECK((W % BLOCK_X == 0 && H % BLOCK_Y == 0) || (viewport_x == 0 && viewport_y == 0 && W == full_W && H == full_H), "Viewport extent must align with native tiles");\n')
            t = replace(t, '      W, H,\n', '      W, H, full_W, full_H, viewport_x, viewport_y,\n')
        return t
    s = section(s, 'RasterizeGaussiansCUDA(', 'RasterizeGaussiansSimpCUDA(', cppforward)
    def cppback(t):
        t = replace(t, '    const torch::Tensor& dL_dout_color,\n',
            '    const int full_image_height,\n    const int full_image_width,\n'
            '    const int viewport_x,\n    const int viewport_y,\n'
            '    const torch::Tensor& dL_dout_color,\n')
        if name.endswith('.cu'):
            t = replace(t, '  const int W = dL_dout_color.size(2);\n',
                '  const int W = dL_dout_color.size(2);\n'
                '  const int full_W = full_image_width > 0 ? full_image_width : W;\n'
                '  const int full_H = full_image_height > 0 ? full_image_height : H;\n')
            t = replace(t, '    W, H, \n', '    W, H, full_W, full_H, viewport_x, viewport_y,\n')
        return t
    s = section(s, 'RasterizeGaussiansBackwardCUDA(', None if name.endswith('.h') else 'torch::Tensor markVisible(', cppback)
    changed[name] = s

for name in ['cuda_rasterizer/rasterizer.h', 'cuda_rasterizer/rasterizer_impl.cu']:
    s = read(name)
    def signature(t):
        return replace(t, 'const int width, int height,\n',
            'const int width, int height,\n'
            '\tconst int full_width, int full_height, int viewport_x, int viewport_y,\n')
    s = section(s, 'std::tuple<int,int> ' + ('forward(' if name.endswith('.h') else 'CudaRasterizer::Rasterizer::forward('),
        'static int forward_simp(' if name.endswith('.h') else 'int CudaRasterizer::Rasterizer::forward_simp(', signature)
    s = section(s, 'static void backward(' if name.endswith('.h') else 'void CudaRasterizer::Rasterizer::backward(', None, signature)
    changed[name] = s

# Shared preprocess keeps complete-image geometry. Only its tile rectangle is
# clipped to the aligned ROI; the other native render modes keep their defaults.
for name in ['cuda_rasterizer/forward.h', 'cuda_rasterizer/forward.cu']:
    s = read(name)
    tail = ('bool prefiltered,\n\t\tint viewport_x = 0, int viewport_y = 0,\n'
            '\t\tint viewport_width = 0, int viewport_height = 0);') if name.endswith('.h') else (
            'bool prefiltered,\n\tint viewport_x, int viewport_y,\n'
            '\tint viewport_width, int viewport_height)')
    s = replace(s, 'bool prefiltered);' if name.endswith('.h') else 'bool prefiltered)', tail,
                1 if name.endswith('.h') else 2)
    if name.endswith('.cu'):
        s = replace(s, '\tgetRect(point_image, rect_dims, rect_min, rect_max, grid);\t\n',
            '\tgetRect(point_image, rect_dims, rect_min, rect_max, grid);\n'
            '\tif (viewport_width > 0)\n\t{\n'
            '\t\tconst uint32_t tx = viewport_x / BLOCK_X, ty = viewport_y / BLOCK_Y;\n'
            '\t\trect_min.x = max(rect_min.x, tx);\n\t\trect_min.y = max(rect_min.y, ty);\n'
            '\t\trect_max.x = max(rect_min.x, min(rect_max.x, tx + (uint32_t)(viewport_width / BLOCK_X)));\n'
            '\t\trect_max.y = max(rect_min.y, min(rect_max.y, ty + (uint32_t)(viewport_height / BLOCK_Y)));\n'
            '\t}\n')
        s = replace(s, '\t\tprefiltered\n\t\t);',
            '\t\tprefiltered,\n\t\tviewport_x, viewport_y, viewport_width, viewport_height\n\t\t);')
    # Only differentiable renderCUDA and its host wrapper gain ROI coordinates.
    def render_sig(t):
        return replace(t, 'int W, int H,\n',
            'int W, int H,\n\tint full_W, int full_H, int viewport_x, int viewport_y,\n')
    if name.endswith('.h'):
        s = section(s, '\tvoid render(', '\tvoid render_simp(', render_sig)
    else:
        def render_kernel(t):
            t = render_sig(t)
            t = replace(t, '\tfloat2 pixf = { (float)pix.x, (float)pix.y };',
                '\tfloat2 pixf = { (float)(pix.x + viewport_x), (float)(pix.y + viewport_y) };')
            t = replace(t, '\tbool done = !inside;',
                '\tbool done = !inside || pix.x + viewport_x >= full_W || pix.y + viewport_y >= full_H;')
            return t
        s = section(s, 'renderCUDA(\n', 'render_simpCUDA(', render_kernel)
        def hostrender(t):
            t = render_sig(t)
            return replace(t, '\t\tW, H,\n', '\t\tW, H, full_W, full_H, viewport_x, viewport_y,\n')
        s = section(s, 'void FORWARD::render(', 'void FORWARD::render_simp(', hostrender)
    changed[name] = s

# Binning uses original global tile candidates/culling and converts only IDs.
name = 'cuda_rasterizer/rasterizer_impl.cu'
s = changed[name]
s = replace(s, '\tfloat2* rects)\n',
    '\tfloat2* rects,\n\tint viewport_x, int viewport_y, dim3 viewport_grid)\n')
s = replace(s, '\t\tconst float2 xy = points_xy[idx];',
    '\t\tconst uint32_t tx = viewport_x / BLOCK_X, ty = viewport_y / BLOCK_Y;\n'
    '\t\trect_min.x = max(rect_min.x, tx);\n\t\trect_min.y = max(rect_min.y, ty);\n'
    '\t\trect_max.x = max(rect_min.x, min(rect_max.x, tx + viewport_grid.x));\n'
    '\t\trect_max.y = max(rect_min.y, min(rect_max.y, ty + viewport_grid.y));\n'
    '\t\tconst float2 xy = points_xy[idx];')
s = replace(s, '\t\t\t\tuint64_t key = y * grid.x + x;',
    '\t\t\t\tuint64_t key = (y - ty) * viewport_grid.x + (x - tx);')
# Skip sentinel keys safely even when the ROI has no contributing tile.
s = replace(s, '\tif (idx == 0)\n\t\tranges[currtile].x = 0;',
    '\tif (idx == 0)\n\t{\n\t\tif (valid_tile) ranges[currtile].x = 0;\n\t}')
s = replace(s, '\t\t\tranges[prevtile].y = idx;',
    '\t\t\tif (prevtile != (uint32_t)-1) ranges[prevtile].y = idx;')
def forward_impl(t):
    t = replace(t, 'const float focal_y = height /', 'const float focal_y = full_height /')
    t = replace(t, 'const float focal_x = width /', 'const float focal_x = full_width /')
    t = replace(t, '\tdim3 block(BLOCK_X, BLOCK_Y, 1);',
        '\tdim3 block(BLOCK_X, BLOCK_Y, 1);\n'
        '\tdim3 full_tile_grid((full_width + BLOCK_X - 1) / BLOCK_X, (full_height + BLOCK_Y - 1) / BLOCK_Y, 1);')
    t = replace(t, '\t\twidth, height,\n', '\t\tfull_width, full_height,\n', 2)
    # restore render call dimensions to output ROI and pass its full geometry.
    t = replace(t, '\t\tfull_width, full_height,\n\t\tgeomState.means2D,',
        '\t\twidth, height, full_width, full_height, viewport_x, viewport_y,\n\t\tgeomState.means2D,')
    t = replace(t, '\t\ttile_grid,\n\t\tgeomState.tiles_touched,',
        '\t\tfull_tile_grid,\n\t\tgeomState.tiles_touched,')
    t = replace(t, '\t\tprefiltered\n', '\t\tprefiltered, viewport_x, viewport_y, width, height\n')
    t = replace(t, '\t\ttile_grid,\n\t\tgeomState.rects2D)',
        '\t\tfull_tile_grid,\n\t\tgeomState.rects2D, viewport_x, viewport_y, tile_grid)')
    return t
s = section(s, 'std::tuple<int,int> CudaRasterizer::Rasterizer::forward(',
    'int CudaRasterizer::Rasterizer::forward_simp(', forward_impl)
s = replace(s, '\t\tgeomState.rects2D)', '\t\tgeomState.rects2D, 0, 0, tile_grid)', 2)
def backward_impl(t):
    t = replace(t, 'const float focal_y = height /', 'const float focal_y = full_height /')
    t = replace(t, 'const float focal_x = width /', 'const float focal_x = full_width /')
    return replace(t, '\t\twidth, height, R, B,\n',
        '\t\twidth, height, R, B, full_width, full_height, viewport_x, viewport_y,\n')
s = section(s, 'void CudaRasterizer::Rasterizer::backward(', None, backward_impl)
changed[name] = s

for name in ['cuda_rasterizer/backward.h', 'cuda_rasterizer/backward.cu']:
    s = read(name)
    s = replace(s, 'int W, int H, int R, int B,\n',
        'int W, int H, int R, int B,\n\tint full_W, int full_H, int viewport_x, int viewport_y,\n')
    if name.endswith('.cu'):
        s = replace(s, 'int W, int H, int B,\n',
            'int W, int H, int B,\n\tint full_W, int full_H, int viewport_x, int viewport_y,\n')
        # Unused per-pixel fallback is also correct if enabled later.
        s = replace(s, '\tint W, int H,\n',
            '\tint W, int H,\n\tint full_W, int full_H, int viewport_x, int viewport_y,\n')
        s = replace(s, 'const float ddelx_dx = 0.5 * W;', 'const float ddelx_dx = 0.5 * full_W;', 2)
        s = replace(s, 'const float ddely_dy = 0.5 * H;', 'const float ddely_dy = 0.5 * full_H;', 2)
        s = replace(s, 'const float2 pixf = {(float) pix.x, (float) pix.y};',
            'const float2 pixf = {(float)(pix.x + viewport_x), (float)(pix.y + viewport_y)};')
        s = replace(s, 'const float2 pixf = { (float)pix.x, (float)pix.y };',
            'const float2 pixf = { (float)(pix.x + viewport_x), (float)(pix.y + viewport_y) };')
        s = replace(s, 'bool valid_pixel = pix.x < W && pix.y < H;',
            'bool valid_pixel = pix.x < W && pix.y < H && pix.x + viewport_x < full_W && pix.y + viewport_y < full_H;')
        s = replace(s, 'const bool inside = pix.x < W&& pix.y < H;',
            'const bool inside = pix.x < W&& pix.y < H && pix.x + viewport_x < full_W && pix.y + viewport_y < full_H;')
        s = replace(s, '\tconst int THREADS = 32;', '\tif (B == 0) return;\n\tconst int THREADS = 32;')
        s = replace(s, '\t\tW, H, B,\n', '\t\tW, H, B, full_W, full_H, viewport_x, viewport_y,\n')
    changed[name] = s

name = 'setup.py'
s = read(name).replace('diff_gaussian_rasterization_ms', 'diff_gaussian_rasterization_msv2_roi')
changed[name] = s

patch = ''.join(''.join(difflib.unified_diff(original[name].splitlines(True), changed[name].splitlines(True),
    fromfile='a/' + name, tofile='b/' + name)) for name in sorted(changed))
(ROOT / 'msv2/roi_backend.patch').write_text(patch, encoding='utf-8', newline='\n')
stage = Path('D:/GS-SLAM-Data/msv2-roi-patch-static-20261106')
stage.mkdir(exist_ok=True)
for name, text in changed.items():
    path = stage / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
receipt = json.loads((REPO / '.msv2-source.json').read_text())
core = sorted(set(changed) | {'ext.cpp', 'conv.cu', 'cuda_rasterizer/adam.cu', 'cuda_rasterizer/adam.h', 'cuda_rasterizer/rasterizer_impl.h', 'cuda_rasterizer/config.h', 'cuda_rasterizer/auxiliary.h', 'LICENSE.md'})
pins = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in core}
headers = sorted((SOURCE / 'third_party/glm/glm').rglob('*'))
glm = {str(path.relative_to(SOURCE)).replace('\\', '/'):hashlib.sha256(path.read_bytes()).hexdigest() for path in headers if path.is_file()}
glm['third_party/glm/copying.txt'] = hashlib.sha256((SOURCE/'third_party/glm/copying.txt').read_bytes()).hexdigest()
digest = hashlib.sha256(json.dumps(glm, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
print(json.dumps({'core':pins,'glm_digest':digest,'glm_files':len(glm),'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'files':len(changed)},indent=2))
