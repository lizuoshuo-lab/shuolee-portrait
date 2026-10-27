# 标准 3DGS PLY 第三方互操作验收

这是独立于主查看器和 Spark 的本机验收入口。两个上游库自己解码和绘制高斯；没有调用 Three.js 的通用点云 PLYLoader 代替高斯。

| 工具 | 固定 npm 版本 | 许可证 | 使用的实际接口 | 限制 |
|---|---|---|---|---|
| [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | `@mkkellogg/gaussian-splats-3d@0.4.7` | MIT, Copyright 2023 Mark Kellogg | `Viewer.addSplatScene`、`PlyParser.parseToUncompressedSplatBuffer` | 渲染最高二阶 SH；输入三阶时会截断，不能作为三阶精确回归 |
| [PlayCanvas Engine](https://github.com/playcanvas/engine) | `playcanvas@2.23.3` | MIT, Copyright 2011-2026 PlayCanvas Ltd. | `Asset('gsplat')`、`GSplatComponent`、引擎 `PlyParser` | 此入口针对未压缩、二进制小端标准 3DGS PLY；兼容预览与原生训练渲染不等价 |

版本、许可证及接口已从上述版本 npm 包的 `package.json`、`LICENSE`、README/类型声明/实际源码核查。依赖许可证随 npm 包保留；如分发打包产物，必须一起保留完整 MIT 通知。没有向外站上传模型，也不需要注册外站账号。

## 运行

在 `apps/viewer` 下执行；依赖由本工程的锁文件固定：

```powershell
npm ci
node --test tests/interop/third-party-read.test.mjs tests/interop/harness-state.test.mjs
npx vite --config tests/interop/vite.config.js
```

打开 [本机入口](http://127.0.0.1:4873/tests/interop/index.html)。点击两个工具分别确认合成模型出现 3 行 × 3 列彩色、大小和角度不同的椭圆高斯。合成 PLY 只有 9 个高斯，62 个 float 字段，包含非等轴尺度、非单位四元数、opacity logit，以及所有 45 个非零 SH3 剩余系数。

导入实际人物时使用文件选择器，两个入口选择**同一份导出的标准 `.ply`**，无须转换为 `.ksplat`、`.sog` 或其他工具专用文件。也可填同源本机 HTTP 地址；不同本机端口需要模型服务器允许 CORS。入口拒绝远程模型 URL 和带用户名密码的 URL。模型在浏览器本机读取，不发送到服务器。

导出的实际文件可先执行 `node tests/interop/probe-ply.mjs "本机导出文件的绝对路径.ply"`，得到 SHA256、两个上游解码点数及 SH 阶数。探测只读，明确输出 `guiVerified: false`；仍须完成下面的浏览器目视验收。

可用 `?fixture=1` 自动载入合成控制，或 `?model=URL编码后的本机HTTP地址` 指定模型。两个查看器都按各自解码结果给出 `Gaussian count`；机器可读值为 `window.__interop`，DOM 为 `#status[data-state][data-loaded-count][data-render-frames]`。

- `loaded`：第三方解码完成，`loadedCount` 来自第三方数据对象。
- `rendered`：该工具已执行含模型的绘制调用；仍需截图目视检查实际外观。
- `error`：保留完整错误；不能把页面出现或画布存在当作载入成功。
- `qualityAccepted` 始终为 `false`：互操作通过不代表人物几何、物理面部点精度或训练结果通过。

生产打包验证可单独执行 `npx vite build --config tests/interop/vite.config.js`。产物在已忽略的 `tests/interop/dist`，入口路径保留为 `tests/interop/index.html`；此验收工具不加入主产品 UI。

## 验收记录的边界

CPU 测试确实调用两个独立上游 PLY 解码器，核验点数、位置、非等轴尺度、四元数顺序、opacity，PlayCanvas 另核验全部 SH3 字段。PlayCanvas CPU 测试在上游 `load:data` 事件停止，避免分配 GPU；这项测试不计为 GPU/GUI 显示通过。

另有运行状态回归：切换模型失败时必须清空旧点数、SH 与包围盒；已销毁实例的迟到回调不能覆盖新实例结果。本机 URL 输入只接受 localhost，拒绝远程地址或嵌入凭据。两套入口只释放自己持有的 GPU、观察器与 canvas，保留 `#viewport`，连续载入也需做实际浏览器复验。

浏览器验收应记录模型 SHA256、工具固定版本、点数、`state`、`renderFrames` 与截图。需要轮流载入合成控制和最终人物 PLY，并确认旋转与缩放可用。没有真实浏览器运行和截图时，状态必须写“GUI 尚未验证”。不能只凭二进制可读、Vite 构建成功或文件扩展名一致宣称显示验收完成。

本目录的 `anisotropic-sh3.ply` 可由 `node tests/interop/generate-fixture.mjs` 重建；其 SHA256 及约定在 `anisotropic-sh3.json`。`private/` 被忽略，只供临时本机检查，禁止提交人物模型、视频或私密日志。
