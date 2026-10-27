# 高斯与点云查看器

独立应用。模型在本机浏览器读取，不上传到服务端；无内置人物和训练报告。

需要 Node.js 22 或 24 LTS。从本仓库取得源码后：

```bash
cd apps/viewer
npm ci
npm run check
npm start
```

打开 http://127.0.0.1:4870。`npm start` 只提供构建后的 `dist`，绑定本机地址。
开发修改使用 `npm run dev`。更新版本用 `git pull --ff-only`，再运行以上命令。
Git 中不提交 `node_modules`、`dist`、视频或人物模型。

导入支持标准 Graphdeco 3DGS PLY（SH0–3）及普通 PLY、PCD、XYZ、PTS 点云。
PCD 支持 ASCII 和 binary；不支持 binary_compressed。单文件上限 512 MiB，最多 1000 万点。
高斯属性缺失、非有限参数、零四元数和不支持的自定义字段会明确报错，不静默当作普通点云。

导出提供两项：

- **通用 PLY**：保留原世界坐标和完整高斯参数。标准二进制小端 float32 文件保持字节一致；ASCII、大端或字段重排会规范为标准二进制格式，不激活 log scale/logit alpha，不量化旋转或球谐。
- **模型与相机包**：`point_cloud.ply`、可选 `cameras.json` 和 COLMAP `cameras.txt/images.txt/points3D.txt`，附可选元数据。不携带工具专用 HTML，不要求本查看器才能打开。

相机支持常见 3DGS 数组和显式 K+c2w/w2c JSON。COLMAP 输出保留 fx/fy/cx/cy；旧 3DGS 读者可能只使用中心主点。缺少主点时显示采用图像中心的说明。模型与相机必须处在同一个世界坐标系。
预览只支持已去畸变针孔相机，非零畸变和原生鱼眼会被拒绝。文件兼容与浏览器显示通过不等于原生渲染一致或重建质量通过。

源文件、导出与渲染分离在 `src/io`、`src/render`。导出始终从保留的原始输入读取，不从渲染器压缩缓冲反向生成。
第三方依赖由 `package-lock.json` 锁定，构建自动包含其许可通知。

独立第三方验收见 [tests/interop/README.md](tests/interop/README.md)。
两套上游高斯解码器各自读取导出 PLY；GaussianSplats3D 最高渲染 SH2，PlayCanvas 支持 SH3。
