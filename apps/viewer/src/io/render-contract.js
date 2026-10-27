// This viewer is a pinhole / screen-space Gaussian preview, not a native 3DGUT renderer.
// Keep projection validation independent of the preview backend.
export function assertPinholeCamera(c){
  if(!c||typeof c!=='object')return;
  const models=[c.projectionModel,c.camera_model,c.cameraModel,c.projection?.model,
    typeof c.projection==='string'?c.projection:null,(c.K||c.fx!=null)&&typeof c.model==='string'?c.model:null];
  for(const model of models){
    if(model!=null&&!['pinhole','simple_pinhole','pinhole_export_assumption'].includes(String(model).toLowerCase()))
      throw Error('当前查看器只支持已去畸变的针孔投影，不能将 '+String(model)+' 静默转换为 PINHOLE；请使用匹配的原生渲染器');
  }
  const arrays=[c.distortion_coefficients,c.distortion_params,c.D,c.radial_coeffs,c.tangential_coeffs,c.thin_prism_coeffs,
    Array.isArray(c.distortion)?c.distortion:c.distortion?.coefficients];
  for(const value of arrays){
    if(value==null)continue;
    if(!Array.isArray(value)||value.flat(Infinity).some(x=>typeof x!=='number'||!Number.isFinite(x)||x!==0))
      throw Error('当前针孔预览不支持非零或未知畸变参数；请使用匹配的原生渲染器');
  }
  for(const key of ['k1','k2','k3','k4','k5','k6','p1','p2']){
    if(c[key]!=null&&(typeof c[key]!=='number'||!Number.isFinite(c[key])||c[key]!==0))
      throw Error('当前针孔预览不支持畸变参数 '+key+'；请使用匹配的原生渲染器');
  }
}
export function assertPreviewCompatible(data){
  if(!data||typeof data!=='object')return;
  assertPinholeCamera(data);
  const c=data.rendererContract;
  if(c==null)return; // Legacy PLYs do not establish their native rendering semantics.
  if(c.schema!=='shuolee-renderer-contract-v1'||c.cameraProjection!=='pinhole'||c.gaussianEvaluation!=='screen-space-2d')
    throw Error('模型声明的相机或高斯求值方式与当前针孔预览不一致，请使用模型指定的原生渲染器');
}
export function previewRendererContract(){
  return {schema:'shuolee-renderer-contract-v1',cameraProjection:'pinhole',gaussianEvaluation:'screen-space-2d',
    previewBackend:'spark',scope:'preview-only',trainingRenderer:'not-recorded',nativeImageEquivalence:'not-verified'};
}
