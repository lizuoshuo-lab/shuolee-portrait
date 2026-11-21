"""Preserve calibrated image extent when reviewing a photographed camera."""

def apply_capture_frame(js):
    if '// calibrated-capture-frame-v1' in js:return js
    js=js.replace("window.viewerState={status:'loading'};",
        "// calibrated-capture-frame-v1\nlet capturedFrame=false;\nwindow.viewerState={status:'loading'};")
    old='function size(){if(!renderer)return;const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}'
    new='''function size(){
 if(!renderer)return;
 const w=viewport.clientWidth,h=viewport.clientHeight,c=cameras?.[selected];
 let rw=w,rh=h;
 if(capturedFrame&&c){const factor=Math.min(w/c.width,h/c.height);rw=Math.max(1,Math.round(c.width*factor));rh=Math.max(1,Math.round(c.height*factor));camera.aspect=(c.width/c.height)*(c.fy/c.fx);}
 else camera.aspect=w/h;
 renderer.setSize(rw,rh);
 renderer.domElement.style.position='absolute';renderer.domElement.style.left=Math.round((w-rw)/2)+'px';renderer.domElement.style.top=Math.round((h-rh)/2)+'px';
 viewport.dataset.projection=capturedFrame?'captured-image':'free-view';
 camera.updateProjectionMatrix();
}'''
    assert old in js;js=js.replace(old,new)
    js=js.replace("controls.addEventListener('start',()=>{stop();$('mode').textContent='自由浏览';});",
        "controls.addEventListener('start',()=>{stop();capturedFrame=false;size();$('mode').textContent='自由浏览 · 含拍摄范围外区域';});")
    js=js.replace('function choose(i){stop();selected=i;', 'function choose(i){stop();selected=i;capturedFrame=true;size();')
    js=js.replace("$('mode').textContent=label(i);window.viewerState.view=c.img_name;",
        "$('mode').textContent=label(i)+' · 原照片画幅';window.viewerState.view=c.img_name;")
    js=js.replace('function overview(){stop();selected=24;', 'function overview(){stop();capturedFrame=false;selected=24;size();')
    js=js.replace("playing=true;controls.enabled=false;clockStart=", "playing=true;capturedFrame=true;size();controls.enabled=false;clockStart=")
    js=js.replace("fov:camera.fov,selected,mode:", "fov:camera.fov,capturedFrame,selected,mode:")
    js=js.replace("selected=p.selected??24;$('views')", "selected=p.selected??24;capturedFrame=!!p.capturedFrame;size();$('views')")
    js=js.replace("model:meta.model,view:'overview'", "model:meta.model,view:window.viewerState.view||'overview'")
    js=js.replace('请通过“打开已训练3DGS.exe”启动','请使用本文件夹的查看器启动程序')
    return js

if __name__=='__main__':
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
    p=root/'viewer/viewer.js'
    (root/'viewer-before-capture-frame.js').write_bytes(p.read_bytes())
    p.write_text(apply_capture_frame(p.read_text(encoding='utf-8')),encoding='utf-8')
