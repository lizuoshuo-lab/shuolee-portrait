from pathlib import Path
import zipfile, json, hashlib, ast
H=Path(__file__).parent
src=Path.home()/'Downloads/new-capture-119-sofa-clothed_scene.zip'
dest=H/'scene-browser-roundtrip121'
with zipfile.ZipFile(src) as z:
    assert z.testzip() is None
    for name in z.namelist():
        p=(dest/name).resolve()
        assert p.is_relative_to(dest.resolve()) and ':' not in name and '\\' not in name
    m=json.loads(z.read('scene.json'))
    assert hashlib.sha256(z.read(m['model']['file'])).hexdigest()==m['model']['sha256']
    assert hashlib.sha256(z.read(m['cameras']['file'])).hexdigest()==m['cameras']['sha256']
    cams=json.loads(z.read('cameras.json'))
    assert len(cams)==m['cameras']['count']==89
    tree=ast.parse(z.read('cfg_args').decode(),mode='eval')
    assert isinstance(tree.body,ast.Call) and tree.body.func.id=='Namespace'
    config={x.arg:ast.literal_eval(x.value) for x in tree.body.keywords}
    assert config['sh_degree']==3
    z.extractall(dest)
result={'zip':str(src),'crcPassed':True,'modelSha256':m['model']['sha256'],'cameraCount':len(cams),'cfg_args_parsed_without_eval':config,'sibrGuiVerified':False,'sparkHostedCameraImportSupported':False}
(H/'scene-package121-download.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
