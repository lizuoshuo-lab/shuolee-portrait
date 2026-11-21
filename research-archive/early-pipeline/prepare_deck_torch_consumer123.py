from pathlib import Path
import hashlib,json

B=Path(__file__).parent
D=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport')
tag='free-transport123-torch-consumer'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
c=read(B/'free-transport123-gpu-content-final.json')
gpu=read(D/'torch-consumer-gpu/result.json')
cpu=read(D/'torch-consumer-result-completed.json')
accept=read(D/'torch-consumer-gpu/acceptance.json')
assert cpu['tests_run']==14 and cpu['outcome']=='passed'
assert gpu['passed'] and len(gpu['checks'])==83 and accept['passed']
c['final_test_evidence']=str(D/'torch-consumer-gpu/result.json')
c['body']='上表为原CPU参考的原生桥接。新消费者另通过14项CPU与83项CUDA数值检查。'
c['footer']='新消费者尚未接原生训练器，未测20万点性能。无实拍拟合或高斯训练。'
c['notes_required'][0]='原生GPU主表对应原CPU参考实现的桥接控制，运输源码SHA256 b0e271a3f66c15551e9aa0d0203f9fb31f8b46fced890eb4efed063f0e596c81。fixture为six-free-offset-gaussians-one-directed-triangle-v1，6个合成自由偏移高斯与1个定向三角形。14项原生检查通过，11次forward和1次backward均完成。本主表保留该次证据，不将新Torch消费者的CUDA数学核验当成再次调用native raster。'
c['notes_required']=[n for n in c['notes_required'] if not n.startswith('00另一对话既有world-unit')]
c['notes_required'].extend([
 '新Torch GPU消费者另行验证：14项纯CPU控制全部通过，耗时0.44518889799999783秒。随后独立CUDA数学比较通过83项字段断言，11次forward、1次backward完成，耗时4.198576817秒。检查中心、完整协方差、SH参考方向、尺度gauge、半周旋转、autocast与梯度。所有输入均为已知合成数据。',
 '新消费者来源：free_clothing_torch.py SHA256 c509cb70eeda9833f829dd4c5b74ed34b59686179fb73f0e50f4c1a7adfe630a；CPU参照free_clothing_transport.py SHA256 c1a3b98842bf1b3087222ec4723a18e6ae9edd7bd3ba45ee74a1b84f5ae4b492。它与上表先前原生桥接所用b0e版本明确区分。',
 '新消费者CUDA数学验收的native_raster_calls=0，native_bridge_validated=false，performance_200k_validated=false。尚未接入原生训练器，没有20万点性能结论，也没有实拍人体模型前向、拟合、高斯优化或PLY导出。training_admitted=false、physical_camera_accepted=false。不能合并两次不同消费者验证后宣称新训练器已经接通。',
 '00侧已发布CPU运输提交d4a3d73fdf8dbec8aed4f661eccd3bbf95f9210b，25项专项控制与164项Linux CPU/CI回归通过，跳过0项。CI：https://github.com/lizuoshuo-lab/shuolee-portrait/actions/runs/40870556238。补丁回传后本地25项通过由主任务复核。此记录不构成00服务器GPU或实拍质量验收。'
])
for name in ['torch-consumer-result-completed.json','torch-consumer-gpu/result.json','torch-consumer-gpu/acceptance.json']:
 p=D/name
 c['notes_required'].append('来源：'+str(p)+'，SHA256 '+sha(p))
(B/(tag+'-content-final.json')).write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
s=(B/'update_deck_free_transport123_gpu.mjs').read_text(encoding='utf-8')
s=s.replace("tag='free-transport123-gpu'","tag='"+tag+"'")
s=s.replace("const prior=await read(b+'/free-transport123-native-deck-receipt.json');","const prior=await read(b+'/free-transport123-gpu-deck-receipt.json');")
s=s.replace("b+'/free-transport123-gpu-content-final.json'","b+'/"+tag+"-content-final.json'")
s=s.replace('8ab6d82f1cde620a2da0826f4f78484ed6ca1e8201206e8c6f983862de9f6433','a10b4f67907a585a7aa68bf9bc68881ed553b457d08aa8327dd6022918c2c6da')
start=s.index("replace('自由衣物高斯的逐帧一致性'")
end=s.index('slide.speakerNotes.text=',start)
s=s[:start]+"replace('11次前向均完成，1次反向完成。',content.body);\nreplace('未加载实拍人物，未做拟合、高斯训练或PLY导出。',content.footer);\nconst tables=last.elements.filter(e=>e.kind==='table');\nif(tables.length!==1)throw Error('Expected preserved native table');\n"+s[end:]
s=s.replace('02_有限视角重建优化路线_123自由衣物原生GPU验证_审阅.pptx','02_有限视角重建优化路线_123原生与GPU消费者分项验证_审阅.pptx')
s=s.replace('preserve_deck_free_transport123_gpu.py','preserve_deck_free_transport123_torch_consumer.py')
(B/'update_deck_free_transport123_torch_consumer.mjs').write_text(s,encoding='utf-8')
print(json.dumps({'new_tag':tag,'cpu_controls':cpu['tests_run'],'gpu_field_assertions':len(gpu['checks']),'scope':'bottom text and notes only'},ensure_ascii=False))
