"""Real pinned asset/source I/O, with the GPU-only dependency probe stubbed."""
from pathlib import Path
import json
import sys
from unittest.mock import patch

project=Path('/mnt/d/Projects/Shuolee')
base=Path('/mnt/d/Projects/Shuolee-validation-20261029/one-shot-reconstruction/official-deployment-cpu-check')
base.mkdir(exist_ok=False)
sys.path[:0]=[str(project/'src'),str(project/'scripts')]
from shuolee_portrait import official_deployment as deployment, official_smplifyx, runtime as rt
import prepare_assets

assets=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
source=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official')
manifest=prepare_assets.read(prepare_assets.MANIFEST)
selected=deployment.selected_manifest(manifest,deployment.PROFILE)
before={str(assets/r['path']):rt.sha(assets/r['path']) for r in selected['files']}
before.update({str(source/name):rt.sha(source/name) for name in deployment.official_pin()[1]})
forbidden=AssertionError('Network/copy/source cloning is forbidden in this real-input CPU check')
with patch.object(prepare_assets,'download',side_effect=forbidden) as download, \
     patch.object(prepare_assets,'copy_verified',side_effect=forbidden) as copying, \
     patch.object(deployment.subprocess,'run',side_effect=forbidden) as clone, \
     patch.object(deployment,'gpu_dependencies',return_value={'CPU_test_stub':True,'cuda_probe_executed':False}):
    resolved=prepare_assets.prepare(base/'runtime',[assets],profile=deployment.PROFILE)
    assert resolved==assets
    prepared_source=deployment.prepare_source(base/'runtime',explicit=source)
    profile=deployment.configure_deployment(resolved,base/'runtime/runs',base/'deployment-fixture.json',source)
    assert profile['doctor']['assets']['files']==7
    assert profile['doctor']['gvhmr_required'] is False
    options=deployment.fitting_options({'shape_policy':{'mode':'fit_reference'}},profile)
    official_smplifyx.adapted_entry_source(options['source_root'])
    assert options['shape_policy']=={'mode':'fit_reference'}
    download.assert_not_called();copying.assert_not_called();clone.assert_not_called()
for path,digest in before.items():assert rt.sha(path)==digest,path
result=dict(state='passed',scope='CPU real-input hashes and deployment routing; GPU dependency probe stubbed',
    actual_asset_files=7,actual_asset_bytes=sum(r['bytes'] for r in selected['files']),
    actual_source_files=7,official_commit=prepared_source['commit'],
    default_source_metadata_reaches_fitter=True,source_adapter_validation_passed=True,
    shared_shape_policy_unchanged=True,original_asset_and_source_hashes_unchanged=True,
    model_downloads=0,source_network_fetches=0,model_copies=0,GPU_calls=0,model_inference_calls=0,
    fitting_calls=0,gaussian_optimizer_steps=0,
    source_network_fetch_verified=False,server_verified=False,quality_accepted=False,
    fresh_video_end_to_end_verified=False,end_to_end_15min_accepted=False,
    deployment_fixture=str(base/'deployment-fixture.json'))
rt.save(base/'result.json',result)
print(json.dumps(result,indent=2))
