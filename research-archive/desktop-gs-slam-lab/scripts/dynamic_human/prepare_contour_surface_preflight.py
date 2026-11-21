from pathlib import Path
import json,hashlib
L=Path(__file__).resolve().parents[2];D=Path(__file__).parent;O=L/'outputs/dynamic-human-enerf-contour-surface-103';p=O/'protocol.json';s=json.loads(p.read_text());
for k in ['MANO_sha256','reset_mask_sha256','paired_difference','selection_result']:s.pop(k,None)
s['pose']='Fixed102branch-specificcontinuousMANOposeandtranslation;pose_deltaandtranslation_deltaremainexactzero.'
s['regularization']='0.0003mean(tanh(offset)^2)+0.00001mean(tanh(scale_delta)^2); fixedposecorrectionzero.'
s['acceptance']='134originalnativecontextsand26consecutivehandviews; compare103pairand079/100historicalreferences. Checkactualskintexture,fingerseparation,toy/armdamageandheldoutviewstogether. Contourcoverageisnotclarity.'
s['reset_mask_sha256']=hashlib.sha256((O/'context-reset-mask.npy').read_bytes()).hexdigest();p.write_text(json.dumps(s,indent=2)+'\n')
s=(D/'verify_hand_visibility_initialization.py').read_text();s=s.replace('dynamic-human-enerf-hand-visibility-100','dynamic-human-enerf-contour-surface-103').replace("['narrow_reset','context_reset']","['prior_control','contour_prior']")
old="ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(O/'protocol.json')['initial_color'],True);reset=torch.from_numpy(np.load(O/('reset-mask.npy' if b=='narrow_reset' else 'context-reset-mask.npy'))).cuda()"
assert old in s;s=s.replace(old,"ManoGaussianSurface(L/read(O/'protocol.json')['priors'][b]['path'],S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(O/'protocol.json')['initial_color'],False);reset=active")
s=s.replace("['Original','1937 reset initial','6330 reset initial']","['Original','Prior control initial','Contour prior initial']")
old="rr,al,_=render_hand_surface(pp,old,hand,c,f,W,K,160,160);pred="
assert old in s;s=s.replace(old,"reference=old.render(pp,c,f,W,K,160,160,2);disabled=render_hand_surface(pp,old,hand,c,f,W,K,160,160,disable_new=True);assert all(torch.equal(reference[k],disabled[k]) for k in [0,1]);assert not hand.pose_delta.requires_grad and not hand.translation_delta.requires_grad\n  rr,al,_=render_hand_surface(pp,old,hand,c,f,W,K,160,160);pred=")
s+="\n(O/'smoke.json').write_text((D/'summary.json').read_text())\n"
(D/'verify_contour_surface_initialization.py').write_text(s,encoding='utf8');print('Fixed103protocolreferencesandpreparedactualCUDApreflight.')
