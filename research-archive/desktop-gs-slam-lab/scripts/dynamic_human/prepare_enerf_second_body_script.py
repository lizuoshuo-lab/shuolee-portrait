"""Derive the fixed-budget body fitter; rebase its canonical surface to frame726."""
from pathlib import Path
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human'
text=(S/'fit_enerf_rgb_body.py').read_text()
def once(a,b):
 global text
 assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b,1)
once('"""Two matched continuous SMPL motion hypotheses from training-only trajectories."""','"""New-interval continuous SMPL; fixed shape prior, refitted motion, canonical726."""')
once("SOURCE=LAB/'outputs/dynamic-human-enerf-rgb-pose-correction-074'","SOURCE=LAB/'outputs/dynamic-human-enerf-second-interval-079/motion'")
once("OUT=LAB/'outputs/dynamic-human-enerf-rgb-pose-correction-074/body'","OUT=LAB/'outputs/dynamic-human-enerf-second-interval-079/body'")
text=text.replace('estimated_offsets','fixed_clock').replace('742,759','718,735').replace('742..758','718..734').replace('744,757','720,733').replace('ts-750','ts-726').replace('times-750','times-726')
once("source='074 RGB-corrected cubic3Djoint trajectories from seven even trainingtimes. No RGB or reservedodd-timekeypoints added tofit.'","source='079RGB joint trajectories fitted from new720..732even times with074clock offsets frozen. Old060shape/scale andpose are initialization/prior only; no old continuous motion extrapolation.'")
once("canonical='Exact060referencepose, shape,scaleandvertices shared by bothbranches. Referenceworldcoordinates usedforGaussianbinding.'","canonical='Rebase to newly fitted physicalframe726. Shared by both future appearance branches; no old canonical mesh or appearance checkpoint reused.'")
once('targets evaluatedanalytically from062training-only trajectory','targets evaluated analytically from079training-only trajectory')
once("OUT/'reference.npz'","OUT/'initial-pose-prior.npz'")
once(" # Precompute exactforward at every existing raw image's capturetime; no imagepixels read here.",""" # Rebase material coordinates to the newly fitted physical frame726.
 with torch.no_grad():
  vref,jref,Mref,_,_=query(tensor([[1.,0.,0.,0.]]));v0=vref[:,0];j0=jref[:,0];M0=Mref[:,0];inv0=torch.linalg.inv(M0)
  identity_error=float(abs(M0@inv0-torch.eye(3,device='cuda')).max());assert identity_error<1e-4
  reference_bias_error=float(abs(v0-((M0@inv0)@v0[...,None])[...,0]).max());assert reference_bias_error<2e-5
 np.savez_compressed(OUT/'reference.npz',vertices=v0.cpu().numpy(),joints=j0.cpu().numpy(),linear=M0.cpu().numpy(),inverse_linear=inv0.cpu().numpy(),faces=model.faces,betas=old['betas'],scale=old['scale'],basepose=(basepose+posecoef[:,0].detach()*mask).cpu().numpy(),translation=(basetrans+transcoef[:,0].detach()).cpu().numpy(),reference_frame=np.array(726))
 # Precompute actual new capturetimes relative to this reference.
""")
once('reference_vertex_error_world=reference_error','initial_pose_prior_reproduction_error_world=reference_error,reference_physical_frame=726,reference_bias_max_error=reference_bias_error')
once("status='RGB_corrected_estimated_offset_body_fitted'","status='new_interval_body_fitted_canonical726_projection_review_pending'") if "status='RGB_corrected_estimated_offset_body_fitted'" in text else None
text=text.replace("status='RGB_corrected_fixed_clock_body_fitted'","status='new_interval_body_fitted_canonical726_projection_review_pending'")
(S/'fit_enerf_second_body.py').write_text(text)
(S/'run_enerf_second_body.sh').write_text('#!/bin/bash\nset -euo pipefail\nsource /opt/gs-slam-lab/surface-gs/env.sh\npython /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/fit_enerf_second_body.py\n',newline='\n')
print('Prepared fixed2500update body fit, with new canonical726; not trained yet.')
