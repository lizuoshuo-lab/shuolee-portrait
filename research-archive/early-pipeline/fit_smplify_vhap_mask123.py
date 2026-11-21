"""One all59 fullbody mask term around the unchanged VHAP joint-stage runner.

Keep RGB, landmark, prior, optimizer and schedule. Normalize absolute alpha
error by the fixed observed foreground area, not by the 4.1M-pixel canvas.
"""
import types
import torch
import fit_smplify_vhap_joint123 as joint
from run_official_gaussians123 import read,save,now

ORIGINAL_TRACKER=joint.make_tracker
joint.OUT=joint.ROOT/'smplify-vhap-mask'
joint.PLAN=joint.ROOT/'smplify-vhap-mask-plan.json'

def make_tracker(out):
    t,rows=ORIGINAL_TRACKER(out)
    original=t.compute_energy
    def energy(self,sample,step_i=None,stage=None):
        total,logs,*rest=original(sample,step_i=step_i,stage=stage)
        if stage is not None:
            pred=rest[-1]['pred_alpha'];target=sample['alpha_map'].to(pred)
            assert target.shape==pred.shape and float(target.sum())>0
            silhouette=self.cfg.w.photo*(pred-target).abs().sum()/target.sum().clamp_min(1)
            assert torch.isfinite(silhouette),'Nonfinite silhouette loss'
            logs['silhouette']=silhouette;total=total+silhouette;logs['total']=total
        return total,logs,*rest
    t.compute_energy=types.MethodType(energy,t)
    return t,rows

def main():
    plan=read(joint.PLAN)
    assert plan['changes']==['Add explicit foreground-area-normalized absolute alpha loss, weight=existing photo30.']
    joint.make_tracker=make_tracker
    joint.main()
    for filename in ('result.json','progress.json'):
        d=read(joint.OUT/filename)
        d.update(updated=now(),original_optimizer_loss_scheduler=False,
            original_optimizer_and_scheduler=True,original_energy_terms_preserved=True,
            added_loss='photo30 * sum(abs(pred_alpha-target_alpha)) / sum(target_alpha)',
            loss_adapter=str(__file__),scope='Full59 matched mask control. Requires actual whole-body/face/hand/foot review.')
        save(joint.OUT/filename,d)

if __name__=='__main__':main()
