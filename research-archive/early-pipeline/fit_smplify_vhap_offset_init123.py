"""Restore the upstream offset initialization before the same mask/global run."""
import fit_smplify_vhap_mask123 as mask
from run_official_gaussians123 import read,save,now

mask.joint.OUT=mask.joint.ROOT/'smplify-vhap-offset-init'
mask.joint.PLAN=mask.joint.ROOT/'smplify-vhap-offset-init-plan.json'

def main():
    plan=read(mask.joint.PLAN)
    assert plan['restore_original_offset_initialization'] and plan['initialization_frame']=='frame_000033'
    mask.joint.make_tracker=mask.make_tracker
    mask.joint.main()
    for name in ('result.json','progress.json'):
        d=read(mask.joint.OUT/name)
        d.update(updated=now(),original_optimizer_loss_scheduler=False,
            original_optimizer_and_scheduler=True,original_energy_terms_preserved=True,
            added_loss='photo30 * sum(abs(pred_alpha-target_alpha)) / sum(target_alpha)',
            restored_stage='rgb_init_offset:500 steps at original lr_scale1; then global59x30 at original lr_scale0.1',
            loss_adapter=str(mask.__file__),scope='Same explicit mask and all59 initialization. Only restore the omitted upstream offset initialization stage. Quality unaccepted until review.')
        save(mask.joint.OUT/name,d)

if __name__=='__main__':main()
