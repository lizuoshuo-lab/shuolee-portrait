from pathlib import Path
repo=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee')
target=repo/'fit_convergence_from_scratch.py'
assert not target.exists()
src=(repo/'fit_motion_consistency_from_scratch.py').read_text('utf-8')
src=src.replace('smplx-motion-consistency-fresh-017','smplx-convergence-fresh-018')
src=src.replace('Fresh motion consistency comparison with a matched contact-loss control.','Fresh 5000-step optimization-budget comparison with a matched 1400-step milestone.')
src=src.replace("choices=['control','shared_lower','stable_feet']", "choices=['control','stable_feet']")
src=src.replace("'All three axis-angle arms have identical initial arrays. No arm uses VPoser.'", "'Both axis-angle arms have identical initial arrays. No arm uses VPoser.'")
src=src.replace('all three arms use identical lower-limb visibility weights.','both arms use identical lower-limb visibility weights.')
src=src.replace('Contact loss enabled identically for all three arms.','Contact loss enabled identically for both arms.')
src=src.replace("        if staged and step==750:", "        if step==1400:\n            with torch.no_grad():np.savez_compressed(ROOT/'milestone1400.npz',**arrays(forward()))\n        if staged and step==750:")
target.write_text(src,'utf-8')
runner='''"""Two independent 5000-step fits; short-run states checked only afterwards."""
import os,time,traceback
import fit_convergence_from_scratch as fit
import shuolee_common as c

def main():
    root=fit.EXPERIMENT;completed=[];start=time.monotonic()
    assert not (root/'run-state.json').exists()
    status=c.read(c.BASE/'work-status.json')
    status.update(updated=c.stamp(),local_training_running=True,current_phase='convergence_fresh_018',
                  current_decision='018从原始观测重新初始化，两组各5000步；只检验优化预算与脚底稳定约束。1400步中间结果仅在生成后与017比较，不读取旧参数初始化。')
    status['current_experiment'].update(current_training_running=True,status='convergence_running',accepted=False,latest_fitting_trial='018',latest_fitting_arms=2,latest_fitting_steps=5000)
    c.save(c.BASE/'work-status.json',status)
    try:
        c.save(root/'run-state.json',{'created':c.stamp(),'state':'preparing','pid':os.getpid(),'from_scratch':True})
        fit.prepare()
        for arm in ['control','stable_feet']:
            c.save(root/'run-state.json',{'created':c.stamp(),'state':'running','pid':os.getpid(),'arm':arm,'completed':completed,'steps_per_arm':5000})
            fit.fit_body(5000,arm);completed.append(arm)
        milestone={}
        for arm in completed:
            old=c.read(c.BASE/'current/smplx-motion-consistency-fresh-017'/arm/'fit-training-result.json')
            h=c.sha(root/arm/'milestone1400.npz');assert h==old['fit_sha256']
            milestone[arm]={'sha256':h,'exact_017_reproduction':True}
        c.save(root/'run-state.json',{'created':c.stamp(),'state':'complete_needs_review','completed':completed,'steps_per_arm':5000,
             'from_scratch':True,'quality_accepted':False,'gaussian_updates':0,'milestone1400':milestone,'seconds':time.monotonic()-start})
    except Exception:
        c.save(root/'run-state.json',{'created':c.stamp(),'state':'failed','completed':completed,'traceback':traceback.format_exc()});raise
    finally:
        status=c.read(c.BASE/'work-status.json');status.update(updated=c.stamp(),local_training_running=False)
        status['current_experiment']['current_training_running']=False
        status['current_experiment']['status']=c.read(root/'run-state.json')['state'];c.save(c.BASE/'work-status.json',status)

if __name__=='__main__':main()
'''
(repo/'run_convergence.py').write_text(runner,'utf-8')
print('Prepared 018: two fresh fits, 5000 steps each, matching 1400-step milestone checks.')
