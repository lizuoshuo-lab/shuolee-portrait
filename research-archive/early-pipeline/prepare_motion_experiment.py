from pathlib import Path

repo=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee')
source=(repo/'fit_floor_contact_from_scratch.py').read_text('utf-8')
target=repo/'fit_motion_consistency_from_scratch.py'
assert not target.exists()
source=source.replace('Fresh measured-floor contact comparison with explicit shoe-clearance slack.', 'Fresh motion consistency comparison with a matched contact-loss control.')
source=source.replace('smplx-floor-contact-fresh-016', 'smplx-motion-consistency-fresh-017')
source=source.replace("use_contact=arm=='floor_contact'", "use_contact=True\n    use_shared_lower=arm=='shared_lower'\n    use_stable_feet=arm=='stable_feet'")
source=source.replace("'Both axis-angle arms have identical initial arrays. Neither arm uses VPoser.'", "'All three axis-angle arms have identical initial arrays. No arm uses VPoser.'")
source=source.replace("'None: both arms use identical lower-limb visibility weights.'", "'None: all three arms use identical lower-limb visibility weights.'")
source=source.replace("'control_difference':'Only contact loss enabled; initialization and every other loss unchanged.'},", """'control_difference':'Contact loss enabled identically for all three arms.'},
        'motion_consistency':{'mode':arm,'start_step':250,'ramp_steps':200,'loss_weight':1.,
            'shared_lower_enabled':use_shared_lower,'stable_feet_enabled':use_stable_feet,
            'shared_lower_local_joint_indices':[0,1,3,4,6,7,9,10],
            'shared_root_rotation_matrix_normalizer':.075,'shared_lower_rotation_normalizer_rad':.05235987755982989,
            'shared_translation_normalizer_model_m':.01,'sole_world_position_normalizer_model_m':.01,
            'scope':'shared_lower penalizes across-frame global rotation/translation and local hip,knee,ankle,foot pose variance; stable_feet penalizes across-frame world-position variance of each sole vertex.',
            'assumptions':'Measured stable shoe tracks motivate a nearly planted-foot hypothesis. Not independent 3D GT and not a general motion model. Head/arm local motion remains enabled.',
            'same_initialization':True,'old_fitted_state_used':False},""")
needle="        assert torch.isfinite(loss), f'Non-finite loss at {step}'"
replacement="""        motion_loss=torch.zeros((),device='cuda')
        motion_ramp=min(1.,max(0.,(step-250)/200.))
        if use_shared_lower:
            # Common configuration is still optimized, rather than imposed as a
            # standing pose. Existing raw initialization is identical to control.
            root_center=s['rot'].mean(dim=0,keepdim=True)
            trans_center=s['trans'].mean(dim=0,keepdim=True)
            lower=s['body'].reshape(T,21,3)[:,[0,1,3,4,6,7,9,10]]
            lower_center=lower.mean(dim=0,keepdim=True)
            motion_loss=(huber((s['rot']-root_center)/.075).mean()+
                         huber((s['trans']-trans_center)/(.01*scale0)).mean()+
                         huber((lower-lower_center)/.05235987755982989).mean())/3.
        elif use_stable_feet:
            for indices in sole.values():
                points=s['vertices'][:,indices]/scale0
                motion_loss=motion_loss+.5*huber((points-points.mean(dim=0,keepdim=True))/.01).mean()
        if use_shared_lower or use_stable_feet:
            loss=loss+motion_ramp*motion_loss
        assert torch.isfinite(loss), f'Non-finite loss at {step}'"""
assert source.count(needle)==1
source=source.replace(needle,replacement)
source=source.replace("'floor_contact_loss':float(contact_loss.detach()),'floor_contact_ramp':contact_ramp,", "'floor_contact_loss':float(contact_loss.detach()),'floor_contact_ramp':contact_ramp,\n                'motion_consistency_loss':float(motion_loss.detach()),'motion_consistency_ramp':motion_ramp,")
source=source.replace("choices=['control','floor_contact']", "choices=['control','shared_lower','stable_feet']")
target.write_text(source,'utf-8')

runner=(repo/'run_floor_contact.py').read_text('utf-8')
runner=runner.replace('fit_floor_contact_from_scratch', 'fit_motion_consistency_from_scratch')
runner=runner.replace('floor_contact_fresh_016', 'motion_consistency_fresh_017')
runner=runner.replace("'floor_contact_running'", "'motion_consistency_running'")
runner=runner.replace("latest_fitting_trial='016'", "latest_fitting_trial='017',latest_fitting_arms=3")
runner=runner.replace("current_decision='016两组从原始输入和新优化器开始，各1400步；唯一新增项为实测地面下的脚底接触假设，允许1至7厘米模型尺度的鞋底余量。先检查脚部和全身，不新增高斯或部署00。'", "current_decision='017三组从相同原始输入和新优化器开始，各1400步；比较接触控制、整体与下身共享约束、脚底世界位置稳定约束。无新增高斯，不部署00。'")
runner=runner.replace("['control','floor_contact']", "['control','shared_lower','stable_feet']")
runner=runner.replace("current/smplx-leg-visibility-fresh-015/axis_visible/fit-training-result.json", "current/smplx-floor-contact-fresh-016/floor_contact/fit-training-result.json")
runner=runner.replace('control_exactly_reproduces_015B', 'control_exactly_reproduces_016_contact')
(repo/'run_motion_consistency.py').write_text(runner,'utf-8')

review=(repo/'review_floor_contact.py').read_text('utf-8')
review=review.replace('smplx-floor-contact-fresh-016','smplx-motion-consistency-fresh-017')
review=review.replace("ARMS=['control','floor_contact']", "ARMS=['control','shared_lower','stable_feet']")
review=review.replace("LABELS={'control':'A：无脚底接触约束','floor_contact':'B：加入脚底接触余量'}", "LABELS={'control':'A：接触控制','shared_lower':'B：整体与下身共享约束','stable_feet':'C：脚底世界位置稳定'}")
review=review.replace("assert all(np.array_equal(inits['control'][v],inits['floor_contact'][v]) for v in inits['control'])", "assert all(np.array_equal(inits['control'][v],inits[k][v]) for k in ARMS for v in inits['control'])")
review=review.replace('control_exactly_reproduces_015B','control_exactly_reproduces_016_contact')
review=review.replace('control_contact_initialization_identical','all_initialization_arrays_identical')
review=review.replace("'known_limit':'A/B differ only in floor-contact loss; initial arrays exactly identical. Clearance1-7cm and within-foot height slack12mm are bounded assumptions for apparently planted shoes, not measured anatomy. Floor orientation comes from SfM; not independent IMU gravity. Fits are not accepted based on minimized contact or2D residual alone.'", "'known_limit':'All initial arrays identical. Three arms share contact and all observation losses; B adds global/lower-body across-frame variance penalty, C adds sole world-position variance penalty. Stable-shoe assumption is limited to this sequence. Fitted temporal metrics and training errors are not independent GT.'")
(repo/'review_motion_consistency.py').write_text(review,'utf-8')
print('Prepared fresh 017 control/shared-lower/stable-feet experiment; no learned arrays copied.')
