from pathlib import Path
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';src=(S/'compare_enerf_hand_fine.py').read_text(encoding='utf8')
src=src.replace('dynamic-human-enerf-hand-fine-085','dynamic-human-enerf-hand-time-scope-088').replace("['parent','fine_photo','fine_flow']","['parent','single_time','multi_time']").replace('fine_photo','single_time').replace('fine_flow','multi_time').replace('flow_vs_photo_db','multi_vs_single_db')
src=src.replace("roots={'parent':S/'local_bspline',**{b:T/b for b in branches[1:]}}","roots={b:T/b for b in branches};checkpoints={'parent':S/'local_bspline/resume.pt',**{b:T/b/'resume.pt' for b in branches[1:]}}")
src=src.replace("['Original','Parent model','Fine photo','Fine + motion']","['Original','Parent model','Fit frame732','Fit seven times']")
src=src.replace("read(S/'input-audit.json')['records']","read(T/'evaluation-protocol.json')['records']").replace('len(reports[\'parent\'])==120','len(reports[\'parent\'])==134')
src=src.replace("full={(r['camera'],r['frame']) for r in plan['full_person']}","full={(r['camera'],r['frame']) for r in plan['full_person']}|{(f'{c:02d}',732) for c in range(18)}")
src=src.replace("read(T/'target-hand-review-plan.json')","read(L/'outputs/dynamic-human-enerf-hand-fine-085/target-hand-review-plan.json')")
needle="before={b:sha(roots[b]/'resume.pt') for b in branches}"
repl="""for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records']:
 if r['frame']==732:detail.setdefault((r['camera'],r['frame']),[]).append(dict(name='same_time_hand',bbox=r['crop']))
def source(c,f):return (T if inputs[c,f].get('input_location')=='088' else S)/'native-input'/f'camera{c}-frame{f:06d}.npz'
def prediction_dir(b,c,f):return S/'local_bspline/native-evaluation' if reports[b][c,f]['reused079parent_prediction'] else roots[b]/'native-evaluation'
before={b:sha(checkpoints[b]) for b in branches}"""
assert needle in src;src=src.replace(needle,repl)
src=src.replace("ip=S/'native-input'/f'{stem}.npz'","ip=source(c,f)").replace("ep=roots[b]/'native-evaluation'","ep=prediction_dir(b,c,f)").replace("np.load(S/'native-input'/f'{stem}.npz')","np.load(source(c,f))").replace("np.load(roots[br]/'native-evaluation'/f'{stem}-prediction.npz')","np.load(prediction_dir(br,c,f)/f'{stem}-prediction.npz')")
src=src.replace("['train','validation_time','validation_camera']","['train','validation_time','validation_camera','validation_camera_same_time']")
src=src.replace('len(rows)==120 and pngchecks==720','len(rows)==134 and pngchecks==804').replace("sha(roots[b]/'resume.pt')==before[b]","sha(checkpoints[b])==before[b]")
src=src.replace("a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a)","a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in ['camera','background']) and np.all(a['frame']==732) and set(b['frame'])==set(range(720,733,2))")
src=src.replace("out=dict(status='120contexts_3models_720PNGs_independently_verified_visual_review_pending'","same_time={label:{b:float(np.mean([r['metrics'][b]['foreground']['psnr_db'] for r in rows if r['frame']==732 and (r['camera'] in ['00','06','12'])==other])) for b in branches} for label,other in [('known',False),('other',True)]}\nout=dict(status='134contexts_3models_804PNGs_independently_verified_visual_review_pending',same_time_foreground_psnr_db=same_time")
src=src.replace('identical_initialization_and_sampling=True','identical_initialization_cameras_background=True,training_times_deliberately_different=True')
src=src.replace("scope='Same-scene development. Original079eval RGB/mattes; targethandadditionalwindows reused unchanged from084. Nojointstaticbackground ornewsceneacceptance.'","scope='Same-scene frozen-motion shape fit diagnostic. Single-time branch trainedonly732,notcontinuousmodel. Original120plus14contexts732; 3newothercamera732photographsrectified identically. Nojointstaticbackground ornewsceneacceptance.'")
compile(src,'compare_enerf_hand_time_scope.py','exec');(S/'compare_enerf_hand_time_scope.py').write_text(src,encoding='utf8');print('134context independent comparison prepared')
