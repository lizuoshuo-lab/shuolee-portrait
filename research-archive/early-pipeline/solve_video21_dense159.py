"""Compare fixed-K registration with a bounded map update at a PnP stall."""
import argparse
import inspect
from pathlib import Path

import global_fixedk121 as geometry
import solve_video21_background155 as prior
from update_leafwoman129_state import read


def replace(code, old, new):
    assert code.count(old)==1, old
    return code.replace(old,new)


def refreshed_solve():
    code=inspect.getsource(geometry.one)
    code=replace(code,'poses={};points={};ba=None;steps=[]',
                      'poses={};points={};ba=None;steps=[];map_updates=[]')
    code=replace(code,'                if not options or max(o[0] for o in options)<20: break',
        """                if not options or max(o[0] for o in options)<20:
                    advanced = not map_updates or map_updates[-1]['registered_count'] < len(poses)
                    if advanced and len(map_updates)<6 and len(poses)>=3 and len(points)>=20:
                        poses,points,refresh=bundle(K,poses,points,tracks,seed,deadline)
                        map_updates.append(dict(registered_count=len(poses),background_points=len(points),bundle=refresh))
                        attempted.clear()
                        print('UPDATED_MAP_AT_STALL',len(poses),len(points),flush=True)
                        continue
                    break""")
    code=replace(code,"registration=steps,bundle_adjustment=ba)","registration=steps,bundle_adjustment=ba,map_updates=map_updates)")
    exec(compile(code,'<existing-fixedK-plus-map-refresh>','exec'),geometry.__dict__)


def main(track_dir,mode):
    out_name='fixedk-dense159-'+mode
    if mode=='refresh':
        reference=track_dir.with_name('fixedk-dense159-reference')/'seated021/global-fixedk/result.json'
        previous=read(reference)
        assert previous['state']=='stopped_insufficient_3d_pnp_support'
        assert previous['registered_count']>=3 and previous['background_points']>=20
        refreshed_solve()
    code=inspect.getsource(prior.main)
    code=replace(code,"out = track_dir.with_name('fixedk-background155')","out = track_dir.with_name(output_name)")
    code=replace(code,"for b in range(a+1,min(a+3,len(frames)))",
                      "for gap in (1,2,4) for b in (a+gap,) if b<len(frames)")
    code=replace(code,"(Path(__file__), Path(original.__file__),", "(Path(__file__), Path(prior_script), Path(original.__file__),")
    code=replace(code,"cpu_seconds_per_subject=420, subject=subject,","cpu_seconds_per_subject=420, subject=subject, map_refresh_mode=mode, max_map_refresh=6 if mode=='refresh' else 0,")
    code=replace(code,"Short-interval background families support all 55 cameras in a common gauge and predict independent material families.",
                      "Real auxiliary source cameras bridge sparse overlap; updating the existing map at a stall may allow later PnP. Independent family validation required.")
    scope=dict(prior.__dict__,__file__=__file__,output_name=out_name,prior_script=prior.__file__,mode=mode)
    exec(compile(code,'<dense-background-input-only-adapter>','exec'),scope)
    scope['main'](track_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track_dir',type=Path)
    parser.add_argument('mode',choices=('reference','refresh'))
    args=parser.parse_args()
    main(args.track_dir,args.mode)
