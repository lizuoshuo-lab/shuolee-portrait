"""Use the existing CPU review, preserving previous grass record images."""
import argparse, inspect
import review_cross_subject_sh0124 as existing

def main(subject,step):
    if subject!='grass-selected-sh0-pipeline124':
        return existing.main(subject,step)
    source=inspect.getsource(existing.main)
    assert source.count('_跨人物SH0_')==1
    source=source.replace('_跨人物SH0_','_跨人物SH0_共享七帧124_')
    scope=dict(existing.__dict__)
    exec(compile(source,__file__+':record-filename-only','exec'),scope)
    scope['main'](subject,step)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('subject');p.add_argument('--step',type=int,required=True)
    args=p.parse_args();main(args.subject,args.step)
