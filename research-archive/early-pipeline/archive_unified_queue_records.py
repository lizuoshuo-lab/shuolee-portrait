"""Archive additional unified runs through the pinned 016/006 verifier."""

import argparse
import inspect
import json
from pathlib import Path
import sys

from PIL import Image,ImageOps

sys.path.insert(0,str(Path(__file__).resolve().parent))
import archive_unified_bound_records as original


ARCHIVER_SHA='9c262fe013378dfdb8a9147e47fb5a03bf392ef3be051bccfad9d77237b9f8c4'
EXPECTED={'001':24,'005':23,'008':19}
require,sha=original.require,original.sha


def contact_sheet(rows,panels,target):
    """Keep the verified four-column order and omit image labels."""
    pairs=sorted(zip(rows,panels,strict=True),key=lambda pair:pair[0]['source_frame_index'])
    require(len(pairs) in EXPECTED.values(),'Unexpected unified frame count.')
    width,height=900,325
    sheet=Image.new('RGB',(3*width,((len(pairs)+2)//3)*height),'#161616')
    for i,(_,path) in enumerate(pairs):
        x,y=i%3*width,i//3*height
        with Image.open(path) as source:
            tile=ImageOps.contain(source.convert('RGB'),(width-8,height-8),Image.Resampling.LANCZOS)
        sheet.paste(tile,(x+(width-tile.width)//2,y+(height-tile.height)//2))
    sheet.save(target,quality=95,subsampling=0)


def archive(subject,publish):
    path=Path(original.__file__).resolve()
    require(sha(path)==ARCHIVER_SHA,'Original verified archiver changed.')
    # Keep all original identity and publication checks. Adapt only frozen counts
    # and the fact that every additional subject copies a fresh original fit.
    source=inspect.getsource(original.archive)
    replacements=(
        ("result['frames']==24","result['frames']==EXPECTED_FRAMES"),
        ("len(order)==len(set(order))==24","len(order)==len(set(order))==EXPECTED_FRAMES"),
        ("fit['new_fitting_started']==(subject=='006')","fit['new_fitting_started'] is True"),
        ("_入选四合一总览_24帧.jpg","_入选四合一总览_{EXPECTED_FRAMES}帧.jpg"),
    )
    for old,new in replacements:
        require(source.count(old)==1,f'Cannot bind pinned archive rule: {old}')
        source=source.replace(old,new)
    expected=EXPECTED[subject]
    namespace=dict(vars(original),EXPECTED_FRAMES=expected,contact_sheet=contact_sheet)
    exec(compile(source,str(path),'exec'),namespace)
    return namespace['archive'](subject,publish)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=EXPECTED)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    result=archive(args.subject,args.publish)
    if args.publish:
        require(result['state']=='archived' and result['frames']==EXPECTED[args.subject]
                and len(result['images'])==result['frames']+1,'Incomplete exact-frame archive.')
        root=original.layout.root_for(args.subject)
        receipt=root/'record-delivery.json'
        if receipt.exists():
            require(original.layout.read(receipt)==result,'Existing delivery receipt differs.')
        else:
            original.layout.write(receipt,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
