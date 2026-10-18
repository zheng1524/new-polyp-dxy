"""Pure frame numerical call trace; reject historical structured-data reads."""
import argparse,json,sys,collections
from pathlib import Path
import pandas as pd
from dxy_mid import measure_files

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    z=pd.read_csv(a.manifest).iloc[0];base=a.manifest.resolve().parent
    ip=(base/z.image_path).resolve();mp=(base/z.mask_path).resolve();opened=set();calls=collections.Counter()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        mode=str(args[1]);path=Path(args[0]).resolve()
        if any(x in mode for x in ('w','a','+')):return
        opened.add(path)
        if path.suffix in ('.csv','.pkl','.json'):
            raise RuntimeError('Unexpected structured input read during frame computation: '+str(path))
    def profile(frame,event,arg):
        name=frame.f_code.co_filename
        if event=='call' and '/dxy_mid/' in name:calls[('dxy_mid/'+name.split('/dxy_mid/',1)[1],frame.f_code.co_name)]+=1
    sys.addaudithook(hook);sys.setprofile(profile)
    try:r=measure_files(ip,mp,inner_diameter_mm=z.ring_mm,source_id=z.source_id,stem=z.stem)
    finally:sys.setprofile(None)
    assert ip in opened and mp in opened
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:json.dump(dict(calls=[dict(file=k[0],function=k[1],count=v) for k,v in sorted(calls.items())],
        source_id=z.source_id,measurement_input_sha256=dict(image=r.audit['image_sha256'],mask=r.audit['mask_sha256']),
        unexpected_structured_reads=0,GT_read=False,historical_prediction_fallback=0),f,indent=2)
    print('Pure-frame IO audit PASS')
