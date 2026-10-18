"""Raw video → JPEG95 → INIT mask → frozen C_joint or explicit frame replay."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from .segmentation import INITSegmentation
from .selection import Config, evaluate, select
from .candidates import _candidate_indices, _sample_indices
from .pipeline import sha256


def prepare_videos(video_manifest,output,model_dir,*,frame_manifest=None,device='cpu'):
    """videos.csv columns group,video,path,ring_mm; frame replay uses video key.

    With frame_manifest the already-frozen frame indices/source IDs are replayed
    without reselection, and image/mask hashes must match. Otherwise use exactly
    the approved C_joint quality rule. Missing selections remain explicit.
    """
    vm=Path(video_manifest);v=pd.read_csv(vm,keep_default_na=False)
    required={'group','video','path','ring_mm'}
    if not required<=set(v.columns):raise ValueError('videos.csv needs '+','.join(sorted(required)))
    if v.video.duplicated().any():raise ValueError('Video keys must be unique, not basename-only matching')
    frozen=pd.read_csv(frame_manifest,keep_default_na=False) if frame_manifest else None
    if frozen is not None:
        if frozen.source_id.duplicated().any():raise ValueError('Duplicate frozen source_id')
        if not set(frozen.video)<=set(v.video):raise ValueError('Frozen video keys missing from index')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use an empty video preparation directory')
    (output/'images').mkdir(parents=True,exist_ok=True);(output/'masks').mkdir(exist_ok=True)
    model=INITSegmentation(model_dir,device);cfg=Config();selected=[];scored=[];failures=[]
    for video in v.to_dict('records'):
        path=Path(video['path']);path=path if path.is_absolute() else vm.resolve().parent/path
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise FileNotFoundError('Cannot open video '+str(path))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not count>0 or not fps>0:raise ValueError('Invalid video frame count/fps')
        if frozen is None:
            schedule=[]
            for segment,indices in enumerate(_candidate_indices(count,5,.05),1):
                for ordinal,index in enumerate(_sample_indices(indices,24),1):
                    schedule.append(dict(group=video['group'],video=video['video'],ring_mm=video['ring_mm'],
                        segment_id=segment,frame_index=index,stem=f'{Path(video["video"]).stem}-raw{segment:02d}-{ordinal:02d}'))
        else:
            schedule=frozen[frozen.video==video['video']].to_dict('records')
            for row in schedule:
                if row['group']!=video['group'] or float(row['ring_mm'])!=float(video['ring_mm']):
                    raise ValueError('Frozen group/reference metadata differs from video index')
        video_hash=sha256(path)
        this=[]
        for row in schedule:
            index=int(row['frame_index']);cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
            if not ok:
                if frozen is not None:raise ValueError('Frozen frame decode failed '+str(index))
                failures.append(dict(video=video['video'],frame_index=index,reason='decode_failed'));continue
            identifier=row.get('source_id') or hashlib.sha256((video_hash+'|'+str(index)).encode()).hexdigest()
            ip=output/'images'/(identifier+'.jpg');mp=output/'masks'/(identifier+'.png')
            if not cv2.imwrite(str(ip),image,[cv2.IMWRITE_JPEG_QUALITY,95]):raise OSError('JPEG write failed')
            label=model.predict(ip)
            Image.fromarray(label,mode='L').save(mp)
            ih,mh=sha256(ip),sha256(mp)
            for key,value in [('image_sha256',ih),('mask_sha256',mh)]:
                if frozen is not None and row.get(key) and row[key]!=value:
                    raise ValueError('Raw replay '+key+' mismatch: '+row['stem'])
            record=dict(row,source_id=identifier,image_path='images/'+ip.name,mask_path='masks/'+mp.name,
                image_sha256=ih,mask_sha256=mh,timestamp_sec=index/fps)
            if frozen is None:record.update(evaluate(image,label,cfg))
            this.append(record);scored.append(record)
        cap.release()
        if frozen is not None:selected.extend(this)
        else:
            previous=-np.inf
            for segment in range(1,6):
                eligible=[z for z in this if z['segment_id']==segment and z['timestamp_sec']-previous>=cfg.min_interval_sec-1e-8]
                chosen=select(eligible,'C_joint',cfg)
                if chosen:previous=max(z['timestamp_sec'] for z in chosen)
                selected.extend(chosen)
                if len(chosen)<cfg.per_segment:failures.append(dict(video=video['video'],segment=segment,reason='insufficient_C_joint_quality_or_time_spaced_candidates',selected=len(chosen)))
    pd.DataFrame(scored).to_csv(output/'candidate_inputs.csv',index=False)
    cols=['source_id','group','video','stem','ring_mm','frame_index','timestamp_sec','image_path','mask_path','image_sha256','mask_sha256']
    pd.DataFrame(selected,columns=cols).to_csv(output/'frames.csv',index=False)
    (output/'preparation.json').write_text(json.dumps(dict(mode='frozen_frame_replay' if frozen is not None else 'C_joint',
        videos=len(v),frames=len(selected),failures=failures,model_sha256=sha256(Path(model_dir)/'model.safetensors'),
        GT_read=False,history_prediction_fallback=0),indent=2))
    return output/'frames.csv'


def run_videos(video_manifest,output,model_dir,*,frame_manifest=None,device='cpu',workers=4,render=False):
    output=Path(output);manifest=prepare_videos(video_manifest,output/'prepared',model_dir,
        frame_manifest=frame_manifest,device=device)
    if not len(pd.read_csv(manifest)):raise ValueError('No selected frames; see preparation.json')
    # CUDA/OpenCV initialization can leave native locks in a forked child.
    # Start the existing CPU CLI in a fresh interpreter; measurement mathematics
    # and its tested process-pool protocol remain exactly the same.
    destination=output/'measurements'
    command=[sys.executable,'-m','dxy_mid','measure','--manifest',str(manifest.resolve()),
             '--output',str(destination.resolve()),'--workers',str(workers)]
    if render:command.append('--render')
    subprocess.run(command,check=True)
    return tuple(pd.read_csv(destination/name) for name in ['frames.csv','groups.csv','videos.csv'])
