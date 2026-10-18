"""DXY-MID command line."""
import argparse
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(prog='dxy-mid',description='Frozen V9.1 Geometry + native Mid Scale')
    sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('measure',help='Level 1: recompute measurements from image/mask manifest')
    run.add_argument('--manifest',type=Path,required=True);run.add_argument('--data-root',type=Path)
    run.add_argument('--output',type=Path,required=True);run.add_argument('--workers',type=int,default=4)
    run.add_argument('--render',action='store_true')
    run.add_argument('--config',type=Path,help='Validate frozen config; no parameter tuning')
    video=sub.add_parser('video',help='Level 2: raw video, INIT segmentation, C_joint/replay, measurement')
    video.add_argument('--videos',type=Path,required=True);video.add_argument('--model-dir',type=Path,required=True)
    video.add_argument('--frame-manifest',type=Path);video.add_argument('--output',type=Path,required=True)
    video.add_argument('--device',default='cpu');video.add_argument('--workers',type=int,default=4)
    video.add_argument('--render',action='store_true')
    args=p.parse_args(argv)
    if args.command=='measure':
        if args.config:
            import json
            from .pipeline import MeasurementConfig
            MeasurementConfig(**json.loads(args.config.read_text())).validate()
        from .batch import run_manifest
        run_manifest(args.manifest,args.output,data_root=args.data_root,workers=args.workers,render=args.render)
    elif args.command=='video':
        from .video import run_videos
        run_videos(args.videos,args.output,args.model_dir,frame_manifest=args.frame_manifest,
            device=args.device,workers=args.workers,render=args.render)

if __name__=='__main__':main()
