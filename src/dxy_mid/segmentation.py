"""Frozen INIT model adapter. Optional torch stack; no binary-model substitute."""
from pathlib import Path
import hashlib
import sys
import numpy as np
from PIL import Image
from .pipeline import sha256
from ._model_manifest import MODEL_CODE_SHA256

WEIGHT_SHA256='981aa596e8906edc4457219581c606d4a5a22bfc6f7e4924d3d9a4b9aff64464'

class INITSegmentation:
    def __init__(self,model_dir,device='cpu'):
        import torch
        from transformers import AutoConfig, AutoModel
        from safetensors.torch import load_file
        root=Path(model_dir).resolve();weight=root/'model.safetensors'
        if not weight.exists():raise FileNotFoundError('Frozen INIT model.safetensors missing; see docs/DATA_MODEL.md')
        if sha256(weight)!=WEIGHT_SHA256:raise ValueError('Not the frozen INIT multiclass checkpoint')
        for relative,digest in MODEL_CODE_SHA256.items():
            code=root/relative
            if not code.is_file() or sha256(code)!=digest:
                raise ValueError('Not the frozen INIT architecture: '+relative)
        self.device=torch.device(device);self.torch=torch
        # Same loading procedure as verified polypseg.common, local only.
        previous=list(sys.path);sys.path.insert(0,str(root))
        try:
            config=AutoConfig.from_pretrained(str(root),trust_remote_code=True,local_files_only=True)
            if (config.out_channels,config.img_size,config.in_channels,config.inter_ch,config.backbone)!=(3,256,3,64,'efficientnet'):
                raise ValueError('Wrong frozen model architecture/configuration')
            model=AutoModel.from_config(config,trust_remote_code=True)
            model.load_state_dict(load_file(str(weight)),strict=True)
        finally:sys.path[:]=previous
        self.model=model.to(self.device).eval()

    def predict(self,image_path):
        torch=self.torch
        image=Image.open(image_path).convert('RGB');w,h=image.size
        # Exact historical preprocessing: PIL bilinear, RGB /255; no mean/std.
        arr=np.asarray(image.resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255.
        x=torch.from_numpy(np.transpose(arr,(2,0,1))).unsqueeze(0).to(self.device)
        with torch.no_grad():logits=self.model(pixel_values=x)['logits'].cpu()
        logits=torch.nn.functional.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False)
        return torch.argmax(logits,dim=1).squeeze(0).numpy().astype(np.uint8)
