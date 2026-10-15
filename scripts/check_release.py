"""Check staged release files without printing potential secret values."""
import re,subprocess
from pathlib import Path

patterns=[('private_key',re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
          ('github_token',re.compile(rb'gh[pousr]_[A-Za-z0-9]{24,}')),
          ('openai_token',re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{32,}')),
          ('aws_access_key',re.compile(rb'AKIA[0-9A-Z]{16}'))]
paths=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACM','-z']).decode().split('\0')
issues=[];count=0
for value in paths:
    if not value:continue
    p=Path(value);data=p.read_bytes();count+=1
    if p.suffix in ('.safetensors','.pt','.pth','.pkl','.mp4','.pem','.key'):issues.append((value,'unreviewed_large_or_private_asset'))
    if len(data)>10*1024*1024:issues.append((value,'new_asset_over_10MiB'))
    for name,pattern in patterns:
        if pattern.search(data):issues.append((value,name))
    if str(p).startswith(('src/dxy_mid/','configs/dxy_mid')) and b'/home/' in data:issues.append((value,'local_absolute_path_in_formal_runtime'))
print('Scanned staged files:',count)
for path,kind in issues:print(path,kind)
print('PASS' if not issues else 'FAIL')
raise SystemExit(bool(issues))
