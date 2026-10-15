"""Maintainer-only extraction of the reachable verified numerical source closure.

Run against the original research workspace. Generated modules have ordinary
relative imports and do NOT load that workspace at runtime. No predictions read.
"""
import argparse, ast, dis, hashlib, inspect, json, sys, textwrap, types
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('research_root',type=Path);args=p.parse_args()
ROOT=args.research_root.resolve();DEST=Path(__file__).resolve().parents[2]/'src/dxy_mid/_engine'
sys.path.insert(0,str(ROOT/'experiments/v9_1_short_Q_mid_median_20261014/scripts'))
import v91_context as c
import q_arc_filter

namespaces={};pending=[];sources=[];overrides={};special={}
def ns(g,hint):
 key=id(g)
 if key not in namespaces:
  name=hint.replace('.','_').replace('-','_')
  used={n['name'] for n in namespaces.values()}
  while name in used:name+='_slice'
  namespaces[key]=dict(name=name,g=g,defs={},bindings={},imports={})
 return namespaces[key]
def local(obj):
 file=getattr(obj,'__file__',None) or (obj.__globals__.get('__file__') if inspect.isfunction(obj) else None) or (inspect.getsourcefile(obj) if inspect.isfunction(obj) or inspect.isclass(obj) else None)
 return file is not None and str(Path(file).resolve()).startswith(str(ROOT))
def binding(g,key,value):
 n=ns(g,g.get('__name__','helpers'));b=n['bindings']
 if key in b:return
 b[key]=None
 if isinstance(value,types.ModuleType):
  if local(value):
   other=ns(vars(value),value.__name__);b[key]=('module',other['name'])
  else:b[key]=('external_module',value.__name__)
 elif inspect.isfunction(value) and local(value):
  other=register(value);b[key]=('function',other['name'],value.__name__)
 elif inspect.isclass(value) and local(value):
  other=register(value);b[key]=('function',other['name'],value.__name__)
 elif inspect.isfunction(value) or inspect.isclass(value) or inspect.isbuiltin(value):
  module=value.__module__;name=value.__name__
  if module=='builtins':b[key]=('literal',name)
  else:b[key]=('external_symbol',module,name)
 else:
  import numpy as np
  if getattr(value,'__module__',None)=='typing':
   b[key]=('external_symbol','typing',key);return
  if isinstance(value,np.ndarray):s='np.array('+repr(value.tolist())+', dtype='+repr(str(value.dtype))+')';binding(g,'np',np)
  elif isinstance(value,Path):raise RuntimeError('Reachable path constant: '+key+'='+str(value))
  else:s=repr(value)
  try:ast.parse(s,mode='eval')
  except SyntaxError:raise RuntimeError('Unsupported binding '+key+' '+s)
  b[key]=('literal',s)

def walk_code(code):
 out={i.argval for i in dis.get_instructions(code) if i.opname in ('LOAD_GLOBAL','LOAD_NAME')}
 for sub in code.co_consts:
  if isinstance(sub,types.CodeType):out|=walk_code(sub)
 return out

def register(obj):
 g=obj.__globals__ if inspect.isfunction(obj) else vars(sys.modules[obj.__module__])
 n=ns(g,obj.__module__)
 if obj.__name__ not in n['defs']:
  n['defs'][obj.__name__]=obj;pending.append((obj,n))
 return n

# Materialise the already-approved V8.2 single-block Boundary adaptation.
import v82_boundary, q_evidence_policy
overrides[id(q_evidence_policy.optimise43)]=q_evidence_policy.adapted_source
adapted=ast.parse(inspect.getsource(c.r.v81.repair_boundary))
for node in ast.walk(adapted):
 if isinstance(node,ast.If) and ast.unparse(node.test)=="distance[y, x] <= RULES['ring_clearance_px']":
  node.body=ast.parse('if not near_ring_evidence(rec,d,polyp,rm,q,check):\n checks.append(check)\n continue').body
ast.fix_missing_locations(adapted)
overrides[id(v82_boundary.FROZEN_ADAPTED_REPAIR)]=ast.unparse(adapted)

# Explicitly materialise the exact AST prefixes already used by V9.1. This
# removes runtime source-file slicing, cv2/open mocks, and fake file paths.
v2=c.r.legacy.v2
orig=inspect.getsource(v2.process_view);tree=ast.parse(orig).body[0]
body=[]
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Tuple) and any(isinstance(x,ast.Name) and x.id=='fitA' for x in t.elts) or isinstance(t,ast.Name) and t.id=='fitA' for t in node.targets):break
 body.append(node)
start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ringinfo' for t in n.targets))
body=ast.parse("original=labels\nrings=[dict(inner_ellipse=geometry['inner'],outer_ellipse=geometry['outer'])]\nr=dict(stem=source_id,group_key='current_frame')").body+body[start:]
ret='dict(A=A,B=B,band=band,center=center,rr=rr,crop=(x0,y0,x1,y1),components=comps,edges=edges,candidate_A_source="broad_outside" if broad_out else "all_substantial")'
fn=ast.FunctionDef(name='process_candidates',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ['labels','geometry','source_id']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body+[ast.Return(ast.parse(ret,mode='eval').body)],decorator_list=[])
ast.fix_missing_locations(fn)
g=dict(vars(v2));g['__name__']='v2_candidates';exec(compile(ast.Module(body=[fn],type_ignores=[]),str(inspect.getfile(v2)), 'exec'),g)
special['process_candidates']=g['process_candidates'];overrides[id(g['process_candidates'])]=ast.unparse(fn)

edge=c.r.exact_v2.EDGE
ft=ast.parse(inspect.getsource(edge)).body[0];b=[]
for node in ft.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fit' for t in node.targets):break
 b.append(node)
ft.body=b+[ast.Return(ast.parse('(visible,removed,geom,extent)',mode='eval').body)]
overrides[id(edge)]=ast.unparse(ft)

obj=c.r.exact_v2.boundary
s=inspect.getsource(obj)
start=s.index(' payload=');end=s.index(';x0,y0,x1,y1=')
s=s[:start]+" p=process_candidates(frame.labels,ring.geometry,frame.source_id)"+s[end:]
overrides[id(obj)]=s
obj.__globals__['process_candidates']=special['process_candidates']
# The old bytecode contains deleted IO bindings. Use the migrated function's
# bytecode for dependency discovery while keeping its original source hash.
newns=dict(obj.__globals__);exec(compile(s,inspect.getsourcefile(obj),'exec'),newns);replacement=newns['boundary'];overrides[id(replacement)]=s
c.r.exact_v2.boundary=replacement

roots=dict(FrameInput=c.r.legacy.FrameInput,RingResult=c.r.RingResult,
 ring=c.r.v81.existing_ring_check,boundary=c.approved.obs.boundary_native,
 cross_ring=c.approved.run,make_data=c.approved.make_data,
 configure=c.r.configs84,choose=c.r.choose_distance_body,
 audit=c.r.audit_distance_variants,baseline=c.r.current_baseline,
 axes=c.r.ellipse_axes,holes=c.frozen_ring_preprocessing.run,
 short_Q=q_arc_filter.apply,short_Q_analyze=q_arc_filter.analyze,
 scale=c.fixed_scale.run,fusion=c.fusion.aggregate)
# Configurations swap the data builder exactly as the approved V8.5 wrapper.
# These are global function bindings, not external files.
for obj in roots.values():register(obj)
# Dynamic namespace lookups are explicit here, making the migration auditable.
for obj in [c.r.make_data84,c.r.q_evidence_policy.run_policy] if hasattr(c.r,'q_evidence_policy') else [c.r.make_data84]:register(obj)
while pending:
 obj,n=pending.pop(0);source=textwrap.dedent(overrides[id(obj)] if id(obj) in overrides else inspect.getsource(obj))
 if 'v81.sha(v81.__file__)' in source:
  source=source.replace('v81.sha(v81.__file__)',repr(hashlib.sha256(Path(c.r.v81.__file__).read_bytes()).hexdigest()))
 tree=ast.parse(source)
 # Future annotations avoid eager dependencies on the original import scaffolding.
 compiled=compile(source,'<relocated>','exec')
 codes=[co for co in compiled.co_consts if isinstance(co,types.CodeType)]
 keys=set().union(*(walk_code(co) for co in codes)) if codes else set()
 for dec in tree.body[0].decorator_list:
  keys|={x.id for x in ast.walk(dec) if isinstance(x,ast.Name)}
 for key in keys:
  if key not in ('__name__','__module__','__qualname__','__annotations__','__file__') and key in n['g']:binding(n['g'],key,n['g'][key])
 # Attribute references, including module-global monkeypatch targets, must be
 # present in the corresponding relocated module. Walk chains through modules.
 for node in ast.walk(tree):
  if not isinstance(node,ast.Attribute):continue
  parts=[];x=node
  while isinstance(x,ast.Attribute):parts.insert(0,x.attr);x=x.value
  if not isinstance(x,ast.Name) or x.id not in keys:continue
  val=n['g'].get(x.id)
  for attr in parts:
   if not isinstance(val,types.ModuleType) or not local(val):break
   if not hasattr(val,attr):break
   v=getattr(val,attr);binding(vars(val),attr,v);val=v
 n.setdefault('source',{})[obj.__name__]=source
 f=Path(obj.__globals__.get('__file__',inspect.getsourcefile(obj))) if inspect.isfunction(obj) else Path(inspect.getsourcefile(obj));sources.append(dict(module=n['name'],symbol=obj.__name__,source_file=str(f.relative_to(ROOT)),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),source_function_sha256=hashlib.sha256((overrides[id(obj)] if id(obj) in overrides else inspect.getsource(obj)).encode()).hexdigest(),relocated_function_sha256=hashlib.sha256(source.encode()).hexdigest(),adapted=id(obj) in overrides))

# Supply names accessed via function.__globals__ string keys in the two frozen
# override wrappers. Their object identities must be retained, not snapshotted.
for fn,key,value in [(c.r.configs84,'make_data84',c.r.make_data84)]:
 binding(fn.__globals__,key,value)
# All dependencies from this last binding should already have been traversed.
assert not pending,'Unexpanded dynamic dependency'
DEST.mkdir(parents=True,exist_ok=True)
for n in namespaces.values():
 external=[];literals=[];links=[]
 for key,b in n['bindings'].items():
  if b[0]=='external_module':external.append(f'import {b[1]} as {key}')
  elif b[0]=='external_symbol':external.append(f'from {b[1]} import {b[2]} as {key}')
  elif b[0]=='literal':literals.append(f'{key} = {b[1]}')
  elif b[0]=='module':links.append(f'from . import {b[1]} as {key}')
  elif b[0]=='function':
   if b[1]==n['name'] and b[2]==key:continue
   links.append(f'from .{b[1]} import {b[2]} as {key}')
 source='"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""\nfrom __future__ import annotations\n'+ '\n'.join(sorted(set(external)))+'\n'+ '\n'.join(literals)+'\n\n'+'\n\n'.join(n.get('source',{}).values())+'\n\n# Bind relocated dependencies after defining functions (including override hooks).\n'+'\n'.join(sorted(set(links)))+'\n'
 compile(source,str(DEST/(n['name']+'.py')),'exec')
 (DEST/(n['name']+'.py')).write_text(source.rstrip()+'\n')
exports=[]
for key,obj in roots.items():exports.append(f'from .{namespaces[id(obj.__globals__) if inspect.isfunction(obj) else id(vars(sys.modules[obj.__module__]))]["name"]} import {obj.__name__} as {key}')
(DEST/'__init__.py').write_text('"""Reachable numerical closure; no experimental runtime imports."""\n'+'\n'.join(exports)+'\n')
import csv
with open(DEST.parents[2]/'docs/SOURCE_PROVENANCE.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=sources[0],lineterminator='\n');w.writeheader();w.writerows(sources)
print('Extracted',len(sources),'definitions across',len(namespaces),'modules')
