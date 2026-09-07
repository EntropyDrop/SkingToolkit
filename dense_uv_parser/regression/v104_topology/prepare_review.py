import json,hashlib,subprocess,os
from pathlib import Path
root=Path('/home/ds/llms/SkingToolkitDev');run=root/'dense_uv_parser/runs/v104_topology_20260907';teacher=root/'dense_uv_parser/runs/v103_generalization_20260907/candidate_risk_1000_beard_protected'
review=json.loads((teacher/'full_review.json').read_text());rows=[];seen=set()
for name,row in review['cases'].items():
 m=json.loads((Path(row['output'])/'manifest.json').read_text());seen.add(m['input_sha256'])
 rows.append({'name':name,'input':m['input'],'input_sha256':m['input_sha256'],'probability':m['foreground_probability']['path'],'role':row['role'],'old_output':row['output']})
fresh=[]
for p in (root/'dense_uv_parser/runs/v103_all_edited_20260907T040559Z/artifacts').glob('*/manifest.json'):
 m=json.loads(p.read_text())
 if m['input_sha256'] in seen:continue
 fresh.append({'name':Path(m['input']).stem,'input':m['input'],'input_sha256':m['input_sha256'],'probability':m['foreground_probability']['path'],'role':'new_real_development','directory':Path(m['input']).parent.name})
fresh.sort(key=lambda r:hashlib.sha256(('v104-new-real-fixed-selection'+r['input_sha256']).encode()).hexdigest())
selected=[]
for folder,count in [('SKING_DDJ_v54',6),('SKING_DDJ_v61',4),('SKING_DDJ_v66',2)]:selected += [r for r in fresh if r['directory']==folder][:count]
assert len(selected)==12
(run/'new_real_selection.json').write_text(json.dumps({'policy':'Fixed hash order and directory quotas; excludes all 31 previously reviewed inputs. No predictions inspected during selection.','rows':selected},indent=2)+'\n')
output=root/'dense_uv_parser/output_history/v104_new_real_v103_baseline_20260907'
args=['/home/ds/miniconda3/envs/sking-v61-worker/bin/python','dense_uv_parser/run_local.py','batch_accessories','--checkpoint',str(teacher/'parser.pt'),'--pipeline',str(teacher/'pipeline.json'),'--output-dir',str(output),'--foreground-probabilities',*[r['probability'] for r in selected],'--inputs',*[r['input'] for r in selected]]
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_DISABLE_PROGRESS_BARS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
with (run/'new_real_baseline.log').open('w') as log:subprocess.run(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
folders={json.loads(p.read_text())['input_sha256']:str(p.parent) for p in output.glob('*/manifest.json')}
for r in selected:r['old_output']=folders[r['input_sha256']]
rows+=selected;assert len(rows)==43 and len({r['input_sha256'] for r in rows})==43
(run/'review_inputs.json').write_text(json.dumps(rows,indent=2)+'\n');print('v104_review_inputs_ready',len(rows))
