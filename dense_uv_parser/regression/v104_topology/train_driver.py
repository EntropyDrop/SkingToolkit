import json,time,subprocess,os
from pathlib import Path
root=Path('/home/ds/llms/SkingToolkitDev');run=root/'dense_uv_parser/runs/v104_topology_20260907';py='/home/ds/miniconda3/envs/sking-v61-worker/bin/python'
def status(state,**kw):
 p=run/'driver_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'state':state,'pid':os.getpid(),'updated_at':time.time(),**kw},indent=2)+'\n');t.replace(p)
try:
 status('waiting_for_preparation')
 while True:
  p=run/'cache/status.json';s=json.loads(p.read_text()) if p.exists() else {}
  if s.get('state')=='failed':raise RuntimeError('Preparation failed: '+s.get('error',''))
  if s.get('state')=='complete':break
  time.sleep(20)
 assert json.loads((run/'smoke_relations_v2/status.json').read_text())['state']=='complete'
 # The preparation was started from v103-compatible code at 9b7adc5, before
 # optional v104 modules were added. Its frozen parent runs without a decoder.
 mp=run/'cache/manifest.json';m=json.loads(mp.read_text());m['source_sha256_at_completion']=m.pop('source_sha256');m['preparation_runtime_base_commit']='9b7adc5';m['preparation_runtime_note']='Frozen v102 parent under v103 deployed pipeline; prepare_v104.py was the sole uncommitted preparation change at launch. Optional final decoder changes during preparation do not run in this parent.';mp.write_text(json.dumps(m,indent=2)+'\n')
 args=[py,'dense_uv_parser/run_local.py','train_final_head_uv','--version','v104','--cache',str(run/'cache'),'--output-dir',str(run/'training'),'--steps','6000','--first-eval','200','--eval-every','1000','--batch-size','8','--decoder-revision','2','--robust-edits','--semantic-geometry','--edit-risk-weight','5','--topology-context','--mask-unknown-relations','--boundary-loss-weight','.5','--paired-every','4','--anchor-every','2','--native-fraction','.7','--learning-rate','0.00008','--init-checkpoint',str(root/'dense_uv_parser/runs/v103_generalization_20260907/candidate_risk_1000_beard_protected/parser.pt')]
 (run/'training_command.json').write_text(json.dumps(args,indent=2)+'\n');status('training_started',command=args)
 env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_DISABLE_PROGRESS_BARS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
 with (run/'training.log').open('w') as log:subprocess.run(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 status('training_complete_pending_quality_review',training_status=json.loads((run/'training/status.json').read_text()),release_promoted=False)
except BaseException as e:status('failed',error=repr(e));raise
