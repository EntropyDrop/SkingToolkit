"""Durable two-worker export of pinned v104 outputs beside edited source files."""
from pathlib import Path
from collections import deque,Counter
import datetime,fcntl,hashlib,json,os,signal,subprocess,sys,tempfile,time,traceback
from PIL import Image

JOB=Path(__file__).resolve().parent
PYTHON='/home/ds/miniconda3/envs/sking-v61-worker/bin/python'
PROJECT=Path('/home/ds/llms/SkingToolkitDev')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def valid_png(path):
    with Image.open(path) as im:
        if im.size!=(64,64) or im.mode!='RGBA':raise ValueError('Expected 64x64 RGBA: '+str(path))
        im.verify()
    with Image.open(path) as im:
        if not im.getchannel('A').getbbox():raise ValueError('Completely transparent output: '+str(path))

def atomic_export(artifact,target,previous_digest):
    valid_png(artifact);digest=sha(artifact)
    if target.exists() and sha(target) not in (previous_digest,digest):raise ValueError('Another writer changed target: '+str(target))
    if not target.exists() or sha(target)!=digest:
        fd,tmp=tempfile.mkstemp(prefix='.'+target.name+'.',suffix='.tmp',dir=target.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(artifact.read_bytes());f.flush();os.fsync(f.fileno())
            os.chmod(tmp,0o644);os.replace(tmp,target)
        finally:Path(tmp).unlink(missing_ok=True)
    return digest

def main():
    inherited=os.environ.pop('V104_JOB_LOCK_FD',None)
    lock=os.fdopen(int(inherited),'a') if inherited else (PROJECT/'dense_uv_parser/runs/v104_all_edited.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run_job()
    finally:
        lock.close()

def run_job():
    settings=json.loads((JOB/'job.json').read_text());inputs=json.loads((JOB/'inputs.json').read_text())
    config=json.loads((JOB/'pipeline.json').read_text());by_input={i['source']:i for i in inputs}
    if config.get('final_head_material_refine_steps')!=64:raise ValueError('Missing final v104 material refinement')
    for relative,digest in settings['pinned_sha256'].items():
        if sha(JOB/relative)!=digest:raise ValueError('Pinned asset changed: '+relative)
    for item in inputs:
        if item['backup']:
            backup=Path(item['backup'])
            if not backup.exists():raise ValueError('Missing prior result backup')
            if sha(backup)!=item['previous_result_sha256']:raise ValueError('Prior result backup changed')
    done={};failed={};records={};active=[];processes=[];seen=set();start=time.time()
    if (JOB/'results.jsonl').exists():
        journal=JOB/'results.jsonl'
        data=journal.read_bytes();offset=0
        for index,line in enumerate(data.splitlines(keepends=True)):
            try:row=json.loads(line)
            except (json.JSONDecodeError,UnicodeDecodeError):
                if offset+len(line)!=len(data):raise
                with journal.open('r+b') as f:f.truncate(offset)
                break
            records[row['source']]=row;offset+=len(line)
        if journal.stat().st_size and not journal.read_bytes().endswith(b'\n'):
            with journal.open('ab') as f:f.write(b'\n')
        for name,row in records.items():
            if row['status']=='complete' and Path(row['target']).exists() and sha(row['target'])==row['result_sha256'] and sha(name)==row['source_sha256']:
                valid_png(row['target']);done[name]=row
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Stopped by signal {signum}; resume this job to continue')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    progress={'status':'running','pid':os.getpid(),'started_at_utc':now(),'total':len(inputs),'initially_complete':len(done),'workers':settings['workers'],'checkpoint_sha256':settings['checkpoint_sha256'],'version':'v104','backup_count':sum(bool(i['backup']) for i in inputs)}
    def update(phase):
        progress.update(phase=phase,completed=len(done),failed=len(failed),remaining=len(inputs)-len(done)-len(failed),updated_at_utc=now(),elapsed_seconds=round(time.time()-start,1),active=[{'batch':b['label'],'pid':b['process'].pid,'count':len(b['items'])} for b in active]);write(JOB/'progress.json',progress)
    def record(row):
        with (JOB/'results.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        records[row['source']]=row
    def export(item,folder):
        manifest=json.loads((folder/'manifest.json').read_text())
        if not manifest['complete'] or manifest['input']!=item['source'] or manifest['input_sha256']!=item['source_sha256']:raise ValueError('Input provenance mismatch')
        if manifest['checkpoint_sha256']!=settings['checkpoint_sha256'] or manifest['pipeline']!=config:raise ValueError('Model/pipeline provenance mismatch')
        if manifest['source_sha256']!=settings['parser_source_sha256']:raise ValueError('Parser source snapshot mismatch')
        if manifest['foreground_provider']['adaptation_sha256']!=settings['foreground_sha256']:raise ValueError('Foreground model provenance mismatch')
        material=json.loads((folder/'final_head_material_refit.json').read_text())
        if not material.get('alpha_exact') or not material.get('body_exact'):raise ValueError('Final material stage changed geometry or body')
        if sha(item['source'])!=item['source_sha256']:raise ValueError('Input changed during inference')
        artifact=folder/'parser_pred_uv_simple_inpainting.png';valid_png(artifact);digest=sha(artifact);target=Path(item['target'])
        atomic_export(artifact,target,item['previous_result_sha256'])
        row={**item,'status':'complete','result_sha256':digest,'artifact':str(artifact),'checkpoint_sha256':settings['checkpoint_sha256'],'pipeline_sha256':settings['pipeline_sha256'],'completed_at_utc':now()}
        record(row);done[item['source']]=row;failed.pop(item['source'],None)
        print(f"completed={len(done)}/{len(inputs)} target={target}",flush=True)
    def collect():
        errors={}
        for mp in (JOB/'artifacts').glob('*/manifest.json'):
            if mp.parent.name.startswith('.') or mp in seen:continue
            m=json.loads(mp.read_text());name=m['input']
            if name in done:seen.add(mp);continue
            try:export(by_input[name],mp.parent);seen.add(mp)
            except Exception as e:errors[name]=str(e)
        return errors
    collect()  # Recover completed artifacts before allocating GPU workers.
    pending=[i for i in inputs if i['source'] not in done]
    batches=[pending[:8],pending[8:16]]+[pending[k:k+64] for k in range(16,len(pending),64)]
    queue=deque({'label':f'batch_{n:03d}','items':items,'retry':False} for n,items in enumerate(batches) if items)
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def launch(batch):
        log=(JOB/'logs'/(batch['label']+'.log')).open('a')
        args=[PYTHON,str(JOB/'code/dense_uv_parser/run_local.py'),'foreground_batch','--checkpoint',str(JOB/'parser.pt'),'--foreground-checkpoint',str(JOB/'foreground.pt'),'--foreground-model-dir',settings['foreground_model_dir'],'--probability-dir',str(PROJECT/'dense_uv_parser/cache/released_foreground'),'--pipeline',str(JOB/'pipeline.json'),'--output-dir',str(JOB/'artifacts'),'--inputs',*[i['source'] for i in batch['items']]]
        p=subprocess.Popen(args,cwd=JOB/'code',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(p);return {**batch,'process':p,'log':log}
    update('starting')
    try:
        while queue or active:
            while queue and len(active)<settings['workers']:active.append(launch(queue.popleft()))
            errors=collect()
            for batch in list(active):
                if batch['process'].poll() is None:continue
                active.remove(batch);batch['log'].close()
                for index,item in enumerate(batch['items']):
                    if item['source'] in done:continue
                    if not batch['retry']:queue.appendleft({'label':batch['label']+f'_retry_{index:03d}','items':[item],'retry':True})
                    else:
                        row={**item,'status':'failed','error':errors.get(item['source'],f"No completed output; exit={batch['process'].returncode}"),'at_utc':now()}
                        record(row);failed[item['source']]=row;print(json.dumps(row,ensure_ascii=False),flush=True)
            update('inference');time.sleep(3)
        update('final_audit');counts=Counter()
        for item in inputs:
            if sha(item['source'])!=item['source_sha256']:raise ValueError('Final edited input checksum changed')
            if item['source'] not in done:continue
            row=done[item['source']]
            if sha(item['source'])!=item['source_sha256'] or sha(item['target'])!=row['result_sha256']:raise ValueError('Final input/result checksum changed')
            valid_png(item['target']);counts[str(Path(item['source']).parent.relative_to(settings['data_root']))]+=1
        for item in json.loads((JOB/'protected_files.json').read_text()):
            if sha(item['path'])!=item['sha256']:raise ValueError('Original source or prior-version result changed: '+item['path'])
        report={'status':'complete' if len(done)==len(inputs) else 'completed_with_errors','inputs':len(inputs),'verified_outputs':len(done),'failed':len(failed),'failures':list(failed.values()),'by_directory':dict(counts),'all_edited_inputs_unchanged':True,'protected_files_unchanged':True,'checkpoint_sha256':settings['checkpoint_sha256'],'pipeline_sha256':settings['pipeline_sha256'],'output_format':'64x64 RGBA','blank_outputs':0,'finished_at_utc':now(),'scope':'File/provenance/format audit; not a manual semantic review of every prediction.'}
        write(JOB/'final_audit.json',report);progress.update(status=report['status'],verified=len(done),finished_at_utc=now());update('finished');write(JOB/'summary.json',{**progress,'failures':list(failed.values())})
        return 0 if not failed else 2
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)

if __name__=='__main__':
    try:sys.exit(main())
    except BaseException:
        error=traceback.format_exc();(JOB/'fatal_error.txt').write_text(error)
        p=JOB/'progress.json';state=json.loads(p.read_text()) if p.exists() else {};state.update(status='failed',error=error,updated_at_utc=now());write(p,state);raise
