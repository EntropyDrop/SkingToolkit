"""Run a checksum-pinned v101 release, including its trained foreground model."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,help='Explicit parser checkpoint override')
    p.add_argument('--release',type=Path,default=Path(__file__).with_name('v101_release.json'))
    p.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'output_history/v101')
    p.add_argument('--inputs',type=Path,nargs='+')
    p.add_argument('--foreground-probabilities',type=Path,nargs='+')
    p.add_argument('--save-tensors',action='store_true')
    p.add_argument('--pipeline',type=Path)
    opt=p.parse_args()
    release=json.loads(opt.release.read_text())
    if not release['passed']:raise ValueError('This release has not passed validation')
    checkpoint=opt.checkpoint.resolve() if opt.checkpoint else root/release['checkpoint']
    foreground=release['foreground']
    delta=root/foreground['checkpoint']
    checks=[(delta,foreground['checkpoint_sha256'])]
    if not opt.checkpoint:checks.append((checkpoint,release['checkpoint_sha256']))
    for path,expected in checks:
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:raise ValueError('Release checkpoint changed: '+str(path))
    pipeline=opt.pipeline.resolve() if opt.pipeline else root/release['pipeline']
    if not opt.pipeline and hashlib.sha256(pipeline.read_bytes()).hexdigest()!=release['pipeline_sha256']:
        raise ValueError('Release pipeline changed')
    entry=root/'dense_uv_parser/run_local.py'
    command=['/home/ds/miniconda3/envs/sking-v61-worker/bin/python',str(entry)]
    if opt.foreground_probabilities:
        command+=['batch_accessories','--foreground-probabilities',*[str(x.resolve()) for x in opt.foreground_probabilities]]
    else:
        command+=['foreground_batch','--foreground-checkpoint',str(delta),
                  '--foreground-model-dir',foreground['model_dir'],
                  '--probability-dir',str(root/'dense_uv_parser/cache/released_foreground')]
    command+=['--checkpoint',str(checkpoint),'--pipeline',str(pipeline),'--output-dir',str(opt.output_dir.resolve())]
    if opt.inputs:command+=['--inputs',*[str(x.resolve()) for x in opt.inputs]]
    if opt.save_tensors:command+=['--save-tensors']
    subprocess.run(command,cwd=root,check=True)


if __name__=='__main__':main()
