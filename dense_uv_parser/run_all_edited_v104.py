#!/usr/bin/env python3
"""Recursively export *_edited.png to sibling *_result_v104.png files.

Default: create an immutable job snapshot and start a detached two-worker run.
Use --dry-run to list the scope, --status JOB to inspect, or --resume JOB after
interruption. Existing v104 outputs are backed up before this run replaces them.
"""
import argparse
from collections import Counter
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path('/home/ds/llms/SKING_DDJ_Dataset/entropydrop_website_generations')
MODEL = PROJECT / 'dense_uv_parser/runs/v104_topology_20260907/v104_model.json'
RUNS = PROJECT / 'dense_uv_parser/runs'
PYTHON = '/home/ds/miniconda3/envs/sking-v61-worker/bin/python'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def discover(root):
    result = []
    for source in sorted(root.rglob('*_edited.png')):
        if not source.is_file():
            continue
        # Export only within the selected root, even if it contains symlinks.
        source.resolve().relative_to(root)
        with Image.open(source) as im:
            if im.width % 2:
                raise ValueError('Two-view input width must be even: ' + str(source))
            im.verify()
        target = source.with_name(source.name[:-len('_edited.png')] + '_result_v104.png')
        if target.is_symlink():
            raise ValueError('Output must not be a symlink: ' + str(target))
        result.append({'source': str(source), 'source_sha256': sha(source), 'target': str(target),
                       'previous_result_sha256': sha(target) if target.exists() else None, 'backup': None})
    if not result:
        raise ValueError('No *_edited.png inputs found under ' + str(root))
    if len({r['target'] for r in result}) != len(result):
        raise ValueError('Output name collision')
    return result


def prepare(root, workers):
    candidate = json.loads(MODEL.read_text())
    if candidate['version'] != 'v104':
        raise ValueError('Expected the reviewed v104 model')
    for key in ('checkpoint', 'pipeline'):
        if sha(candidate[key]) != candidate[key + '_sha256']:
            raise ValueError('Reviewed model changed: ' + key)
    review = json.loads((Path(candidate['checkpoint']).parent / 'full_review.json').read_text())
    if not review['scoped_checks_passed'] or review['checkpoint_sha256'] != candidate['checkpoint_sha256']:
        raise ValueError('Candidate does not match its completed review')
    reference = json.loads((Path(review['cases']['beard']['output']) / 'manifest.json').read_text())
    for name, digest in reference['source_sha256'].items():
        if sha(PROJECT / 'dense_uv_parser' / name) != digest:
            raise ValueError('Source differs from reviewed v104 code: ' + name)
    pipeline = json.loads(Path(candidate['pipeline']).read_text())
    if pipeline['version'] != 'v104' or not pipeline['final_head_material_protect_inner_footprints']:
        raise ValueError('Wrong v104 pipeline')
    foreground = pipeline['foreground_release']
    foreground_path = PROJECT / foreground['checkpoint']
    if not foreground['passed'] or sha(foreground_path) != foreground['checkpoint_sha256']:
        raise ValueError('Paired foreground model changed')
    inputs = discover(root)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = RUNS / ('v104_all_edited_' + stamp)
    job.mkdir(exist_ok=False)
    for directory in ('code', 'logs', 'artifacts'):
        (job / directory).mkdir()
    archive = subprocess.check_output(['git', 'archive', '--format=tar', 'HEAD'], cwd=PROJECT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(job / 'code', filter='data')
    for name, digest in reference['source_sha256'].items():
        if sha(job / 'code/dense_uv_parser' / name) != digest:
            raise ValueError('Committed code differs from reviewed source: ' + name)
    for source, name in [(candidate['checkpoint'], 'parser.pt'), (candidate['pipeline'], 'pipeline.json'),
                         (foreground_path, 'foreground.pt')]:
        shutil.copy2(source, job / name)
    shutil.copy2(PROJECT / 'dense_uv_parser/regression/v104_batch/runner.py', job / 'runner.py')
    for row in inputs:
        if row['previous_result_sha256']:
            backup = job / 'previous_v104_results' / Path(row['target']).relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row['target'], backup)
            if sha(backup) != row['previous_result_sha256']:
                raise ValueError('Existing output changed during backup')
            row['backup'] = str(backup)
    priority = ['99LBZPR14PEYLZ69_edited.png', '254NJBPVUEM759NA_edited.png',
                '1ZW63H5Z8YNV57MT_edited.png', '1WQEQRUDRLMLVX3E_edited.png',
                'PU418PLGERMFC487_edited.png', '1ZDZB1UAMS1DEZLD_edited.png']
    inputs.sort(key=lambda r: (priority.index(Path(r['source']).name)
                              if Path(r['source']).name in priority else len(priority), r['source']))
    write(job / 'inputs.json', inputs)
    protected = {}
    for pattern in ('*_source.png', '*_result*.png'):
        for path in root.rglob(pattern):
            if path.is_file() and not path.name.endswith('_result_v104.png'):
                protected[str(path)] = sha(path)
    for row in inputs:
        if row['backup']:
            protected[row['backup']] = row['previous_result_sha256']
    write(job / 'protected_files.json', [{'path': p, 'sha256': digest} for p, digest in sorted(protected.items())])
    parser_sources = {p.name: sha(p) for p in (job / 'code/dense_uv_parser').glob('*.py')}
    pinned = {str(p.relative_to(job)): sha(p) for p in (job / 'code').rglob('*') if p.is_file()}
    pinned.update({name: sha(job / name) for name in ('parser.pt', 'pipeline.json', 'foreground.pt',
                                                    'runner.py', 'inputs.json', 'protected_files.json')})
    settings = {'version': 'v104', 'data_root': str(root), 'job_directory': str(job), 'workers': workers,
                'checkpoint_sha256': candidate['checkpoint_sha256'], 'pipeline_sha256': candidate['pipeline_sha256'],
                'foreground_sha256': foreground['checkpoint_sha256'], 'foreground_model_dir': foreground['model_dir'],
                'reviewed_candidate': candidate, 'parser_source_sha256': parser_sources, 'pinned_sha256': pinned,
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()}
    write(job / 'job.json', settings)
    write(job / 'progress.json', {'status': 'prepared', 'version': 'v104', 'total': len(inputs), 'completed': 0})
    return job


def launch(job, lock_fd):
    settings = json.loads((job / 'job.json').read_text())
    if settings['version'] != 'v104' or Path(settings['job_directory']) != job:
        raise ValueError('Not a v104 job directory')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1',
               HF_HUB_DISABLE_PROGRESS_BARS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
               V104_JOB_LOCK_FD=str(lock_fd))
    with (job / 'runner.log').open('a') as log:
        process = subprocess.Popen([PYTHON, str(job / 'runner.py')], cwd=PROJECT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock_fd,))
    record = {'pid': process.pid, 'job': str(job), 'version': 'v104',
              'total': len(json.loads((job / 'inputs.json').read_text())), 'workers': settings['workers']}
    write(job / 'launch.json', record)
    write(RUNS / 'v104_all_edited_latest.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--resume', type=Path, metavar='JOB')
    modes.add_argument('--status', type=Path, metavar='JOB')
    p.add_argument('--data-root', type=Path, default=DATA)
    p.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    opt = p.parse_args()
    if opt.status:
        print((opt.status.resolve() / 'progress.json').read_text())
        return
    if opt.dry_run:
        root = opt.data_root.resolve()
        rows = discover(root)
        print(json.dumps({'data_root': str(root), 'total': len(rows),
                          'by_directory': dict(Counter(str(Path(r['source']).parent.relative_to(root)) for r in rows)),
                          'existing_v104_outputs_to_backup': sum(r['previous_result_sha256'] is not None for r in rows),
                          'pattern': '*_edited.png', 'output_pattern': '*_result_v104.png'}, ensure_ascii=False, indent=2))
        return
    RUNS.mkdir(exist_ok=True)
    # Serialize launchers, then refuse a second job while a worker job is active.
    with (RUNS / 'v104_all_edited_launch.lock').open('a') as launcher:
        fcntl.flock(launcher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (RUNS / 'v104_all_edited.lock').open('a') as worker:
            try:
                fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('A v104 batch is already running; inspect runs/v104_all_edited_latest.json.')
            job = opt.resume.resolve() if opt.resume else prepare(opt.data_root.resolve(), opt.workers)
            if opt.prepare_only:
                print(json.dumps({'job': str(job), 'status': 'prepared'}, indent=2))
            else:
                # The child inherits the held lock; there is no launch gap in
                # which a second batch can begin exporting the same targets.
                launch(job, worker.fileno())


if __name__ == '__main__':
    main()
