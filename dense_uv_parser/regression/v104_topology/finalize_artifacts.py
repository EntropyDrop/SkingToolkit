"""Record the reviewed v104 candidate and export the two feedback examples.

This does not change a deployment default or rerun the complete image dataset.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path('/home/ds/llms/SkingToolkitDev')
run = root / 'dense_uv_parser/runs/v104_topology_20260907'
candidate = run / 'candidate_step_5000'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')

training = read(run / 'training/status.json')
selection = read(run / 'training_selection.json')
review = read(candidate / 'full_review.json')
visual = read(candidate / 'visual_review.json')
test = read(run / 'locked_test_report.json')
manifest = read(candidate / 'manifest.json')
assert training['state'] == 'complete' and training['step'] == 6000
assert training['validation_warning_count'] == 0
assert selection['selected']['step'] == 5000 and selection['improved_validation']
assert review['scoped_checks_passed'] and review['repeat_count'] == 19
assert visual['status'] == 'completed_with_limitations'
checkpoint_sha = sha(candidate / 'parser.pt')
assert checkpoint_sha == manifest['checkpoint_sha256'] == review['checkpoint_sha256']
assert checkpoint_sha == test['models']['v104']['checkpoint_sha256']
assert checkpoint_sha == read(run / 'test_opened.json')['checkpoint_sha256']

exports = []
for name in ('99LBZPR14PEYLZ69', '254NJBPVUEM759NA'):
    row = review['cases'][name]
    source = Path(row['input'])
    uv = Path(row['output']) / 'pred_uv.png'
    destination = source.with_name(name + '_result_v104.png')
    preserved = [source] + sorted(source.parent.glob(name + '_result*.png'))
    hashes = {str(p): sha(p) for p in preserved}
    data = uv.read_bytes()
    if destination.exists():
        assert destination.read_bytes() == data, 'An existing v104 output differs; do not overwrite it.'
    else:
        destination.write_bytes(data)
    assert sha(destination) == sha(uv)
    assert all(sha(Path(p)) == digest for p, digest in hashes.items())
    exports.append({'input': str(source), 'input_sha256': sha(source), 'source_uv': str(uv),
                    'output': str(destination), 'output_sha256': sha(destination),
                    'preserved_existing_sha256': hashes})
write(candidate / 'export_manifest.json', {'checkpoint_sha256': checkpoint_sha, 'count': len(exports), 'exports': exports})

limitations = [
    'The sealed synthetic native test improves view-covered outer-head IoU, but whole-head IoU is effectively unchanged.',
    'Unobserved native head-bottom IoU falls from 43.94% to 37.56%; do not claim full-head generalization improvement.',
    'Some top and side/back hair fragments and blurred hat text remain in the real development review.',
    'The 43 real images are development/fit checks, not independent per-texel human ground truth.',
]
manifest.update(status='reviewed_candidate_with_known_limitations', version='v104', selected_step=5000,
                training_steps_completed=6000, validation_warning_count=0,
                scoped_review_passed=True, real_review_count=43, repeat_count=19,
                locked_test_count=112, test_result='mixed: visible-native and paired improvement; unobserved-native regression',
                limitations=limitations, exported_feedback_count=len(exports), release_promoted=False,
                source_commit_at_review=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                training_code_commit='3f4a846', reviewed_at=time.time())
write(candidate / 'manifest.json', manifest)
write(run / 'v104_model.json', {'version':'v104', 'status':manifest['status'], 'checkpoint':str(candidate/'parser.pt'),
                              'pipeline':str(candidate/'pipeline.json'), 'checkpoint_sha256':checkpoint_sha,
                              'pipeline_sha256':sha(candidate/'pipeline.json'), 'selected_step':5000,
                              'limitations':limitations, 'release_promoted':False})
write(run / 'driver_status.json', {'state':'complete_reviewed_with_limitations', 'training_status':training,
                                  'selected_checkpoint':str(candidate/'parser.pt'), 'scoped_checks_passed':True,
                                  'locked_test_result':manifest['test_result'], 'updated_at':time.time(),
                                  'release_promoted':False})
print(json.dumps({'status':manifest['status'], 'checkpoint_sha256':checkpoint_sha,
                  'exports':[r['output'] for r in exports], 'release_promoted':False}, indent=2))
